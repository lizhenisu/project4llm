from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from rag_core.config import RagConfig, _resolve_model_path
from rag_core.embeddings import post_json, resolve_device, resolve_torch_dtype, siliconflow_url
from rag_core.types import SearchHit


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[SearchHit], *, limit: int) -> list[SearchHit]: ...


class PassthroughReranker:
    def rerank(self, query: str, hits: list[SearchHit], *, limit: int) -> list[SearchHit]:
        del query
        return [replace(hit, rerank_score=hit.score) for hit in hits[:limit]]


class SiliconFlowReranker:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model_name: str,
    ) -> None:
        if not api_key:
            raise RuntimeError("SILICONFLOW_API_KEY must be configured for SiliconFlow rerank.")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model_name = model_name

    def rerank(self, query: str, hits: list[SearchHit], *, limit: int) -> list[SearchHit]:
        if not hits:
            return []
        payload = {
            "model": self._model_name,
            "query": query,
            "documents": [hit.text for hit in hits],
            "return_documents": False,
            "top_n": limit,
        }
        data = post_json(
            siliconflow_url(self._base_url, "/rerank"),
            api_key=self._api_key,
            payload=payload,
        )
        reranked: list[SearchHit] = []
        for result in data.get("results", []):
            index = int(result.get("index", -1))
            if index < 0 or index >= len(hits):
                continue
            score = result.get("relevance_score", result.get("score", 0.0))
            reranked.append(replace(hits[index], rerank_score=float(score)))
        return reranked[:limit]


class TransformersBGEReranker:
    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int,
        max_length: int,
        device: str,
        dtype: str,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._batch_size = max(1, batch_size)
        self._max_length = max(1, max_length)
        self._device = resolve_device(torch, device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_kwargs = {}
        torch_dtype = resolve_torch_dtype(torch, device=self._device, dtype=dtype)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **model_kwargs,
        )
        self._model.to(self._device)
        self._model.eval()

    def rerank(self, query: str, hits: list[SearchHit], *, limit: int) -> list[SearchHit]:
        if not hits:
            return []

        pairs = [(query, hit.text) for hit in hits]
        scores: list[float] = []
        with self._torch.no_grad():
            for start in range(0, len(pairs), self._batch_size):
                batch_pairs = pairs[start : start + self._batch_size]
                batch = self._tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt",
                )
                batch = {key: value.to(self._device) for key, value in batch.items()}
                logits = self._model(**batch).logits.squeeze(-1)
                batch_scores = logits.detach().cpu().tolist()
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
                scores.extend(float(score) for score in batch_scores)

        reranked = [
            replace(hit, rerank_score=float(score))
            for hit, score in zip(hits, scores, strict=True)
        ]
        return sorted(reranked, key=lambda hit: hit.rerank_score or 0.0, reverse=True)[
            :limit
        ]


def build_reranker(config: RagConfig) -> Reranker:
    if config.rerank_backend == "none":
        return PassthroughReranker()
    if config.rerank_backend == "siliconflow":
        return SiliconFlowReranker(
            base_url=config.siliconflow_base_url,
            api_key=config.siliconflow_api_key,
            model_name=config.rerank_model,
        )
    if config.rerank_backend == "bge":
        model_path = _resolve_model_path(config.rerank_model, ms_subdir="bge-reranker-v2-m3")
        return TransformersBGEReranker(
            model_path,
            batch_size=config.rerank_batch_size,
            max_length=config.rerank_max_length,
            device=config.model_device,
            dtype=config.model_dtype,
        )
    raise ValueError(
        f"Unsupported RAG_RERANK_BACKEND={config.rerank_backend!r}; use none/siliconflow/bge"
    )
