from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from rag_core.config import RagConfig
from rag_core.text_utils import lexical_overlap_score
from rag_core.types import SearchHit


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[SearchHit], *, limit: int) -> list[SearchHit]: ...


class LexicalReranker:
    """Small local reranker for smoke tests; production should use BGE reranker."""

    def rerank(self, query: str, hits: list[SearchHit], *, limit: int) -> list[SearchHit]:
        scored = [
            replace(hit, rerank_score=lexical_overlap_score(query, hit.text))
            for hit in hits
        ]
        return sorted(
            scored,
            key=lambda hit: (hit.rerank_score or 0.0, hit.score),
            reverse=True,
        )[:limit]


class TransformersBGEReranker:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()

    def rerank(self, query: str, hits: list[SearchHit], *, limit: int) -> list[SearchHit]:
        if not hits:
            return []

        pairs = [(query, hit.text) for hit in hits]
        with self._torch.no_grad():
            batch = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )
            batch = {key: value.to(self._device) for key, value in batch.items()}
            logits = self._model(**batch).logits.squeeze(-1)
            scores = logits.detach().cpu().tolist()
            if isinstance(scores, float):
                scores = [scores]

        reranked = [
            replace(hit, rerank_score=float(score))
            for hit, score in zip(hits, scores, strict=True)
        ]
        return sorted(reranked, key=lambda hit: hit.rerank_score or 0.0, reverse=True)[
            :limit
        ]


def build_reranker(config: RagConfig) -> Reranker:
    if config.rerank_backend == "bge":
        return TransformersBGEReranker(config.rerank_model)
    if config.rerank_backend == "lexical":
        return LexicalReranker()
    raise ValueError(f"Unsupported RAG_RERANK_BACKEND={config.rerank_backend!r}")

