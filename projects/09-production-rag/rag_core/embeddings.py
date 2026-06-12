from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rag_core.config import RagConfig, _resolve_model_path


class EmbeddingModel(Protocol):
    @property
    def dim(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    def encode(self, texts: list[str]) -> list[list[float]]: ...

    def tokenize(self, text: str) -> list[int]: ...

    def count_tokens(self, text: str) -> int: ...


class ImageEmbeddingModel(EmbeddingModel, Protocol):
    def encode_images(self, image_paths: list[Path]) -> list[list[float]]: ...


class HashEmbeddingModel:
    def __init__(self, *, model_name: str, dim: int) -> None:
        self._model_name = model_name
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]

    def tokenize(self, text: str) -> list[int]:
        tokens = lexical_tokens(text)
        return [stable_token_bucket(token, self._dim) for token in tokens]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        tokens = lexical_tokens(text)
        if not tokens:
            return vector
        for token in tokens:
            bucket = stable_token_bucket(token, self._dim)
            sign = 1.0 if stable_token_bucket(f"{token}:sign", 2) == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


class SiliconFlowEmbeddingModel:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model_name: str,
        dim: int,
        batch_size: int,
    ) -> None:
        if not api_key:
            raise RuntimeError("SILICONFLOW_API_KEY must be configured for SiliconFlow embeddings.")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model_name = model_name
        self._dim = dim
        self._batch_size = max(1, batch_size)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self._batch_size]))
        for vector in vectors:
            if len(vector) != self._dim:
                raise ValueError(
                    f"Embedding dim mismatch: expected {self._dim}, got {len(vector)}. "
                    "Set EMBEDDING_DIM to match the SiliconFlow model."
                )
        return vectors

    def tokenize(self, text: str) -> list[int]:
        tokens = lexical_tokens(text)
        return [stable_token_bucket(token, self._dim) for token in tokens]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {
            "model": self._model_name,
            "input": texts,
        }
        data = post_json(
            siliconflow_url(self._base_url, "/embeddings"),
            api_key=self._api_key,
            payload=payload,
        )
        items = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
        return [list(item["embedding"]) for item in items]


def zero_image_vector(config: RagConfig) -> list[float]:
    return [0.0] * config.image_embedding_dim


class TransformersBGEEmbeddingModel:
    def __init__(
        self,
        model_name: str,
        dim: int,
        *,
        model_path: str | None = None,
        batch_size: int,
        max_length: int,
        device: str,
        dtype: str,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._model_name = model_name
        self._dim = dim
        self._batch_size = max(1, batch_size)
        self._max_length = max(1, max_length)
        self._torch = torch
        self._device = resolve_device(torch, device)
        load_from = model_path or model_name
        self._tokenizer = AutoTokenizer.from_pretrained(load_from)
        model_kwargs = {}
        torch_dtype = resolve_torch_dtype(torch, device=self._device, dtype=dtype)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        self._model = AutoModel.from_pretrained(load_from, **model_kwargs)
        self._model.to(self._device)
        self._model.eval()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        with self._torch.no_grad():
            for start in range(0, len(texts), self._batch_size):
                batch_texts = texts[start : start + self._batch_size]
                self._raise_if_truncated(batch_texts, offset=start)
                batch = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                batch = {key: value.to(self._device) for key, value in batch.items()}
                outputs = self._model(**batch)
                token_embeddings = outputs.last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).float()
                pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
                pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.extend(pooled.detach().cpu().tolist())

        for vector in vectors:
            if len(vector) != self._dim:
                raise ValueError(
                    f"Embedding dim mismatch: expected {self._dim}, got {len(vector)}. "
                    "Set EMBEDDING_DIM to match the model."
                )
        return vectors

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def tokenize(self, text: str) -> list[int]:
        encoded = self._tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )
        return list(encoded["input_ids"])

    def _raise_if_truncated(self, texts: list[str], *, offset: int) -> None:
        encoded = self._tokenizer(
            texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        for index, input_ids in enumerate(encoded["input_ids"]):
            if len(input_ids) > self._max_length:
                preview = texts[index][:120].replace("\n", " ")
                raise ValueError(
                    "Embedding input exceeds RAG_EMBED_MAX_LENGTH: "
                    f"batch_index={offset + index} tokens={len(input_ids)} "
                    f"max_length={self._max_length}. Re-chunk the document instead "
                    f"of relying on tokenizer truncation. preview={preview!r}"
                )


class TransformersCLIPImageEmbeddingModel:
    def __init__(
        self,
        model_name: str,
        dim: int,
        *,
        batch_size: int,
        device: str,
        dtype: str,
    ) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._model_name = model_name
        self._dim = dim
        self._batch_size = max(1, batch_size)
        self._torch = torch
        self._device = resolve_device(torch, device)
        self._processor = CLIPProcessor.from_pretrained(model_name)
        model_kwargs = {}
        torch_dtype = resolve_torch_dtype(torch, device=self._device, dtype=dtype)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        self._model = CLIPModel.from_pretrained(model_name, **model_kwargs)
        self._model.to(self._device)
        self._model.eval()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        with self._torch.no_grad():
            for start in range(0, len(texts), self._batch_size):
                batch_texts = texts[start : start + self._batch_size]
                inputs = self._processor(
                    text=batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self._device) for key, value in inputs.items()}
                features = pooled_clip_features(self._model.get_text_features(**inputs))
                features = self._torch.nn.functional.normalize(features, p=2, dim=1)
                vectors.extend(features.detach().cpu().tolist())
        self._check_dims(vectors)
        return vectors

    def encode_images(self, image_paths: list[Path]) -> list[list[float]]:
        if not image_paths:
            return []

        from PIL import Image

        vectors: list[list[float]] = []
        with self._torch.no_grad():
            for start in range(0, len(image_paths), self._batch_size):
                batch_paths = image_paths[start : start + self._batch_size]
                images = [Image.open(path).convert("RGB") for path in batch_paths]
                inputs = self._processor(images=images, return_tensors="pt")
                inputs = {key: value.to(self._device) for key, value in inputs.items()}
                features = pooled_clip_features(self._model.get_image_features(**inputs))
                features = self._torch.nn.functional.normalize(features, p=2, dim=1)
                vectors.extend(features.detach().cpu().tolist())
        self._check_dims(vectors)
        return vectors

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def tokenize(self, text: str) -> list[int]:
        inputs = self._processor.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )
        return list(inputs["input_ids"])

    def _check_dims(self, vectors: list[list[float]]) -> None:
        for vector in vectors:
            if len(vector) < self._dim:
                vector.extend([0.0] * (self._dim - len(vector)))
                continue
            if len(vector) != self._dim:
                raise ValueError(
                    f"Image embedding dim mismatch: expected {self._dim}, got {len(vector)}. "
                    "Set IMAGE_EMBEDDING_DIM to match the model or a larger padded dimension."
                )


def build_embedding_model(config: RagConfig) -> EmbeddingModel:
    if config.embedding_backend == "hash":
        return HashEmbeddingModel(model_name=config.embedding_model, dim=config.embedding_dim)
    if config.embedding_backend == "siliconflow":
        return SiliconFlowEmbeddingModel(
            base_url=config.siliconflow_base_url,
            api_key=config.siliconflow_api_key,
            model_name=config.embedding_model,
            dim=config.embedding_dim,
            batch_size=config.embedding_batch_size,
        )
    if config.embedding_backend == "bge":
        local_path = _resolve_model_path(config.embedding_model, ms_subdir="bge-m3")
        return TransformersBGEEmbeddingModel(
            model_name=config.embedding_model,
            model_path=local_path,
            dim=config.embedding_dim,
            batch_size=config.embedding_batch_size,
            max_length=config.embedding_max_length,
            device=config.model_device,
            dtype=config.model_dtype,
        )
    raise ValueError(
        "Unsupported RAG_EMBEDDING_BACKEND="
        f"{config.embedding_backend!r}. Use 'hash', 'siliconflow', or 'bge'."
    )


def lexical_tokens(text: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


def stable_token_bucket(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def siliconflow_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}{path}"
    return f"{base}/v1{path}"


def post_json(url: str, *, api_key: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SiliconFlow API error {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"SiliconFlow API request failed: {exc.reason}") from exc


def pooled_clip_features(features):
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    return features


def build_image_embedding_model(config: RagConfig) -> ImageEmbeddingModel:
    if config.image_embedding_backend == "clip":
        return TransformersCLIPImageEmbeddingModel(
            model_name=config.image_embedding_model,
            dim=config.image_embedding_dim,
            batch_size=config.image_embedding_batch_size,
            device=config.model_device,
            dtype=config.model_dtype,
        )
    raise ValueError(
        "Unsupported RAG_IMAGE_EMBEDDING_BACKEND="
        f"{config.image_embedding_backend!r}. Use 'clip'."
    )


def resolve_device(torch, requested: str) -> str:
    if requested and requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_torch_dtype(torch, *, device: str, dtype: str):
    if not dtype or dtype == "auto":
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    resolved = mapping.get(dtype)
    if resolved is None:
        raise ValueError(
            f"Unsupported RAG_MODEL_DTYPE={dtype!r}; use auto/fp16/bf16/fp32"
        )
    if device.startswith("cpu") and resolved != torch.float32:
        return torch.float32
    return resolved
