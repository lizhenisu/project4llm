from __future__ import annotations

from pathlib import Path
from typing import Protocol

from rag_core.config import RagConfig, _resolve_model_path
from rag_core.text_utils import hash_dense_embedding


class EmbeddingModel(Protocol):
    @property
    def dim(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class ImageEmbeddingModel(EmbeddingModel, Protocol):
    def encode_images(self, image_paths: list[Path]) -> list[list[float]]: ...


class HashEmbeddingModel:
    def __init__(self, dim: int, model_name: str = "hash-teaching-backend") -> None:
        self._dim = dim
        self._model_name = model_name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [hash_dense_embedding(text, self._dim) for text in texts]

    def encode_images(self, image_paths: list[Path]) -> list[list[float]]:
        return self.encode([str(path) for path in image_paths])


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
                batch = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
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
                features = self._model.get_text_features(**inputs)
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
                features = self._model.get_image_features(**inputs)
                features = self._torch.nn.functional.normalize(features, p=2, dim=1)
                vectors.extend(features.detach().cpu().tolist())
        self._check_dims(vectors)
        return vectors

    def _check_dims(self, vectors: list[list[float]]) -> None:
        for vector in vectors:
            if len(vector) != self._dim:
                raise ValueError(
                    f"Image embedding dim mismatch: expected {self._dim}, got {len(vector)}. "
                    "Set IMAGE_EMBEDDING_DIM to match the model."
                )


def build_embedding_model(config: RagConfig) -> EmbeddingModel:
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
    if config.embedding_backend == "hash":
        return HashEmbeddingModel(
            dim=config.embedding_dim,
            model_name=f"hash:{config.embedding_model}",
        )
    raise ValueError(f"Unsupported RAG_EMBEDDING_BACKEND={config.embedding_backend!r}")


def build_image_embedding_model(config: RagConfig) -> ImageEmbeddingModel:
    if config.image_embedding_backend == "hash":
        return HashEmbeddingModel(
            dim=config.image_embedding_dim,
            model_name="hash:image-teaching-backend",
        )
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
        f"{config.image_embedding_backend!r}. Use 'hash' or 'clip'."
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
