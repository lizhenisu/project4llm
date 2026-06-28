from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rag_core.config import load_config
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.pipeline import retrieve_and_rerank
from rag_core.types import Chunk, RewriteResult, SearchHit
from search_multimodal import retrieve_multimodal


TENANT_ID = "multi-source-real-milvus"
TARGET_SOURCE = "relevant-second"
SELECTED_DOC_IDS = [
    "irrelevant-first/page-1",
    "irrelevant-first/page-2",
    "irrelevant-first/page-1/image-1",
    "relevant-second/page-1",
    "relevant-second/page-1/image-1",
    "third-paper/page-1",
    "third-paper/page-1/image-1",
]


class DeterministicEmbeddingModel:
    model_name = "deterministic-multi-source"

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [vector(self.dim, 1.0, 0.0) for _ in texts]

    def count_tokens(self, text: str) -> int:
        return len(text)


class DeterministicImageModel(DeterministicEmbeddingModel):
    def encode_images(self, image_paths: list[Path]) -> list[list[float]]:
        return [vector(self.dim, 1.0, 0.0) for _ in image_paths]


class TargetFactReranker:
    def rerank(self, query: str, hits: list[SearchHit], *, limit: int) -> list[SearchHit]:
        scored = [
            replace(
                hit,
                rerank_score=1.0 if "TARGET_FACT_7741" in hit.text else 0.5 - index * 0.001,
            )
            for index, hit in enumerate(hits)
        ]
        return sorted(scored, key=lambda hit: hit.rerank_score or 0.0, reverse=True)[:limit]


def main() -> None:
    old_milvus_uri = os.environ.get("MILVUS_URI")
    old_collection = os.environ.get("RAG_COLLECTION")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MILVUS_URI"] = str(Path(tmp) / "multi_source_real.db")
        os.environ["RAG_COLLECTION"] = "rag_smoke_multi_source_real"
        try:
            run_smoke(Path(tmp))
        finally:
            restore_env("MILVUS_URI", old_milvus_uri)
            restore_env("RAG_COLLECTION", old_collection)
    print("smoke_multi_source_real_milvus=ok")


def run_smoke(tmp_dir: Path) -> None:
    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=True)
    text_model = DeterministicEmbeddingModel(config.embedding_dim)
    image_model = DeterministicImageModel(config.image_embedding_dim)
    chunks = fixture_chunks()
    entities = [
        chunk_to_entity(
            chunk,
            dense_vector=text_vector_for(chunk, config.embedding_dim),
            image_vector=image_vector_for(chunk, config.image_embedding_dim),
            embedding_model=text_model.model_name,
            embedding_dim=text_model.dim,
        )
        for chunk in chunks
    ]
    assert upsert_entities(
        client,
        collection_name=config.collection_name,
        entities=entities,
    ) == len(entities)

    with (
        patch("rag_core.pipeline.load_config", return_value=config),
        patch("rag_core.pipeline.build_embedding_model", return_value=text_model),
        patch("rag_core.pipeline.load_source_guides_for_rewrite", return_value=[]),
        patch(
            "rag_core.pipeline.rewrite_query",
            return_value=RewriteResult("TARGET_FACT_7741", "TARGET_FACT_7741", "deterministic"),
        ),
        patch("rag_core.pipeline.build_reranker", return_value=TargetFactReranker()),
    ):
        text_result = retrieve_and_rerank(
            "TARGET_FACT_7741",
            tenant_id=TENANT_ID,
            acl_groups=["engineering"],
            doc_ids=SELECTED_DOC_IDS,
            doc_version=1,
            candidate_limit=12,
            context_limit=5,
            request_id="real-milvus-text-multi-source",
        )

    assert text_result.trace.retrieval_mode == "hybrid_dense_sparse_source_fanout_rerank"
    assert source_ids(text_result.candidates) == {
        "irrelevant-first",
        "relevant-second",
        "third-paper",
    }
    assert text_result.hits[0].doc_id.startswith(f"{TARGET_SOURCE}/")
    assert source_ids(text_result.hits) == {
        "irrelevant-first",
        "relevant-second",
        "third-paper",
    }

    query_image = tmp_dir / "query.png"
    query_image.write_bytes(b"synthetic-image-query")
    with (
        patch("search_multimodal.load_config", return_value=config),
        patch("search_multimodal.build_embedding_model", return_value=text_model),
        patch("search_multimodal.build_image_embedding_model", return_value=image_model),
        patch(
            "search_multimodal.rewrite_query",
            return_value=RewriteResult("TARGET_FACT_7741", "TARGET_FACT_7741", "deterministic"),
        ),
        patch("search_multimodal.build_reranker", return_value=TargetFactReranker()),
    ):
        multimodal_result = retrieve_multimodal(
            text_query="TARGET_FACT_7741",
            image_query_path=query_image,
            tenant_id=TENANT_ID,
            acl_groups=["engineering"],
            doc_ids=SELECTED_DOC_IDS,
            doc_version=1,
            candidate_limit=12,
            context_limit=5,
            request_id="real-milvus-multimodal-multi-source",
        )

    assert multimodal_result.trace.retrieval_mode.endswith("_source_fanout")
    assert source_ids(multimodal_result.candidates) == {
        "irrelevant-first",
        "relevant-second",
        "third-paper",
    }
    assert multimodal_result.hits[0].doc_id == f"{TARGET_SOURCE}/page-1/image-1", [
        (hit.doc_id, hit.score, hit.rerank_score)
        for hit in multimodal_result.hits
    ]
    assert source_ids(multimodal_result.hits) == {
        "irrelevant-first",
        "relevant-second",
        "third-paper",
    }


def fixture_chunks() -> list[Chunk]:
    rows = [
        ("irrelevant-first/page-1", "Unrelated overview with common retrieval terminology.", "pdf"),
        ("irrelevant-first/page-2", "Another unrelated page about generic architecture.", "pdf"),
        ("irrelevant-first/page-1/image-1", "Unrelated diagram caption.", "image"),
        (
            "relevant-second/page-1",
            "TARGET_FACT_7741 is the exact fact that must be recalled from the second PDF.",
            "pdf",
        ),
        (
            "relevant-second/page-1/image-1",
            "TARGET_FACT_7741 architecture diagram and visual evidence.",
            "image",
        ),
        ("third-paper/page-1", "Third document about forecasting baselines.", "pdf"),
        ("third-paper/page-1/image-1", "Third document chart.", "image"),
    ]
    return [
        Chunk(
            tenant_id=TENANT_ID,
            doc_id=doc_id,
            doc_version=1,
            chunk_index=0,
            source_type=source_type,
            source_uri=f"memory://{doc_id}",
            title=doc_id,
            text=text,
            language="en",
            acl_groups=["engineering"],
            metadata={},
        )
        for doc_id, text, source_type in rows
    ]


def text_vector_for(chunk: Chunk, dim: int) -> list[float]:
    if chunk.doc_id.startswith(TARGET_SOURCE):
        return vector(dim, 1.0, 0.0)
    if chunk.doc_id.startswith("irrelevant-first"):
        return vector(dim, 0.8, 0.6)
    return vector(dim, 0.0, 1.0)


def image_vector_for(chunk: Chunk, dim: int) -> list[float]:
    if chunk.source_type != "image":
        return vector(dim, 0.0, 0.0)
    return text_vector_for(chunk, dim)


def vector(dim: int, first: float, second: float) -> list[float]:
    values = [0.0] * dim
    if dim > 0:
        values[0] = first
    if dim > 1:
        values[1] = second
    return values


def source_ids(hits: list[SearchHit]) -> set[str]:
    return {
        str(hit.metadata.get("retrieval_source_id") or hit.doc_id.split("/page-", 1)[0])
        for hit in hits
    }


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
