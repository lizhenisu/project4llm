from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rag_core.context import explain_context_packing
from rag_core.pipeline import retrieve_and_rerank
from rag_core.types import RewriteResult, SearchHit
from search_multimodal import retrieve_multimodal


class FakeEmbeddingModel:
    model_name = "BAAI/bge-m3"

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] for _ in texts]


class FakeImageEmbeddingModel(FakeEmbeddingModel):
    def encode_images(self, image_paths: list[Path]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in image_paths]


class FakeReranker:
    def rerank(self, query: str, hits: list[SearchHit], *, limit: int) -> list[SearchHit]:
        scored = [
            replace(hit, rerank_score=score)
            for hit, score in zip(hits, [0.95, 0.90, 0.85], strict=True)
        ]
        return scored[:limit]


def main() -> None:
    test_context_helper_backfills_later_hits()
    test_text_pipeline_backfills_after_doc_limit()
    test_multimodal_pipeline_backfills_after_doc_limit()
    print("smoke_context_backfill=ok")


def test_context_helper_backfills_later_hits() -> None:
    hits, stats, decisions = explain_context_packing(
        make_ranked_hits(),
        max_selected=2,
        max_chars=10_000,
        max_chunks_per_doc=1,
        min_rerank_score=None,
    )
    assert [hit.doc_id for hit in hits] == ["doc-a", "doc-b"]
    assert stats.selected_count == 2
    assert stats.dropped_by_doc_limit == 1
    assert [decision.reason for decision in decisions] == [
        "fits_budget",
        "max_chunks_per_doc",
        "fits_budget",
    ]


def test_text_pipeline_backfills_after_doc_limit() -> None:
    config = make_config()
    candidates = make_ranked_hits()
    with (
        patch("rag_core.pipeline.load_config", return_value=config),
        patch("rag_core.pipeline.connect", return_value=object()),
        patch("rag_core.pipeline.ensure_collection"),
        patch("rag_core.pipeline.build_embedding_model", return_value=FakeEmbeddingModel()),
        patch(
            "rag_core.pipeline.rewrite_query",
            return_value=RewriteResult("query", "query", "llm"),
        ),
        patch("rag_core.pipeline.mentions_other_tenant", return_value=False),
        patch("rag_core.pipeline.load_current_versions", return_value={}),
        patch("rag_core.pipeline.build_filter_expr", return_value='tenant_id == "team_a"'),
        patch("rag_core.pipeline.hybrid_search", return_value=candidates),
        patch("rag_core.pipeline.build_reranker", return_value=FakeReranker()),
    ):
        result = retrieve_and_rerank(
            "query",
            tenant_id="team_a",
            acl_groups=["ops"],
            candidate_limit=3,
            context_limit=2,
            request_id="smoke-context-backfill-text",
        )
    assert [hit.doc_id for hit in result.reranked] == ["doc-a", "doc-a", "doc-b"]
    assert [hit.doc_id for hit in result.hits] == ["doc-a", "doc-b"]
    assert result.trace.reranked_count == 3
    assert result.trace.context_count == 2
    assert result.trace.dropped_by_doc_limit == 1


def test_multimodal_pipeline_backfills_after_doc_limit() -> None:
    config = make_config()
    candidates = make_ranked_hits()
    with (
        patch("search_multimodal.load_config", return_value=config),
        patch("search_multimodal.connect", return_value=object()),
        patch("search_multimodal.ensure_collection"),
        patch("search_multimodal.build_embedding_model", return_value=FakeEmbeddingModel()),
        patch(
            "search_multimodal.build_image_embedding_model",
            return_value=FakeImageEmbeddingModel(),
        ),
        patch(
            "search_multimodal.rewrite_query",
            return_value=RewriteResult("query", "query", "llm"),
        ),
        patch("search_multimodal.mentions_other_tenant", return_value=False),
        patch("search_multimodal.load_current_versions", return_value={}),
        patch("search_multimodal.build_filter_expr", return_value='tenant_id == "team_a"'),
        patch("search_multimodal.hybrid_search", return_value=candidates),
        patch("search_multimodal.image_search", return_value=candidates),
        patch("search_multimodal.reciprocal_rank_fusion", return_value=candidates),
    ):
        result = retrieve_multimodal(
            "query",
            tenant_id="team_a",
            acl_groups=["ops"],
            candidate_limit=3,
            context_limit=2,
            request_id="smoke-context-backfill-multimodal",
        )
    assert [hit.doc_id for hit in result.candidates] == ["doc-a", "doc-a", "doc-b"]
    assert [hit.doc_id for hit in result.hits] == ["doc-a", "doc-b"]
    assert result.trace.candidate_count == 3
    assert result.trace.context_count == 2
    assert result.trace.dropped_by_doc_limit == 1


def make_config():
    return SimpleNamespace(
        collection_name="rag_smoke_context_backfill",
        object_store_dir=Path("/tmp/rag_smoke_context_backfill"),
        max_context_chars=10_000,
        max_chunks_per_doc=1,
        min_rerank_score=None,
    )


def make_ranked_hits() -> list[SearchHit]:
    return [
        make_hit("doc-a", 0, score=1.0),
        make_hit("doc-a", 1, score=0.9),
        make_hit("doc-b", 0, score=0.8),
    ]


def make_hit(doc_id: str, chunk_index: int, *, score: float) -> SearchHit:
    return SearchHit(
        id=f"{doc_id}:{chunk_index}",
        score=score,
        text=f"{doc_id} retrieval evidence",
        doc_id=doc_id,
        title=doc_id,
        source_uri=f"memory://{doc_id}",
        source_type="md",
        chunk_index=chunk_index,
        tenant_id="team_a",
        acl_groups=["ops"],
        metadata={},
    )


if __name__ == "__main__":
    main()
