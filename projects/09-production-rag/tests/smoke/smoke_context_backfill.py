from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rag_core.context import explain_context_packing
from rag_core.pipeline import retrieve_and_rerank
from rag_core.retrieval_scope import (
    context_chunks_per_source,
    group_selected_doc_ids,
    round_robin_hit_groups,
)
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


class IdentityReranker:
    def rerank(self, query: str, hits: list[SearchHit], *, limit: int) -> list[SearchHit]:
        return [
            replace(hit, rerank_score=1.0 - index * 0.01)
            for index, hit in enumerate(hits[:limit])
        ]


def main() -> None:
    test_context_helper_backfills_later_hits()
    test_context_limit_counts_pdf_pages_as_one_source()
    test_selected_doc_groups_round_robin_across_sources()
    test_single_selected_source_can_fill_context()
    test_text_pipeline_fans_out_small_multi_source_scope()
    test_multimodal_fans_out_small_multi_source_scope()
    test_text_pipeline_backfills_after_doc_limit()
    test_multimodal_pipeline_backfills_after_doc_limit()
    test_multimodal_image_and_text_query_fuses_then_reranks()
    test_multimodal_image_file_query_uses_image_scores_only()
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


def test_context_limit_counts_pdf_pages_as_one_source() -> None:
    hits = [
        make_hit("attention/page-1", 0, score=1.0),
        make_hit("attention/page-2", 0, score=0.9),
        make_hit("autoformer/page-1", 0, score=0.8),
    ]
    selected, stats, _ = explain_context_packing(
        hits,
        max_selected=3,
        max_chars=10_000,
        max_chunks_per_doc=1,
        min_rerank_score=None,
    )
    assert [hit.doc_id for hit in selected] == ["attention/page-1", "autoformer/page-1"]
    assert stats.dropped_by_doc_limit == 1


def test_selected_doc_groups_round_robin_across_sources() -> None:
    groups = group_selected_doc_ids([
        "attention/page-1",
        "attention/page-2",
        "autoformer/page-1",
        "third-paper/page-1/image-1",
    ])
    assert groups == [
        ("attention", ["attention/page-1", "attention/page-2"]),
        ("autoformer", ["autoformer/page-1"]),
        ("third-paper", ["third-paper/page-1/image-1"]),
    ]
    interleaved = round_robin_hit_groups([
        [make_hit("attention/page-1", 0, score=1.0), make_hit("attention/page-2", 0, score=0.9)],
        [make_hit("autoformer/page-1", 0, score=0.8)],
        [make_hit("third-paper/page-1", 0, score=0.7)],
    ])
    assert [hit.doc_id for hit in interleaved] == [
        "attention/page-1",
        "autoformer/page-1",
        "third-paper/page-1",
        "attention/page-2",
    ]


def test_single_selected_source_can_fill_context() -> None:
    one_source = group_selected_doc_ids([
        "attention/page-1",
        "attention/page-2",
    ])
    three_sources = group_selected_doc_ids([
        "attention/page-1",
        "autoformer/page-1",
        "third-paper/page-1",
    ])
    assert context_chunks_per_source(2, 5, one_source) == 5
    assert context_chunks_per_source(2, 5, three_sources) == 2
    assert context_chunks_per_source(2, 5, []) == 2


def test_multimodal_fans_out_small_multi_source_scope() -> None:
    config = make_config()
    selected_ids = [
        "attention/page-1",
        "attention/page-2",
        "autoformer/page-1",
        "third-paper/page-1",
    ]

    def scoped_hits(*args, filter_expr: str, **kwargs):
        if '"attention/page-1"' in filter_expr:
            return [
                make_hit("attention/page-1", 0, score=0.99),
                make_hit("attention/page-2", 0, score=0.98),
            ]
        if '"autoformer/page-1"' in filter_expr:
            return [make_hit("autoformer/page-1", 0, score=0.80)]
        if '"third-paper/page-1"' in filter_expr:
            return [make_hit("third-paper/page-1", 0, score=0.70)]
        raise AssertionError(filter_expr)

    with (
        patch("search_multimodal.load_config", return_value=config),
        patch("search_multimodal.connect", return_value=object()),
        patch("search_multimodal.ensure_collection"),
        patch("search_multimodal.build_embedding_model", return_value=FakeEmbeddingModel()),
        patch("search_multimodal.build_image_embedding_model", return_value=FakeImageEmbeddingModel()),
        patch("search_multimodal.rewrite_query", return_value=RewriteResult("query", "query", "llm")),
        patch("search_multimodal.mentions_other_tenant", return_value=False),
        patch("search_multimodal.load_current_versions", return_value={}),
        patch("search_multimodal.hybrid_search", side_effect=scoped_hits) as hybrid,
        patch("search_multimodal.image_search", side_effect=scoped_hits) as image,
        patch("search_multimodal.build_reranker", return_value=IdentityReranker()),
    ):
        result = retrieve_multimodal(
            "query",
            tenant_id="team_a",
            acl_groups=["ops"],
            doc_ids=selected_ids,
            candidate_limit=3,
            context_limit=3,
            request_id="smoke-multi-source-fanout",
        )

    assert hybrid.call_count == 3
    assert image.call_count == 3
    assert {hit.metadata["retrieval_source_id"] for hit in result.candidates} == {
        "attention",
        "autoformer",
        "third-paper",
    }
    assert [hit.doc_id for hit in result.hits] == [
        "attention/page-1",
        "autoformer/page-1",
        "third-paper/page-1",
    ]
    assert result.trace.retrieval_mode.endswith("_source_fanout")


def test_text_pipeline_fans_out_small_multi_source_scope() -> None:
    config = make_config()
    selected_ids = [
        "attention/page-1",
        "attention/page-2",
        "autoformer/page-1",
        "third-paper/page-1",
    ]

    def scoped_hits(*args, filter_expr: str, **kwargs):
        if '"attention/page-1"' in filter_expr:
            return [
                make_hit("attention/page-1", 0, score=0.99),
                make_hit("attention/page-2", 0, score=0.98),
            ]
        if '"autoformer/page-1"' in filter_expr:
            return [make_hit("autoformer/page-1", 0, score=0.80)]
        if '"third-paper/page-1"' in filter_expr:
            return [make_hit("third-paper/page-1", 0, score=0.70)]
        raise AssertionError(filter_expr)

    with (
        patch("rag_core.pipeline.load_config", return_value=config),
        patch("rag_core.pipeline.connect", return_value=object()),
        patch("rag_core.pipeline.ensure_collection"),
        patch("rag_core.pipeline.build_embedding_model", return_value=FakeEmbeddingModel()),
        patch("rag_core.pipeline.rewrite_query", return_value=RewriteResult("query", "query", "llm")),
        patch("rag_core.pipeline.mentions_other_tenant", return_value=False),
        patch("rag_core.pipeline.load_current_versions", return_value={}),
        patch("rag_core.pipeline.load_source_guides_for_rewrite", return_value=[]),
        patch("rag_core.pipeline.hybrid_search", side_effect=scoped_hits) as hybrid,
        patch("rag_core.pipeline.build_reranker", return_value=IdentityReranker()),
    ):
        result = retrieve_and_rerank(
            "query",
            tenant_id="team_a",
            acl_groups=["ops"],
            doc_ids=selected_ids,
            candidate_limit=3,
            context_limit=3,
            request_id="smoke-text-multi-source-fanout",
        )

    assert hybrid.call_count == 3
    assert {hit.metadata["retrieval_source_id"] for hit in result.candidates} == {
        "attention",
        "autoformer",
        "third-paper",
    }
    assert [hit.doc_id for hit in result.hits] == [
        "attention/page-1",
        "autoformer/page-1",
        "third-paper/page-1",
    ]
    assert result.trace.retrieval_mode == "hybrid_dense_sparse_source_fanout_rerank"


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
        patch("search_multimodal.build_reranker", return_value=FakeReranker()),
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
    assert [hit.doc_id for hit in result.reranked] == ["doc-a", "doc-a", "doc-b"]
    assert [hit.doc_id for hit in result.hits] == ["doc-a", "doc-b"]
    assert result.trace.candidate_count == 3
    assert result.trace.reranked_count == 3
    assert result.trace.context_count == 2
    assert result.trace.dropped_by_doc_limit == 1


def test_multimodal_image_and_text_query_fuses_then_reranks() -> None:
    config = make_config()
    candidates = make_ranked_hits()
    with tempfile.TemporaryDirectory() as tmp:
        image_path = Path(tmp) / "query.png"
        image_path.write_bytes(b"fake-png")
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
            patch("search_multimodal.hybrid_search", return_value=candidates) as hybrid,
            patch("search_multimodal.image_search", return_value=candidates) as image,
            patch("search_multimodal.reciprocal_rank_fusion", return_value=candidates) as fusion,
            patch("search_multimodal.build_reranker", return_value=FakeReranker()) as reranker,
        ):
            result = retrieve_multimodal(
                text_query="query",
                image_query_path=image_path,
                tenant_id="team_a",
                acl_groups=["ops"],
                candidate_limit=3,
                context_limit=2,
                request_id="smoke-image-text-query",
            )
    hybrid.assert_called_once()
    image.assert_called_once()
    fusion.assert_called_once()
    reranker.assert_called_once()
    assert result.trace.retrieval_mode == "multimodal_text_image_file_fusion_rerank"
    assert result.trace.stage_latency_ms["text_embedding"] >= 0
    assert result.trace.stage_latency_ms["image_embedding"] >= 0
    assert result.trace.reranked_count == 3
    assert [hit.doc_id for hit in result.hits] == ["doc-a", "doc-b"]


def test_multimodal_image_file_query_uses_image_scores_only() -> None:
    config = make_config()
    image_hits = [
        make_hit("figure-1-transformer", 0, score=0.91),
        make_hit("figure-2-attention", 0, score=0.72),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        image_path = Path(tmp) / "query.png"
        image_path.write_bytes(b"fake-png")
        with (
            patch("search_multimodal.load_config", return_value=config),
            patch("search_multimodal.connect", return_value=object()),
            patch("search_multimodal.ensure_collection"),
            patch("search_multimodal.build_embedding_model", return_value=FakeEmbeddingModel()),
            patch(
                "search_multimodal.build_image_embedding_model",
                return_value=FakeImageEmbeddingModel(),
            ),
            patch("search_multimodal.load_current_versions", return_value={}),
            patch("search_multimodal.build_filter_expr", return_value='tenant_id == "team_a"'),
            patch("search_multimodal.hybrid_search", side_effect=AssertionError("text search should be skipped")),
            patch("search_multimodal.reciprocal_rank_fusion", side_effect=AssertionError("RRF should be skipped")),
            patch("search_multimodal.image_search", return_value=image_hits),
        ):
            result = retrieve_multimodal(
                str(image_path),
                tenant_id="team_a",
                acl_groups=["ops"],
                candidate_limit=2,
                context_limit=2,
                request_id="smoke-image-file-query",
            )
    assert [hit.doc_id for hit in result.candidates] == ["figure-1-transformer", "figure-2-attention"]
    assert [hit.score for hit in result.candidates] == [0.91, 0.72]
    assert result.trace.retrieval_mode == "image_vector_file_query"
    assert result.trace.stage_latency_ms["text_embedding"] == 0.0
    assert result.trace.stage_latency_ms["text_search"] == 0.0
    assert result.candidates[0].metadata["fusion"]["mode"] == "image_only"
    assert result.candidates[0].metadata["fusion"]["channels"] == {"image_vector": 1}
    assert result.candidates[0].metadata["fusion"]["channel_scores"]["image_vector"] == 0.91


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
