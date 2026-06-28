from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from answer_multimodal import answer_multimodal_query
from rag_core.answering import AnswerGeneration
from rag_core.types import SearchHit, TraceInfo
from search_multimodal import anchor_query_image_evidence


def main() -> None:
    test_multimodal_answer_emits_document_route_before_retrieval()
    test_uploaded_image_keeps_nearest_image_as_context_anchor()
    print("smoke_multimodal_scope_routing=ok")


def test_multimodal_answer_emits_document_route_before_retrieval() -> None:
    stages: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        config = SimpleNamespace(object_store_dir=Path(tmp))
        with (
            patch("answer_multimodal.load_config", return_value=config),
            patch("answer_multimodal.retrieve_multimodal", return_value=fake_retrieval()),
            patch("answer_multimodal.PdfImageCaptioner.from_query_env", return_value=None),
            patch("answer_multimodal.generate_answer", return_value=fake_generation()),
        ):
            answer_multimodal_query(
                "解释这张系统架构图",
                text_query="解释这张系统架构图",
                image_query_path="/tmp/query-image.png",
                tenant_id="team_a",
                candidate_limit=10,
                context_limit=5,
                doc_ids=["attention/page-1", "autoformer/page-1"],
                doc_version=1,
                request_id="multimodal-route-smoke",
                stage_callback=stages.append,
            )

    assert [stage["stage"] for stage in stages[:2]] == ["intent_router", "scope_resolution"]
    assert stages[0]["intent"] == "local_qa"
    assert stages[1]["resolved_doc_ids"] == ["attention/page-1", "autoformer/page-1"]


def test_uploaded_image_keeps_nearest_image_as_context_anchor() -> None:
    nearest_image = fake_hit("attention-image", source_type="image", rerank_score=-0.8)
    text_hit = fake_hit("autoformer-text", source_type="pdf", rerank_score=0.9)

    anchored = anchor_query_image_evidence(
        [text_hit, nearest_image],
        image_hits=[nearest_image],
        has_image_file_query=True,
    )

    assert [hit.id for hit in anchored] == ["attention-image", "autoformer-text"]
    assert anchored[0].rerank_score is None


def fake_retrieval() -> SimpleNamespace:
    hit = fake_hit("attention-image", source_type="image")
    return SimpleNamespace(
        request_id="multimodal-route-smoke",
        hits=[hit],
        candidates=[hit],
        reranked=[hit],
        trace=TraceInfo(
            request_id="multimodal-route-smoke",
            original_query="解释这张系统架构图",
            rewritten_query="解释这张系统架构图",
            rewrite_backend="fake",
            tenant_id="team_a",
            acl_groups=[],
            doc_version=1,
            current_versions={},
            embedding_model="fake",
            source_types=[],
            doc_ids=["attention/page-1", "autoformer/page-1"],
            filter_expr="",
            retrieval_mode="multimodal_text_image_file_fusion_rerank",
            candidate_count=1,
            reranked_count=1,
            context_count=1,
            dropped_by_score=0,
            dropped_by_doc_limit=0,
            dropped_by_budget=0,
            stage_latency_ms={},
        ),
    )


def fake_hit(
    hit_id: str,
    *,
    source_type: str,
    rerank_score: float | None = None,
) -> SearchHit:
    return SearchHit(
        id=hit_id,
        score=0.9,
        text=f"{hit_id} evidence",
        doc_id=hit_id,
        title=hit_id,
        source_uri=f"memory://{hit_id}",
        source_type=source_type,
        chunk_index=0,
        tenant_id="team_a",
        acl_groups=[],
        metadata={},
        rerank_score=rerank_score,
    )


def fake_generation() -> AnswerGeneration:
    return AnswerGeneration(
        answer="ok",
        llm_model="fake",
        llm_backend="fake",
        latency_ms=1.0,
        token_usage={},
    )


if __name__ == "__main__":
    main()
