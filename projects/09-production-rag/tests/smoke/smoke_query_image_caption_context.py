from __future__ import annotations

import tempfile
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from answer_multimodal import answer_multimodal_query
from rag_core.answering import AnswerGeneration
from rag_core.types import SearchHit, TraceInfo


def main() -> None:
    test_query_image_caption_is_added_to_answer_query()
    print("smoke_query_image_caption_context=ok")


def test_query_image_caption_is_added_to_answer_query() -> None:
    captured: dict[str, str] = {}

    with tempfile.TemporaryDirectory() as tmp:
        image_path = Path(tmp) / "query.png"
        image_path.write_bytes(b"fake-png")

        with (
            patch("answer_multimodal.retrieve_multimodal", return_value=fake_retrieval()),
            patch("answer_multimodal.load_config", return_value=SimpleNamespace()),
            patch("answer_multimodal.PdfImageCaptioner.from_query_env", return_value=FakeCaptioner()),
            patch("answer_multimodal.generate_answer", side_effect=capture_generate_answer(captured)),
        ):
            result = answer_multimodal_query(
                "这张图和资料有什么关系？",
                text_query="这张图和资料有什么关系？",
                image_query_path=str(image_path),
                tenant_id="team_a",
                candidate_limit=5,
                context_limit=3,
                request_id="smoke-query-image-caption",
            )

    assert result.answer == "ok"
    final_query = captured["query"]
    assert "用户上传图片的文字化描述" in final_query
    assert "图片里有一张折线图，显示延迟从 20ms 上升到 80ms。" in final_query
    assert "这张图和资料有什么关系？" in final_query
    assert "不要说你无法查看图片" in final_query


class FakeCaptioner:
    def caption_image_path(self, image_path: Path, *, query: str = "", language_hint: str = "zh") -> str:
        assert image_path.name == "query.png"
        assert query == "这张图和资料有什么关系？"
        assert language_hint == "zh"
        return "图片里有一张折线图，显示延迟从 20ms 上升到 80ms。"


def capture_generate_answer(captured: dict[str, str]):
    def _generate_answer(config, query: str, hits: list[SearchHit]) -> AnswerGeneration:
        captured["query"] = query
        return AnswerGeneration(
            answer="ok",
            llm_model="fake",
            llm_backend="fake",
            latency_ms=1.0,
            token_usage={},
        )

    return _generate_answer


def fake_retrieval() -> SimpleNamespace:
    hit = SearchHit(
        id="hit-1",
        score=0.9,
        text="资料证据：系统延迟监控图表。",
        doc_id="latency-doc",
        title="延迟监控资料",
        source_uri="memory://latency-doc",
        source_type="image",
        chunk_index=0,
        tenant_id="team_a",
        acl_groups=["engineering"],
        metadata={},
    )
    trace = TraceInfo(
        request_id="smoke-query-image-caption",
        original_query="这张图和资料有什么关系？",
        rewritten_query="这张图和资料有什么关系？",
        rewrite_backend="none",
        tenant_id="team_a",
        acl_groups=["engineering"],
        doc_version=None,
        current_versions={},
        embedding_model="fake",
        source_types=["image"],
        doc_ids=[],
        filter_expr="",
        retrieval_mode="multimodal_text_image_file_fusion_rerank",
        candidate_count=1,
        reranked_count=1,
        context_count=1,
        dropped_by_score=0,
        dropped_by_doc_limit=0,
        dropped_by_budget=0,
        stage_latency_ms={},
    )
    return SimpleNamespace(
        request_id="smoke-query-image-caption",
        hits=[hit],
        candidates=[hit],
        reranked=[hit],
        trace=trace,
    )


if __name__ == "__main__":
    main()
