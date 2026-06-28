from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from answer import answer_query
from rag_core.answering import AnswerGeneration
from rag_core.document_scope import (
    LOCAL_QA,
    PER_DOC_MAP_REDUCE,
    SELECTED_DOC_SUMMARY,
    SELECTED_DOC_SYNTHESIS,
    build_scope_plan,
)
from rag_core.io import write_jsonl
from rag_core.object_store import archive_source_documents
from rag_core.pipeline import RetrievalResult
from rag_core.types import SearchHit, SourceDocument, TraceInfo


def main() -> None:
    test_router_requires_coverage_for_selected_doc_summary()
    test_scope_plan_reads_guides_through_object_store_backend()
    test_page_ids_collapse_to_unique_uploaded_documents()
    test_ambiguous_risk_question_requires_selected_doc_coverage()
    test_missing_source_guides_fall_back_to_archived_documents()
    test_answer_query_uses_all_selected_doc_guides_for_summary()
    test_explicit_local_question_narrows_scope_then_uses_top_k()
    test_zero_document_open_chat_skips_retrieval()
    test_large_scope_uses_map_reduce_when_guides_exceed_budget()
    print("smoke_document_scope_routing=ok")


def test_scope_plan_reads_guides_through_object_store_backend() -> None:
    rows = [
        {
            "tenant_id": "team_a",
            "source_doc_id": "s3-doc",
            "doc_version": 1,
            "title": "S3 Document",
            "guide": "Guide loaded through the object-store abstraction.",
        }
    ]
    with tempfile.TemporaryDirectory() as tmp:
        config = fake_config(tmp)
        with (
            patch("rag_core.document_scope.object_exists", return_value=True) as exists,
            patch("rag_core.document_scope.read_object_jsonl", return_value=rows) as read_rows,
        ):
            plan = build_scope_plan(
                config=config,
                tenant_id="team_a",
                query="总结这份资料",
                doc_ids=["s3-doc"],
                doc_version=1,
                include_all_sources=False,
            )
    exists.assert_called_once_with(config.object_store_dir, Path("canonical/source_guides.jsonl"))
    read_rows.assert_called_once_with(config.object_store_dir, Path("canonical/source_guides.jsonl"))
    assert plan.route.coverage_required is True
    assert plan.coverage()["covered_doc_count"] == 1
    assert plan.guides[0].guide == "Guide loaded through the object-store abstraction."


def test_router_requires_coverage_for_selected_doc_summary() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = fake_config(tmp)
        write_guides(config.object_store_dir, count=6)
        plan = build_scope_plan(
            config=config,
            tenant_id="team_a",
            query="总结这些资料的主要内容",
            doc_ids=[f"doc-{index}" for index in range(1, 7)],
            doc_version=1,
            include_all_sources=False,
        )
    assert plan.route.intent == SELECTED_DOC_SUMMARY
    assert plan.route.coverage_required is True
    assert plan.resolved_doc_ids == [f"doc-{index}" for index in range(1, 7)]
    assert len(plan.guides) == 6
    assert plan.coverage()["covered_doc_count"] == 6


def test_page_ids_collapse_to_unique_uploaded_documents() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = fake_config(tmp)
        rows = []
        selected = []
        for document_index in range(1, 4):
            source_id = f"paper-{document_index}"
            selected.extend(f"{source_id}/page-{page_index}" for page_index in range(1, 20))
            for revision in range(1, 5):
                rows.append(
                    {
                        "tenant_id": "team_a",
                        "source_doc_id": source_id,
                        "doc_version": 1,
                        "title": f"Paper {document_index}",
                        "guide": f"Paper {document_index} summary revision {revision}.",
                    }
                )
        path = config.object_store_dir / "canonical" / "source_guides.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(path, rows)

        plan = build_scope_plan(
            config=config,
            tenant_id="team_a",
            query="总结这些资料的核心内容",
            doc_ids=selected,
            doc_version=1,
            include_all_sources=False,
        )

    assert len(selected) == 57
    assert plan.selected_doc_ids == ["paper-1", "paper-2", "paper-3"]
    assert plan.resolved_doc_ids == ["paper-1", "paper-2", "paper-3"]
    assert len(plan.guides) == 3
    assert plan.coverage()["covered_doc_count"] == 3
    assert plan.guides[0].guide.endswith("revision 4.")


def test_ambiguous_risk_question_requires_selected_doc_coverage() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = fake_config(tmp)
        write_guides(config.object_store_dir, count=14)
        selected = [f"doc-{index}" for index in range(1, 15)]
        plan = build_scope_plan(
            config=config,
            tenant_id="team_a",
            query="帮我看看这里面有哪些风险",
            doc_ids=selected,
            doc_version=1,
            include_all_sources=False,
        )
    assert plan.route.intent == SELECTED_DOC_SYNTHESIS
    assert plan.route.coverage_required is True
    assert plan.resolved_doc_ids == selected
    assert len(plan.guides) == 14


def test_missing_source_guides_fall_back_to_archived_documents() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = fake_config(tmp)
        archive_source_documents(
            config.object_store_dir,
            [
                SourceDocument(
                    tenant_id="team_a",
                    doc_id="doc-1",
                    doc_version=1,
                    source_type="pdf",
                    source_uri="memory://doc-1",
                    title="Document 1",
                    text="Document 1 archived source text about deployment risk.",
                ),
                SourceDocument(
                    tenant_id="team_a",
                    doc_id="doc-2",
                    doc_version=1,
                    source_type="pdf",
                    source_uri="memory://doc-2",
                    title="Document 2",
                    text="Document 2 archived source text about data risk.",
                ),
            ],
        )
        plan = build_scope_plan(
            config=config,
            tenant_id="team_a",
            query="总结这些资料",
            doc_ids=["doc-1", "doc-2"],
            doc_version=1,
            include_all_sources=False,
        )
    assert plan.route.coverage_required is True
    assert len(plan.guides) == 2
    assert plan.missing_doc_ids == []
    assert "archived source text" in plan.guides[0].guide


def test_answer_query_uses_all_selected_doc_guides_for_summary() -> None:
    captured: dict[str, object] = {}
    stages: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        config = fake_config(tmp)
        write_guides(config.object_store_dir, count=6)
        with (
            patch("answer.load_config", return_value=config),
            patch("rag_core.document_scope.generate_answer", side_effect=capture_document_answer(captured)),
        ):
            result = answer_query(
                "总结这些资料的主要内容",
                tenant_id="team_a",
                candidate_limit=20,
                context_limit=5,
                doc_ids=[f"doc-{index}" for index in range(1, 7)],
                doc_version=1,
                request_id="smoke-doc-scope-summary",
                stage_callback=stages.append,
            )

    assert result.answer == "document answer [1] [2] [3] [4] [5] [6]"
    assert result.trace.retrieval_mode == "document_scope_coverage"
    assert result.trace.intent_router["intent"] == SELECTED_DOC_SUMMARY
    assert result.trace.coverage_plan["covered_doc_count"] == 6
    assert len(captured["hits"]) == 6
    assert all(hit.source_type == "source_summary" for hit in captured["hits"])
    assert [stage["stage"] for stage in stages] == [
        "intent_router",
        "scope_resolution",
        "coverage_plan",
        "search",
        "search",
        "context",
        "context",
        "answer",
        "answer",
        "coverage_plan",
    ]
    assert all("False" not in str(stage["detail"]) for stage in stages)
    assert all("True" not in str(stage["detail"]) for stage in stages)


def test_explicit_local_question_narrows_scope_then_uses_top_k() -> None:
    captured: dict[str, object] = {}
    with tempfile.TemporaryDirectory() as tmp:
        config = fake_config(tmp)
        write_guides(config.object_store_dir, count=6)
        with (
            patch("answer.load_config", return_value=config),
            patch("answer.retrieve_and_rerank", side_effect=capture_retrieval(captured)),
            patch("answer.generate_answer", return_value=fake_generation("local answer")),
        ):
            result = answer_query(
                "doc-2 里面的违约金是多少？",
                tenant_id="team_a",
                candidate_limit=20,
                context_limit=5,
                doc_ids=[f"doc-{index}" for index in range(1, 7)],
                doc_version=1,
                request_id="smoke-doc-scope-local",
            )

    assert result.answer == "local answer"
    assert captured["doc_ids"] == ["doc-2"]
    assert result.trace.retrieval_mode == "hybrid_dense_sparse_rerank"
    assert result.trace.doc_ids == ["doc-2"]
    assert result.trace.context_count == 1


def test_zero_document_open_chat_skips_retrieval() -> None:
    stages: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        config = fake_config(tmp)
        with (
            patch("answer.load_config", return_value=config),
            patch("answer.retrieve_and_rerank", side_effect=AssertionError("retrieval must be skipped")),
            patch("answer.generate_chat", return_value=fake_generation("direct answer")),
        ):
            result = answer_query(
                "给我讲一个与知识库无关的笑话",
                tenant_id="team_a",
                candidate_limit=20,
                context_limit=5,
                doc_version=1,
                source_types=["pdf"],
                request_id="smoke-zero-document-open-chat",
                stage_callback=stages.append,
            )

    assert result.answer == "direct answer"
    assert result.trace.retrieval_mode == "direct_llm_no_retrieval"
    assert result.trace.context_count == 0
    assert [stage["stage"] for stage in stages] == [
        "intent_router",
        "scope_resolution",
        "answer",
        "answer",
    ]


def test_large_scope_uses_map_reduce_when_guides_exceed_budget() -> None:
    captured: dict[str, object] = {"calls": []}
    stages: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        config = fake_config(tmp, max_context_chars=180)
        write_guides(config.object_store_dir, count=5, guide_suffix=" " + ("long evidence " * 20))
        with (
            patch("answer.load_config", return_value=config),
            patch("rag_core.document_scope.generate_answer", side_effect=capture_map_reduce_answer(captured)),
        ):
            result = answer_query(
                "整理一下这几份材料",
                tenant_id="team_a",
                candidate_limit=20,
                context_limit=5,
                doc_ids=[f"doc-{index}" for index in range(1, 6)],
                doc_version=1,
                request_id="smoke-doc-scope-map-reduce",
                stage_callback=stages.append,
            )

    assert result.trace.coverage_plan["coverage_mode"] == PER_DOC_MAP_REDUCE
    assert result.trace.coverage_plan["covered_doc_count"] == 5
    assert result.trace.coverage_plan["document_map_batches"] >= 2
    assert len(captured["calls"]) >= 3
    assert result.answer == "final reduced answer [1]"
    reduce_stages = [stage for stage in stages if stage["stage"] == "document_reduce"]
    assert [stage["status"] for stage in reduce_stages] == ["active", "done"]
    assert [stage["status"] for stage in stages if stage["stage"] == "answer"] == ["active", "done"]
    assert [stage["status"] for stage in stages if stage["stage"] == "context"] == ["active", "done"]


def write_guides(object_store_dir: Path, *, count: int, guide_suffix: str = "") -> None:
    rows = [
        {
            "tenant_id": "team_a",
            "source_doc_id": f"doc-{index}",
            "doc_version": 1,
            "title": f"Document {index}",
            "guide": f"Document {index} summary with important facts.{guide_suffix}",
            "model": "fake",
            "updated_at": 1,
        }
        for index in range(1, count + 1)
    ]
    path = object_store_dir / "canonical" / "source_guides.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, rows)


def fake_config(tmp: str, *, max_context_chars: int = 20_000) -> SimpleNamespace:
    return SimpleNamespace(
        object_store_dir=Path(tmp) / "object_store",
        max_context_chars=max_context_chars,
    )


def capture_document_answer(captured: dict[str, object]):
    def _generate_answer(config, query: str, hits: list[SearchHit]) -> AnswerGeneration:
        captured["query"] = query
        captured["hits"] = hits
        return fake_generation("document answer [1] [2] [3] [4] [5] [6]")

    return _generate_answer


def capture_map_reduce_answer(captured: dict[str, object]):
    def _generate_answer(config, query: str, hits: list[SearchHit]) -> AnswerGeneration:
        calls = captured.setdefault("calls", [])
        calls.append({"query": query, "hits": hits})
        if all(hit.doc_id.startswith("partial-summary-") for hit in hits):
            return fake_generation("final reduced answer [1]")
        return fake_generation(f"partial answer covering {','.join(hit.doc_id for hit in hits)}")

    return _generate_answer


def capture_retrieval(captured: dict[str, object]):
    def _retrieve_and_rerank(query: str, **kwargs) -> RetrievalResult:
        captured["doc_ids"] = kwargs.get("doc_ids")
        hit = fake_hit(doc_id="doc-2")
        trace = fake_trace(doc_ids=kwargs.get("doc_ids") or [])
        return RetrievalResult(
            request_id=kwargs.get("request_id") or "smoke-doc-scope-local",
            hits=[hit],
            candidates=[hit],
            reranked=[hit],
            trace=trace,
        )

    return _retrieve_and_rerank


def fake_generation(answer: str) -> AnswerGeneration:
    return AnswerGeneration(
        answer=answer,
        llm_model="fake",
        llm_backend="fake",
        latency_ms=1.0,
        token_usage={},
    )


def fake_hit(*, doc_id: str) -> SearchHit:
    return SearchHit(
        id=f"hit-{doc_id}",
        score=1.0,
        text=f"{doc_id} evidence",
        doc_id=doc_id,
        title=doc_id,
        source_uri=f"memory://{doc_id}",
        source_type="pdf",
        chunk_index=0,
        tenant_id="team_a",
        acl_groups=[],
        metadata={},
    )


def fake_trace(*, doc_ids: list[str]) -> TraceInfo:
    return TraceInfo(
        request_id="smoke-doc-scope-local",
        original_query="doc-2 里面的违约金是多少？",
        rewritten_query="doc-2 里面的违约金是多少？",
        rewrite_backend="fake",
        tenant_id="team_a",
        acl_groups=[],
        doc_version=1,
        current_versions={},
        embedding_model="fake",
        source_types=[],
        doc_ids=doc_ids,
        filter_expr="",
        retrieval_mode="hybrid_dense_sparse_rerank",
        candidate_count=1,
        reranked_count=1,
        context_count=1,
        dropped_by_score=0,
        dropped_by_doc_limit=0,
        dropped_by_budget=0,
        stage_latency_ms={},
    )


if __name__ == "__main__":
    main()
