from __future__ import annotations

import tempfile
from pathlib import Path

from build_eval_from_feedback import build_eval_rows_from_feedback
from rag_core.events import append_event


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        runtime_dir = Path(tmp) / "runtime"
        append_event(
            runtime_dir,
            "answer_events",
            {
                "request_id": "answer-good-1",
                "query": "RAG 延迟怎么排查？",
                "auth_context": {"tenant_id": "team_a", "acl_groups": ["ops"]},
                "doc_version": 2,
                "source_types": ["md"],
                "trace": {"tenant_id": "team_a"},
                "final_context": [
                    {"doc_id": "rag-runbook"},
                    {"doc_id": "hybrid-search"},
                ],
            },
        )
        append_event(
            runtime_dir,
            "feedback_events",
            {
                "request_id": "answer-good-1",
                "rating": 1,
                "comment": "答案正确",
                "selected_doc_ids": ["rag-runbook"],
            },
        )
        append_event(
            runtime_dir,
            "retrieval_events",
            {
                "request_id": "search-bad-1",
                "query": "不能回答的问题",
                "auth_context": {"tenant_id": "team_a", "acl_groups": ["ops"]},
                "trace": {"tenant_id": "team_a"},
                "final_context": [{"doc_id": "wrong-doc"}],
            },
        )
        append_event(
            runtime_dir,
            "feedback_events",
            {
                "request_id": "search-bad-1",
                "rating": -1,
                "comment": "没有证据",
                "selected_doc_ids": [],
            },
        )
        append_event(
            runtime_dir,
            "feedback_events",
            {
                "request_id": "missing-request",
                "rating": 1,
                "comment": "orphan feedback should be skipped",
            },
        )

        positive_only = build_eval_rows_from_feedback(runtime_dir)
        with_negative = build_eval_rows_from_feedback(
            runtime_dir,
            include_negative=True,
        )

    assert len(positive_only) == 1
    assert positive_only[0]["query"] == "RAG 延迟怎么排查？"
    assert positive_only[0]["tenant_id"] == "team_a"
    assert positive_only[0]["expected_doc_ids"] == ["rag-runbook"]
    assert positive_only[0]["answerable"] is True
    assert positive_only[0]["query_type"] == "feedback_positive"
    assert positive_only[0]["doc_version"] == 2
    assert positive_only[0]["source_types"] == ["md"]

    assert len(with_negative) == 2
    negative = with_negative[1]
    assert negative["query"] == "不能回答的问题"
    assert negative["expected_doc_ids"] == []
    assert negative["answerable"] is False
    assert negative["query_type"] == "feedback_negative"
    print("smoke_feedback_eval_export=ok")


if __name__ == "__main__":
    main()
