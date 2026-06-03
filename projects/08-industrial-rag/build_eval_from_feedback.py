from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rag_core.config import DATA_DIR, load_config
from rag_core.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an eval JSONL from runtime answer/retrieval feedback events."
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        help="Runtime event directory. Defaults to RAG_RUNTIME_DIR.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "feedback_eval_queries.jsonl",
        help="Output eval JSONL compatible with eval_retrieval.py and eval_answer.py.",
    )
    parser.add_argument(
        "--min-positive-rating",
        type=int,
        default=1,
        help="Feedback rating >= this value is treated as positive evidence.",
    )
    parser.add_argument(
        "--include-negative",
        action="store_true",
        help="Include negative feedback rows. Rows without selected docs become unanswerable examples.",
    )
    args = parser.parse_args()

    config = load_config()
    runtime_dir = args.runtime_dir or config.runtime_dir
    rows = build_eval_rows_from_feedback(
        runtime_dir,
        min_positive_rating=args.min_positive_rating,
        include_negative=args.include_negative,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    print(f"runtime_dir={runtime_dir}")
    print(f"output={args.output}")
    print(f"eval_rows={len(rows)}")


def build_eval_rows_from_feedback(
    runtime_dir: Path,
    *,
    min_positive_rating: int = 1,
    include_negative: bool = False,
) -> list[dict[str, Any]]:
    request_events = index_request_events(runtime_dir)
    feedback_events = read_events(runtime_dir / "feedback_events.jsonl")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...], bool]] = set()

    for feedback in feedback_events:
        request_id = str(feedback.get("request_id", ""))
        if not request_id:
            continue
        request_event = request_events.get(request_id)
        if not request_event:
            continue
        rating = int(feedback.get("rating", 0))
        selected_doc_ids = unique_strings(feedback.get("selected_doc_ids", []))
        final_doc_ids = unique_strings(
            hit.get("doc_id")
            for hit in request_event.get("final_context", [])
        )

        if rating >= min_positive_rating:
            expected_doc_ids = selected_doc_ids or final_doc_ids
            answerable = bool(expected_doc_ids)
            query_type = "feedback_positive"
        elif include_negative:
            expected_doc_ids = selected_doc_ids
            answerable = bool(selected_doc_ids)
            query_type = "feedback_negative_with_correction" if selected_doc_ids else "feedback_negative"
        else:
            continue

        row = {
            "query": request_event.get("query", ""),
            "tenant_id": resolve_tenant_id(request_event),
            "expected_doc_ids": expected_doc_ids,
            "answerable": answerable,
            "query_type": query_type,
            "source_request_id": request_id,
            "feedback_rating": rating,
        }
        if feedback.get("comment"):
            row["feedback_comment"] = feedback["comment"]
        if request_event.get("source_types"):
            row["source_types"] = request_event["source_types"]
        if request_event.get("doc_version") is not None:
            row["doc_version"] = request_event["doc_version"]

        key = (row["query"], tuple(row["expected_doc_ids"]), row["answerable"])
        if row["query"] and key not in seen:
            seen.add(key)
            rows.append(row)

    return rows


def index_request_events(runtime_dir: Path) -> dict[str, dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    for path in (runtime_dir / "retrieval_events.jsonl", runtime_dir / "answer_events.jsonl"):
        for event in read_events(path):
            request_id = str(event.get("request_id", ""))
            if request_id:
                events[request_id] = event
    return events


def resolve_tenant_id(event: dict[str, Any]) -> str:
    auth_context = event.get("auth_context", {})
    if isinstance(auth_context, dict) and auth_context.get("tenant_id"):
        return str(auth_context["tenant_id"])
    trace = event.get("trace", {})
    if isinstance(trace, dict) and trace.get("tenant_id"):
        return str(trace["tenant_id"])
    return "team_a"


def unique_strings(values) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        if value is None:
            continue
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


if __name__ == "__main__":
    main()
