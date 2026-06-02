from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from answer import answer_query
from rag_core.citations import citation_accuracy, is_refusal
from rag_core.config import DATA_DIR
from rag_core.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate answer citation accuracy, evidence hit, and refusal quality."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "eval_queries.jsonl",
        help="JSONL eval set.",
    )
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--context-limit", type=int, default=5)
    parser.add_argument(
        "--force-refusal-threshold",
        action="store_true",
        help="Temporarily set a high rerank threshold to exercise refusal behavior.",
    )
    parser.add_argument("--json-output", type=Path, help="Write metrics as JSON.")
    args = parser.parse_args()

    previous_threshold = os.environ.get("RAG_MIN_RERANK_SCORE")
    if args.force_refusal_threshold:
        os.environ["RAG_MIN_RERANK_SCORE"] = "999"

    try:
        metrics = evaluate_answers(
            input_path=args.input,
            candidate_limit=args.candidate_limit,
            context_limit=args.context_limit,
        )
        print(f"citation_accuracy: {metrics['citation_accuracy']:.3f}")
        print(f"evidence_hit_rate: {metrics['evidence_hit_rate']:.3f}")
        print(f"refusal_quality: {metrics['refusal_quality']:.3f}")
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    finally:
        if args.force_refusal_threshold:
            if previous_threshold is None:
                os.environ.pop("RAG_MIN_RERANK_SCORE", None)
            else:
                os.environ["RAG_MIN_RERANK_SCORE"] = previous_threshold


def evaluate_answers(
    *,
    input_path: Path,
    candidate_limit: int,
    context_limit: int,
) -> dict[str, float | int]:
    rows = read_jsonl(input_path)
    citation_scores: list[float] = []
    evidence_hits = 0
    refusal_correct = 0
    answerable_count = 0
    unanswerable_count = 0

    for row in rows:
        result = answer_query(
            row["query"],
            tenant_id=row["tenant_id"],
            candidate_limit=candidate_limit,
            context_limit=context_limit,
        )
        expected = set(row.get("expected_doc_ids", []))
        returned = {hit.doc_id for hit in result.hits}
        answerable = bool(row.get("answerable", bool(expected)))
        refused = is_refusal(result.answer)
        citation_scores.append(citation_accuracy(result.answer, len(result.hits)))

        if answerable:
            answerable_count += 1
            if returned & expected:
                evidence_hits += 1
        else:
            unanswerable_count += 1
            if refused or not returned:
                refusal_correct += 1

        print(
            f"query={row['query']} answerable={answerable} refused={refused} "
            f"expected={sorted(expected)} returned={sorted(returned)}"
        )

    return {
        "query_count": len(rows),
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
        "citation_accuracy": avg(citation_scores),
        "evidence_hit_rate": evidence_hits / answerable_count if answerable_count else 0.0,
        "refusal_quality": refusal_correct / unanswerable_count if unanswerable_count else 1.0,
    }


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
