from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from answer import answer_query
from answer_multimodal import answer_multimodal_query
from rag_core.citations import citation_accuracy, faithfulness_score, is_refusal, term_coverage
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
        "--mode",
        choices=["text", "multimodal"],
        default="text",
        help="Answer pipeline to evaluate.",
    )
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
            mode=args.mode,
        )
        print(f"citation_accuracy: {metrics['citation_accuracy']:.3f}")
        print(f"evidence_hit_rate: {metrics['evidence_hit_rate']:.3f}")
        print(f"refusal_quality: {metrics['refusal_quality']:.3f}")
        print(f"answer_correctness: {metrics['answer_correctness']:.3f}")
        print(f"faithfulness: {metrics['faithfulness']:.3f}")
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
    mode: str = "text",
) -> dict[str, float | int]:
    rows = read_jsonl(input_path)
    citation_scores: list[float] = []
    evidence_hits = 0
    refusal_correct = 0
    answerable_count = 0
    unanswerable_count = 0
    correctness_scores: list[float] = []
    faithfulness_scores: list[float] = []

    for row in rows:
        result = run_answer_eval_query(
            row,
            candidate_limit=candidate_limit,
            context_limit=context_limit,
            mode=mode,
        )
        expected, returned = eval_targets(row, result.hits)
        answerable = bool(row.get("answerable", bool(expected)))
        refused = is_refusal(result.answer)
        citation_scores.append(citation_accuracy(result.answer, len(result.hits)))
        evidence_text = "\n".join(hit.text for hit in result.hits)
        correctness_scores.append(
            term_coverage(result.answer, list(row.get("expected_answer_terms", [])))
        )
        faithfulness_scores.append(
            faithfulness_score(
                result.answer,
                evidence_text,
                list(row.get("unsupported_answer_terms", [])),
            )
        )

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
        "mode": mode,
        "query_count": len(rows),
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
        "citation_accuracy": avg(citation_scores),
        "evidence_hit_rate": evidence_hits / answerable_count if answerable_count else 0.0,
        "refusal_quality": refusal_correct / unanswerable_count if unanswerable_count else 1.0,
        "answer_correctness": avg(correctness_scores),
        "faithfulness": avg(faithfulness_scores),
    }


def run_answer_eval_query(
    row: dict,
    *,
    candidate_limit: int,
    context_limit: int,
    mode: str,
):
    if mode == "multimodal":
        return answer_multimodal_query(
            row["query"],
            tenant_id=row["tenant_id"],
            candidate_limit=candidate_limit,
            context_limit=context_limit,
            acl_groups=row.get("acl_groups") or None,
            doc_version=row.get("doc_version"),
            source_types=row.get("source_types") or None,
            history=row.get("history") or None,
        )
    return answer_query(
        row["query"],
        tenant_id=row["tenant_id"],
        candidate_limit=candidate_limit,
        context_limit=context_limit,
        acl_groups=row.get("acl_groups") or None,
        doc_version=row.get("doc_version"),
        source_types=row.get("source_types") or None,
        history=row.get("history") or None,
    )


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def eval_targets(row: dict, hits) -> tuple[set[str], set[str]]:
    expected_chunk_ids = set(row.get("expected_chunk_ids", []))
    if expected_chunk_ids:
        return expected_chunk_ids, {hit_eval_chunk_id(hit, expected_chunk_ids) for hit in hits}
    return set(row.get("expected_doc_ids", [])), {hit.doc_id for hit in hits}


def hit_eval_chunk_id(hit, expected: set[str]) -> str:
    if hit.id in expected:
        return hit.id
    metadata_chunk_id = str((hit.metadata or {}).get("chunk_id", ""))
    if metadata_chunk_id in expected:
        return metadata_chunk_id
    return f"{hit.doc_id}:{hit.chunk_index}"


if __name__ == "__main__":
    main()
