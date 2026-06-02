from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_answer import evaluate_answers
from eval_retrieval import evaluate_retrieval
from rag_core.config import DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail fast if retrieval/answer metrics do not meet release gates."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "eval_queries.jsonl",
        help="JSONL eval set.",
    )
    parser.add_argument("--retrieval-mode", choices=["dense", "hybrid", "rerank"], default="rerank")
    parser.add_argument("--retrieval-limit", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--context-limit", type=int, default=5)
    parser.add_argument("--min-recall", type=float, default=1.0)
    parser.add_argument("--min-mrr", type=float, default=1.0)
    parser.add_argument("--min-ndcg", type=float, default=1.0)
    parser.add_argument("--max-leakage-failures", type=int, default=0)
    parser.add_argument("--max-p95-retrieval-ms", type=float, default=800.0)
    parser.add_argument("--min-citation-accuracy", type=float, default=1.0)
    parser.add_argument("--min-evidence-hit-rate", type=float, default=1.0)
    parser.add_argument("--min-refusal-quality", type=float, default=1.0)
    parser.add_argument("--json-output", type=Path, help="Write gate report as JSON.")
    args = parser.parse_args()

    retrieval = evaluate_retrieval(
        input_path=args.input,
        limit=args.retrieval_limit,
        mode=args.retrieval_mode,
    )
    answer = evaluate_answers(
        input_path=args.input,
        candidate_limit=args.candidate_limit,
        context_limit=args.context_limit,
    )
    failures = gate_failures(args, retrieval, answer)
    report = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "retrieval": retrieval,
        "answer": answer,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if failures:
        raise SystemExit("release_gate=failed")
    print("release_gate=ok")


def gate_failures(args, retrieval: dict, answer: dict) -> list[str]:
    checks = [
        ("recall", retrieval["recall"], ">=", args.min_recall),
        ("mrr", retrieval["mrr"], ">=", args.min_mrr),
        ("ndcg", retrieval["ndcg"], ">=", args.min_ndcg),
        (
            "permission_leakage_failures",
            retrieval["permission_leakage_failures"],
            "<=",
            args.max_leakage_failures,
        ),
        ("p95_retrieval_ms", retrieval["p95_latency_ms"], "<=", args.max_p95_retrieval_ms),
        ("citation_accuracy", answer["citation_accuracy"], ">=", args.min_citation_accuracy),
        ("evidence_hit_rate", answer["evidence_hit_rate"], ">=", args.min_evidence_hit_rate),
        ("refusal_quality", answer["refusal_quality"], ">=", args.min_refusal_quality),
    ]
    failures: list[str] = []
    for name, value, operator, threshold in checks:
        passed = value >= threshold if operator == ">=" else value <= threshold
        if not passed:
            failures.append(f"{name}={value} expected {operator} {threshold}")
    return failures


if __name__ == "__main__":
    main()
