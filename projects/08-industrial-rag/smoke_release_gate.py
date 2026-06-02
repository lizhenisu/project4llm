from __future__ import annotations

from pathlib import Path

from release_gate import gate_failures


def main() -> None:
    retrieval = {
        "recall": 1.0,
        "mrr": 1.0,
        "ndcg": 1.0,
        "permission_leakage_failures": 0,
        "p95_latency_ms": 10.0,
    }
    answer = {
        "citation_accuracy": 1.0,
        "evidence_hit_rate": 1.0,
        "refusal_quality": 1.0,
    }
    args = Args()
    assert gate_failures(args, retrieval, answer) == []
    failing = dict(retrieval, recall=0.5, permission_leakage_failures=1)
    failures = gate_failures(args, failing, answer)
    assert any(item.startswith("recall=") for item in failures)
    assert any(item.startswith("permission_leakage_failures=") for item in failures)
    print("smoke_release_gate=ok")


class Args:
    input = Path("unused")
    min_recall = 1.0
    min_mrr = 1.0
    min_ndcg = 1.0
    max_leakage_failures = 0
    max_p95_retrieval_ms = 800.0
    min_citation_accuracy = 1.0
    min_evidence_hit_rate = 1.0
    min_refusal_quality = 1.0


if __name__ == "__main__":
    main()
