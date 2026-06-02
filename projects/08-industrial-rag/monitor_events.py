from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rag_core.config import load_config
from rag_core.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize runtime retrieval/answer/feedback events."
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        help="Runtime event directory. Defaults to RAG_RUNTIME_DIR.",
    )
    args = parser.parse_args()

    config = load_config()
    runtime_dir = args.runtime_dir or config.runtime_dir
    summary = summarize_runtime_events(runtime_dir)
    print_summary(summary)


def summarize_runtime_events(runtime_dir: Path) -> dict[str, Any]:
    retrieval_events = read_events(runtime_dir / "retrieval_events.jsonl")
    answer_events = read_events(runtime_dir / "answer_events.jsonl")
    feedback_events = read_events(runtime_dir / "feedback_events.jsonl")

    stage_latencies = collect_stage_latencies([*retrieval_events, *answer_events])
    llm_latencies = [
        float(event.get("llm", {}).get("latency_ms", 0.0))
        for event in answer_events
        if "llm" in event
    ]
    retrieval_modes = Counter(
        event.get("trace", {}).get("retrieval_mode", "<missing>")
        for event in retrieval_events
    )
    context_counts = [
        int(event.get("trace", {}).get("context_count", 0))
        for event in [*retrieval_events, *answer_events]
        if "trace" in event
    ]
    doc_counts = Counter(
        hit.get("doc_id", "<missing>")
        for event in [*retrieval_events, *answer_events]
        for hit in event.get("final_context", [])
    )
    ratings = Counter(str(event.get("rating", "<missing>")) for event in feedback_events)

    return {
        "runtime_dir": str(runtime_dir),
        "retrieval_events": len(retrieval_events),
        "answer_events": len(answer_events),
        "feedback_events": len(feedback_events),
        "retrieval_modes": dict(sorted(retrieval_modes.items())),
        "context": {
            "avg": avg(context_counts),
            "zero_context_events": sum(1 for count in context_counts if count == 0),
        },
        "stage_latency_ms": {
            name: latency_summary(values)
            for name, values in sorted(stage_latencies.items())
        },
        "llm_latency_ms": latency_summary(llm_latencies),
        "top_context_docs": dict(doc_counts.most_common(10)),
        "feedback_ratings": dict(sorted(ratings.items())),
    }


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def collect_stage_latencies(events: list[dict[str, Any]]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for event in events:
        stages = event.get("trace", {}).get("stage_latency_ms", {})
        for stage, latency in stages.items():
            values[str(stage)].append(float(latency))
    return values


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "count": len(values),
        "avg": round(avg(values), 2),
        "p50": round(percentile(values, 0.50), 2),
        "p95": round(percentile(values, 0.95), 2),
        "p99": round(percentile(values, 0.99), 2),
    }


def avg(values: list[float] | list[int]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
    return ordered[index]


def print_summary(summary: dict[str, Any]) -> None:
    print(f"runtime_dir={summary['runtime_dir']}")
    print(f"retrieval_events={summary['retrieval_events']}")
    print(f"answer_events={summary['answer_events']}")
    print(f"feedback_events={summary['feedback_events']}")
    print(f"retrieval_modes={summary['retrieval_modes']}")
    print(f"context={summary['context']}")
    print(f"stage_latency_ms={summary['stage_latency_ms']}")
    print(f"llm_latency_ms={summary['llm_latency_ms']}")
    print(f"top_context_docs={summary['top_context_docs']}")
    print(f"feedback_ratings={summary['feedback_ratings']}")


if __name__ == "__main__":
    main()
