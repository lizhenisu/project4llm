from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tests.load.milvus_search_load import SearchSample, build_summary, tenant_for_index


def main() -> None:
    args = SimpleNamespace(
        base_url="http://127.0.0.1:8008",
        concurrency=2,
        tenant_count=3,
    )
    samples = [
        SearchSample(
            index=0,
            tenant_id="tenant-load-0000",
            ok=True,
            latency_ms=100.0,
            status_code=200,
            hit_count=5,
            candidate_count=20,
            reranked_count=20,
            stage_latency_ms={"milvus_search": 40.0, "rerank": 20.0},
        ),
        SearchSample(
            index=1,
            tenant_id="tenant-load-0001",
            ok=True,
            latency_ms=200.0,
            status_code=200,
            hit_count=4,
            candidate_count=18,
            reranked_count=18,
            stage_latency_ms={"milvus_search": 80.0, "rerank": 30.0},
        ),
    ]
    summary = build_summary(args, samples, wall_ms=220.0)

    assert tenant_for_index("tenant-load", 3, 4) == "tenant-load-0001"
    assert summary["success"] == 2
    assert summary["failed"] == 0
    assert summary["throughput_rps"] == 9.09
    assert summary["latency_ms"]["p95"] == 195.0
    assert summary["stage_latency_ms"]["milvus_search"]["p95"] == 78.0
    assert summary["status_counts"] == {"200": 2}
    print("smoke_milvus_search_load=ok")


if __name__ == "__main__":
    main()
