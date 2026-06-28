from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tests.load.rag_query_load import (  # noqa: E402
    RequestSample,
    build_summary,
    classify_http_error,
    run_batch,
    tenant_resolution_error,
)


def main() -> None:
    args = SimpleNamespace(
        base_url="http://api.test",
        endpoint="/query/stream",
        external_mode="mock",
        concurrency=2,
        max_failure_rate=0.5,
        max_p95_ms=500.0,
        max_first_event_p95_ms=100.0,
        min_throughput_rps=1.0,
        min_accepted_rate=0.5,
    )
    samples = [
        RequestSample(
            index=0,
            request_id="ok",
            ok=True,
            total_ms=200.0,
            status_code=200,
            resolved_tenant_id="tenant-a",
            first_event_ms=50.0,
            answer_chars=100,
            citations=2,
            stages_ms={"answer": 120.0},
            stage_events=4,
        ),
        RequestSample(
            index=1,
            request_id="rejected",
            ok=False,
            total_ms=10.0,
            status_code=503,
            error="HTTP 503: Query service is busy for this tenant",
            error_kind="rejected_tenant",
        ),
    ]
    warmup = [
        RequestSample(
            index=0,
            request_id="warmup",
            ok=True,
            total_ms=300.0,
            status_code=200,
        )
    ]
    summary = build_summary(args, samples, wall_ms=500.0, warmup_samples=warmup)
    assert summary["warmup"] == {"requests": 1, "success": 1, "failed": 0}
    assert summary["accepted"] == 1
    assert summary["accepted_rate"] == 0.5
    assert summary["failure_rate"] == 0.5
    assert summary["status_counts"] == {"200": 1, "503": 1}
    assert summary["error_kind_counts"] == {"rejected_tenant": 1}
    assert summary["capacity_gate"]["passed"] is True
    assert classify_http_error(503, "busy for this user") == "rejected_user"
    assert tenant_resolution_error("tenant-a", "tenant-b").startswith(
        "tenant_resolution_mismatch"
    )

    args.max_failure_rate = 0.0
    failed = build_summary(args, samples, wall_ms=500.0)
    assert failed["capacity_gate"]["passed"] is False
    verify_explicit_concurrency()
    print("smoke_rag_query_load=ok")


def verify_explicit_concurrency() -> None:
    lock = threading.Lock()
    active = 0
    peak = 0

    def fake_send(_args, _question: str, index: int, _label: str) -> RequestSample:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return RequestSample(
            index=index,
            request_id=f"concurrency-{index}",
            ok=True,
            total_ms=50.0,
            status_code=200,
        )

    args = SimpleNamespace(concurrency=8, fail_fast=False)
    with patch("tests.load.rag_query_load.send_stream_request", side_effect=fake_send):
        samples = asyncio.run(run_batch(args, ["synthetic"], total=8, label="smoke"))
    assert len(samples) == 8
    assert peak == 8


if __name__ == "__main__":
    main()
