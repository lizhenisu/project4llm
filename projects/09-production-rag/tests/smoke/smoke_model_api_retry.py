from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.model_api_retry import call_model_api_with_retries, model_api_metrics_snapshot


class ApiError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"api status {status_code}")
        self.status_code = status_code


def main() -> None:
    old_retries = os.environ.get("RAG_MODEL_API_RETRIES")
    old_backoff = os.environ.get("RAG_MODEL_API_BACKOFF_SECONDS")
    old_max_backoff = os.environ.get("RAG_MODEL_API_BACKOFF_MAX_SECONDS")
    try:
        os.environ["RAG_MODEL_API_RETRIES"] = "3"
        os.environ["RAG_MODEL_API_BACKOFF_SECONDS"] = "0"
        os.environ["RAG_MODEL_API_BACKOFF_MAX_SECONDS"] = "0"
        test_transient_errors_retry_until_success()
        test_non_transient_errors_do_not_retry()
    finally:
        restore_env("RAG_MODEL_API_RETRIES", old_retries)
        restore_env("RAG_MODEL_API_BACKOFF_SECONDS", old_backoff)
        restore_env("RAG_MODEL_API_BACKOFF_MAX_SECONDS", old_max_backoff)
    print("smoke_model_api_retry=ok")


def test_transient_errors_retry_until_success() -> None:
    attempts = {"count": 0}
    before = operation_metrics("flaky")

    def flaky_call() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ApiError(429)
        return "ok"

    with patch("rag_core.model_api_retry.time.sleep", return_value=None):
        assert call_model_api_with_retries("flaky", flaky_call) == "ok"
    assert attempts["count"] == 3
    after = operation_metrics("flaky")
    assert after["calls_total"] == before["calls_total"] + 1
    assert after["attempts_total"] == before["attempts_total"] + 3
    assert after["retries_total"] == before["retries_total"] + 2
    assert after["successes_total"] == before["successes_total"] + 1
    assert after["failures_total"] == before["failures_total"]


def test_non_transient_errors_do_not_retry() -> None:
    attempts = {"count": 0}
    before = operation_metrics("bad_request")

    def bad_request() -> str:
        attempts["count"] += 1
        raise ApiError(400)

    try:
        call_model_api_with_retries("bad_request", bad_request)
    except ApiError as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("expected ApiError")
    assert attempts["count"] == 1
    after = operation_metrics("bad_request")
    assert after["calls_total"] == before["calls_total"] + 1
    assert after["attempts_total"] == before["attempts_total"] + 1
    assert after["retries_total"] == before["retries_total"]
    assert after["failures_total"] == before["failures_total"] + 1
    assert after["latency_max_ms"] >= 0


def operation_metrics(operation: str) -> dict[str, int | float]:
    return model_api_metrics_snapshot()["operations"].get(
        operation,
        {
            "calls_total": 0,
            "attempts_total": 0,
            "retries_total": 0,
            "successes_total": 0,
            "failures_total": 0,
            "latency_total_ms": 0.0,
            "latency_max_ms": 0.0,
            "latency_avg_ms": 0.0,
        },
    )


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
