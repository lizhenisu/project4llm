from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.model_api_retry import model_api_slot
from serve import query_stream_semaphore
from serve import query_stream_tenant_semaphore


def main() -> None:
    test_model_api_slot_times_out_when_full()
    test_query_stream_semaphore_rejects_when_full()
    test_query_stream_tenant_semaphore_rejects_when_full()
    print("smoke_concurrency_backpressure=ok")


def test_model_api_slot_times_out_when_full() -> None:
    old_limit = os.environ.get("RAG_MODEL_API_MAX_CONCURRENCY")
    old_timeout = os.environ.get("RAG_MODEL_API_QUEUE_TIMEOUT_SECONDS")
    os.environ["RAG_MODEL_API_MAX_CONCURRENCY"] = "1"
    os.environ["RAG_MODEL_API_QUEUE_TIMEOUT_SECONDS"] = "0"
    try:
        with model_api_slot("outer"):
            try:
                with model_api_slot("inner"):
                    raise AssertionError("inner model API slot should not be acquired")
            except RuntimeError as exc:
                assert "Model API concurrency limit reached" in str(exc)
    finally:
        restore_env("RAG_MODEL_API_MAX_CONCURRENCY", old_limit)
        restore_env("RAG_MODEL_API_QUEUE_TIMEOUT_SECONDS", old_timeout)


def test_query_stream_semaphore_rejects_when_full() -> None:
    semaphore = query_stream_semaphore(1)
    acquired = semaphore.acquire(blocking=False)
    assert acquired
    try:
        assert not semaphore.acquire(blocking=False)
    finally:
        semaphore.release()


def test_query_stream_tenant_semaphore_rejects_when_full() -> None:
    semaphore = query_stream_tenant_semaphore(1, "tenant-smoke")
    acquired = semaphore.acquire(blocking=False)
    assert acquired
    try:
        assert not semaphore.acquire(blocking=False)
    finally:
        semaphore.release()


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
