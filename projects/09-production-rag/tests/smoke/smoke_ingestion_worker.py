from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.ingestion_jobs import (  # noqa: E402
    IngestionJobRunner,
    ingestion_execution_mode,
    run_ingestion_worker,
    submit_upload_ingestion_job,
)
from rag_core.sources import SourceSummary  # noqa: E402


def main() -> None:
    test_external_mode_leaves_persisted_task_for_worker()
    test_worker_polls_and_shuts_down_cleanly()
    test_embedded_runner_polling_is_singleton_and_stoppable()
    test_invalid_execution_mode_is_rejected()
    print("smoke_ingestion_worker=ok")


def test_external_mode_leaves_persisted_task_for_worker() -> None:
    old_mode = os.environ.get("RAG_INGEST_EXECUTION_MODE")
    os.environ["RAG_INGEST_EXECUTION_MODE"] = "external"
    try:
        with patch("rag_core.ingestion_jobs.ingestion_job_runner") as embedded_runner:
            accepted = submit_upload_ingestion_job(
                pending_source=source(),
                saved_path=Path("/tmp/synthetic-worker.txt"),
                tenant_id="synthetic-worker-tenant",
                acl_groups=["engineering"],
                doc_version=1,
                language="zh",
            )
        assert accepted is True
        embedded_runner.assert_not_called()
    finally:
        restore_env("RAG_INGEST_EXECUTION_MODE", old_mode)


def test_worker_polls_and_shuts_down_cleanly() -> None:
    stop_event = threading.Event()

    class FakeRunner:
        def __init__(self) -> None:
            self.drain_count = 0
            self.shutdown_calls: list[bool] = []

        def drain_pending(self) -> None:
            self.drain_count += 1
            if self.drain_count == 2:
                stop_event.set()

        def shutdown(self, *, wait: bool = True) -> None:
            self.shutdown_calls.append(wait)

    runner = FakeRunner()
    run_ingestion_worker(stop_event=stop_event, poll_seconds=0.01, runner=runner)
    assert runner.drain_count == 2
    assert runner.shutdown_calls == [True]


def test_embedded_runner_polling_is_singleton_and_stoppable() -> None:
    runner = IngestionJobRunner(workers=1, queue_limit=1)
    with patch.object(runner, "drain_pending") as drain_pending:
        runner.start_polling(poll_seconds=0.1)
        first_thread = runner._poll_thread
        runner.start_polling(poll_seconds=0.1)
        assert runner._poll_thread is first_thread
        wait_for(lambda: drain_pending.call_count >= 1)
        runner.shutdown(wait=True)
    assert first_thread is not None
    assert not first_thread.is_alive()


def test_invalid_execution_mode_is_rejected() -> None:
    old_mode = os.environ.get("RAG_INGEST_EXECUTION_MODE")
    os.environ["RAG_INGEST_EXECUTION_MODE"] = "not-a-mode"
    try:
        try:
            ingestion_execution_mode()
        except ValueError as exc:
            assert "embedded" in str(exc)
            assert "external" in str(exc)
        else:
            raise AssertionError("invalid ingestion execution mode was accepted")
    finally:
        restore_env("RAG_INGEST_EXECUTION_MODE", old_mode)


def source() -> SourceSummary:
    return SourceSummary(
        doc_id="synthetic-worker-doc",
        title="synthetic-worker.txt",
        source_type="txt",
        source_uri="/tmp/synthetic-worker.txt",
        doc_version=1,
        chunk_count=0,
        acl_groups=["engineering"],
        status="queued",
        current=False,
        created_at=1,
        updated_at=1,
    )


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def wait_for(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not reached before timeout")


if __name__ == "__main__":
    main()
