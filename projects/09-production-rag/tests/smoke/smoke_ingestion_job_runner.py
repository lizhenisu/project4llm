from __future__ import annotations

import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.ingestion_jobs import IngestionJobRunner
from rag_core.sources import SourceSummary


def main() -> None:
    test_ingestion_runner_limits_concurrency_and_cleans_successful_tasks()
    test_ingestion_runner_limits_per_tenant_inflight_tasks()
    test_ingestion_runner_persists_failures()
    print("smoke_ingestion_job_runner=ok")


def test_ingestion_runner_limits_concurrency_and_cleans_successful_tasks() -> None:
    lock = threading.Lock()
    started = threading.Event()
    release = threading.Event()
    state = {"active": 0, "max_active": 0, "deleted": [], "statuses": []}

    def fake_ingest_uploaded_path(**kwargs):
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        started.set()
        release.wait(timeout=5)
        with lock:
            state["active"] -= 1

    def fake_delete_source_task(**kwargs):
        state["deleted"].append(kwargs["task_id"])

    def fake_update_source_task_status(**kwargs):
        state["statuses"].append((kwargs["source"].doc_id, kwargs["status"]))
        return kwargs["source"]

    runner = IngestionJobRunner(workers=1, queue_limit=1, tenant_queue_limit=1)
    with tempfile.TemporaryDirectory() as tmp:
        patches = (
            patch("rag_core.ingestion_jobs.load_config", return_value=SimpleNamespace()),
            patch("rag_core.ingestion_jobs.ingest_uploaded_path", side_effect=fake_ingest_uploaded_path),
            patch("rag_core.ingestion_jobs.delete_source_task", side_effect=fake_delete_source_task),
            patch("rag_core.ingestion_jobs.update_source_task_status", side_effect=fake_update_source_task_status),
        )
        for manager in patches:
            manager.__enter__()
        try:
            first = runner.submit_upload(
                pending_source=fake_source("task-1"),
                saved_path=Path(tmp) / "first.txt",
                tenant_id="team_a",
                acl_groups=["engineering"],
                doc_version=None,
                language="zh",
            )
            assert first is True
            assert started.wait(timeout=5)
            second = runner.submit_upload(
                pending_source=fake_source("task-2"),
                saved_path=Path(tmp) / "second.txt",
                tenant_id="team_a",
                acl_groups=["engineering"],
                doc_version=None,
                language="zh",
            )
            assert second is False
            release.set()
            wait_for(lambda: state["deleted"] == ["task-1"])
        finally:
            for manager in reversed(patches):
                manager.__exit__(None, None, None)

    assert state["max_active"] == 1
    assert state["statuses"] == [("task-1", "processing")]


def test_ingestion_runner_limits_per_tenant_inflight_tasks() -> None:
    release = threading.Event()
    started = threading.Event()
    state = {"active_by_tenant": {}, "deleted": []}

    def fake_ingest_uploaded_path(**kwargs):
        tenant_id = kwargs["tenant_id"]
        state["active_by_tenant"][tenant_id] = state["active_by_tenant"].get(tenant_id, 0) + 1
        started.set()
        release.wait(timeout=5)
        state["active_by_tenant"][tenant_id] -= 1

    def fake_delete_source_task(**kwargs):
        state["deleted"].append(kwargs["task_id"])

    runner = IngestionJobRunner(workers=2, queue_limit=4, tenant_queue_limit=1)
    with tempfile.TemporaryDirectory() as tmp:
        patches = (
            patch("rag_core.ingestion_jobs.load_config", return_value=SimpleNamespace()),
            patch("rag_core.ingestion_jobs.ingest_uploaded_path", side_effect=fake_ingest_uploaded_path),
            patch("rag_core.ingestion_jobs.delete_source_task", side_effect=fake_delete_source_task),
            patch("rag_core.ingestion_jobs.update_source_task_status", side_effect=lambda **kwargs: kwargs["source"]),
        )
        for manager in patches:
            manager.__enter__()
        try:
            assert runner.submit_upload(
                pending_source=fake_source("task-a1"),
                saved_path=Path(tmp) / "a1.txt",
                tenant_id="team_a",
                acl_groups=["engineering"],
                doc_version=None,
                language="zh",
            )
            assert started.wait(timeout=5)
            assert not runner.submit_upload(
                pending_source=fake_source("task-a2"),
                saved_path=Path(tmp) / "a2.txt",
                tenant_id="team_a",
                acl_groups=["engineering"],
                doc_version=None,
                language="zh",
            )
            assert runner.submit_upload(
                pending_source=fake_source("task-b1"),
                saved_path=Path(tmp) / "b1.txt",
                tenant_id="team_b",
                acl_groups=["engineering"],
                doc_version=None,
                language="zh",
            )
            release.set()
            wait_for(lambda: sorted(state["deleted"]) == ["task-a1", "task-b1"])
        finally:
            for manager in reversed(patches):
                manager.__exit__(None, None, None)


def test_ingestion_runner_persists_failures() -> None:
    state = {"failed": []}

    def fake_ingest_uploaded_path(**kwargs):
        raise RuntimeError("boom")

    def fake_fail_source_task(**kwargs):
        state["failed"].append((kwargs["source"].doc_id, kwargs["error"]))

    runner = IngestionJobRunner(workers=1, queue_limit=1, tenant_queue_limit=1)
    with tempfile.TemporaryDirectory() as tmp:
        patches = (
            patch("rag_core.ingestion_jobs.load_config", return_value=SimpleNamespace()),
            patch("rag_core.ingestion_jobs.ingest_uploaded_path", side_effect=fake_ingest_uploaded_path),
            patch("rag_core.ingestion_jobs.fail_source_task", side_effect=fake_fail_source_task),
            patch("rag_core.ingestion_jobs.update_source_task_status", side_effect=lambda **kwargs: kwargs["source"]),
        )
        for manager in patches:
            manager.__enter__()
        try:
            accepted = runner.submit_upload(
                pending_source=fake_source("task-failed"),
                saved_path=Path(tmp) / "failed.txt",
                tenant_id="team_a",
                acl_groups=["engineering"],
                doc_version=None,
                language="zh",
            )
            assert accepted is True
            wait_for(lambda: bool(state["failed"]))
        finally:
            for manager in reversed(patches):
                manager.__exit__(None, None, None)

    assert state["failed"] == [("task-failed", "boom")]


def fake_source(doc_id: str) -> SourceSummary:
    return SourceSummary(
        doc_id=doc_id,
        title=f"{doc_id}.txt",
        source_type="txt",
        source_uri=f"memory://{doc_id}",
        doc_version=1,
        chunk_count=0,
        acl_groups=["engineering"],
        status="queued",
        current=False,
        created_at=1,
        updated_at=1,
        child_doc_ids=[],
    )


def wait_for(predicate, *, timeout: float = 5.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not reached before timeout")


if __name__ == "__main__":
    main()
