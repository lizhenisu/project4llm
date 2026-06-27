from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.config import load_config  # noqa: E402
from rag_core.database import connect_metadata_db  # noqa: E402
from rag_core.ingestion_jobs import IngestionJobRunner  # noqa: E402
from rag_core.sources import (  # noqa: E402
    SourceSummary,
    claim_source_task_for_processing,
    delete_source_task,
    fail_source_task,
    renew_source_task_lease,
    requeue_stale_processing_source_tasks,
    save_source_task_for_tenant,
)
from rag_core.text_utils import now_ms  # noqa: E402


TENANT_ID = f"lease-smoke-{uuid.uuid4().hex[:10]}"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = replace(
            load_config(),
            metadata_database_url=os.environ.get("SMOKE_METADATA_DATABASE_URL") or None,
            object_store_dir=Path(tmp) / "object_store",
            runtime_dir=Path(tmp) / "runtime",
        )
        ensure_schema(config)
        test_atomic_claim_and_owner_guard(config)
        test_expired_lease_can_be_reclaimed(config)
        test_two_runners_execute_once_and_renew(config)
    print("smoke_ingestion_task_leases=ok")


def test_atomic_claim_and_owner_guard(config) -> None:
    source = save_task(config, "atomic")
    barrier = threading.Barrier(2)

    def claim(owner: str):
        barrier.wait(timeout=5)
        return claim_source_task_for_processing(
            config=config,
            tenant_id=TENANT_ID,
            source=source,
            lease_owner=owner,
            lease_ms=60_000,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(executor.map(claim, ["owner-a", "owner-b"]))
    assert sum(item is not None for item in claims) == 1
    row = task_row(config, source.doc_id)
    owner = str(row["lease_owner"])
    wrong_owner = "owner-b" if owner == "owner-a" else "owner-a"
    assert int(row["attempt_count"]) == 1
    assert int(row["lease_expires_at"]) > now_ms()
    assert renew_source_task_lease(
        config=config,
        tenant_id=TENANT_ID,
        task_id=source.doc_id,
        lease_owner=wrong_owner,
        lease_ms=60_000,
    ) is False
    old_expiry = int(row["lease_expires_at"])
    assert renew_source_task_lease(
        config=config,
        tenant_id=TENANT_ID,
        task_id=source.doc_id,
        lease_owner=owner,
        lease_ms=120_000,
    ) is True
    assert int(task_row(config, source.doc_id)["lease_expires_at"]) >= old_expiry
    assert delete_source_task(
        config=config,
        tenant_id=TENANT_ID,
        task_id=source.doc_id,
        lease_owner=wrong_owner,
    ) is False
    assert delete_source_task(
        config=config,
        tenant_id=TENANT_ID,
        task_id=source.doc_id,
        lease_owner=owner,
    ) is True


def test_expired_lease_can_be_reclaimed(config) -> None:
    source = save_task(config, "reclaim")
    assert claim_source_task_for_processing(
        config=config,
        tenant_id=TENANT_ID,
        source=source,
        lease_owner="expired-owner",
        lease_ms=60_000,
    )
    with connect_metadata_db(config) as conn:
        conn.execute(
            "UPDATE source_tasks SET lease_expires_at = ? WHERE tenant_id = ? AND id = ?",
            (now_ms() - 1, TENANT_ID, source.doc_id),
        )
    assert requeue_stale_processing_source_tasks(
        config=config,
        stale_after_ms=60_000,
        limit=10,
    ) == 1
    queued = replace(source, status="queued", updated_at=now_ms())
    assert claim_source_task_for_processing(
        config=config,
        tenant_id=TENANT_ID,
        source=queued,
        lease_owner="replacement-owner",
        lease_ms=60_000,
    )
    assert fail_source_task(
        config=config,
        tenant_id=TENANT_ID,
        source=queued,
        error="stale worker must not overwrite",
        lease_owner="expired-owner",
    ) is False
    assert fail_source_task(
        config=config,
        tenant_id=TENANT_ID,
        source=queued,
        error="replacement failure",
        lease_owner="replacement-owner",
    ) is True
    row = task_row(config, source.doc_id)
    assert row["status"] == "failed"
    assert row["error"] == "replacement failure"
    assert row["lease_owner"] == ""
    assert int(row["attempt_count"]) == 2
    assert delete_source_task(config=config, tenant_id=TENANT_ID, task_id=source.doc_id)


def test_two_runners_execute_once_and_renew(config) -> None:
    source = save_task(config, "runner")
    started = threading.Event()
    release = threading.Event()
    executions: list[str] = []
    lock = threading.Lock()

    def fake_ingest_uploaded_path(**kwargs):
        with lock:
            executions.append(kwargs["tenant_id"])
        started.set()
        release.wait(timeout=5)

    runner_a = IngestionJobRunner(
        workers=1,
        queue_limit=1,
        tenant_queue_limit=1,
        runner_id="runner-a",
    )
    runner_b = IngestionJobRunner(
        workers=1,
        queue_limit=1,
        tenant_queue_limit=1,
        runner_id="runner-b",
    )
    runner_a.processing_stale_ms = 1_000
    runner_b.processing_stale_ms = 1_000
    with (
        patch("rag_core.ingestion_jobs.load_config", return_value=config),
        patch("rag_core.ingestion_jobs.ingest_uploaded_path", side_effect=fake_ingest_uploaded_path),
    ):
        barrier = threading.Barrier(2)

        def submit(runner: IngestionJobRunner):
            barrier.wait(timeout=5)
            return runner.submit_upload(
                pending_source=source,
                saved_path=Path(source.source_uri),
                tenant_id=TENANT_ID,
                acl_groups=source.acl_groups,
                doc_version=source.doc_version,
                language="zh",
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            accepted = list(executor.map(submit, [runner_a, runner_b]))
        assert accepted == [True, True]
        assert started.wait(timeout=5)
        first_expiry = int(task_row(config, source.doc_id)["lease_expires_at"])
        time.sleep(0.5)
        renewed_expiry = int(task_row(config, source.doc_id)["lease_expires_at"])
        assert renewed_expiry > first_expiry
        assert executions == [TENANT_ID]
        release.set()
        wait_for(lambda: task_row(config, source.doc_id, required=False) is None)
    runner_a._executor.shutdown(wait=True)
    runner_b._executor.shutdown(wait=True)


def save_task(config, label: str) -> SourceSummary:
    timestamp = now_ms()
    task_id = f"{TENANT_ID}-{label}-{uuid.uuid4().hex[:8]}"
    source = SourceSummary(
        doc_id=task_id,
        title=f"{label}.txt",
        source_type="txt",
        source_uri=str(config.object_store_dir / f"{label}.txt"),
        doc_version=1,
        chunk_count=0,
        acl_groups=["engineering"],
        status="queued",
        current=False,
        created_at=timestamp,
        updated_at=timestamp,
        child_doc_ids=[],
    )
    save_source_task_for_tenant(config=config, tenant_id=TENANT_ID, source=source)
    return source


def ensure_schema(config) -> None:
    with connect_metadata_db(config):
        pass


def task_row(config, task_id: str, *, required: bool = True):
    with connect_metadata_db(config) as conn:
        row = conn.execute(
            """
            SELECT status, error, lease_owner, lease_expires_at, attempt_count, updated_at
            FROM source_tasks
            WHERE tenant_id = ? AND id = ?
            """,
            (TENANT_ID, task_id),
        ).fetchone()
    if required:
        assert row is not None
    return row


def wait_for(predicate, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not reached before timeout")


if __name__ == "__main__":
    main()
