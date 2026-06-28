from __future__ import annotations

import os
import sys
import tempfile
import threading
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
from rag_core.sources import (  # noqa: E402
    SourceSummary,
    create_source_task,
    delete_source_task,
    save_source_task_for_tenant,
)
from rag_core.sources import lock_upload_admission as source_lock_upload_admission  # noqa: E402
from rag_core.text_utils import now_ms  # noqa: E402
from rag_core.upload_admission import (  # noqa: E402
    acquire_upload_admission_reservation,
    release_upload_admission_reservation,
    upload_admission_metrics_snapshot,
    UploadAdmissionRejected,
)


PREFIX = f"upload-admission-smoke-{uuid.uuid4().hex[:10]}"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = replace(
            load_config(),
            metadata_database_url=os.environ.get("SMOKE_METADATA_DATABASE_URL") or None,
            object_store_dir=Path(tmp) / "object_store",
            runtime_dir=Path(tmp) / "runtime",
        )
        with connect_metadata_db(config):
            pass
        test_atomic_global_reservation(config)
        test_tenant_reservation_isolation(config)
        test_active_tasks_reduce_available_capacity(config)
        test_task_creation_consumes_reservation_atomically(config)
        test_conversion_serializes_with_new_admission(config)
        test_expired_reservation_takeover(config)
    print("smoke_upload_admission_leases=ok")


def test_atomic_global_reservation(config) -> None:
    barrier = threading.Barrier(2)

    def acquire(owner: str):
        barrier.wait(timeout=5)
        try:
            return acquire_upload_admission_reservation(
                config=config,
                tenant_id=f"{PREFIX}-global",
                global_limit=1,
                tenant_limit=2,
                lease_ms=60_000,
                owner=owner,
            )
        except UploadAdmissionRejected as exc:
            return exc.kind

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(acquire, [f"{PREFIX}-owner-a", f"{PREFIX}-owner-b"]))
    reservations = [result for result in results if not isinstance(result, str)]
    rejected = [result for result in results if isinstance(result, str)]
    assert len(reservations) == 1
    assert rejected == ["global"]
    assert release_upload_admission_reservation(config=config, reservation=reservations[0])


def test_tenant_reservation_isolation(config) -> None:
    first = acquire_upload_admission_reservation(
        config=config,
        tenant_id=f"{PREFIX}-tenant-a",
        global_limit=2,
        tenant_limit=1,
        lease_ms=60_000,
    )
    try:
        assert rejection_kind(
            config,
            tenant_id=f"{PREFIX}-tenant-a",
            global_limit=2,
            tenant_limit=1,
        ) == "tenant"
        second = acquire_upload_admission_reservation(
            config=config,
            tenant_id=f"{PREFIX}-tenant-b",
            global_limit=2,
            tenant_limit=1,
            lease_ms=60_000,
        )
        assert upload_admission_metrics_snapshot(config=config)["global_reservations"] >= 2
        assert release_upload_admission_reservation(config=config, reservation=second)
    finally:
        assert release_upload_admission_reservation(config=config, reservation=first)


def test_active_tasks_reduce_available_capacity(config) -> None:
    tenant_id = f"{PREFIX}-active"
    source = queued_source(config, tenant_id)
    save_source_task_for_tenant(config=config, tenant_id=tenant_id, source=source)
    try:
        assert rejection_kind(
            config,
            tenant_id=tenant_id,
            global_limit=1,
            tenant_limit=10,
        ) == "global"
        assert rejection_kind(
            config,
            tenant_id=tenant_id,
            global_limit=10,
            tenant_limit=1,
        ) == "tenant"
    finally:
        assert delete_source_task(config=config, tenant_id=tenant_id, task_id=source.doc_id)


def test_task_creation_consumes_reservation_atomically(config) -> None:
    tenant_id = f"{PREFIX}-consume"
    reservation = acquire_upload_admission_reservation(
        config=config,
        tenant_id=tenant_id,
        global_limit=2,
        tenant_limit=2,
        lease_ms=60_000,
    )
    source = create_source_task(
        config=config,
        tenant_id=tenant_id,
        path=config.object_store_dir / "synthetic.txt",
        acl_groups=["engineering"],
        doc_version=None,
        upload_reservation_owner=reservation.owner,
    )
    assert reservation_slot_count(config, reservation.owner) == 0
    with connect_metadata_db(config) as conn:
        row = conn.execute(
            "SELECT status FROM source_tasks WHERE tenant_id = ? AND id = ?",
            (tenant_id, source.doc_id),
        ).fetchone()
    assert row is not None and row["status"] == "queued"
    assert delete_source_task(config=config, tenant_id=tenant_id, task_id=source.doc_id)


def test_expired_reservation_takeover(config) -> None:
    tenant_id = f"{PREFIX}-expiry"
    expired = acquire_upload_admission_reservation(
        config=config,
        tenant_id=tenant_id,
        global_limit=1,
        tenant_limit=1,
        lease_ms=60_000,
    )
    with connect_metadata_db(config) as conn:
        conn.execute(
            "UPDATE upload_admission_slots SET lease_expires_at = ? WHERE reservation_owner = ?",
            (now_ms() - 1, expired.owner),
        )
    replacement = acquire_upload_admission_reservation(
        config=config,
        tenant_id=tenant_id,
        global_limit=1,
        tenant_limit=1,
        lease_ms=60_000,
    )
    assert release_upload_admission_reservation(config=config, reservation=expired) is False
    assert release_upload_admission_reservation(config=config, reservation=replacement)


def test_conversion_serializes_with_new_admission(config) -> None:
    tenant_id = f"{PREFIX}-serialized"
    reservation = acquire_upload_admission_reservation(
        config=config,
        tenant_id=tenant_id,
        global_limit=1,
        tenant_limit=1,
        lease_ms=60_000,
    )
    lock_acquired = threading.Event()
    allow_conversion = threading.Event()
    admission_started = threading.Event()

    def hold_conversion_lock(conn, *, timestamp=None):
        source_lock_upload_admission(conn, timestamp=timestamp)
        lock_acquired.set()
        assert allow_conversion.wait(timeout=5)

    def convert():
        return create_source_task(
            config=config,
            tenant_id=tenant_id,
            path=config.object_store_dir / "serialized.txt",
            acl_groups=["engineering"],
            doc_version=None,
            upload_reservation_owner=reservation.owner,
        )

    def attempt_admission():
        admission_started.set()
        return rejection_kind(
            config,
            tenant_id=tenant_id,
            global_limit=1,
            tenant_limit=1,
        )

    with (
        patch("rag_core.sources.lock_upload_admission", side_effect=hold_conversion_lock),
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        conversion_future = executor.submit(convert)
        assert lock_acquired.wait(timeout=5)
        admission_future = executor.submit(attempt_admission)
        assert admission_started.wait(timeout=5)
        allow_conversion.set()
        source = conversion_future.result(timeout=5)
        assert admission_future.result(timeout=5) == "global"
    assert delete_source_task(config=config, tenant_id=tenant_id, task_id=source.doc_id)


def queued_source(config, tenant_id: str) -> SourceSummary:
    timestamp = now_ms()
    return SourceSummary(
        doc_id=f"{tenant_id}-queued",
        title="synthetic.txt",
        source_type="txt",
        source_uri=str(config.object_store_dir / "synthetic.txt"),
        doc_version=1,
        chunk_count=0,
        acl_groups=["engineering"],
        status="queued",
        current=False,
        created_at=timestamp,
        updated_at=timestamp,
    )


def rejection_kind(config, *, tenant_id: str, global_limit: int, tenant_limit: int) -> str:
    try:
        reservation = acquire_upload_admission_reservation(
            config=config,
            tenant_id=tenant_id,
            global_limit=global_limit,
            tenant_limit=tenant_limit,
            lease_ms=60_000,
        )
    except UploadAdmissionRejected as exc:
        return exc.kind
    release_upload_admission_reservation(config=config, reservation=reservation)
    raise AssertionError("upload admission unexpectedly succeeded")


def reservation_slot_count(config, owner: str) -> int:
    with connect_metadata_db(config) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM upload_admission_slots WHERE reservation_owner = ?",
            (owner,),
        ).fetchone()
    assert row is not None
    return int(row["count"] or 0)


if __name__ == "__main__":
    main()
