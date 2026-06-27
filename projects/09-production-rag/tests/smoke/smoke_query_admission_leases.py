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


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.config import load_config  # noqa: E402
from rag_core.database import connect_metadata_db  # noqa: E402
from rag_core.query_admission import (  # noqa: E402
    QueryAdmissionRejected,
    QueryAdmissionLeaseGuard,
    acquire_query_admission_lease,
    query_admission_metrics_snapshot,
    release_query_admission_lease,
    renew_query_admission_lease,
)
from rag_core.text_utils import now_ms  # noqa: E402


PREFIX = f"query-admission-smoke-{uuid.uuid4().hex[:10]}"


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
        test_atomic_global_limit(config)
        test_tenant_and_user_limits(config)
        test_lease_guard_renews_and_releases(config)
        test_expired_lease_takeover(config)
    print("smoke_query_admission_leases=ok")


def test_atomic_global_limit(config) -> None:
    barrier = threading.Barrier(2)

    def acquire(owner: str):
        barrier.wait(timeout=5)
        try:
            return acquire_query_admission_lease(
                config=config,
                tenant_id=f"{PREFIX}-global-tenant",
                user_key=f"{PREFIX}:user:{owner}",
                global_limit=1,
                tenant_limit=2,
                user_limit=1,
                lease_ms=60_000,
                owner=owner,
            )
        except QueryAdmissionRejected as exc:
            return exc.kind

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(acquire, [f"{PREFIX}-owner-a", f"{PREFIX}-owner-b"]))
    leases = [result for result in results if not isinstance(result, str)]
    rejected = [result for result in results if isinstance(result, str)]
    assert len(leases) == 1
    assert rejected == ["global"]
    assert release_query_admission_lease(config=config, lease=leases[0])


def test_tenant_and_user_limits(config) -> None:
    first = acquire_query_admission_lease(
        config=config,
        tenant_id=f"{PREFIX}-tenant-a",
        user_key=f"{PREFIX}-tenant-a:user:a",
        global_limit=3,
        tenant_limit=1,
        user_limit=1,
        lease_ms=60_000,
    )
    try:
        rejected_kind = acquire_rejection_kind(
            config,
            tenant_id=f"{PREFIX}-tenant-a",
            user_key=f"{PREFIX}-tenant-a:user:b",
            global_limit=3,
            tenant_limit=1,
            user_limit=1,
        )
        assert rejected_kind == "tenant"
        second_tenant = acquire_query_admission_lease(
            config=config,
            tenant_id=f"{PREFIX}-tenant-b",
            user_key=f"{PREFIX}-tenant-b:user:a",
            global_limit=3,
            tenant_limit=2,
            user_limit=1,
            lease_ms=60_000,
        )
        try:
            rejected_kind = acquire_rejection_kind(
                config,
                tenant_id=f"{PREFIX}-tenant-b",
                user_key=f"{PREFIX}-tenant-b:user:a",
                global_limit=3,
                tenant_limit=2,
                user_limit=1,
            )
            assert rejected_kind == "user"
            metrics = query_admission_metrics_snapshot(config=config)
            assert metrics["global_slots"] >= 2
            assert metrics["tenant_slots"] >= 2
            assert metrics["user_slots"] >= 2
        finally:
            assert release_query_admission_lease(config=config, lease=second_tenant)
    finally:
        assert release_query_admission_lease(config=config, lease=first)


def test_expired_lease_takeover(config) -> None:
    old = acquire_query_admission_lease(
        config=config,
        tenant_id=f"{PREFIX}-expiry",
        user_key="",
        global_limit=1,
        tenant_limit=1,
        user_limit=1,
        lease_ms=60_000,
        owner=f"{PREFIX}-expired-owner",
    )
    with connect_metadata_db(config) as conn:
        conn.execute(
            "UPDATE query_admission_slots SET lease_expires_at = ? WHERE lease_owner = ?",
            (now_ms() - 1, old.owner),
        )
    replacement = acquire_query_admission_lease(
        config=config,
        tenant_id=f"{PREFIX}-expiry",
        user_key="",
        global_limit=1,
        tenant_limit=1,
        user_limit=1,
        lease_ms=60_000,
        owner=f"{PREFIX}-replacement-owner",
    )
    assert renew_query_admission_lease(config=config, lease=old, lease_ms=60_000) is False
    assert release_query_admission_lease(config=config, lease=replacement)


def test_lease_guard_renews_and_releases(config) -> None:
    lease = acquire_query_admission_lease(
        config=config,
        tenant_id=f"{PREFIX}-heartbeat",
        user_key="",
        global_limit=2,
        tenant_limit=1,
        user_limit=1,
        lease_ms=1000,
    )
    first_expiry = owner_expiry(config, lease.owner)
    guard = QueryAdmissionLeaseGuard(config=config, lease=lease, lease_ms=1000)
    guard.start()
    time.sleep(0.5)
    assert owner_expiry(config, lease.owner) > first_expiry
    assert guard.valid.is_set()
    guard.close()
    assert owner_slot_count(config, lease.owner) == 0


def acquire_rejection_kind(
    config,
    *,
    tenant_id: str,
    user_key: str,
    global_limit: int,
    tenant_limit: int,
    user_limit: int,
) -> str:
    try:
        lease = acquire_query_admission_lease(
            config=config,
            tenant_id=tenant_id,
            user_key=user_key,
            global_limit=global_limit,
            tenant_limit=tenant_limit,
            user_limit=user_limit,
            lease_ms=60_000,
        )
    except QueryAdmissionRejected as exc:
        return exc.kind
    release_query_admission_lease(config=config, lease=lease)
    raise AssertionError("query admission unexpectedly succeeded")


def owner_expiry(config, owner: str) -> int:
    with connect_metadata_db(config) as conn:
        row = conn.execute(
            "SELECT MAX(lease_expires_at) AS expires_at FROM query_admission_slots WHERE lease_owner = ?",
            (owner,),
        ).fetchone()
    assert row is not None
    return int(row["expires_at"] or 0)


def owner_slot_count(config, owner: str) -> int:
    with connect_metadata_db(config) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM query_admission_slots WHERE lease_owner = ?",
            (owner,),
        ).fetchone()
    assert row is not None
    return int(row["count"] or 0)


if __name__ == "__main__":
    main()
