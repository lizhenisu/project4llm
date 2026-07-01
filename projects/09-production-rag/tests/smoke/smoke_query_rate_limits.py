from __future__ import annotations

import os
import sys
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.config import load_config  # noqa: E402
from rag_core.database import connect_metadata_db  # noqa: E402
from rag_core.query_rate_limits import (  # noqa: E402
    QueryRateLimitRejected,
    acquire_query_rate_limit,
)
from rag_core.text_utils import now_ms  # noqa: E402


PREFIX = f"query-rate-limit-smoke-{uuid.uuid4().hex[:10]}"
GLOBAL_KEY = f"{PREFIX}-global"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="rag-query-rate-limit-") as tmp:
        config = replace(
            load_config(),
            metadata_database_url=os.environ.get("SMOKE_METADATA_DATABASE_URL") or None,
            object_store_dir=Path(tmp) / "object_store",
            runtime_dir=Path(tmp) / "runtime",
        )
        with connect_metadata_db(config):
            pass
        try:
            test_atomic_global_limit(config)
            cleanup(config)
            test_scope_rejection_rolls_back_earlier_counts(config)
            cleanup(config)
            test_user_isolation(config)
            cleanup(config)
            test_window_rollover(config)
        finally:
            cleanup(config)
    print("smoke_query_rate_limits=ok")


def test_atomic_global_limit(config) -> None:
    workers = 20
    limit = 5
    barrier = threading.Barrier(workers)

    def acquire(index: int) -> str:
        barrier.wait(timeout=10)
        try:
            acquire_query_rate_limit(
                config=config,
                tenant_id=f"{PREFIX}-tenant-{index}",
                user_key=f"{PREFIX}-user-{index}",
                global_limit=limit,
                tenant_limit=workers,
                user_limit=workers,
                global_scope_key=GLOBAL_KEY,
            )
            return "accepted"
        except QueryRateLimitRejected as exc:
            return exc.kind

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(acquire, range(workers)))
    assert results.count("accepted") == limit, results
    assert results.count("global") == workers - limit, results
    assert window_count(config, "global", GLOBAL_KEY) == limit


def test_scope_rejection_rolls_back_earlier_counts(config) -> None:
    tenant_a = f"{PREFIX}-tenant-a"
    tenant_b = f"{PREFIX}-tenant-b"
    acquire_query_rate_limit(
        config=config,
        tenant_id=tenant_a,
        user_key=f"{PREFIX}-user-a",
        global_limit=2,
        tenant_limit=1,
        user_limit=1,
        global_scope_key=GLOBAL_KEY,
    )
    assert rejection_kind(
        config,
        tenant_id=tenant_a,
        user_key=f"{PREFIX}-user-b",
        global_limit=2,
        tenant_limit=1,
        user_limit=1,
    ) == "tenant"
    acquire_query_rate_limit(
        config=config,
        tenant_id=tenant_b,
        user_key=f"{PREFIX}-user-b",
        global_limit=2,
        tenant_limit=1,
        user_limit=1,
        global_scope_key=GLOBAL_KEY,
    )
    assert rejection_kind(
        config,
        tenant_id=f"{PREFIX}-tenant-c",
        user_key=f"{PREFIX}-user-c",
        global_limit=2,
        tenant_limit=1,
        user_limit=1,
    ) == "global"
    assert window_count(config, "global", GLOBAL_KEY) == 2
    assert window_count(config, "tenant", tenant_a) == 1
    assert window_count(config, "tenant", tenant_b) == 1


def test_user_isolation(config) -> None:
    tenant_id = f"{PREFIX}-user-tenant"
    user_a = f"{PREFIX}-isolated-user-a"
    user_b = f"{PREFIX}-isolated-user-b"
    acquire_query_rate_limit(
        config=config,
        tenant_id=tenant_id,
        user_key=user_a,
        global_limit=10,
        tenant_limit=10,
        user_limit=1,
        global_scope_key=GLOBAL_KEY,
    )
    assert rejection_kind(
        config,
        tenant_id=tenant_id,
        user_key=user_a,
        global_limit=10,
        tenant_limit=10,
        user_limit=1,
    ) == "user"
    acquire_query_rate_limit(
        config=config,
        tenant_id=tenant_id,
        user_key=user_b,
        global_limit=10,
        tenant_limit=10,
        user_limit=1,
        global_scope_key=GLOBAL_KEY,
    )
    assert window_count(config, "user", user_a) == 1
    assert window_count(config, "user", user_b) == 1


def test_window_rollover(config) -> None:
    tenant_id = f"{PREFIX}-rollover-tenant"
    user_key = f"{PREFIX}-rollover-user"
    timestamp = now_ms()
    first_window = timestamp - (timestamp % 60_000)
    grant = acquire_query_rate_limit(
        config=config,
        tenant_id=tenant_id,
        user_key=user_key,
        global_limit=1,
        tenant_limit=1,
        user_limit=1,
        window_seconds=60,
        timestamp_ms=first_window,
        global_scope_key=GLOBAL_KEY,
    )
    try:
        acquire_query_rate_limit(
            config=config,
            tenant_id=tenant_id,
            user_key=user_key,
            global_limit=1,
            tenant_limit=1,
            user_limit=1,
            window_seconds=60,
            timestamp_ms=first_window + 1,
            global_scope_key=GLOBAL_KEY,
        )
    except QueryRateLimitRejected as exc:
        assert exc.kind == "global"
        assert exc.retry_after_seconds == 60
    else:
        raise AssertionError("same fixed window unexpectedly accepted another request")
    next_grant = acquire_query_rate_limit(
        config=config,
        tenant_id=tenant_id,
        user_key=user_key,
        global_limit=1,
        tenant_limit=1,
        user_limit=1,
        window_seconds=60,
        timestamp_ms=first_window + 60_000,
        global_scope_key=GLOBAL_KEY,
    )
    assert next_grant.window_started_at == grant.window_expires_at


def rejection_kind(
    config,
    *,
    tenant_id: str,
    user_key: str,
    global_limit: int,
    tenant_limit: int,
    user_limit: int,
) -> str:
    try:
        acquire_query_rate_limit(
            config=config,
            tenant_id=tenant_id,
            user_key=user_key,
            global_limit=global_limit,
            tenant_limit=tenant_limit,
            user_limit=user_limit,
            global_scope_key=GLOBAL_KEY,
        )
    except QueryRateLimitRejected as exc:
        return exc.kind
    raise AssertionError("query rate limit unexpectedly accepted a request")


def window_count(config, scope_type: str, scope_key: str) -> int:
    with connect_metadata_db(config) as conn:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(request_count), 0) AS count
            FROM query_rate_limit_windows
            WHERE scope_type = ? AND scope_key = ?
            """,
            (scope_type, scope_key),
        ).fetchone()
    assert row is not None
    return int(row["count"] or 0)


def cleanup(config) -> None:
    with connect_metadata_db(config) as conn:
        conn.execute(
            "DELETE FROM query_rate_limit_windows WHERE scope_key LIKE ?",
            (f"{PREFIX}%",),
        )


if __name__ == "__main__":
    main()
