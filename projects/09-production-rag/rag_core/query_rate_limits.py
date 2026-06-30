from __future__ import annotations

import math
from dataclasses import dataclass

from rag_core.config import RagConfig
from rag_core.database import connect_metadata_db
from rag_core.text_utils import now_ms


@dataclass(frozen=True)
class QueryRateLimitGrant:
    scopes: tuple[str, ...]
    window_started_at: int
    window_expires_at: int


class QueryRateLimitRejected(RuntimeError):
    def __init__(self, kind: str, *, retry_after_seconds: int) -> None:
        self.kind = kind
        self.retry_after_seconds = max(1, int(retry_after_seconds))
        super().__init__(f"Shared query rate limit rejected at {kind} scope")


def query_rate_limit_window_snapshot(
    *,
    config: RagConfig,
    window_seconds: int = 60,
    timestamp_ms: int | None = None,
) -> dict[str, object]:
    timestamp = now_ms() if timestamp_ms is None else int(timestamp_ms)
    window_ms = max(1, int(window_seconds)) * 1000
    window_started_at = timestamp - (timestamp % window_ms)
    requests = {"global": 0, "tenant": 0, "user": 0}
    active_keys = {"global": 0, "tenant": 0, "user": 0}
    with connect_metadata_db(config) as conn:
        rows = conn.execute(
            """
            SELECT scope_type, SUM(request_count) AS request_count, COUNT(*) AS active_keys
            FROM query_rate_limit_windows
            WHERE window_started_at = ?
            GROUP BY scope_type
            """,
            (window_started_at,),
        ).fetchall()
    for row in rows:
        scope_type = str(row["scope_type"])
        if scope_type not in requests:
            continue
        requests[scope_type] = int(row["request_count"] or 0)
        active_keys[scope_type] = int(row["active_keys"] or 0)
    return {
        "window_started_at": window_started_at,
        "window_expires_at": window_started_at + window_ms,
        "requests": requests,
        "active_keys": active_keys,
    }


def acquire_query_rate_limit(
    *,
    config: RagConfig,
    tenant_id: str,
    user_key: str,
    global_limit: int,
    tenant_limit: int,
    user_limit: int,
    window_seconds: int = 60,
    timestamp_ms: int | None = None,
    global_scope_key: str = "*",
) -> QueryRateLimitGrant:
    timestamp = now_ms() if timestamp_ms is None else int(timestamp_ms)
    window_ms = max(1, int(window_seconds)) * 1000
    window_started_at = timestamp - (timestamp % window_ms)
    window_expires_at = window_started_at + window_ms
    retry_after_seconds = max(
        1,
        math.ceil((window_expires_at - timestamp) / 1000),
    )
    scopes: list[tuple[str, str, int]] = []
    if int(global_limit) > 0:
        scopes.append(("global", global_scope_key, int(global_limit)))
    if int(tenant_limit) > 0:
        scopes.append(("tenant", tenant_id, int(tenant_limit)))
    if user_key and int(user_limit) > 0:
        scopes.append(("user", user_key, int(user_limit)))

    try:
        with connect_metadata_db(config) as conn:
            conn.execute(
                "DELETE FROM query_rate_limit_windows WHERE window_started_at < ?",
                (window_started_at - window_ms,),
            )
            for scope_type, scope_key, limit in scopes:
                if not increment_scope_window(
                    conn,
                    scope_type=scope_type,
                    scope_key=scope_key,
                    window_started_at=window_started_at,
                    limit=limit,
                    timestamp=timestamp,
                ):
                    raise QueryRateLimitRejected(
                        scope_type,
                        retry_after_seconds=retry_after_seconds,
                    )
    except QueryRateLimitRejected:
        raise

    return QueryRateLimitGrant(
        scopes=tuple(scope_type for scope_type, _scope_key, _limit in scopes),
        window_started_at=window_started_at,
        window_expires_at=window_expires_at,
    )


def increment_scope_window(
    conn,
    *,
    scope_type: str,
    scope_key: str,
    window_started_at: int,
    limit: int,
    timestamp: int,
) -> bool:
    cursor = conn.execute(
        """
        INSERT INTO query_rate_limit_windows(
            scope_type, scope_key, window_started_at, request_count, updated_at
        )
        VALUES (?, ?, ?, 1, ?)
        ON CONFLICT(scope_type, scope_key, window_started_at) DO UPDATE SET
            request_count = query_rate_limit_windows.request_count + 1,
            updated_at = excluded.updated_at
        WHERE query_rate_limit_windows.request_count < ?
        """,
        (
            scope_type,
            scope_key,
            window_started_at,
            timestamp,
            max(1, int(limit)),
        ),
    )
    return int(cursor.rowcount or 0) == 1
