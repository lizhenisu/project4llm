from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Literal

from rag_core.config import RagConfig
from rag_core.database import connect_metadata_db
from rag_core.text_utils import now_ms


@dataclass(frozen=True)
class QueryResultClaim:
    mode: Literal["owner", "cached", "waiting"]
    owner: str
    response: dict[str, Any] | None = None


class QueryResultConflictError(ValueError):
    pass


class QueryResultWaitTimeout(TimeoutError):
    pass


class QueryResultLeaseGuard:
    def __init__(
        self,
        *,
        config: RagConfig,
        tenant_id: str,
        request_id: str,
        owner: str,
        lease_ms: int,
    ) -> None:
        self.config = config
        self.tenant_id = tenant_id
        self.request_id = request_id
        self.owner = owner
        self.lease_ms = max(1000, int(lease_ms))
        self.valid = threading.Event()
        self.valid.set()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._renew_loop,
            name="rag-query-result-lease",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=1)

    def _renew_loop(self) -> None:
        interval_seconds = max(0.25, self.lease_ms / 3000)
        while not self._stop.wait(interval_seconds):
            try:
                renewed = renew_query_result_lease(
                    config=self.config,
                    tenant_id=self.tenant_id,
                    request_id=self.request_id,
                    owner=self.owner,
                    lease_ms=self.lease_ms,
                )
            except Exception:
                renewed = False
            if not renewed:
                self.valid.clear()
                return


def query_result_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def claim_query_result(
    *,
    config: RagConfig,
    tenant_id: str,
    request_id: str,
    fingerprint: str,
    lease_ms: int,
    ttl_ms: int,
    owner: str | None = None,
) -> QueryResultClaim:
    timestamp = now_ms()
    resolved_owner = owner or f"query-result-{uuid.uuid4().hex}"
    lease_expires_at = timestamp + max(1000, int(lease_ms))
    expires_at = timestamp + max(1000, int(ttl_ms))
    with connect_metadata_db(config) as conn:
        conn.execute(
            "DELETE FROM query_result_cache WHERE expires_at < ?",
            (timestamp,),
        )
        inserted = conn.execute(
            """
            INSERT INTO query_result_cache(
                tenant_id, request_id, request_fingerprint, status, lease_owner,
                lease_expires_at, response_json, error, created_at, updated_at, expires_at
            )
            VALUES (?, ?, ?, 'processing', ?, ?, '', '', ?, ?, ?)
            ON CONFLICT(tenant_id, request_id) DO NOTHING
            """,
            (
                tenant_id,
                request_id,
                fingerprint,
                resolved_owner,
                lease_expires_at,
                timestamp,
                timestamp,
                expires_at,
            ),
        )
        if int(inserted.rowcount or 0) == 1:
            return QueryResultClaim(mode="owner", owner=resolved_owner)
        row = conn.execute(
            """
            SELECT request_fingerprint, status, lease_expires_at, response_json
            FROM query_result_cache
            WHERE tenant_id = ? AND request_id = ?
            """,
            (tenant_id, request_id),
        ).fetchone()
        if row is None:
            return QueryResultClaim(mode="waiting", owner=resolved_owner)
        if str(row["request_fingerprint"]) != fingerprint:
            raise QueryResultConflictError("Request ID is already associated with a different query")
        if str(row["status"]) == "completed":
            return QueryResultClaim(
                mode="cached",
                owner=resolved_owner,
                response=json.loads(str(row["response_json"] or "{}")),
            )
        if str(row["status"]) == "failed" or int(row["lease_expires_at"] or 0) < timestamp:
            takeover = conn.execute(
                """
                UPDATE query_result_cache
                SET status = 'processing', lease_owner = ?, lease_expires_at = ?,
                    response_json = '', error = '', updated_at = ?, expires_at = ?
                WHERE tenant_id = ? AND request_id = ? AND request_fingerprint = ?
                  AND status != 'completed'
                  AND (status = 'failed' OR lease_expires_at < ?)
                """,
                (
                    resolved_owner,
                    lease_expires_at,
                    timestamp,
                    expires_at,
                    tenant_id,
                    request_id,
                    fingerprint,
                    timestamp,
                ),
            )
            if int(takeover.rowcount or 0) == 1:
                conn.execute(
                    "DELETE FROM query_result_events WHERE tenant_id = ? AND request_id = ?",
                    (tenant_id, request_id),
                )
                return QueryResultClaim(mode="owner", owner=resolved_owner)
    return QueryResultClaim(mode="waiting", owner=resolved_owner)


def wait_for_query_result(
    *,
    config: RagConfig,
    tenant_id: str,
    request_id: str,
    fingerprint: str,
    lease_ms: int,
    ttl_ms: int,
    owner: str,
    timeout_seconds: float,
    cancelled: threading.Event | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> QueryResultClaim:
    deadline = time.monotonic() + max(0.1, float(timeout_seconds))
    poll_seconds = query_result_poll_seconds()
    max_poll_seconds = query_result_poll_max_seconds()
    last_sequence = 0
    while time.monotonic() < deadline:
        if cancelled is not None and cancelled.is_set():
            raise QueryResultWaitTimeout("Client disconnected while waiting for the original query")
        events = list_query_result_events(
            config=config,
            tenant_id=tenant_id,
            request_id=request_id,
            after_sequence=last_sequence,
        )
        for sequence, event in events:
            last_sequence = sequence
            if on_event is not None:
                on_event(event)
        claim = claim_query_result(
            config=config,
            tenant_id=tenant_id,
            request_id=request_id,
            fingerprint=fingerprint,
            lease_ms=lease_ms,
            ttl_ms=ttl_ms,
            owner=owner,
        )
        if claim.mode != "waiting":
            events = list_query_result_events(
                config=config,
                tenant_id=tenant_id,
                request_id=request_id,
                after_sequence=last_sequence,
            )
            for sequence, event in events:
                last_sequence = sequence
                if on_event is not None:
                    on_event(event)
            return claim
        time.sleep(poll_seconds)
        poll_seconds = min(max_poll_seconds, poll_seconds * 1.5)
    raise QueryResultWaitTimeout("Timed out waiting for the original query result")


def append_query_result_event(
    *,
    config: RagConfig,
    tenant_id: str,
    request_id: str,
    owner: str,
    event: dict[str, Any],
) -> int | None:
    timestamp = now_ms()
    with connect_metadata_db(config) as conn:
        cursor = conn.execute(
            """
            INSERT INTO query_result_events(
                tenant_id, request_id, sequence, event_json, created_at
            )
            SELECT ?, ?, COALESCE((
                SELECT MAX(sequence)
                FROM query_result_events
                WHERE tenant_id = ? AND request_id = ?
            ), 0) + 1, ?, ?
            WHERE EXISTS (
                SELECT 1
                FROM query_result_cache
                WHERE tenant_id = ? AND request_id = ?
                  AND status = 'processing' AND lease_owner = ?
            )
            """,
            (
                tenant_id,
                request_id,
                tenant_id,
                request_id,
                json.dumps(event, ensure_ascii=False, separators=(",", ":")),
                timestamp,
                tenant_id,
                request_id,
                owner,
            ),
        )
        if int(cursor.rowcount or 0) != 1:
            return None
        row = conn.execute(
            """
            SELECT MAX(sequence) AS sequence
            FROM query_result_events
            WHERE tenant_id = ? AND request_id = ?
            """,
            (tenant_id, request_id),
        ).fetchone()
    return int(row["sequence"]) if row is not None and row["sequence"] is not None else None


def list_query_result_events(
    *,
    config: RagConfig,
    tenant_id: str,
    request_id: str,
    after_sequence: int = 0,
) -> list[tuple[int, dict[str, Any]]]:
    with connect_metadata_db(config) as conn:
        rows = conn.execute(
            """
            SELECT sequence, event_json
            FROM query_result_events
            WHERE tenant_id = ? AND request_id = ? AND sequence > ?
            ORDER BY sequence
            """,
            (tenant_id, request_id, max(0, int(after_sequence))),
        ).fetchall()
    events: list[tuple[int, dict[str, Any]]] = []
    for row in rows:
        sequence = int(row["sequence"])
        event = json.loads(str(row["event_json"]))
        event["sequence"] = sequence
        events.append((sequence, event))
    return events


def complete_query_result(
    *,
    config: RagConfig,
    tenant_id: str,
    request_id: str,
    owner: str,
    response: dict[str, Any],
    ttl_ms: int,
) -> bool:
    timestamp = now_ms()
    with connect_metadata_db(config) as conn:
        cursor = conn.execute(
            """
            UPDATE query_result_cache
            SET status = 'completed', response_json = ?, error = '',
                lease_owner = '', lease_expires_at = 0, updated_at = ?, expires_at = ?
            WHERE tenant_id = ? AND request_id = ? AND status = 'processing' AND lease_owner = ?
            """,
            (
                json.dumps(response, ensure_ascii=False, separators=(",", ":")),
                timestamp,
                timestamp + max(1000, int(ttl_ms)),
                tenant_id,
                request_id,
                owner,
            ),
        )
    return int(cursor.rowcount or 0) == 1


def fail_query_result(
    *,
    config: RagConfig,
    tenant_id: str,
    request_id: str,
    owner: str,
    error: str,
) -> bool:
    timestamp = now_ms()
    with connect_metadata_db(config) as conn:
        cursor = conn.execute(
            """
            UPDATE query_result_cache
            SET status = 'failed', error = ?, lease_owner = '', lease_expires_at = 0, updated_at = ?
            WHERE tenant_id = ? AND request_id = ? AND status = 'processing' AND lease_owner = ?
            """,
            (error[:500], timestamp, tenant_id, request_id, owner),
        )
    return int(cursor.rowcount or 0) == 1


def renew_query_result_lease(
    *,
    config: RagConfig,
    tenant_id: str,
    request_id: str,
    owner: str,
    lease_ms: int,
) -> bool:
    timestamp = now_ms()
    with connect_metadata_db(config) as conn:
        cursor = conn.execute(
            """
            UPDATE query_result_cache
            SET lease_expires_at = ?, updated_at = ?
            WHERE tenant_id = ? AND request_id = ? AND status = 'processing'
              AND lease_owner = ? AND lease_expires_at >= ?
            """,
            (
                timestamp + max(1000, int(lease_ms)),
                timestamp,
                tenant_id,
                request_id,
                owner,
                timestamp,
            ),
        )
    return int(cursor.rowcount or 0) == 1


def query_result_cache_snapshot(*, config: RagConfig) -> dict[str, int]:
    timestamp = now_ms()
    with connect_metadata_db(config) as conn:
        rows = conn.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM query_result_cache
            WHERE expires_at >= ?
            GROUP BY status
            """,
            (timestamp,),
        ).fetchall()
        expired = conn.execute(
            "SELECT COUNT(*) AS count FROM query_result_cache WHERE expires_at < ?",
            (timestamp,),
        ).fetchone()
        stale_processing = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM query_result_cache
            WHERE status = 'processing'
              AND lease_expires_at < ?
              AND expires_at >= ?
            """,
            (timestamp, timestamp),
        ).fetchone()
        event_count = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM query_result_events AS events
            JOIN query_result_cache AS cache
              ON cache.tenant_id = events.tenant_id
             AND cache.request_id = events.request_id
            WHERE cache.expires_at >= ?
            """,
            (timestamp,),
        ).fetchone()
    counts = {str(row["status"]): int(row["count"] or 0) for row in rows}
    return {
        "processing": counts.get("processing", 0),
        "completed": counts.get("completed", 0),
        "failed": counts.get("failed", 0),
        "expired": int(expired["count"] or 0) if expired is not None else 0,
        "stale_processing": (
            int(stale_processing["count"] or 0)
            if stale_processing is not None
            else 0
        ),
        "events": int(event_count["count"] or 0) if event_count is not None else 0,
    }


def query_result_poll_seconds() -> float:
    value = os.environ.get("RAG_QUERY_RESULT_POLL_SECONDS", "0.1")
    try:
        return max(0.02, float(value))
    except ValueError:
        return 0.1


def query_result_poll_max_seconds() -> float:
    value = os.environ.get("RAG_QUERY_RESULT_POLL_MAX_SECONDS", "1")
    try:
        return max(query_result_poll_seconds(), float(value))
    except ValueError:
        return 1.0
