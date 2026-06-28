from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass

from rag_core.config import RagConfig
from rag_core.database import connect_metadata_db
from rag_core.text_utils import now_ms


@dataclass(frozen=True)
class QueryAdmissionLease:
    owner: str
    scopes: tuple[tuple[str, str], ...]
    expires_at: int


class QueryAdmissionRejected(RuntimeError):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        super().__init__(f"Shared query admission rejected at {kind} scope")


class QueryAdmissionLeaseGuard:
    def __init__(
        self,
        *,
        config: RagConfig,
        lease: QueryAdmissionLease,
        lease_ms: int,
    ) -> None:
        self.config = config
        self.lease = lease
        self.lease_ms = max(1000, int(lease_ms))
        self.valid = threading.Event()
        self.valid.set()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._renew_loop,
            name="rag-query-admission-lease",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=1)
        try:
            release_query_admission_lease(config=self.config, lease=self.lease)
        except Exception:
            pass

    def _renew_loop(self) -> None:
        interval_seconds = max(0.25, self.lease_ms / 3000)
        while not self._stop.wait(interval_seconds):
            try:
                renewed = renew_query_admission_lease(
                    config=self.config,
                    lease=self.lease,
                    lease_ms=self.lease_ms,
                )
            except Exception:
                renewed = False
            if not renewed:
                self.valid.clear()
                return


def acquire_query_admission_lease(
    *,
    config: RagConfig,
    tenant_id: str,
    user_key: str,
    global_limit: int,
    tenant_limit: int,
    user_limit: int,
    lease_ms: int,
    owner: str | None = None,
) -> QueryAdmissionLease:
    timestamp = now_ms()
    resolved_owner = owner or f"query-{uuid.uuid4().hex}"
    scopes = [
        ("global", "*", max(1, int(global_limit))),
        ("tenant", tenant_id, max(1, int(tenant_limit))),
    ]
    if user_key:
        scopes.append(("user", user_key, max(1, int(user_limit))))
    expires_at = timestamp + max(1000, int(lease_ms))
    try:
        with connect_metadata_db(config) as conn:
            conn.execute(
                "DELETE FROM query_admission_slots WHERE lease_expires_at < ?",
                (timestamp,),
            )
            for scope_type, scope_key, limit in scopes:
                if not claim_scope_slot(
                    conn,
                    scope_type=scope_type,
                    scope_key=scope_key,
                    limit=limit,
                    owner=resolved_owner,
                    expires_at=expires_at,
                    timestamp=timestamp,
                ):
                    raise QueryAdmissionRejected(scope_type)
    except QueryAdmissionRejected:
        raise
    return QueryAdmissionLease(
        owner=resolved_owner,
        scopes=tuple((scope_type, scope_key) for scope_type, scope_key, _limit in scopes),
        expires_at=expires_at,
    )


def claim_scope_slot(
    conn,
    *,
    scope_type: str,
    scope_key: str,
    limit: int,
    owner: str,
    expires_at: int,
    timestamp: int,
) -> bool:
    for slot_index in range(limit):
        cursor = conn.execute(
            """
            INSERT INTO query_admission_slots(
                scope_type, scope_key, slot_index, lease_owner,
                lease_expires_at, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scope_type, scope_key, slot_index) DO NOTHING
            """,
            (
                scope_type,
                scope_key,
                slot_index,
                owner,
                expires_at,
                timestamp,
                timestamp,
            ),
        )
        if int(cursor.rowcount or 0) == 1:
            return True
    return False


def renew_query_admission_lease(
    *,
    config: RagConfig,
    lease: QueryAdmissionLease,
    lease_ms: int,
) -> bool:
    timestamp = now_ms()
    expires_at = timestamp + max(1000, int(lease_ms))
    with connect_metadata_db(config) as conn:
        cursor = conn.execute(
            """
            UPDATE query_admission_slots
            SET lease_expires_at = ?, updated_at = ?
            WHERE lease_owner = ? AND lease_expires_at >= ?
            """,
            (expires_at, timestamp, lease.owner, timestamp),
        )
    return int(cursor.rowcount or 0) == len(lease.scopes)


def release_query_admission_lease(
    *,
    config: RagConfig,
    lease: QueryAdmissionLease,
) -> bool:
    with connect_metadata_db(config) as conn:
        cursor = conn.execute(
            "DELETE FROM query_admission_slots WHERE lease_owner = ?",
            (lease.owner,),
        )
    return int(cursor.rowcount or 0) == len(lease.scopes)


def query_admission_metrics_snapshot(*, config: RagConfig) -> dict[str, int]:
    timestamp = now_ms()
    with connect_metadata_db(config) as conn:
        rows = conn.execute(
            """
            SELECT scope_type, COUNT(*) AS count
            FROM query_admission_slots
            WHERE lease_expires_at >= ?
            GROUP BY scope_type
            """,
            (timestamp,),
        ).fetchall()
        expired = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM query_admission_slots
            WHERE lease_expires_at < ?
            """,
            (timestamp,),
        ).fetchone()
    counts = {str(row["scope_type"]): int(row["count"] or 0) for row in rows}
    return {
        "global_slots": counts.get("global", 0),
        "tenant_slots": counts.get("tenant", 0),
        "user_slots": counts.get("user", 0),
        "expired_slots": int(expired["count"] or 0) if expired is not None else 0,
    }
