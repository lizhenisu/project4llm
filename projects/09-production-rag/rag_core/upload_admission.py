from __future__ import annotations

import uuid
from dataclasses import dataclass

from rag_core.config import RagConfig
from rag_core.database import connect_metadata_db
from rag_core.text_utils import now_ms


ACTIVE_SOURCE_TASK_STATUSES = ("queued", "processing", "uploading")


@dataclass(frozen=True)
class UploadAdmissionReservation:
    owner: str
    tenant_id: str
    expires_at: int


class UploadAdmissionRejected(RuntimeError):
    def __init__(self, kind: str, *, active_tasks: int, limit: int) -> None:
        self.kind = kind
        self.active_tasks = active_tasks
        self.limit = limit
        super().__init__(f"Upload admission rejected at {kind} scope")


def acquire_upload_admission_reservation(
    *,
    config: RagConfig,
    tenant_id: str,
    global_limit: int,
    tenant_limit: int,
    lease_ms: int,
    owner: str | None = None,
) -> UploadAdmissionReservation:
    timestamp = now_ms()
    resolved_owner = owner or f"upload-{uuid.uuid4().hex}"
    expires_at = timestamp + max(1000, int(lease_ms))
    with connect_metadata_db(config) as conn:
        lock_upload_admission(conn, timestamp=timestamp)
        conn.execute(
            "DELETE FROM upload_admission_slots WHERE lease_expires_at < ?",
            (timestamp,),
        )
        global_active = active_source_task_count(conn)
        tenant_active = active_source_task_count(conn, tenant_id=tenant_id)
        scopes = (
            ("global", "*", max(1, int(global_limit)), global_active),
            ("tenant", tenant_id, max(1, int(tenant_limit)), tenant_active),
        )
        for scope_type, scope_key, limit, active_tasks in scopes:
            available_reservations = max(0, limit - active_tasks)
            if available_reservations == 0 or not claim_upload_slot(
                conn,
                scope_type=scope_type,
                scope_key=scope_key,
                available_reservations=available_reservations,
                owner=resolved_owner,
                expires_at=expires_at,
                timestamp=timestamp,
            ):
                raise UploadAdmissionRejected(
                    scope_type,
                    active_tasks=active_tasks,
                    limit=limit,
                )
    return UploadAdmissionReservation(
        owner=resolved_owner,
        tenant_id=tenant_id,
        expires_at=expires_at,
    )


def lock_upload_admission(conn, *, timestamp: int | None = None) -> None:
    resolved_timestamp = timestamp if timestamp is not None else now_ms()
    conn.execute(
        """
        INSERT INTO admission_locks(name, updated_at)
        VALUES ('upload', ?)
        ON CONFLICT(name) DO NOTHING
        """,
        (resolved_timestamp,),
    )
    conn.execute(
        "UPDATE admission_locks SET updated_at = ? WHERE name = 'upload'",
        (resolved_timestamp,),
    )


def active_source_task_count(conn, *, tenant_id: str | None = None) -> int:
    params: list[object] = [*ACTIVE_SOURCE_TASK_STATUSES]
    where = "WHERE status IN (?, ?, ?)"
    if tenant_id:
        where += " AND tenant_id = ?"
        params.append(tenant_id)
    row = conn.execute(
        f"SELECT COUNT(*) AS count FROM source_tasks {where}",
        tuple(params),
    ).fetchone()
    return int(row["count"] or 0) if row is not None else 0


def claim_upload_slot(
    conn,
    *,
    scope_type: str,
    scope_key: str,
    available_reservations: int,
    owner: str,
    expires_at: int,
    timestamp: int,
) -> bool:
    for slot_index in range(available_reservations):
        cursor = conn.execute(
            """
            INSERT INTO upload_admission_slots(
                scope_type, scope_key, slot_index, reservation_owner,
                lease_expires_at, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(scope_type, scope_key, slot_index) DO NOTHING
            """,
            (
                scope_type,
                scope_key,
                slot_index,
                owner,
                expires_at,
                timestamp,
            ),
        )
        if int(cursor.rowcount or 0) == 1:
            return True
    return False


def release_upload_admission_reservation(
    *,
    config: RagConfig,
    reservation: UploadAdmissionReservation,
) -> bool:
    with connect_metadata_db(config) as conn:
        cursor = conn.execute(
            "DELETE FROM upload_admission_slots WHERE reservation_owner = ?",
            (reservation.owner,),
        )
    return int(cursor.rowcount or 0) == 2


def upload_admission_metrics_snapshot(*, config: RagConfig) -> dict[str, int]:
    timestamp = now_ms()
    with connect_metadata_db(config) as conn:
        rows = conn.execute(
            """
            SELECT scope_type, COUNT(*) AS count
            FROM upload_admission_slots
            WHERE lease_expires_at >= ?
            GROUP BY scope_type
            """,
            (timestamp,),
        ).fetchall()
        expired = conn.execute(
            "SELECT COUNT(*) AS count FROM upload_admission_slots WHERE lease_expires_at < ?",
            (timestamp,),
        ).fetchone()
    counts = {str(row["scope_type"]): int(row["count"] or 0) for row in rows}
    return {
        "global_reservations": counts.get("global", 0),
        "tenant_reservations": counts.get("tenant", 0),
        "expired_reservations": int(expired["count"] or 0) if expired is not None else 0,
    }
