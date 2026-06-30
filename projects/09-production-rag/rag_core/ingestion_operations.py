from __future__ import annotations

import uuid
from dataclasses import dataclass

from rag_core.config import RagConfig
from rag_core.database import connect_metadata_db
from rag_core.text_utils import now_ms


INGESTION_OPERATION_OUTCOMES = (
    "queued",
    "not_found",
    "not_retryable",
    "admission_rejected_global",
    "admission_rejected_tenant",
    "admission_unavailable",
    "reservation_lost",
    "retry_unavailable",
)


@dataclass(frozen=True)
class IngestionOperationAudit:
    id: str
    actor_user_id: str
    tenant_id: str
    task_id: str
    operation: str
    outcome: str
    detail: str
    created_at: int


@dataclass(frozen=True)
class DeadLetterSourceTask:
    tenant_id: str
    task_id: str
    title: str
    source_type: str
    error: str
    attempt_count: int
    dead_lettered_at: int
    updated_at: int


def append_ingestion_operation_audit(
    *,
    config: RagConfig,
    actor_user_id: str,
    tenant_id: str,
    task_id: str,
    operation: str,
    outcome: str,
    detail: str = "",
    retention_days: int = 90,
    timestamp_ms: int | None = None,
) -> IngestionOperationAudit:
    timestamp = now_ms() if timestamp_ms is None else int(timestamp_ms)
    event = new_ingestion_operation_audit(
        actor_user_id=actor_user_id,
        tenant_id=tenant_id,
        task_id=task_id,
        operation=operation,
        outcome=outcome,
        detail=detail,
        timestamp_ms=timestamp,
    )
    with connect_metadata_db(config) as conn:
        prune_ingestion_operation_audit(
            conn,
            retention_days=retention_days,
            timestamp_ms=timestamp,
        )
        insert_ingestion_operation_audit(conn, event)
    return event


def new_ingestion_operation_audit(
    *,
    actor_user_id: str,
    tenant_id: str,
    task_id: str,
    operation: str,
    outcome: str,
    detail: str = "",
    timestamp_ms: int | None = None,
) -> IngestionOperationAudit:
    timestamp = now_ms() if timestamp_ms is None else int(timestamp_ms)
    return IngestionOperationAudit(
        id=f"ingestion-audit-{uuid.uuid4().hex}",
        actor_user_id=bounded(actor_user_id, 200),
        tenant_id=bounded_required(tenant_id, 200, "tenant_id"),
        task_id=bounded_required(task_id, 500, "task_id"),
        operation=bounded_required(operation, 80, "operation"),
        outcome=bounded_required(outcome, 80, "outcome"),
        detail=bounded(detail, 300),
        created_at=timestamp,
    )


def insert_ingestion_operation_audit(conn, event: IngestionOperationAudit) -> None:
    conn.execute(
        """
        INSERT INTO ingestion_operation_audit(
            id, actor_user_id, tenant_id, task_id, operation, outcome, detail, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.id,
            event.actor_user_id,
            event.tenant_id,
            event.task_id,
            event.operation,
            event.outcome,
            event.detail,
            event.created_at,
        ),
    )


def prune_ingestion_operation_audit(
    conn,
    *,
    retention_days: int = 90,
    timestamp_ms: int | None = None,
) -> None:
    timestamp = now_ms() if timestamp_ms is None else int(timestamp_ms)
    cutoff = timestamp - max(1, int(retention_days)) * 24 * 60 * 60 * 1000
    conn.execute(
        "DELETE FROM ingestion_operation_audit WHERE created_at < ?",
        (cutoff,),
    )


def list_dead_letter_source_tasks(
    *,
    config: RagConfig,
    limit: int = 50,
    offset: int = 0,
) -> list[DeadLetterSourceTask]:
    with connect_metadata_db(config) as conn:
        rows = conn.execute(
            """
            SELECT tenant_id, id, title, source_type, error, attempt_count,
                   dead_lettered_at, updated_at
            FROM source_tasks
            WHERE status = 'failed' AND dead_lettered_at > 0
            ORDER BY dead_lettered_at DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            (max(1, min(200, int(limit))), max(0, int(offset))),
        ).fetchall()
    return [
        DeadLetterSourceTask(
            tenant_id=str(row["tenant_id"]),
            task_id=str(row["id"]),
            title=str(row["title"]),
            source_type=str(row["source_type"]),
            error=str(row["error"] or ""),
            attempt_count=int(row["attempt_count"] or 0),
            dead_lettered_at=int(row["dead_lettered_at"] or 0),
            updated_at=int(row["updated_at"] or 0),
        )
        for row in rows
    ]


def count_dead_letter_source_tasks(*, config: RagConfig) -> int:
    with connect_metadata_db(config) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM source_tasks
            WHERE status = 'failed' AND dead_lettered_at > 0
            """
        ).fetchone()
    return int(row[0] if row is not None else 0)


def list_ingestion_operation_audit(
    *,
    config: RagConfig,
    limit: int = 50,
    offset: int = 0,
    retention_days: int = 90,
) -> list[IngestionOperationAudit]:
    cutoff = audit_retention_cutoff(retention_days=retention_days)
    with connect_metadata_db(config) as conn:
        rows = conn.execute(
            """
            SELECT id, actor_user_id, tenant_id, task_id, operation, outcome, detail, created_at
            FROM ingestion_operation_audit
            WHERE created_at >= ?
            ORDER BY created_at DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            (
                cutoff,
                max(1, min(200, int(limit))),
                max(0, int(offset)),
            ),
        ).fetchall()
    return [
        IngestionOperationAudit(
            id=str(row["id"]),
            actor_user_id=str(row["actor_user_id"]),
            tenant_id=str(row["tenant_id"]),
            task_id=str(row["task_id"]),
            operation=str(row["operation"]),
            outcome=str(row["outcome"]),
            detail=str(row["detail"]),
            created_at=int(row["created_at"]),
        )
        for row in rows
    ]


def count_ingestion_operation_audit(
    *,
    config: RagConfig,
    retention_days: int = 90,
) -> int:
    cutoff = audit_retention_cutoff(retention_days=retention_days)
    with connect_metadata_db(config) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM ingestion_operation_audit WHERE created_at >= ?",
            (cutoff,),
        ).fetchone()
    return int(row[0] if row is not None else 0)


def ingestion_operation_metrics_snapshot(
    *,
    config: RagConfig,
    retention_days: int = 90,
) -> dict[str, object]:
    outcomes = {outcome: 0 for outcome in INGESTION_OPERATION_OUTCOMES}
    cutoff = audit_retention_cutoff(retention_days=retention_days)
    with connect_metadata_db(config) as conn:
        rows = conn.execute(
            """
            SELECT outcome, COUNT(*) AS event_count
            FROM ingestion_operation_audit
            WHERE operation = 'bulk_redrive' AND created_at >= ?
            GROUP BY outcome
            """,
            (cutoff,),
        ).fetchall()
    for row in rows:
        outcome = str(row["outcome"])
        if outcome in outcomes:
            outcomes[outcome] = int(row["event_count"] or 0)
    return {
        "audit_events": sum(outcomes.values()),
        "bulk_redrive_outcomes": outcomes,
    }


def audit_retention_cutoff(*, retention_days: int, timestamp_ms: int | None = None) -> int:
    timestamp = now_ms() if timestamp_ms is None else int(timestamp_ms)
    return timestamp - max(1, int(retention_days)) * 24 * 60 * 60 * 1000


def bounded(value: str, limit: int) -> str:
    return str(value or "").strip()[:limit]


def bounded_required(value: str, limit: int, name: str) -> str:
    result = bounded(value, limit)
    if not result:
        raise ValueError(f"{name} is required")
    return result
