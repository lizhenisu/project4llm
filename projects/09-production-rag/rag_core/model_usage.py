from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from rag_core.config import RagConfig
from rag_core.database import connect_metadata_db
from rag_core.text_utils import now_ms


@dataclass(frozen=True)
class ModelUsageRow:
    usage_date: str
    tenant_id: str
    principal_key: str
    workload: str
    provider: str
    model: str
    operation: str
    request_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    updated_at: int


@dataclass(frozen=True)
class ModelUsageContext:
    config: RagConfig
    tenant_id: str
    principal_key: str
    workload: str


_MODEL_USAGE_CONTEXT: ContextVar[ModelUsageContext | None] = ContextVar(
    "rag_model_usage_context",
    default=None,
)
_MODEL_USAGE_METRICS_LOCK = threading.Lock()
_MODEL_USAGE_METRICS = {
    "recorded_total": 0,
    "write_failures_total": 0,
}
MODEL_USAGE_WORKLOADS = (
    "query",
    "search",
    "ingestion",
    "studio_mindmap",
    "studio_table",
)


@contextmanager
def model_usage_context(
    *,
    config: RagConfig,
    tenant_id: str,
    principal_key: str,
    workload: str,
):
    context = ModelUsageContext(
        config=config,
        tenant_id=bounded_required(tenant_id, 200, "tenant_id"),
        principal_key=bounded(principal_key, 300),
        workload=bounded_required(workload, 80, "workload"),
    )
    token = _MODEL_USAGE_CONTEXT.set(context)
    try:
        yield context
    finally:
        _MODEL_USAGE_CONTEXT.reset(token)


def record_response_model_usage(
    *,
    response: Any,
    provider: str,
    model: str,
    operation: str,
) -> bool:
    context = _MODEL_USAGE_CONTEXT.get()
    if context is None:
        return False
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    try:
        record_model_usage(
            config=context.config,
            tenant_id=context.tenant_id,
            principal_key=context.principal_key,
            workload=context.workload,
            provider=provider,
            model=model,
            operation=operation,
            usage=usage,
        )
    except Exception:
        with _MODEL_USAGE_METRICS_LOCK:
            _MODEL_USAGE_METRICS["write_failures_total"] += 1
        return False
    with _MODEL_USAGE_METRICS_LOCK:
        _MODEL_USAGE_METRICS["recorded_total"] += 1
    return True


def model_usage_recording_metrics_snapshot() -> dict[str, int]:
    with _MODEL_USAGE_METRICS_LOCK:
        return dict(_MODEL_USAGE_METRICS)


def record_model_usage(
    *,
    config: RagConfig,
    tenant_id: str,
    principal_key: str,
    workload: str,
    provider: str,
    model: str,
    operation: str,
    usage: Any,
    timestamp_ms: int | None = None,
) -> ModelUsageRow:
    timestamp = now_ms() if timestamp_ms is None else int(timestamp_ms)
    prompt_tokens, completion_tokens, total_tokens = normalize_token_usage(usage)
    row = ModelUsageRow(
        usage_date=usage_date(timestamp),
        tenant_id=bounded_required(tenant_id, 200, "tenant_id"),
        principal_key=bounded(principal_key, 300),
        workload=bounded_required(workload, 80, "workload"),
        provider=bounded_required(provider, 80, "provider"),
        model=bounded_required(model, 200, "model"),
        operation=bounded_required(operation, 100, "operation"),
        request_count=1,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        updated_at=timestamp,
    )
    with connect_metadata_db(config) as conn:
        conn.execute(
            """
            INSERT INTO model_usage_daily(
                usage_date, tenant_id, principal_key, workload, provider, model,
                operation, request_count, prompt_tokens, completion_tokens,
                total_tokens, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
            ON CONFLICT(
                usage_date, tenant_id, principal_key, workload, provider, model, operation
            ) DO UPDATE SET
                request_count = model_usage_daily.request_count + excluded.request_count,
                prompt_tokens = model_usage_daily.prompt_tokens + excluded.prompt_tokens,
                completion_tokens = model_usage_daily.completion_tokens + excluded.completion_tokens,
                total_tokens = model_usage_daily.total_tokens + excluded.total_tokens,
                updated_at = excluded.updated_at
            """,
            (
                row.usage_date,
                row.tenant_id,
                row.principal_key,
                row.workload,
                row.provider,
                row.model,
                row.operation,
                row.prompt_tokens,
                row.completion_tokens,
                row.total_tokens,
                row.updated_at,
            ),
        )
    return row


def list_model_usage(
    *,
    config: RagConfig,
    tenant_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> list[ModelUsageRow]:
    conditions: list[str] = []
    parameters: list[object] = []
    if tenant_id:
        conditions.append("tenant_id = ?")
        parameters.append(tenant_id)
    if start_date:
        conditions.append("usage_date >= ?")
        parameters.append(start_date)
    if end_date:
        conditions.append("usage_date <= ?")
        parameters.append(end_date)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    parameters.extend((max(1, min(1000, int(limit))), max(0, int(offset))))
    with connect_metadata_db(config) as conn:
        rows = conn.execute(
            f"""
            SELECT usage_date, tenant_id, principal_key, workload, provider, model,
                   operation, request_count, prompt_tokens, completion_tokens,
                   total_tokens, updated_at
            FROM model_usage_daily
            {where}
            ORDER BY usage_date DESC, total_tokens DESC, tenant_id, operation
            LIMIT ? OFFSET ?
            """,
            tuple(parameters),
        ).fetchall()
    return [model_usage_row_from_db(row) for row in rows]


def count_model_usage_rows(
    *,
    config: RagConfig,
    tenant_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> int:
    conditions: list[str] = []
    parameters: list[object] = []
    if tenant_id:
        conditions.append("tenant_id = ?")
        parameters.append(tenant_id)
    if start_date:
        conditions.append("usage_date >= ?")
        parameters.append(start_date)
    if end_date:
        conditions.append("usage_date <= ?")
        parameters.append(end_date)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with connect_metadata_db(config) as conn:
        row = conn.execute(
            f"SELECT COUNT(*) FROM model_usage_daily {where}",
            tuple(parameters),
        ).fetchone()
    return int(row[0] if row is not None else 0)


def model_usage_totals(
    *,
    config: RagConfig,
    tenant_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, int]:
    conditions: list[str] = []
    parameters: list[object] = []
    if tenant_id:
        conditions.append("tenant_id = ?")
        parameters.append(tenant_id)
    if start_date:
        conditions.append("usage_date >= ?")
        parameters.append(start_date)
    if end_date:
        conditions.append("usage_date <= ?")
        parameters.append(end_date)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with connect_metadata_db(config) as conn:
        row = conn.execute(
            f"""
            SELECT
                COALESCE(SUM(request_count), 0) AS request_count,
                COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                COALESCE(SUM(total_tokens), 0) AS total_tokens
            FROM model_usage_daily
            {where}
            """,
            tuple(parameters),
        ).fetchone()
    return {
        "request_count": int(row["request_count"] or 0),
        "prompt_tokens": int(row["prompt_tokens"] or 0),
        "completion_tokens": int(row["completion_tokens"] or 0),
        "total_tokens": int(row["total_tokens"] or 0),
    }


def model_usage_daily_metrics_snapshot(
    *,
    config: RagConfig,
    selected_date: str | None = None,
) -> dict[str, object]:
    date = selected_date or usage_date(now_ms())
    workloads = {
        workload: {
            "request_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        for workload in MODEL_USAGE_WORKLOADS
    }
    with connect_metadata_db(config) as conn:
        rows = conn.execute(
            """
            SELECT workload,
                   SUM(request_count) AS request_count,
                   SUM(prompt_tokens) AS prompt_tokens,
                   SUM(completion_tokens) AS completion_tokens,
                   SUM(total_tokens) AS total_tokens
            FROM model_usage_daily
            WHERE usage_date = ?
            GROUP BY workload
            """,
            (date,),
        ).fetchall()
    for row in rows:
        workload = str(row["workload"])
        if workload not in workloads:
            continue
        workloads[workload] = {
            "request_count": int(row["request_count"] or 0),
            "prompt_tokens": int(row["prompt_tokens"] or 0),
            "completion_tokens": int(row["completion_tokens"] or 0),
            "total_tokens": int(row["total_tokens"] or 0),
        }
    return {
        "usage_date": date,
        "workloads": workloads,
        "recording": model_usage_recording_metrics_snapshot(),
    }


def model_usage_row_from_db(row) -> ModelUsageRow:
    return ModelUsageRow(
        usage_date=str(row["usage_date"]),
        tenant_id=str(row["tenant_id"]),
        principal_key=str(row["principal_key"]),
        workload=str(row["workload"]),
        provider=str(row["provider"]),
        model=str(row["model"]),
        operation=str(row["operation"]),
        request_count=int(row["request_count"] or 0),
        prompt_tokens=int(row["prompt_tokens"] or 0),
        completion_tokens=int(row["completion_tokens"] or 0),
        total_tokens=int(row["total_tokens"] or 0),
        updated_at=int(row["updated_at"] or 0),
    )


def normalize_token_usage(usage: Any) -> tuple[int, int, int]:
    prompt_tokens = usage_value(usage, "prompt_tokens", "input_tokens")
    completion_tokens = usage_value(usage, "completion_tokens", "output_tokens")
    total_tokens = usage_value(usage, "total_tokens")
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens


def usage_value(usage: Any, *names: str) -> int:
    for name in names:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if value is not None:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                continue
    return 0


def usage_date(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).date().isoformat()


def validate_usage_date(value: str) -> str:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("日期必须是有效的 YYYY-MM-DD") from exc
    return parsed.isoformat()


def bounded(value: str, limit: int) -> str:
    return str(value or "").strip()[:limit]


def bounded_required(value: str, limit: int, name: str) -> str:
    result = bounded(value, limit)
    if not result:
        raise ValueError(f"{name} is required")
    return result
