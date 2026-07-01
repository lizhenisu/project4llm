from __future__ import annotations

import os
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.config import load_config  # noqa: E402
from rag_core.database import connect_metadata_db  # noqa: E402
from rag_core.model_api_retry import call_model_api_with_retries  # noqa: E402
from rag_core.model_usage import (  # noqa: E402
    count_model_usage_rows,
    list_model_usage,
    model_usage_context,
    record_model_usage,
)


PREFIX = f"model-usage-smoke-{uuid.uuid4().hex[:10]}"
TIMESTAMP = 1_782_864_000_000


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="rag-model-usage-") as tmp:
        config = replace(
            load_config(),
            metadata_database_url=os.environ.get("SMOKE_METADATA_DATABASE_URL") or None,
            object_store_dir=Path(tmp) / "object_store",
            runtime_dir=Path(tmp) / "runtime",
        )
        with connect_metadata_db(config):
            pass
        try:
            test_concurrent_daily_aggregation(config)
            test_scope_and_usage_normalization(config)
            test_retry_layer_records_active_context(config)
        finally:
            cleanup(config)
    print("smoke_model_usage=ok")


def test_concurrent_daily_aggregation(config) -> None:
    def record(_index: int) -> None:
        record_model_usage(
            config=config,
            tenant_id=f"{PREFIX}-tenant-a",
            principal_key=f"{PREFIX}-user-a",
            workload="query",
            provider="newapi",
            model="synthetic-chat-model",
            operation="answer_generation",
            usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            timestamp_ms=TIMESTAMP,
        )

    with ThreadPoolExecutor(max_workers=20) as executor:
        list(executor.map(record, range(40)))

    rows = list_model_usage(
        config=config,
        tenant_id=f"{PREFIX}-tenant-a",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.request_count == 40
    assert row.prompt_tokens == 440
    assert row.completion_tokens == 280
    assert row.total_tokens == 720


def test_scope_and_usage_normalization(config) -> None:
    record_model_usage(
        config=config,
        tenant_id=f"{PREFIX}-tenant-b",
        principal_key="",
        workload="ingestion",
        provider="siliconflow",
        model="synthetic-embedding-model",
        operation="text_embedding",
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=None),
        timestamp_ms=TIMESTAMP + 24 * 60 * 60 * 1000,
    )
    rows = list_model_usage(
        config=config,
        start_date="2026-07-01",
        end_date="2026-07-02",
    )
    assert len(rows) == 2
    tenant_b = next(row for row in rows if row.tenant_id.endswith("tenant-b"))
    assert tenant_b.total_tokens == 7
    assert tenant_b.request_count == 1
    assert count_model_usage_rows(config=config) == 2
    assert count_model_usage_rows(
        config=config,
        tenant_id=f"{PREFIX}-tenant-b",
    ) == 1


def test_retry_layer_records_active_context(config) -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=13,
            completion_tokens=3,
            total_tokens=16,
        )
    )
    with model_usage_context(
        config=config,
        tenant_id=f"{PREFIX}-tenant-c",
        principal_key=f"{PREFIX}-user-c",
        workload="studio_table",
    ):
        assert call_model_api_with_retries(
            "table_generation",
            lambda: response,
            usage_provider="synthetic-provider",
            usage_model="synthetic-model",
        ) is response
    rows = list_model_usage(
        config=config,
        tenant_id=f"{PREFIX}-tenant-c",
    )
    assert len(rows) == 1
    assert rows[0].workload == "studio_table"
    assert rows[0].operation == "table_generation"
    assert rows[0].request_count == 1
    assert rows[0].total_tokens == 16


def cleanup(config) -> None:
    with connect_metadata_db(config) as conn:
        conn.execute(
            "DELETE FROM model_usage_daily WHERE tenant_id LIKE ?",
            (f"{PREFIX}%",),
        )


if __name__ == "__main__":
    main()
