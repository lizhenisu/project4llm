from __future__ import annotations

import os
import sys
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.config import load_config  # noqa: E402
from rag_core.database import connect_metadata_db  # noqa: E402
from rag_core.ingestion_operations import (  # noqa: E402
    append_ingestion_operation_audit,
    count_ingestion_operation_audit,
    list_ingestion_operation_audit,
)
from rag_core.text_utils import now_ms  # noqa: E402


PREFIX = f"ingestion-audit-smoke-{uuid.uuid4().hex[:10]}"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="rag-ingestion-audit-") as tmp:
        config = replace(
            load_config(),
            metadata_database_url=os.environ.get("SMOKE_METADATA_DATABASE_URL") or None,
            object_store_dir=Path(tmp) / "object_store",
            runtime_dir=Path(tmp) / "runtime",
        )
        with connect_metadata_db(config):
            pass
        timestamp = now_ms()
        append_ingestion_operation_audit(
            config=config,
            actor_user_id=f"{PREFIX}-actor",
            tenant_id=f"{PREFIX}-tenant",
            task_id=f"{PREFIX}-old",
            operation="bulk_redrive",
            outcome="not_retryable",
            detail="bounded synthetic reason",
            retention_days=1,
            timestamp_ms=timestamp - 2 * 24 * 60 * 60 * 1000,
        )
        append_ingestion_operation_audit(
            config=config,
            actor_user_id=f"{PREFIX}-actor",
            tenant_id=f"{PREFIX}-tenant",
            task_id=f"{PREFIX}-current",
            operation="bulk_redrive",
            outcome="queued",
            retention_days=1,
            timestamp_ms=timestamp,
        )
        rows = list_ingestion_operation_audit(config=config)
        assert len(rows) == 1
        assert rows[0].task_id == f"{PREFIX}-current"
        assert rows[0].actor_user_id == f"{PREFIX}-actor"
        assert rows[0].outcome == "queued"
        assert count_ingestion_operation_audit(config=config) == 1
        cleanup(config)
    print("smoke_ingestion_operation_audit=ok")


def cleanup(config) -> None:
    with connect_metadata_db(config) as conn:
        conn.execute(
            "DELETE FROM ingestion_operation_audit WHERE tenant_id LIKE ?",
            (f"{PREFIX}%",),
        )


if __name__ == "__main__":
    main()
