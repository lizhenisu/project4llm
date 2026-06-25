from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.config import load_config
from rag_core.database import connect_metadata_db
from rag_core.text_utils import now_ms


def main() -> None:
    if not os.environ.get("RAG_METADATA_DATABASE_URL"):
        raise RuntimeError("RAG_METADATA_DATABASE_URL is required for this smoke test.")
    config = load_config()
    tenant_id = f"tenant-postgres-smoke-{now_ms()}"
    with connect_metadata_db(config) as conn:
        conn.execute(
            """
            INSERT INTO current_source_versions(tenant_id, doc_id, doc_version, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(tenant_id, doc_id) DO UPDATE SET
                doc_version = excluded.doc_version,
                updated_at = excluded.updated_at
            """,
            (tenant_id, "doc-a", 7, now_ms()),
        )
        row = conn.execute(
            """
            SELECT doc_version
            FROM current_source_versions
            WHERE tenant_id = ? AND doc_id = ?
            """,
            (tenant_id, "doc-a"),
        ).fetchone()
        assert row is not None
        assert row[0] == 7
        assert row["doc_version"] == 7
        deleted = conn.execute(
            "DELETE FROM current_source_versions WHERE tenant_id = ?",
            (tenant_id,),
        )
        assert deleted.rowcount == 1
    print("smoke_postgres_metadata=ok")


if __name__ == "__main__":
    main()
