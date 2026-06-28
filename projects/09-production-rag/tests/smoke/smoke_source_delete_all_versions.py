from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.config import load_config  # noqa: E402
from rag_core.database import connect_metadata_db  # noqa: E402
from rag_core.object_store import archive_source_documents, load_archived_source_documents  # noqa: E402
from rag_core.sources import (  # noqa: E402
    delete_source,
    list_source_catalog,
    next_source_doc_version,
    save_source_catalog_for_tenant,
    SourceSummary,
)
from rag_core.types import SourceDocument  # noqa: E402
from rag_core.versioning import load_current_versions, publish_current_versions  # noqa: E402


RUN_ID = uuid.uuid4().hex[:10]
LOGICAL_DOC_ID = "synthetic-delete@sha256-fixed"


class FakeMilvusClient:
    def __init__(self) -> None:
        self.delete_filters: list[str] = []

    def delete(self, *, collection_name: str, filter: str):
        self.delete_filters.append(filter)
        return {"delete_count": 3, "collection": collection_name}


def main() -> None:
    run_case(database_url=None, backend_name="sqlite")
    postgres_url = os.environ.get("SMOKE_METADATA_DATABASE_URL", "")
    if postgres_url:
        run_case(database_url=postgres_url, backend_name="postgres")
    print("smoke_source_delete_all_versions=ok")


def run_case(*, database_url: str | None, backend_name: str) -> None:
    tenant_id = f"synthetic-delete-all-{backend_name}-{RUN_ID}"
    with tempfile.TemporaryDirectory(prefix=f"rag-delete-{backend_name}-") as tmp:
        root = Path(tmp)
        config = replace(
            load_config(),
            metadata_database_url=database_url,
            object_store_backend="local",
            object_store_dir=root / "object_store",
            runtime_dir=root / "runtime",
            source_list_cache_ttl_seconds=0,
        )
        v1_docs = [
            source_doc(tenant_id, "page-1", version=1),
            source_doc(tenant_id, "page-old-only", version=1),
        ]
        v2_docs = [
            source_doc(tenant_id, "page-1", version=2),
            source_doc(tenant_id, "page-new-only", version=2),
        ]
        archive_source_documents(config.object_store_dir, [*v1_docs, *v2_docs])
        publish_current_versions(config.object_store_dir, v2_docs, config=config)
        save_source_catalog_for_tenant(
            config=config,
            tenant_id=tenant_id,
            sources=[
                source_summary(tenant_id, v1_docs, version=1, current=False),
                source_summary(tenant_id, v2_docs, version=2, current=True),
            ],
        )
        seed_metadata_rows(config, tenant_id)

        fake_milvus = FakeMilvusClient()
        with (
            patch("rag_core.sources.connect", return_value=fake_milvus),
            patch("rag_core.sources.ensure_collection"),
        ):
            detail = delete_source(
                config=config,
                tenant_id=tenant_id,
                doc_id=LOGICAL_DOC_ID,
                doc_version=None,
            )

        expected_children = sorted({doc.doc_id for doc in [*v1_docs, *v2_docs]})
        assert detail["logical_doc_ids"] == [LOGICAL_DOC_ID]
        assert detail["target_doc_ids"] == expected_children
        assert len(fake_milvus.delete_filters) == 1
        delete_filter = fake_milvus.delete_filters[0]
        assert "doc_version ==" not in delete_filter
        assert all(child_doc_id in delete_filter for child_doc_id in expected_children)
        assert list_source_catalog(config=config, tenant_id=tenant_id) == []
        assert load_current_versions(config.object_store_dir, tenant_id=tenant_id, config=config) == {}
        assert load_archived_source_documents(config.object_store_dir, include_deleted=True) == []
        assert next_source_doc_version(config, docs=[source_doc(tenant_id, "page-1", version=1)]) == 1
        assert_metadata_rows_deleted(config, tenant_id)


def source_doc(tenant_id: str, suffix: str, *, version: int) -> SourceDocument:
    return SourceDocument(
        tenant_id=tenant_id,
        doc_id=f"{LOGICAL_DOC_ID}/{suffix}",
        doc_version=version,
        source_type="pdf",
        source_uri="/synthetic/2023-TIDE.pdf",
        title=f"2023-TIDE {suffix}",
        text=f"synthetic version {version} content for {suffix}",
        acl_groups=["engineering"],
        metadata={
            "relative_path": "synthetic-delete.pdf",
            "content_key": "sha256-fixed",
            "page_no": 1,
        },
    )


def source_summary(
    tenant_id: str,
    docs: list[SourceDocument],
    *,
    version: int,
    current: bool,
) -> SourceSummary:
    return SourceSummary(
        doc_id=LOGICAL_DOC_ID,
        title="2023-TIDE.pdf",
        source_type="pdf",
        source_uri="/synthetic/2023-TIDE.pdf",
        doc_version=version,
        chunk_count=len(docs),
        acl_groups=["engineering"],
        status="ready",
        current=current,
        created_at=version,
        updated_at=version,
        child_doc_ids=[doc.doc_id for doc in docs],
    )


def seed_metadata_rows(config, tenant_id: str) -> None:
    with connect_metadata_db(config) as conn:
        conn.execute(
            """
            INSERT INTO source_title_overrides(tenant_id, doc_id, doc_version, title, updated_at)
            VALUES (?, ?, ?, ?, ?), (?, ?, ?, ?, ?)
            """,
            (
                tenant_id,
                LOGICAL_DOC_ID,
                1,
                "old title",
                1,
                tenant_id,
                LOGICAL_DOC_ID,
                2,
                "new title",
                2,
            ),
        )
        conn.execute(
            """
            INSERT INTO source_tasks(
                id, tenant_id, doc_id, title, source_type, source_uri, doc_version,
                acl_groups, status, error, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"synthetic-delete-task-{RUN_ID}-{tenant_id}",
                tenant_id,
                LOGICAL_DOC_ID,
                "stale task",
                "pdf",
                "/synthetic/2023-TIDE.pdf",
                2,
                json.dumps(["engineering"]),
                "failed",
                "synthetic stale task",
                1,
                1,
            ),
        )


def assert_metadata_rows_deleted(config, tenant_id: str) -> None:
    with connect_metadata_db(config) as conn:
        title_count = conn.execute(
            "SELECT COUNT(*) AS count FROM source_title_overrides WHERE tenant_id = ?",
            (tenant_id,),
        ).fetchone()["count"]
        task_count = conn.execute(
            "SELECT COUNT(*) AS count FROM source_tasks WHERE tenant_id = ?",
            (tenant_id,),
        ).fetchone()["count"]
    assert int(title_count) == 0
    assert int(task_count) == 0


if __name__ == "__main__":
    main()
