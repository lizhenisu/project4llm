from __future__ import annotations

import tempfile
from pathlib import Path

from rag_core.object_store import (
    archive_delete_tombstone,
    archive_source_documents,
    load_archived_source_documents,
    load_delete_tombstones,
)
from rag_core.types import SourceDocument


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        object_store_dir = Path(tmp) / "object_store"
        v1 = SourceDocument(
            tenant_id="team_a",
            doc_id="delete-runbook",
            doc_version=1,
            source_type="md",
            source_uri="memory://delete-runbook-v1",
            title="Delete Runbook v1",
            text="v1 应该被删除 tombstone 跳过。",
            acl_groups=["ops"],
        )
        v2 = SourceDocument(
            tenant_id="team_a",
            doc_id="delete-runbook",
            doc_version=2,
            source_type="md",
            source_uri="memory://delete-runbook-v2",
            title="Delete Runbook v2",
            text="v2 仍然可以重建。",
            acl_groups=["ops"],
        )
        archive_source_documents(object_store_dir, [v1, v2])

        archive_delete_tombstone(
            object_store_dir,
            tenant_id="team_a",
            doc_id="delete-runbook",
            doc_version=1,
        )
        active_docs = load_archived_source_documents(object_store_dir)
        assert [doc.doc_version for doc in active_docs] == [2]
        all_docs = load_archived_source_documents(object_store_dir, include_deleted=True)
        assert sorted(doc.doc_version for doc in all_docs) == [1, 2]

        archive_source_documents(object_store_dir, [v1])
        assert load_delete_tombstones(object_store_dir) == []
        restored_docs = load_archived_source_documents(object_store_dir)
        assert sorted(doc.doc_version for doc in restored_docs) == [1, 2]

        archive_delete_tombstone(
            object_store_dir,
            tenant_id="team_a",
            doc_id="delete-runbook",
        )
        assert load_archived_source_documents(object_store_dir) == []
        assert len(load_archived_source_documents(object_store_dir, include_deleted=True)) == 2

    print("smoke_object_store_delete_tombstone=ok")


if __name__ == "__main__":
    main()
