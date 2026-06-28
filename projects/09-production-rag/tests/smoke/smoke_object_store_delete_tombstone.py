from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import tempfile
from pathlib import Path

from rag_core.object_store import (
    archive_delete_tombstone,
    archive_source_documents,
    load_archived_source_documents,
    load_delete_tombstones,
    purge_source_documents,
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

        duplicate_v2 = SourceDocument(
            tenant_id="team_a",
            doc_id="delete-runbook",
            doc_version=2,
            source_type="md",
            source_uri="memory://delete-runbook-v2-reupload",
            title="Delete Runbook v2",
            text="v2 重新上传后应覆盖同版本归档，而不是重复展示。",
            acl_groups=["ops"],
        )
        archive_source_documents(object_store_dir, [duplicate_v2])
        active_docs_before_delete = load_archived_source_documents(object_store_dir)
        assert len(active_docs_before_delete) == 2
        assert [doc for doc in active_docs_before_delete if doc.doc_version == 2][0].text.startswith("v2 重新上传")

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

        upload_dir = object_store_dir / "uploads" / "team_a" / "upload-1"
        upload_dir.mkdir(parents=True)
        uploaded_file = upload_dir / "delete-runbook.md"
        uploaded_file.write_text("private document text", encoding="utf-8")
        private_doc = SourceDocument(
            tenant_id="team_a",
            doc_id="private-doc",
            doc_version=1,
            source_type="md",
            source_uri=str(uploaded_file),
            title="Private Doc",
            text="private document text",
            acl_groups=["ops"],
        )
        archive_source_documents(object_store_dir, [private_doc])
        archive_delete_tombstone(
            object_store_dir,
            tenant_id="team_a",
            doc_id="private-doc",
            doc_version=1,
        )
        purged = purge_source_documents(
            object_store_dir,
            tenant_id="team_a",
            doc_ids=["private-doc"],
            doc_version=1,
        )
        assert purged == {
            "archived_documents": 1,
            "delete_tombstones": 1,
            "upload_dirs": 1,
            "uploaded_objects": 0,
        }
        assert not upload_dir.exists()
        assert all(doc.doc_id != "private-doc" for doc in load_archived_source_documents(object_store_dir, include_deleted=True))

        s3_work_dir = object_store_dir / "uploads" / "team_a" / "s3-work-copy"
        s3_work_dir.mkdir(parents=True)
        s3_work_file = s3_work_dir / "source.pdf"
        s3_work_file.write_bytes(b"synthetic local parser copy")
        s3_doc = SourceDocument(
            tenant_id="team_a",
            doc_id="s3-backed-doc",
            doc_version=1,
            source_type="pdf",
            source_uri="s3://synthetic-bucket/uploads/team_a/s3-work-copy/source.pdf",
            title="S3 backed doc",
            text="synthetic S3 content",
            acl_groups=["ops"],
            metadata={"source_uri_local_work_path": str(s3_work_file)},
        )
        archive_source_documents(object_store_dir, [s3_doc])
        s3_purged = purge_source_documents(
            object_store_dir,
            tenant_id="team_a",
            doc_ids=["s3-backed-doc"],
        )
        assert s3_purged["upload_dirs"] == 1
        assert not s3_work_dir.exists()

        concurrent_docs = [
            SourceDocument(
                tenant_id="team_a",
                doc_id=f"concurrent-doc-{index}",
                doc_version=1,
                source_type="md",
                source_uri=f"memory://concurrent-doc-{index}",
                title=f"Concurrent Doc {index}",
                text="concurrent archive test",
                acl_groups=["ops"],
            )
            for index in range(30)
        ]
        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(lambda doc: archive_source_documents(object_store_dir, [doc]), concurrent_docs))
        archived_doc_ids = {
            doc.doc_id
            for doc in load_archived_source_documents(object_store_dir, include_deleted=True)
        }
        assert {doc.doc_id for doc in concurrent_docs}.issubset(archived_doc_ids)

    print("smoke_object_store_delete_tombstone=ok")


if __name__ == "__main__":
    main()
