from __future__ import annotations

import os
import sys
import tempfile
from io import BytesIO
from types import SimpleNamespace
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.object_store import archive_source_documents, load_archived_source_documents, purge_source_documents
from rag_core.document_scope import build_scope_plan
from rag_core.jsonl_store import read_object_bytes_by_relative_path
from rag_core.sources import save_uploaded_file
from rag_core.section_summaries import (
    delete_source_section_summaries,
    load_source_section_summaries,
    save_source_section_summaries,
)
from rag_core.source_guides import load_source_guide, save_source_guide
from rag_core.text_utils import now_ms
from rag_core.types import SourceDocument


ENV_KEYS = [
    "RAG_OBJECT_STORE_BACKEND",
    "RAG_S3_ENDPOINT_URL",
    "RAG_S3_ACCESS_KEY_ID",
    "RAG_S3_SECRET_ACCESS_KEY",
    "RAG_S3_BUCKET",
    "RAG_S3_PREFIX",
]


def main() -> None:
    old_env = {key: os.environ.get(key) for key in ENV_KEYS}
    tenant_id = f"tenant-s3-smoke-{now_ms()}"
    prefix = f"smoke/{tenant_id}"
    try:
        os.environ["RAG_OBJECT_STORE_BACKEND"] = "s3"
        os.environ["RAG_S3_ENDPOINT_URL"] = os.environ.get("RAG_S3_ENDPOINT_URL", "http://127.0.0.1:9000")
        os.environ["RAG_S3_ACCESS_KEY_ID"] = os.environ.get("RAG_S3_ACCESS_KEY_ID", "minioadmin")
        os.environ["RAG_S3_SECRET_ACCESS_KEY"] = os.environ.get("RAG_S3_SECRET_ACCESS_KEY", "minioadmin")
        os.environ["RAG_S3_BUCKET"] = os.environ.get("RAG_S3_BUCKET", "production-rag")
        os.environ["RAG_S3_PREFIX"] = prefix
        with tempfile.TemporaryDirectory() as tmp:
            object_store_dir = Path(tmp) / "object_store"
            config = SimpleNamespace(
                object_store_dir=object_store_dir,
                max_context_chars=20_000,
                max_upload_bytes=10 * 1024 * 1024,
            )
            uploaded = save_uploaded_file(
                config=config,
                tenant_id=tenant_id,
                filename="upload-smoke.txt",
                content=BytesIO(b"s3 upload smoke"),
            )
            assert uploaded.is_file()
            stored_body = read_object_bytes_by_relative_path(
                object_store_dir,
                Path("uploads") / tenant_id / uploaded.parent.name / uploaded.name,
            )
            assert stored_body == b"s3 upload smoke"
            doc = SourceDocument(
                tenant_id=tenant_id,
                doc_id="doc-a",
                doc_version=1,
                source_type="txt",
                source_uri="s3://production-rag/smoke/doc-a.txt",
                title="S3 Smoke Doc",
                text="S3 object store smoke content.",
                acl_groups=["engineering"],
            )
            assert archive_source_documents(object_store_dir, [doc]) == 1
            loaded = load_archived_source_documents(object_store_dir)
            assert len(loaded) == 1
            assert loaded[0].doc_id == "doc-a"
            save_source_guide(
                object_store_dir,
                tenant_id=tenant_id,
                source_doc_id="doc-a",
                doc_version=1,
                title="S3 Smoke Doc",
                guide="This guide is stored in S3.",
                model="smoke",
            )
            assert save_source_section_summaries(
                object_store_dir,
                tenant_id=tenant_id,
                source_doc_id="doc-a",
                doc_version=1,
                docs=[doc],
            ) == 1
            assert "stored in S3" in (load_source_guide(
                object_store_dir,
                tenant_id=tenant_id,
                source_doc_id="doc-a",
                doc_version=1,
            ) or "")
            scope_plan = build_scope_plan(
                config=config,
                tenant_id=tenant_id,
                query="总结这份资料",
                doc_ids=["doc-a"],
                doc_version=1,
                include_all_sources=False,
            )
            assert scope_plan.route.coverage_required is True
            assert scope_plan.coverage()["covered_doc_count"] == 1
            assert scope_plan.guides[0].guide == "This guide is stored in S3."
            extraction_plan = build_scope_plan(
                config=config,
                tenant_id=tenant_id,
                query="从这份资料中提取关键内容",
                doc_ids=["doc-a"],
                doc_version=1,
                include_all_sources=False,
            )
            assert len(extraction_plan.section_summaries) == 1
            assert extraction_plan.section_summaries[0].summary == "S3 object store smoke content."
            purged = purge_source_documents(
                object_store_dir,
                tenant_id=tenant_id,
                doc_ids=["doc-a"],
                doc_version=1,
            )
            assert purged["archived_documents"] == 1
            assert load_archived_source_documents(object_store_dir) == []
            assert delete_source_section_summaries(
                object_store_dir,
                tenant_id=tenant_id,
                source_doc_ids={"doc-a"},
                doc_version=1,
            ) == 1
            assert load_source_section_summaries(
                object_store_dir,
                tenant_id=tenant_id,
                source_keys={("doc-a", 1)},
            ) == []
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("smoke_s3_object_store=ok")


if __name__ == "__main__":
    main()
