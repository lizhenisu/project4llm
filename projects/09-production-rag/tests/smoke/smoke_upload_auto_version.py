from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.config import load_config  # noqa: E402
from rag_core.sources import (  # noqa: E402
    IngestSummary,
    apply_uploaded_content_identity,
    create_source_task,
    ingest_uploaded_path,
    list_queued_source_tasks,
)
from rag_core.types import SourceDocument  # noqa: E402
from rag_core.versioning import publish_current_versions  # noqa: E402


TENANT_ID = "synthetic-auto-version-tenant"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = replace(
            load_config(),
            metadata_database_url=None,
            object_store_dir=root / "object_store",
            runtime_dir=root / "runtime",
        )
        upload_path = root / "uploads" / "synthetic-repeat.pdf"
        upload_path.parent.mkdir(parents=True)
        upload_path.write_bytes(b"%PDF-1.4\nsynthetic repeated upload\n")

        pending = create_source_task(
            config=config,
            tenant_id=TENANT_ID,
            path=upload_path,
            acl_groups=["engineering"],
            doc_version=None,
        )
        queued = next(
            record
            for record in list_queued_source_tasks(config=config)
            if record.source.doc_id == pending.doc_id
        )
        assert queued.requested_doc_version is None

        parsed_docs = apply_uploaded_content_identity(
            [synthetic_page(upload_path)],
            path=upload_path,
            input_dir=upload_path.parent,
        )
        stale_current_docs = [replace(doc, doc_version=2) for doc in parsed_docs]
        publish_current_versions(
            config.object_store_dir,
            stale_current_docs,
            config=config,
        )

        captured_versions: list[int] = []

        def capture_ingest(*, config, docs):
            captured_versions.extend(doc.doc_version for doc in docs)
            return IngestSummary(sources=[], document_count=len(docs), chunk_count=0)

        with (
            patch("rag_core.sources.load_documents_for_path", return_value=[synthetic_page(upload_path)]),
            patch("rag_core.sources.ingest_source_documents", side_effect=capture_ingest),
        ):
            ingest_uploaded_path(
                config=config,
                path=upload_path,
                tenant_id=TENANT_ID,
                acl_groups=["engineering"],
                doc_version=queued.requested_doc_version,
            )

    assert captured_versions == [3]
    print("smoke_upload_auto_version=ok")


def synthetic_page(path: Path) -> SourceDocument:
    return SourceDocument(
        tenant_id=TENANT_ID,
        doc_id="synthetic-repeat/page-1",
        doc_version=1,
        source_type="pdf",
        source_uri=str(path),
        title="synthetic-repeat p1",
        text="TiDE is five to ten times faster than the best Transformer baseline.",
        acl_groups=["engineering"],
        metadata={"relative_path": path.name, "page_no": 1},
    )


if __name__ == "__main__":
    main()
