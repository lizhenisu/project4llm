from __future__ import annotations

import tempfile
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.object_store import archive_source_documents
from rag_core.sources import apply_uploaded_content_identity, next_source_doc_version, source_document_identity
from rag_core.types import SourceDocument


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        first_dir = root / "first"
        second_dir = root / "second"
        duplicate_dir = root / "duplicate"
        first_dir.mkdir()
        second_dir.mkdir()
        duplicate_dir.mkdir()
        first_path = first_dir / "自然辩证法.pdf"
        second_path = second_dir / "自然辩证法.pdf"
        duplicate_path = duplicate_dir / "自然辩证法.pdf"
        first_path.write_bytes(b"%PDF-1.4\npage one says alpha\n")
        second_path.write_bytes(b"%PDF-1.4\npage one says beta\n")
        duplicate_path.write_bytes(first_path.read_bytes())

        first_doc = stamped_doc(first_path, first_dir)
        second_doc = stamped_doc(second_path, second_dir)
        duplicate_doc = stamped_doc(duplicate_path, duplicate_dir)
        config = SimpleNamespace(
            object_store_dir=root / "object_store",
            runtime_dir=root / "runtime",
            metadata_database_url=None,
        )
        archive_source_documents(config.object_store_dir, [first_doc])
        first_repeat_version = next_source_doc_version(config, [duplicate_doc])
        second_content_version = next_source_doc_version(config, [second_doc])

    assert first_doc.metadata["content_sha256"] != second_doc.metadata["content_sha256"]
    assert first_doc.metadata["content_sha256"] == duplicate_doc.metadata["content_sha256"]
    assert first_doc.doc_id != second_doc.doc_id
    assert first_doc.doc_id == duplicate_doc.doc_id
    assert first_repeat_version == 2
    assert second_content_version == 1

    first_source_id, first_title = source_document_identity(
        doc_id=first_doc.doc_id,
        title=first_doc.title,
        source_uri=first_doc.source_uri,
        metadata=first_doc.metadata,
    )
    second_source_id, second_title = source_document_identity(
        doc_id=second_doc.doc_id,
        title=second_doc.title,
        source_uri=second_doc.source_uri,
        metadata=second_doc.metadata,
    )
    assert first_title == second_title == "自然辩证法.pdf"
    assert first_source_id != second_source_id
    assert first_source_id.startswith("自然辩证法@sha256-")
    assert second_source_id.startswith("自然辩证法@sha256-")
    print("smoke_upload_content_hash_identity=ok")


def stamped_doc(path: Path, input_dir: Path) -> SourceDocument:
    docs = apply_uploaded_content_identity(
        [
            SourceDocument(
                tenant_id="team_a",
                doc_id="自然辩证法/page-1",
                doc_version=1,
                source_type="pdf",
                source_uri=str(path),
                title="自然辩证法 p1",
                text="测试正文",
                acl_groups=["engineering"],
                metadata={"relative_path": "自然辩证法.pdf", "page_no": 1},
            )
        ],
        path=path,
        input_dir=input_dir,
    )
    assert len(docs) == 1
    return docs[0]


if __name__ == "__main__":
    main()
