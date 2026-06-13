from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

from rag_core.config import RagConfig, load_config
from rag_core.object_store import archive_source_documents
from rag_core.sources import dedupe_source_documents, next_source_doc_version, summarize_ingested_sources
from rag_core.text_utils import chunk_document
from rag_core.types import SourceDocument


def main() -> None:
    docs = [
        SourceDocument(
            tenant_id="team_a",
            doc_id=f"自然辩证法/page-{page_no}",
            doc_version=1,
            source_type="pdf",
            source_uri="/tmp/自然辩证法.pdf",
            title=f"自然辩证法 p{page_no}",
            text=f"第 {page_no} 页正文",
            acl_groups=["engineering"],
            metadata={"relative_path": "自然辩证法.pdf", "page_no": page_no},
        )
        for page_no in (1, 2)
    ]
    chunks = [
        chunk
        for doc in docs
        for chunk in chunk_document(doc, chunk_size=100, overlap=0)
    ]

    summaries = summarize_ingested_sources(docs, chunks)

    assert len(summaries) == 1
    source = summaries[0]
    assert source.doc_id == "自然辩证法"
    assert source.title == "自然辩证法.pdf"
    assert source.chunk_count == 2
    assert source.child_doc_ids == ["自然辩证法/page-1", "自然辩证法/page-2"]
    assert dedupe_source_documents([docs[0], replace(docs[0], source_uri="/tmp/reupload/自然辩证法.pdf", text="新正文")])[
        0
    ].text == "新正文"

    with tempfile.TemporaryDirectory() as tmp:
        config = make_config(Path(tmp))
        archive_source_documents(config.object_store_dir, docs)
        next_docs = [
            replace(doc, doc_version=1, source_uri=f"/tmp/new-upload/{doc.title.replace(' ', '_')}.pdf")
            for doc in docs
        ]
        assert next_source_doc_version(config, next_docs) == 2

    print("source document summary smoke passed")


def make_config(base_dir: Path) -> RagConfig:
    return replace(
        load_config(),
        milvus_uri=str(base_dir / "milvus.db"),
        collection_name="test_collection",
        runtime_dir=base_dir / "runtime",
        object_store_dir=base_dir / "object_store",
    )


if __name__ == "__main__":
    main()
