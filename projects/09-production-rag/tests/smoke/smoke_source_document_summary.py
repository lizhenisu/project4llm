from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

from rag_core.config import RagConfig, load_config
from rag_core.object_store import archive_source_documents
from rag_core.sources import (
    dedupe_source_documents,
    next_source_doc_version,
    pdf_image_chunks,
    source_document_display_blocks,
    summarize_ingested_sources,
)
from rag_core.text_utils import chunk_document
from rag_core.types import Chunk, SourceDocument


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
        image_path = config.object_store_dir / "uploads" / "team_a" / "upload-1" / "paper.assets" / "page-1-image-1.png"
        image_path.parent.mkdir(parents=True)
        image_path.write_bytes(b"fake-png")
        image_doc = replace(
            docs[0],
            metadata={
                **docs[0].metadata,
                "display_blocks": [
                    {
                        "type": "image",
                        "title": "Figure 1",
                        "path": str(image_path),
                        "media_type": "image/png",
                    }
                ],
            },
        )
        blocks = source_document_display_blocks(config=config, tenant_id="team_a", docs=[image_doc])
        assert blocks[-1]["url"].startswith("/source-assets/uploads/team_a/upload-1/paper.assets/page-1-image-1.png?")
        assert "data:image/" not in str(blocks[-1])
        image_chunks = pdf_image_chunks([image_doc])
        assert len(image_chunks) == 1
        assert image_chunks[0].source_type == "image"
        assert image_chunks[0].doc_id == "自然辩证法/page-1/image-1"
        assert image_chunks[0].source_uri == str(image_path)
        assert image_chunks[0].metadata["linked_doc_id"] == "自然辩证法/page-1"
        assert image_chunks[0].metadata["linked_source_type"] == "pdf"
        assert image_chunks[0].metadata["linked_source_uri"] == "/tmp/自然辩证法.pdf"
        assert image_chunks[0].metadata["derived_from_pdf_image"] is True
        assert image_chunks[0].metadata["display_blocks"][0]["path"] == str(image_path)
        derived_image_doc = source_document_from_chunk(image_chunks[0])
        summary_with_image = summarize_ingested_sources(
            [derived_image_doc, image_doc],
            [*chunk_document(image_doc, chunk_size=100, overlap=0), image_chunks[0]],
        )[0]
        assert summary_with_image.source_type == "pdf"
        assert summary_with_image.source_uri == "/tmp/自然辩证法.pdf"
        assert "自然辩证法/page-1/image-1" in summary_with_image.child_doc_ids

    print("source document summary smoke passed")


def make_config(base_dir: Path) -> RagConfig:
    return replace(
        load_config(),
        milvus_uri=str(base_dir / "milvus.db"),
        collection_name="test_collection",
        runtime_dir=base_dir / "runtime",
        object_store_dir=base_dir / "object_store",
    )


def source_document_from_chunk(chunk: Chunk) -> SourceDocument:
    return SourceDocument(
        tenant_id=chunk.tenant_id,
        doc_id=chunk.doc_id,
        doc_version=chunk.doc_version,
        source_type=chunk.source_type,
        source_uri=chunk.source_uri,
        title=chunk.title,
        text=chunk.text,
        language=chunk.language,
        acl_groups=chunk.acl_groups,
        metadata=chunk.metadata,
    )


if __name__ == "__main__":
    main()
