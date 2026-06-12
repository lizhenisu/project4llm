from __future__ import annotations

from rag_core.sources import summarize_ingested_sources
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
    print("source document summary smoke passed")


if __name__ == "__main__":
    main()
