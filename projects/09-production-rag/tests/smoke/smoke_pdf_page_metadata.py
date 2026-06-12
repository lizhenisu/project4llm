from __future__ import annotations

import tempfile
from pathlib import Path

import rag_core.io as io
from rag_core.answering import build_prompt
from rag_core.types import SearchHit


def main() -> None:
    old_extract_pdf_pages = io.extract_pdf_pages
    io.extract_pdf_pages = fake_extract_pdf_pages
    try:
        run_smoke()
    finally:
        io.extract_pdf_pages = old_extract_pdf_pages


def run_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pdf_path = root / "manual.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n% teaching smoke placeholder\n")

        docs = io.load_file_documents(
            root,
            tenant_id="team_a",
            doc_version=4,
            acl_groups=["ops"],
        )

    assert len(docs) == 2
    assert [doc.doc_id for doc in docs] == ["manual/page-1", "manual/page-2"]
    assert [doc.title for doc in docs] == ["manual p1", "manual p2"]
    assert docs[0].source_type == "pdf"
    assert docs[0].metadata["page_no"] == 1
    assert docs[0].metadata["page_start"] == 1
    assert docs[0].metadata["page_end"] == 1
    assert docs[0].metadata["page_count"] == 2
    assert docs[1].metadata["page_no"] == 2

    prompt = build_prompt(
        "退款 SLA 在哪一页？",
        [
            SearchHit(
                id="hit-1",
                score=1.0,
                text="标题路径: manual p2\n来源: pdf\n正文:\n退款 SLA 是 15 分钟。",
                doc_id=docs[1].doc_id,
                title=docs[1].title,
                source_uri=docs[1].source_uri,
                source_type=docs[1].source_type,
                chunk_index=0,
                tenant_id=docs[1].tenant_id,
                acl_groups=docs[1].acl_groups,
                metadata=docs[1].metadata,
            )
        ],
    )
    assert "doc_id=manual/page-2" in prompt
    assert "page=2" in prompt
    print("smoke_pdf_page_metadata=ok")


def fake_extract_pdf_pages(path: Path) -> list[tuple[int, str]]:
    assert path.name == "manual.pdf"
    return [
        (1, "第一页是目录。"),
        (2, "第二页包含退款 SLA 是 15 分钟。"),
    ]


if __name__ == "__main__":
    main()
