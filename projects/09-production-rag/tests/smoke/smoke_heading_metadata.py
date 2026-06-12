from __future__ import annotations

import tempfile
from pathlib import Path

from rag_core.io import load_file_documents
from rag_core.text_utils import chunk_document


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "handbook.md").write_text(
            "\n".join(
                [
                    "# 产品手册",
                    "",
                    "总览说明。",
                    "",
                    "## 退款规则",
                    "",
                    "退款需要订单号和付款凭证。",
                    "",
                    "### SLA",
                    "",
                    "黄金用户退款 SLA 是 15 分钟。",
                ]
            ),
            encoding="utf-8",
        )
        (root / "runbook.html").write_text(
            """
            <html>
              <head><title>Ops Runbook</title></head>
              <body>
                <nav>导航文本也会被 parser 看到，但 heading path 应保留。</nav>
                <h1>RAG 运维</h1>
                <h2>延迟排障</h2>
                <p>先检查 embedding batch，再检查 Milvus。</p>
              </body>
            </html>
            """,
            encoding="utf-8",
        )

        docs = load_file_documents(
            root,
            tenant_id="team_a",
            doc_version=2,
            acl_groups=["ops"],
        )

    md_docs = [doc for doc in docs if doc.source_type == "md"]
    assert [doc.doc_id for doc in md_docs] == [
        "handbook/section-000",
        "handbook/section-001",
        "handbook/section-002",
    ]
    assert md_docs[1].metadata["heading_path"] == ["产品手册", "退款规则"]
    assert md_docs[2].metadata["heading_path"] == ["产品手册", "退款规则", "SLA"]
    assert md_docs[2].title == "产品手册 > 退款规则 > SLA"

    html_doc = next(doc for doc in docs if doc.source_type == "html")
    assert html_doc.title == "Ops Runbook"
    assert html_doc.metadata["heading_path"] == ["RAG 运维", "延迟排障"]
    assert "RAG 运维 > 延迟排障" in html_doc.text
    assert "导航文本" not in html_doc.text

    chunks = chunk_document(md_docs[2], chunk_size=32, overlap=4)
    assert chunks
    assert "标题路径: 产品手册 > 退款规则 > SLA" in chunks[0].text
    print("smoke_heading_metadata=ok")


if __name__ == "__main__":
    main()
