from __future__ import annotations

from rag_core.text_utils import chunk_document
from rag_core.types import SourceDocument


def main() -> None:
    doc = SourceDocument(
        tenant_id="team_a",
        doc_id="chunk-structure",
        doc_version=1,
        source_type="md",
        source_uri="memory://chunk-structure",
        title="Chunk Structure",
        text="""
第一段说明 retrieval pipeline 会先 parse 再 chunk。

| sku | priority | owner |
| --- | --- | --- |
| sku_neon_42 | gold | ops_table_team |

第二段说明代码块应该保持完整。

```python
def build_filter(tenant_id):
    return f'tenant_id == "{tenant_id}"'
```
""",
        acl_groups=["ops"],
    )

    chunks = chunk_document(doc, chunk_size=24, overlap=4)
    texts = [chunk.text for chunk in chunks]

    table_chunks = [
        text
        for text in texts
        if "| sku | priority | owner |" in text
        or "sku_neon_42" in text
        or "ops_table_team" in text
    ]
    assert len(table_chunks) == 1
    assert "| sku | priority | owner |" in table_chunks[0]
    assert "| sku_neon_42 | gold | ops_table_team |" in table_chunks[0]

    code_chunks = [text for text in texts if "build_filter" in text or "tenant_id" in text]
    assert len(code_chunks) == 1
    assert "```python" in code_chunks[0]
    assert "return f'tenant_id ==" in code_chunks[0]
    assert code_chunks[0].rstrip().endswith("```")

    assert all("标题路径: Chunk Structure" in text for text in texts)
    assert all("来源: md" in text for text in texts)
    print("smoke_chunk_structure=ok")


if __name__ == "__main__":
    main()
