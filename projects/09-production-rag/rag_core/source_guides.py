from __future__ import annotations

from pathlib import Path

from rag_core.config import RagConfig
from rag_core.io import read_jsonl, write_jsonl
from rag_core.text_utils import now_ms
from rag_core.types import SourceDocument


SOURCE_GUIDES_PATH = Path("canonical/source_guides.jsonl")
SOURCE_GUIDE_MAX_CHARS = 9000


def get_or_create_source_guide(
    *,
    config: RagConfig,
    tenant_id: str,
    source_doc_id: str,
    doc_version: int,
    title: str,
    docs: list[SourceDocument],
) -> str:
    cached = load_source_guide(
        config.object_store_dir,
        tenant_id=tenant_id,
        source_doc_id=source_doc_id,
        doc_version=doc_version,
    )
    if cached:
        return cached
    guide = generate_source_guide(config=config, title=title, docs=docs)
    save_source_guide(
        config.object_store_dir,
        tenant_id=tenant_id,
        source_doc_id=source_doc_id,
        doc_version=doc_version,
        guide=guide,
        model=config.llm_model,
    )
    return guide


def generate_source_guide(*, config: RagConfig, title: str, docs: list[SourceDocument]) -> str:
    if not config.llm_base_url or not config.llm_api_key:
        raise RuntimeError("NEW_API_URL/NEW_API_KEY must be configured for source guide generation.")
    source_text = build_source_guide_context(docs)
    if not source_text:
        return "该来源暂无可总结的解析正文。"

    from openai import OpenAI

    client = OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)
    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {
                "role": "system",
                "content": "你是企业知识库来源指南助手。你只依据给定原文，为读者生成简洁准确的中文摘要。",
            },
            {
                "role": "user",
                "content": build_source_guide_prompt(title=title, source_text=source_text),
            },
        ],
        temperature=0.2,
    )
    guide = (response.choices[0].message.content or "").strip()
    if not guide:
        raise RuntimeError("LLM source guide generation returned empty content.")
    return normalize_guide_text(guide)


def build_source_guide_prompt(*, title: str, source_text: str) -> str:
    return f"""请为下面这个知识库来源生成“来源指南”摘要。

要求:
- 只依据原文，不编造。
- 输出 2-4 句中文自然语言摘要。
- 说明这份资料主要讲什么、包含哪些关键信息、适合用来回答什么类型的问题。
- 不要输出标题、Markdown、编号列表或引用标记。
- 不要直接复制原文长句。

来源标题: {title}

原文:
{source_text}
"""


def build_source_guide_context(docs: list[SourceDocument]) -> str:
    blocks: list[str] = []
    total = 0
    for doc in docs:
        text = doc.text.strip()
        if not text:
            continue
        page_no = doc.metadata.get("page_no")
        header = f"[第 {page_no} 页]" if page_no is not None else f"[{doc.title}]"
        block = f"{header}\n{text}"
        remaining = SOURCE_GUIDE_MAX_CHARS - total
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = block[:remaining]
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks).strip()


def normalize_guide_text(text: str) -> str:
    normalized = " ".join(line.strip() for line in text.splitlines() if line.strip())
    return normalized.strip().strip('"').strip("'")


def load_source_guide(
    object_store_dir: Path,
    *,
    tenant_id: str,
    source_doc_id: str,
    doc_version: int,
) -> str | None:
    path = object_store_dir / SOURCE_GUIDES_PATH
    if not path.exists():
        return None
    for row in read_jsonl(path):
        if (
            str(row.get("tenant_id")) == tenant_id
            and str(row.get("source_doc_id")) == source_doc_id
            and int(row.get("doc_version", 0)) == int(doc_version)
        ):
            guide = str(row.get("guide") or "").strip()
            return guide or None
    return None


def save_source_guide(
    object_store_dir: Path,
    *,
    tenant_id: str,
    source_doc_id: str,
    doc_version: int,
    guide: str,
    model: str,
) -> None:
    path = object_store_dir / SOURCE_GUIDES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(path) if path.exists() else []
    row = {
        "tenant_id": tenant_id,
        "source_doc_id": source_doc_id,
        "doc_version": int(doc_version),
        "guide": guide,
        "model": model,
        "updated_at": now_ms(),
    }
    merged = {
        source_guide_key(item): item
        for item in [*rows, row]
    }
    write_jsonl(path, merged.values())


def source_guide_key(row: dict) -> tuple[str, str, int]:
    return (
        str(row["tenant_id"]),
        str(row["source_doc_id"]),
        int(row["doc_version"]),
    )
