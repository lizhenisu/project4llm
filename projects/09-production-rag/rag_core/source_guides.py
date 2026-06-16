from __future__ import annotations

from pathlib import Path

from rag_core.config import RagConfig
from rag_core.io import read_jsonl, write_jsonl
from rag_core.prompts import SOURCE_GUIDE_SYSTEM_PROMPT
from rag_core.prompts import build_source_guide_prompt as prompt_source_guide
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
    doc_title: str,
    docs: list[SourceDocument],
) -> SourceGuideResult:
    cached = load_source_guide_full(
        config.object_store_dir,
        tenant_id=tenant_id,
        source_doc_id=source_doc_id,
        doc_version=doc_version,
    )
    if cached:
        return cached
    result = generate_source_guide(config=config, title=doc_title, docs=docs)
    save_source_guide(
        config.object_store_dir,
        tenant_id=tenant_id,
        source_doc_id=source_doc_id,
        doc_version=doc_version,
        title=result.title,
        guide=result.guide,
        model=config.llm_model,
    )
    return result


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
                "content": SOURCE_GUIDE_SYSTEM_PROMPT,
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
    title_text, guide_text = parse_title_and_guide(guide)
    if not title_text:
        title_text = title
    return SourceGuideResult(title=normalize_guide_text(title_text), guide=normalize_guide_text(guide_text))


def parse_title_and_guide(raw: str) -> tuple[str, str]:
    for delimiter in ("\n\n", "\n"):
        parts = raw.split(delimiter, 1)
        if len(parts) == 2:
            first = parts[0].strip()
            second = parts[1].strip()
            if first and second and len(first) < 120:
                return first, second
    if len(raw) < 120:
        return raw, raw
    return "", raw


def build_source_guide_prompt(*, title: str, source_text: str) -> str:
    return prompt_source_guide(title=title, source_text=source_text)


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


def load_source_guide_full(
    object_store_dir: Path,
    *,
    tenant_id: str,
    source_doc_id: str,
    doc_version: int,
) -> SourceGuideResult | None:
    path = object_store_dir / SOURCE_GUIDES_PATH
    if not path.exists():
        return None
    for row in read_jsonl(path):
        if (
            str(row.get("tenant_id")) == tenant_id
            and str(row.get("source_doc_id")) == source_doc_id
            and int(row.get("doc_version", 0)) == int(doc_version)
        ):
            title = str(row.get("title") or "").strip()
            guide = str(row.get("guide") or "").strip()
            if guide:
                return SourceGuideResult(title=title or guide, guide=guide)
    return None

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
            title = str(row.get("title") or "").strip()
            guide = str(row.get("guide") or "").strip()
            if title and guide:
                return f"标题: {title}\n{guide}"
            return guide or None
    return None


def load_source_guides_for_rewrite(
    object_store_dir: Path,
    *,
    tenant_id: str,
    doc_ids: list[str] | None = None,
    doc_version: int | None = None,
    current_doc_versions: dict[str, int] | None = None,
    limit: int = 20,
) -> list[str]:
    path = object_store_dir / SOURCE_GUIDES_PATH
    if not path.exists():
        return []
    allowed_doc_ids = set(doc_ids or [])
    rows = read_jsonl(path)
    matched = source_guide_rows_for_rewrite(
        rows,
        tenant_id=tenant_id,
        allowed_doc_ids=allowed_doc_ids,
        doc_version=doc_version,
        current_doc_versions=current_doc_versions,
        limit=limit,
    )
    if not matched and allowed_doc_ids:
        matched = source_guide_rows_for_rewrite(
            rows,
            tenant_id=tenant_id,
            allowed_doc_ids=set(),
            doc_version=doc_version,
            current_doc_versions=current_doc_versions,
            limit=limit,
        )
    return matched


def source_guide_rows_for_rewrite(
    source_rows: list[dict],
    *,
    tenant_id: str,
    allowed_doc_ids: set[str],
    doc_version: int | None,
    current_doc_versions: dict[str, int] | None,
    limit: int,
) -> list[str]:
    rows: list[str] = []
    for row in source_rows:
        source_doc_id = str(row.get("source_doc_id") or "")
        if str(row.get("tenant_id")) != tenant_id:
            continue
        if allowed_doc_ids and source_doc_id not in allowed_doc_ids:
            continue
        row_version = int(row.get("doc_version", 0))
        if doc_version is not None:
            if row_version != int(doc_version):
                continue
        elif current_doc_versions is not None and current_doc_versions.get(source_doc_id) != row_version:
            continue
        title = str(row.get("title") or "").strip()
        guide = str(row.get("guide") or "").strip()
        if not guide:
            continue
        rows.append(f"标题: {title}\n摘要: {guide}" if title else f"摘要: {guide}")
        if len(rows) >= limit:
            break
    return rows


def save_source_guide(
    object_store_dir: Path,
    *,
    tenant_id: str,
    source_doc_id: str,
    doc_version: int,
    title: str,
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
        "title": title,
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
from dataclasses import dataclass

@dataclass
class SourceGuideResult:
    title: str
    guide: str
