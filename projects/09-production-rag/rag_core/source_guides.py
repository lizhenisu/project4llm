from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from rag_core.config import RagConfig
from rag_core.jsonl_store import object_exists, read_object_jsonl, write_object_jsonl
from rag_core.model_api_retry import call_model_api_with_retries
from rag_core.prompts import SOURCE_GUIDE_SYSTEM_PROMPT
from rag_core.prompts import build_source_guide_prompt as prompt_source_guide
from rag_core.text_utils import now_ms
from rag_core.types import SourceDocument


SOURCE_GUIDES_PATH = Path("canonical/source_guides.jsonl")
DEFAULT_SOURCE_GUIDE_CHUNK_CHARS = 230_000
DEFAULT_SOURCE_GUIDE_LLM_WORKERS = 3
SOURCE_GUIDES_LOCK = threading.Lock()


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


def generate_source_guide(*, config: RagConfig, title: str, docs: list[SourceDocument]) -> SourceGuideResult:
    if not config.llm_base_url or not config.llm_api_key:
        raise RuntimeError("NEW_API_URL/NEW_API_KEY must be configured for source guide generation.")
    source_chunks = build_source_guide_context_chunks(docs)
    if not source_chunks:
        guide = "该来源暂无可总结的解析正文。"
        return SourceGuideResult(title=title or guide, guide=guide)

    partial_guides = parallel_map_ordered(
        [
            lambda source_text=source_text: generate_source_guide_text(
                config=config,
                title=title,
                source_text=source_text,
            )
            for source_text in source_chunks
        ],
        workers=source_guide_llm_workers(),
    )
    if len(partial_guides) == 1:
        guide = partial_guides[0]
    else:
        merged_source_text = "\n\n".join(
            f"[摘要片段 {index}]\n{guide}"
            for index, guide in enumerate(partial_guides, start=1)
        )
        guide = generate_source_guide_text(
            config=config,
            title=title,
            source_text=merged_source_text,
        )
    title_text, guide_text = parse_title_and_guide(guide)
    if not title_text:
        title_text = title
    return SourceGuideResult(title=normalize_guide_text(title_text), guide=normalize_guide_text(guide_text))


def generate_source_guide_text(*, config: RagConfig, title: str, source_text: str) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)
    response = call_model_api_with_retries(
        "source_guide_generation",
        lambda: client.chat.completions.create(
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
        ),
    )
    guide = (response.choices[0].message.content or "").strip()
    if not guide:
        raise RuntimeError("LLM source guide generation returned empty content.")
    return guide


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
    return "\n\n".join(build_source_guide_context_chunks(docs)).strip()


def build_source_guide_context_chunks(docs: list[SourceDocument]) -> list[str]:
    blocks: list[str] = []
    for doc in docs:
        text = doc.text.strip()
        if not text:
            continue
        page_no = doc.metadata.get("page_no")
        header = f"[第 {page_no} 页]" if page_no is not None else f"[{doc.title}]"
        blocks.append(f"{header}\n{text}")
    return split_source_guide_text("\n\n".join(blocks).strip(), chunk_chars=source_guide_chunk_chars())


def split_source_guide_text(text: str, *, chunk_chars: int) -> list[str]:
    chunk_chars = max(1, chunk_chars)
    paragraphs = [paragraph.strip() for paragraph in text.splitlines() if paragraph.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        if len(paragraph) > chunk_chars:
            if current:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            chunks.extend(paragraph[start : start + chunk_chars] for start in range(0, len(paragraph), chunk_chars))
            continue
        if current and current_len + len(paragraph) > chunk_chars:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += len(paragraph)
    if current:
        chunks.append("\n".join(current))
    return chunks


def parallel_map_ordered(tasks: list, *, workers: int) -> list:
    if not tasks:
        return []
    worker_count = min(len(tasks), max(1, workers))
    if worker_count <= 1:
        return [task() for task in tasks]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(lambda task: task(), tasks))


def source_guide_chunk_chars() -> int:
    return env_int("RAG_SOURCE_GUIDE_CHUNK_CHARS", DEFAULT_SOURCE_GUIDE_CHUNK_CHARS)


def source_guide_llm_workers() -> int:
    return env_int("RAG_SOURCE_GUIDE_LLM_WORKERS", DEFAULT_SOURCE_GUIDE_LLM_WORKERS)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        return max(1, int(value))
    except ValueError:
        return default


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
    if not object_exists(object_store_dir, SOURCE_GUIDES_PATH):
        return None
    for row in read_object_jsonl(object_store_dir, SOURCE_GUIDES_PATH):
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
    if not object_exists(object_store_dir, SOURCE_GUIDES_PATH):
        return None
    for row in read_object_jsonl(object_store_dir, SOURCE_GUIDES_PATH):
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
    if not object_exists(object_store_dir, SOURCE_GUIDES_PATH):
        return []
    allowed_doc_ids = set(doc_ids or [])
    rows = read_object_jsonl(object_store_dir, SOURCE_GUIDES_PATH)
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
        if allowed_doc_ids and not source_guide_matches_allowed_doc_ids(source_doc_id, allowed_doc_ids):
            continue
        row_version = int(row.get("doc_version", 0))
        if doc_version is not None:
            if row_version != int(doc_version):
                continue
        elif current_doc_versions is not None:
            current_version = current_source_guide_version(
                source_doc_id,
                current_doc_versions=current_doc_versions,
            )
            if current_version != row_version:
                continue
        title = str(row.get("title") or "").strip()
        guide = str(row.get("guide") or "").strip()
        if not guide:
            continue
        rows.append(f"标题: {title}\n摘要: {guide}" if title else f"摘要: {guide}")
        if len(rows) >= limit:
            break
    return rows


def source_guide_matches_allowed_doc_ids(source_doc_id: str, allowed_doc_ids: set[str]) -> bool:
    if source_doc_id in allowed_doc_ids:
        return True
    child_prefix = f"{source_doc_id}/"
    return any(doc_id.startswith(child_prefix) for doc_id in allowed_doc_ids)


def current_source_guide_version(
    source_doc_id: str,
    *,
    current_doc_versions: dict[str, int],
) -> int | None:
    exact = current_doc_versions.get(source_doc_id)
    if exact is not None:
        return int(exact)
    child_prefix = f"{source_doc_id}/"
    child_versions = [
        int(version)
        for doc_id, version in current_doc_versions.items()
        if str(doc_id).startswith(child_prefix)
    ]
    if not child_versions:
        return None
    return max(child_versions)


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
    with SOURCE_GUIDES_LOCK:
        rows = read_object_jsonl(object_store_dir, SOURCE_GUIDES_PATH)
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
        write_object_jsonl(object_store_dir, SOURCE_GUIDES_PATH, merged.values())


def delete_source_guides(
    object_store_dir: Path,
    *,
    tenant_id: str,
    source_doc_ids: set[str],
    doc_version: int | None = None,
) -> int:
    if not object_exists(object_store_dir, SOURCE_GUIDES_PATH) or not source_doc_ids:
        return 0
    with SOURCE_GUIDES_LOCK:
        rows = read_object_jsonl(object_store_dir, SOURCE_GUIDES_PATH)
        remaining = [
            row
            for row in rows
            if not source_guide_matches_delete(
                row,
                tenant_id=tenant_id,
                source_doc_ids=source_doc_ids,
                doc_version=doc_version,
            )
        ]
        removed = len(rows) - len(remaining)
        if removed:
            write_object_jsonl(object_store_dir, SOURCE_GUIDES_PATH, remaining)
        return removed


def source_guide_matches_delete(
    row: dict,
    *,
    tenant_id: str,
    source_doc_ids: set[str],
    doc_version: int | None,
) -> bool:
    if str(row.get("tenant_id")) != tenant_id:
        return False
    if str(row.get("source_doc_id")) not in source_doc_ids:
        return False
    return doc_version is None or int(row.get("doc_version", 0)) == int(doc_version)


def source_guide_key(row: dict) -> tuple[str, str, int]:
    return (
        str(row["tenant_id"]),
        str(row["source_doc_id"]),
        int(row["doc_version"]),
    )


@dataclass
class SourceGuideResult:
    title: str
    guide: str
