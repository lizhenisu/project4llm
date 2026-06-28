from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

from rag_core.jsonl_store import object_exists, read_object_jsonl, write_object_jsonl
from rag_core.text_utils import now_ms
from rag_core.types import SourceDocument


SOURCE_SECTION_SUMMARIES_PATH = Path("canonical/source_section_summaries.jsonl")
DEFAULT_SECTION_SUMMARY_CHARS = 1_800
DEFAULT_SECTION_SUMMARY_LIMIT = 80
SOURCE_SECTION_SUMMARIES_LOCK = threading.Lock()


@dataclass(frozen=True)
class SourceSectionSummary:
    tenant_id: str
    source_doc_id: str
    doc_version: int
    section_index: int
    title: str
    summary: str


def save_source_section_summaries(
    object_store_dir: Path,
    *,
    tenant_id: str,
    source_doc_id: str,
    doc_version: int,
    docs: list[SourceDocument],
) -> int:
    sections = build_source_section_summaries(
        tenant_id=tenant_id,
        source_doc_id=source_doc_id,
        doc_version=doc_version,
        docs=docs,
    )
    with SOURCE_SECTION_SUMMARIES_LOCK:
        rows = read_object_jsonl(object_store_dir, SOURCE_SECTION_SUMMARIES_PATH)
        rows = [
            row
            for row in rows
            if not section_summary_matches_source(
                row,
                tenant_id=tenant_id,
                source_doc_id=source_doc_id,
                doc_version=doc_version,
            )
        ]
        rows.extend(
            {
                "tenant_id": section.tenant_id,
                "source_doc_id": section.source_doc_id,
                "doc_version": section.doc_version,
                "section_index": section.section_index,
                "title": section.title,
                "summary": section.summary,
                "method": "deterministic-extractive",
                "updated_at": now_ms(),
            }
            for section in sections
        )
        write_object_jsonl(object_store_dir, SOURCE_SECTION_SUMMARIES_PATH, rows)
    return len(sections)


def load_source_section_summaries(
    object_store_dir: Path,
    *,
    tenant_id: str,
    source_keys: set[tuple[str, int]],
) -> list[SourceSectionSummary]:
    if not source_keys or not object_exists(object_store_dir, SOURCE_SECTION_SUMMARIES_PATH):
        return []
    sections: list[SourceSectionSummary] = []
    for row in read_object_jsonl(object_store_dir, SOURCE_SECTION_SUMMARIES_PATH):
        source_doc_id = str(row.get("source_doc_id") or "")
        doc_version = int(row.get("doc_version", 0))
        if str(row.get("tenant_id")) != tenant_id or (source_doc_id, doc_version) not in source_keys:
            continue
        summary = str(row.get("summary") or "").strip()
        if not summary:
            continue
        sections.append(
            SourceSectionSummary(
                tenant_id=tenant_id,
                source_doc_id=source_doc_id,
                doc_version=doc_version,
                section_index=int(row.get("section_index", 0)),
                title=str(row.get("title") or f"章节 {len(sections) + 1}"),
                summary=summary,
            )
        )
    return sorted(sections, key=lambda section: (section.source_doc_id, section.section_index))


def delete_source_section_summaries(
    object_store_dir: Path,
    *,
    tenant_id: str,
    source_doc_ids: set[str],
    doc_version: int | None = None,
) -> int:
    if not source_doc_ids or not object_exists(object_store_dir, SOURCE_SECTION_SUMMARIES_PATH):
        return 0
    with SOURCE_SECTION_SUMMARIES_LOCK:
        rows = read_object_jsonl(object_store_dir, SOURCE_SECTION_SUMMARIES_PATH)
        remaining = [
            row
            for row in rows
            if not (
                str(row.get("tenant_id")) == tenant_id
                and str(row.get("source_doc_id")) in source_doc_ids
                and (doc_version is None or int(row.get("doc_version", 0)) == int(doc_version))
            )
        ]
        removed = len(rows) - len(remaining)
        if removed:
            write_object_jsonl(object_store_dir, SOURCE_SECTION_SUMMARIES_PATH, remaining)
        return removed


def build_source_section_summaries(
    *,
    tenant_id: str,
    source_doc_id: str,
    doc_version: int,
    docs: list[SourceDocument],
) -> list[SourceSectionSummary]:
    sections: list[SourceSectionSummary] = []
    for doc in docs:
        text = normalize_section_text(doc.text)
        if not text:
            continue
        page_no = doc.metadata.get("page_no")
        base_title = f"第 {page_no} 页" if page_no is not None else (doc.title or source_doc_id)
        for part_index, part in enumerate(split_section_text(text), start=1):
            title = base_title if part_index == 1 else f"{base_title}（续 {part_index}）"
            sections.append(
                SourceSectionSummary(
                    tenant_id=tenant_id,
                    source_doc_id=source_doc_id,
                    doc_version=int(doc_version),
                    section_index=len(sections),
                    title=title,
                    summary=extractive_section_summary(part),
                )
            )
            if len(sections) >= section_summary_limit():
                return sections
    return sections


def split_section_text(text: str) -> list[str]:
    chunk_chars = max(section_summary_chars() * 2, section_summary_chars())
    return [text[start : start + chunk_chars] for start in range(0, len(text), chunk_chars)]


def extractive_section_summary(text: str) -> str:
    limit = section_summary_chars()
    if len(text) <= limit:
        return text
    head_chars = max(1, int(limit * 0.75))
    tail_chars = max(1, limit - head_chars - 1)
    return f"{text[:head_chars].rstrip()}…{text[-tail_chars:].lstrip()}"


def normalize_section_text(text: str) -> str:
    return " ".join(part.strip() for part in str(text or "").splitlines() if part.strip())


def section_summary_chars() -> int:
    return env_positive_int("RAG_SECTION_SUMMARY_CHARS", DEFAULT_SECTION_SUMMARY_CHARS)


def section_summary_limit() -> int:
    return env_positive_int("RAG_SECTION_SUMMARY_LIMIT", DEFAULT_SECTION_SUMMARY_LIMIT)


def env_positive_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def section_summary_matches_source(
    row: dict,
    *,
    tenant_id: str,
    source_doc_id: str,
    doc_version: int,
) -> bool:
    return (
        str(row.get("tenant_id")) == tenant_id
        and str(row.get("source_doc_id")) == source_doc_id
        and int(row.get("doc_version", 0)) == int(doc_version)
    )
