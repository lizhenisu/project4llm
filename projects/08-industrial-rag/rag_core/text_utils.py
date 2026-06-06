from __future__ import annotations

import hashlib
import re
import time
from typing import Iterable

from rag_core.types import Chunk, SourceDocument


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def chunk_document(doc: SourceDocument, *, chunk_size: int, overlap: int) -> list[Chunk]:
    blocks = split_structural_blocks(doc.text)
    if not blocks:
        return []

    effective_chunk_size = max(1, chunk_size)
    chunks: list[Chunk] = []
    current_blocks: list[str] = []
    current_tokens = 0

    for block in blocks:
        block_tokens = len(tokenize(block))
        if block_tokens == 0:
            continue

        if block_tokens > effective_chunk_size:
            if current_blocks:
                chunks.append(make_chunk(doc, len(chunks), "\n\n".join(current_blocks)))
                current_blocks = []
                current_tokens = 0
            for body in split_large_block(block, chunk_size=effective_chunk_size, overlap=overlap):
                chunks.append(make_chunk(doc, len(chunks), body))
            continue

        if current_blocks and current_tokens + block_tokens > effective_chunk_size:
            chunks.append(make_chunk(doc, len(chunks), "\n\n".join(current_blocks)))
            current_blocks = []
            current_tokens = 0

        current_blocks.append(block)
        current_tokens += block_tokens

    if current_blocks:
        chunks.append(make_chunk(doc, len(chunks), "\n\n".join(current_blocks)))
    return chunks


def make_chunk(doc: SourceDocument, chunk_index: int, body: str) -> Chunk:
    chunk_text = f"标题路径: {doc.title}\n来源: {doc.source_type}\n正文:\n{body.strip()}"
    return Chunk(
        tenant_id=doc.tenant_id,
        doc_id=doc.doc_id,
        doc_version=doc.doc_version,
        chunk_index=chunk_index,
        source_type=doc.source_type,
        source_uri=doc.source_uri,
        title=doc.title,
        text=chunk_text,
        language=doc.language,
        acl_groups=doc.acl_groups,
        metadata=doc.metadata,
    )


def split_structural_blocks(text: str) -> list[str]:
    """Split text without breaking fenced code blocks or markdown tables."""
    blocks: list[str] = []
    paragraph: list[str] = []
    code_block: list[str] = []
    table_block: list[str] = []
    in_code = False

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            blocks.append(normalize_text(" ".join(paragraph)))
            paragraph = []

    def flush_table() -> None:
        nonlocal table_block
        if table_block:
            blocks.append("\n".join(table_block).strip())
            table_block = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            flush_table()
            code_block.append(line)
            if in_code:
                blocks.append("\n".join(code_block).strip())
                code_block = []
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_block.append(line)
            continue

        if is_markdown_table_line(stripped):
            flush_paragraph()
            table_block.append(line)
            continue

        flush_table()
        if not stripped:
            flush_paragraph()
            continue
        paragraph.append(stripped)

    if in_code and code_block:
        blocks.append("\n".join(code_block).strip())
    flush_table()
    flush_paragraph()
    return [block for block in blocks if block.strip()]


def is_markdown_table_line(line: str) -> bool:
    if "|" not in line:
        return False
    stripped = line.strip()
    if stripped.startswith("|") and stripped.endswith("|"):
        return True
    return bool(re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", stripped))


def split_large_block(block: str, *, chunk_size: int, overlap: int) -> list[str]:
    token_spans = list(TOKEN_PATTERN.finditer(block.lower()))
    if not token_spans:
        return []
    stride = max(1, chunk_size - overlap)
    chunks: list[str] = []
    for start in range(0, len(token_spans), stride):
        end = min(start + chunk_size, len(token_spans))
        if start >= end:
            continue
        char_start = 0 if start == 0 else token_spans[start].start()
        char_end = token_spans[end].start() if end < len(token_spans) else len(block)
        chunk = block[char_start:char_end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_id(chunk: Chunk) -> str:
    raw = (
        f"{chunk.tenant_id}:{chunk.doc_id}:"
        f"{chunk.doc_version}:{chunk.chunk_index}"
    )
    return stable_hash(raw, length=32)


def lexical_overlap_score(query: str, text: str) -> float:
    query_terms = set(tokenize(query))
    if not query_terms:
        return 0.0
    text_terms = set(tokenize(text))
    return len(query_terms & text_terms) / len(query_terms)


def now_ms() -> int:
    return int(time.time() * 1000)


def reciprocal_rank_fusion(rank_lists: Iterable[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for rank_list in rank_lists:
        for rank, item_id in enumerate(rank_list, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores
