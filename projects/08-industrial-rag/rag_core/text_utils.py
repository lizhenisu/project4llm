from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
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
    normalized = normalize_text(doc.text)
    tokens = tokenize(normalized)
    if not tokens:
        return []

    stride = max(1, chunk_size - overlap)
    chunks: list[Chunk] = []
    for chunk_index, start in enumerate(range(0, len(tokens), stride)):
        token_slice = tokens[start : start + chunk_size]
        if not token_slice:
            continue
        body = "".join(token_slice)
        chunk_text = f"标题路径: {doc.title}\n来源: {doc.source_type}\n正文:\n{body}"
        chunks.append(
            Chunk(
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
        )
    return chunks


def chunk_id(chunk: Chunk) -> str:
    raw = (
        f"{chunk.tenant_id}:{chunk.doc_id}:"
        f"{chunk.doc_version}:{chunk.chunk_index}"
    )
    return stable_hash(raw, length=32)


def hash_dense_embedding(text: str, dim: int) -> list[float]:
    vector = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def sparse_embedding(text: str, *, vocab_size: int = 100_000) -> dict[int, float]:
    counts = Counter(tokenize(text))
    if not counts:
        return {}

    max_count = max(counts.values())
    sparse: dict[int, float] = {}
    for token, count in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % vocab_size
        sparse[bucket] = sparse.get(bucket, 0.0) + count / max_count
    return sparse


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

