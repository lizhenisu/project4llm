from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceDocument:
    tenant_id: str
    doc_id: str
    doc_version: int
    source_type: str
    source_uri: str
    title: str
    text: str
    language: str = "zh"
    acl_groups: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageDocument:
    tenant_id: str
    doc_id: str
    doc_version: int
    source_uri: str
    title: str
    ocr_text: str = ""
    caption: str = ""
    language: str = "zh"
    acl_groups: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    tenant_id: str
    doc_id: str
    doc_version: int
    chunk_index: int
    source_type: str
    source_uri: str
    title: str
    text: str
    language: str
    acl_groups: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SearchHit:
    id: str
    score: float
    text: str
    doc_id: str
    title: str
    source_uri: str
    source_type: str
    chunk_index: int
    tenant_id: str
    acl_groups: list[str]
    metadata: dict[str, Any]
    rerank_score: float | None = None


@dataclass(frozen=True)
class TraceInfo:
    request_id: str
    original_query: str
    rewritten_query: str
    rewrite_backend: str
    tenant_id: str
    acl_groups: list[str]
    doc_version: int | None
    filter_expr: str
    retrieval_mode: str
    candidate_count: int
    reranked_count: int
    context_count: int
    dropped_by_score: int
    dropped_by_doc_limit: int
    dropped_by_budget: int


@dataclass(frozen=True)
class PackingStats:
    selected_count: int
    dropped_by_score: int
    dropped_by_doc_limit: int
    dropped_by_budget: int


@dataclass(frozen=True)
class RewriteResult:
    original_query: str
    rewritten_query: str
    backend: str
