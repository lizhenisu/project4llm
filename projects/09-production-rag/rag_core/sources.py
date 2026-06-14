from __future__ import annotations

import shutil
import json
import uuid
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, BinaryIO

from rag_core.config import RagConfig
from rag_core.database import connect_metadata_db
from rag_core.embeddings import build_embedding_model, zero_image_vector
from rag_core.io import load_file_documents, load_table_documents
from rag_core.milvus_store import (
    chunk_to_entity,
    connect,
    ensure_collection,
    milvus_string_literal,
    upsert_entities,
)
from rag_core.object_store import (
    archive_delete_tombstone,
    archive_source_documents,
    load_archived_source_documents,
)
from rag_core.pii import apply_pii_policy
from rag_core.source_guides import get_or_create_source_guide, load_source_guide, load_source_guide_full
from rag_core.text_utils import chunk_document, now_ms
from rag_core.types import Chunk, SourceDocument
from rag_core.versioning import load_current_versions, publish_current_versions, unpublish_current_version


SUPPORTED_FILE_SUFFIXES = {".pdf", ".html", ".htm", ".md", ".txt", ".csv", ".tsv"}


@dataclass(frozen=True)
class SourceSummary:
    doc_id: str
    title: str
    source_type: str
    source_uri: str
    doc_version: int
    chunk_count: int
    acl_groups: list[str]
    status: str
    current: bool
    created_at: int | None = None
    updated_at: int | None = None
    child_doc_ids: list[str] = field(default_factory=list)
    error: str = ""


@dataclass(frozen=True)
class IngestSummary:
    sources: list[SourceSummary]
    document_count: int
    chunk_count: int


@dataclass(frozen=True)
class SourceContent:
    doc_id: str
    title: str
    source_type: str
    source_uri: str
    doc_version: int
    child_doc_ids: list[str]
    guide: str
    tags: list[str]
    suggested_title: str = ""
    text: str


def save_uploaded_file(
    *,
    config: RagConfig,
    tenant_id: str,
    filename: str,
    content: BinaryIO,
) -> Path:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_FILE_SUFFIXES:
        raise ValueError(f"Unsupported source file type: {suffix or '<none>'}")
    safe_name = safe_filename(filename)
    upload_dir = config.object_store_dir / "uploads" / tenant_id / uuid.uuid4().hex
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / safe_name
    with target.open("wb") as file:
        shutil.copyfileobj(content, file)
    return target


def ingest_uploaded_path(
    *,
    config: RagConfig,
    path: Path,
    tenant_id: str,
    acl_groups: list[str],
    doc_version: int | None = None,
    language: str = "zh",
) -> IngestSummary:
    input_dir = path.parent
    docs = load_documents_for_path(
        path,
        input_dir=input_dir,
        tenant_id=tenant_id,
        doc_version=doc_version or 1,
        acl_groups=acl_groups or ["default"],
        language=language,
    )
    if doc_version is None:
        version = next_source_doc_version(config, docs)
        docs = [replace(doc, doc_version=version) for doc in docs]
    return ingest_source_documents(config=config, docs=docs)


def create_source_task(
    *,
    config: RagConfig,
    tenant_id: str,
    path: Path,
    acl_groups: list[str],
    doc_version: int | None,
) -> SourceSummary:
    timestamp = now_ms()
    doc_id = f"upload-{uuid.uuid4().hex[:12]}"
    source = SourceSummary(
        doc_id=doc_id,
        title=path.name,
        source_type=path.suffix.lower().lstrip(".") or "file",
        source_uri=str(path),
        doc_version=doc_version or 1,
        chunk_count=0,
        acl_groups=acl_groups,
        status="processing",
        current=False,
        created_at=timestamp,
        updated_at=timestamp,
        child_doc_ids=[],
    )
    save_source_task_for_tenant(config=config, tenant_id=tenant_id, source=source)
    return source


def save_source_task_for_tenant(*, config: RagConfig, tenant_id: str, source: SourceSummary, error: str = "") -> None:
    with connect_metadata_db(config) as conn:
        conn.execute(
            """
            INSERT INTO source_tasks(
                id, tenant_id, doc_id, title, source_type, source_uri, doc_version,
                acl_groups, status, error, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                error = excluded.error,
                updated_at = excluded.updated_at
            """,
            (
                source.doc_id,
                tenant_id,
                source.doc_id,
                source.title,
                source.source_type,
                source.source_uri,
                source.doc_version,
                json.dumps(source.acl_groups, ensure_ascii=False),
                source.status,
                error,
                source.created_at or now_ms(),
                source.updated_at or now_ms(),
            ),
        )


def delete_source_task(*, config: RagConfig, tenant_id: str, task_id: str) -> None:
    with connect_metadata_db(config) as conn:
        conn.execute("DELETE FROM source_tasks WHERE tenant_id = ? AND id = ?", (tenant_id, task_id))


def fail_source_task(*, config: RagConfig, tenant_id: str, source: SourceSummary, error: str) -> None:
    failed = replace(source, status="failed", updated_at=now_ms())
    save_source_task_for_tenant(config=config, tenant_id=tenant_id, source=failed, error=error[:500])


def load_documents_for_path(
    path: Path,
    *,
    input_dir: Path,
    tenant_id: str,
    doc_version: int,
    acl_groups: list[str],
    language: str,
) -> list[SourceDocument]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return load_table_documents(
            input_dir,
            tenant_id=tenant_id,
            doc_version=doc_version,
            acl_groups=acl_groups,
            language=language,
            recursive=False,
        )
    return load_file_documents(
        input_dir,
        tenant_id=tenant_id,
        doc_version=doc_version,
        acl_groups=acl_groups,
        language=language,
        recursive=False,
    )


def ingest_source_documents(*, config: RagConfig, docs: list[SourceDocument]) -> IngestSummary:
    if not docs:
        return IngestSummary(sources=[], document_count=0, chunk_count=0)

    client = connect(config)
    ensure_collection(client, config, reset=False)
    redacted_docs = [
        SourceDocument(
            **{
                **doc.__dict__,
                "text": apply_pii_policy(
                    doc.text,
                    policy=config.pii_policy,
                    label=f"{doc.doc_id}:text",
                ),
            }
        )
        for doc in docs
    ]
    text_model = build_embedding_model(config)
    chunks = [
        chunk
        for doc in redacted_docs
        for chunk in chunk_document(
            doc,
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
            token_counter=text_model.count_tokens,
        )
    ]
    dense_vectors = text_model.encode([chunk.text for chunk in chunks])
    zero_image = zero_image_vector(config)
    entities = [
        chunk_to_entity(
            chunk,
            dense_vector=dense_vector,
            image_vector=zero_image,
            embedding_model=text_model.model_name,
            embedding_dim=text_model.dim,
        )
        for chunk, dense_vector in zip(chunks, dense_vectors, strict=True)
    ]
    sources = summarize_ingested_sources(redacted_docs, chunks)
    generate_ingested_source_guides(config=config, sources=sources, docs=redacted_docs)
    upsert_entities(client, collection_name=config.collection_name, entities=entities)
    archive_source_documents(config.object_store_dir, redacted_docs)
    publish_current_versions(config.object_store_dir, redacted_docs)
    return IngestSummary(
        sources=sources,
        document_count=len(redacted_docs),
        chunk_count=len(chunks),
    )


def summarize_ingested_sources(
    docs: list[SourceDocument],
    chunks: list[Chunk],
) -> list[SourceSummary]:
    chunk_counts = Counter((chunk.doc_id, chunk.chunk_index) for chunk in chunks)
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for doc in docs:
        document_id, title = source_document_identity(
            doc_id=doc.doc_id,
            title=doc.title,
            source_uri=doc.source_uri,
            metadata=doc.metadata,
        )
        key = (document_id, doc.doc_version)
        item = grouped.setdefault(
            key,
            {
                "doc_id": document_id,
                "title": title,
                "source_type": doc.source_type,
                "source_uri": doc.source_uri,
                "doc_version": doc.doc_version,
                "acl_groups": set(),
                "child_doc_ids": set(),
                "chunk_keys": set(),
            },
        )
        item["acl_groups"].update(doc.acl_groups)
        item["child_doc_ids"].add(doc.doc_id)
        for chunk_key, count in chunk_counts.items():
            if chunk_key[0] == doc.doc_id and count > 0:
                item["chunk_keys"].add(chunk_key)

    return [
        SourceSummary(
            doc_id=str(item["doc_id"]),
            title=str(item["title"]),
            source_type=str(item["source_type"]),
            source_uri=str(item["source_uri"]),
            doc_version=int(item["doc_version"]),
            chunk_count=len(item["chunk_keys"]),
            acl_groups=sorted(item["acl_groups"]),
            status="ready",
            current=True,
            child_doc_ids=sorted(item["child_doc_ids"]),
        )
        for item in grouped.values()
    ]


def generate_ingested_source_guides(
    *,
    config: RagConfig,
    sources: list[SourceSummary],
    docs: list[SourceDocument],
) -> None:
    for source in sources:
        child_ids = set(source.child_doc_ids or [source.doc_id])
        source_docs = dedupe_source_documents(
            [
                doc
                for doc in docs
                if doc.doc_id in child_ids
                and int(doc.doc_version) == int(source.doc_version)
            ]
        )
        source_docs = sorted(source_docs, key=source_document_sort_key)
        if not source_docs:
            continue
        get_or_create_source_guide(
            config=config,
            tenant_id=source_docs[0].tenant_id,
            source_doc_id=source.doc_id,
            doc_version=source.doc_version,
            doc_title=source.title,
            docs=source_docs,
        )


def list_sources(*, config: RagConfig, tenant_id: str) -> list[SourceSummary]:
    client = connect(config)
    ensure_collection(client, config, reset=False)
    rows = client.query(
        collection_name=config.collection_name,
        filter=f"tenant_id == {milvus_string_literal(tenant_id)} and is_active == true",
        output_fields=[
            "doc_id",
            "doc_version",
            "title",
            "source_type",
            "source_uri",
            "chunk_index",
            "acl_groups",
            "created_at",
            "updated_at",
            "metadata",
        ],
        limit=10000,
    )
    current_versions = load_current_versions(config.object_store_dir, tenant_id=tenant_id)
    grouped: dict[tuple[str, int], dict] = defaultdict(
        lambda: {"chunk_keys": set(), "child_doc_ids": set(), "acl_groups": set()}
    )
    for row in rows:
        child_doc_id = str(row["doc_id"])
        metadata = row.get("metadata") or {}
        document_id, title = source_document_identity(
            doc_id=child_doc_id,
            title=str(row["title"]),
            source_uri=str(row["source_uri"]),
            metadata=metadata,
        )
        key = (document_id, int(row["doc_version"]))
        item = grouped[key]
        item["doc_id"] = document_id
        item["doc_version"] = int(row["doc_version"])
        item["title"] = title
        item["source_type"] = str(row["source_type"])
        item["source_uri"] = str(row["source_uri"])
        item["acl_groups"].update(list(row.get("acl_groups") or []))
        item["child_doc_ids"].add(child_doc_id)
        item["chunk_keys"].add((child_doc_id, int(row["chunk_index"])))
        created_at = int(row["created_at"]) if row.get("created_at") else None
        updated_at = int(row["updated_at"]) if row.get("updated_at") else None
        item["created_at"] = min_timestamp(item.get("created_at"), created_at)
        item["updated_at"] = max_timestamp(item.get("updated_at"), updated_at)
    title_overrides = load_source_title_overrides(config=config, tenant_id=tenant_id)
    summaries = [
        SourceSummary(
            doc_id=item["doc_id"],
            title=title_overrides.get((item["doc_id"], item["doc_version"]), item["title"]),
            source_type=item["source_type"],
            source_uri=item["source_uri"],
            doc_version=item["doc_version"],
            chunk_count=len(item["chunk_keys"]),
            acl_groups=sorted(item["acl_groups"]),
            status="ready",
            current=all(
                current_versions.get(child_doc_id) == item["doc_version"]
                for child_doc_id in item["child_doc_ids"]
            ),
            created_at=item["created_at"],
            updated_at=item["updated_at"],
            child_doc_ids=sorted(item["child_doc_ids"]),
        )
        for item in grouped.values()
    ]
    summaries.extend(list_source_tasks(config=config, tenant_id=tenant_id))
    return sorted(summaries, key=lambda item: (item.status == "ready", not item.current, item.title, item.doc_id))


def list_source_tasks(*, config: RagConfig, tenant_id: str) -> list[SourceSummary]:
    with connect_metadata_db(config) as conn:
        rows = conn.execute(
            """
            SELECT doc_id, title, source_type, source_uri, doc_version, acl_groups,
                   status, error, created_at, updated_at
            FROM source_tasks
            WHERE tenant_id = ?
            ORDER BY updated_at DESC
            """,
            (tenant_id,),
        ).fetchall()
    return [
        SourceSummary(
            doc_id=str(row["doc_id"]),
            title=str(row["title"]),
            source_type=str(row["source_type"]),
            source_uri=str(row["source_uri"]),
            doc_version=int(row["doc_version"]),
            chunk_count=0,
            acl_groups=json.loads(row["acl_groups"] or "[]"),
            status=str(row["status"]),
            current=False,
            created_at=int(row["created_at"] or 0),
            updated_at=int(row["updated_at"] or 0),
            child_doc_ids=[],
            error=str(row["error"] or ""),
        )
        for row in rows
    ]


def load_source_title_overrides(*, config: RagConfig, tenant_id: str) -> dict[tuple[str, int], str]:
    with connect_metadata_db(config) as conn:
        rows = conn.execute(
            """
            SELECT doc_id, doc_version, title
            FROM source_title_overrides
            WHERE tenant_id = ?
            """,
            (tenant_id,),
        ).fetchall()
    return {(str(row["doc_id"]), int(row["doc_version"])): str(row["title"]) for row in rows}


def rename_source(
    *,
    config: RagConfig,
    tenant_id: str,
    doc_id: str,
    title: str,
    doc_version: int | None = None,
) -> SourceSummary:
    clean_title = " ".join(title.split()).strip()
    if not clean_title:
        raise ValueError("来源标题不能为空")
    if len(clean_title) > 160:
        raise ValueError("来源标题不能超过 160 个字符")
    source = get_source(config=config, tenant_id=tenant_id, doc_id=doc_id, doc_version=doc_version)
    if source is None or source.status != "ready":
        raise ValueError("来源不存在或尚未解析完成")
    with connect_metadata_db(config) as conn:
        conn.execute(
            """
            INSERT INTO source_title_overrides(tenant_id, doc_id, doc_version, title, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(tenant_id, doc_id, doc_version) DO UPDATE SET
                title = excluded.title,
                updated_at = excluded.updated_at
            """,
            (tenant_id, source.doc_id, source.doc_version, clean_title, now_ms()),
        )
    return replace(source, title=clean_title, updated_at=now_ms())


def get_source(
    *,
    config: RagConfig,
    tenant_id: str,
    doc_id: str,
    doc_version: int | None = None,
) -> SourceSummary | None:
    sources = list_sources(config=config, tenant_id=tenant_id)
    matches = [
        source
        for source in sources
        if (source.doc_id == doc_id or doc_id in source.child_doc_ids)
        and (doc_version is None or source.doc_version == doc_version)
    ]
    if not matches:
        return None
    current = [source for source in matches if source.current]
    return (current or matches)[0]


def get_source_content(
    *,
    config: RagConfig,
    tenant_id: str,
    doc_id: str,
    doc_version: int | None = None,
) -> SourceContent | None:
    source = get_source(config=config, tenant_id=tenant_id, doc_id=doc_id, doc_version=doc_version)
    if source is None:
        return None

    child_ids = set(source.child_doc_ids or [source.doc_id])
    archived_docs = load_archived_source_documents(config.object_store_dir)
    docs = [
        doc
        for doc in archived_docs
        if doc.tenant_id == tenant_id
        and doc.doc_id in child_ids
        and doc.doc_version == source.doc_version
    ]
    docs = dedupe_source_documents(docs)
    if not docs:
        return SourceContent(
            doc_id=source.doc_id,
            title=source.title,
            source_type=source.source_type,
            source_uri=source.source_uri,
            doc_version=source.doc_version,
            child_doc_ids=source.child_doc_ids,
            guide="未找到已归档的解析正文。可以重新上传该来源以恢复文档详情。",
            tags=[],
            text="",
        )

    docs = sorted(docs, key=source_document_sort_key)
    text = "\n\n".join(source_document_text_block(doc) for doc in docs if doc.text.strip()).strip()
    guide_full = load_source_guide_full(
        config.object_store_dir,
        tenant_id=tenant_id,
        source_doc_id=source.doc_id,
        doc_version=source.doc_version,
    )
    guide = guide_full.guide if guide_full else None
    suggested_title = guide_full.title if guide_full else ""
    return SourceContent(
        doc_id=source.doc_id,
        title=source.title,
        source_type=source.source_type,
        source_uri=source.source_uri,
        doc_version=source.doc_version,
        child_doc_ids=source.child_doc_ids,
        guide=guide or "来源指南尚未生成。请重新上传该来源以在入库准备流程中生成摘要。",
        suggested_title=suggested_title,
        tags=extract_source_tags(text),
        text=text,
    )


def delete_source(
    *,
    config: RagConfig,
    tenant_id: str,
    doc_id: str,
    doc_version: int | None = None,
) -> dict[str, object]:
    client = connect(config)
    ensure_collection(client, config, reset=False)
    source = get_source(config=config, tenant_id=tenant_id, doc_id=doc_id, doc_version=doc_version)
    target_doc_ids = source.child_doc_ids if source is not None else [doc_id]
    filter_expr = (
        f"tenant_id == {milvus_string_literal(tenant_id)} "
        f"and doc_id in [{', '.join(milvus_string_literal(item) for item in target_doc_ids)}]"
    )
    if doc_version is not None:
        filter_expr += f" and doc_version == {doc_version}"
    result = client.delete(collection_name=config.collection_name, filter=filter_expr)
    effective_version = doc_version if doc_version is not None else source.doc_version if source is not None else None
    unpublished = {
        target_doc_id: unpublish_current_version(
            config.object_store_dir,
            tenant_id=tenant_id,
            doc_id=target_doc_id,
            doc_version=effective_version,
        )
        for target_doc_id in target_doc_ids
    }
    tombstoned = sum(
        archive_delete_tombstone(
            config.object_store_dir,
            tenant_id=tenant_id,
            doc_id=target_doc_id,
            doc_version=effective_version,
            reason="api_delete_source",
        )
        for target_doc_id in target_doc_ids
    )
    if source is not None:
        with connect_metadata_db(config) as conn:
            conn.execute(
                """
                DELETE FROM source_title_overrides
                WHERE tenant_id = ? AND doc_id = ? AND doc_version = ?
                """,
                (tenant_id, source.doc_id, source.doc_version),
            )
    return {
        "filter": filter_expr,
        "milvus": result,
        "target_doc_ids": target_doc_ids,
        "unpublished": unpublished,
        "tombstoned": tombstoned,
    }


def next_doc_version(config: RagConfig, *, tenant_id: str, doc_id: str) -> int:
    current = load_current_versions(config.object_store_dir, tenant_id=tenant_id)
    return int(current.get(doc_id, 0)) + 1


def next_source_doc_version(config: RagConfig, docs: list[SourceDocument]) -> int:
    if not docs:
        return 1
    identities = {
        (
            doc.tenant_id,
            source_document_identity(
                doc_id=doc.doc_id,
                title=doc.title,
                source_uri=doc.source_uri,
                metadata=doc.metadata,
            )[0],
        )
        for doc in docs
    }
    max_version = 0
    for archived in load_archived_source_documents(config.object_store_dir, include_deleted=True):
        document_id, _ = source_document_identity(
            doc_id=archived.doc_id,
            title=archived.title,
            source_uri=archived.source_uri,
            metadata=archived.metadata,
        )
        if (archived.tenant_id, document_id) in identities:
            max_version = max(max_version, int(archived.doc_version))
    for doc in docs:
        current = load_current_versions(config.object_store_dir, tenant_id=doc.tenant_id)
        max_version = max(max_version, int(current.get(doc.doc_id, 0)))
    return max_version + 1


def dedupe_source_documents(docs: list[SourceDocument]) -> list[SourceDocument]:
    unique = {
        (doc.tenant_id, doc.doc_id, int(doc.doc_version)): doc
        for doc in docs
    }
    return list(unique.values())


def safe_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    cleaned = "".join(char if char.isalnum() or char in ".-_ ()[]" else "_" for char in name)
    return cleaned or "upload.txt"


def source_document_identity(
    *,
    doc_id: str,
    title: str,
    source_uri: str,
    metadata: dict[str, Any],
) -> tuple[str, str]:
    relative_path = str(metadata.get("relative_path") or "").strip()
    if relative_path:
        path = Path(relative_path)
        return path.with_suffix("").as_posix(), path.name

    source_name = Path(source_uri).name
    if source_name:
        return Path(source_name).with_suffix("").as_posix(), source_name

    if "/page-" in doc_id:
        return doc_id.rsplit("/page-", 1)[0], title.rsplit(" p", 1)[0]
    return doc_id, title


def min_timestamp(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def max_timestamp(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def source_document_sort_key(doc: SourceDocument) -> tuple[int, str]:
    page_no = doc.metadata.get("page_no")
    if isinstance(page_no, int):
        return page_no, doc.doc_id
    if isinstance(page_no, str) and page_no.isdigit():
        return int(page_no), doc.doc_id
    return 0, doc.doc_id


def source_document_text_block(doc: SourceDocument) -> str:
    page_no = doc.metadata.get("page_no")
    text = doc.text.strip()
    if page_no is None:
        return text
    return f"第 {page_no} 页\n\n{text}"


def extract_source_tags(text: str, *, limit: int = 5) -> list[str]:
    tags: list[str] = []
    for line in text.splitlines():
        normalized = line.strip().strip("#").strip()
        if not normalized or len(normalized) > 24:
            continue
        if normalized.startswith(("第 ", "第")) and normalized.endswith("页"):
            continue
        if normalized[0].isdigit() or normalized[0] in "一二三四五六七八九十":
            tags.append(normalized)
        if len(tags) >= limit:
            break
    return tags
