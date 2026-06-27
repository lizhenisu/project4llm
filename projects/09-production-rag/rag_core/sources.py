from __future__ import annotations

import shutil
import hashlib
import json
import mimetypes
import os
import threading
import time
import uuid
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import quote

from rag_core.config import RagConfig
from rag_core.database import connect_metadata_db
from rag_core.embeddings import build_embedding_model, build_image_embedding_model, zero_image_vector
from rag_core.io import image_bytes_are_informative, load_file_documents, load_table_documents
from rag_core.jsonl_store import (
    object_store_backend,
    object_uri_for_relative_path,
    quote_object_uri,
    upload_file_to_object_store,
)
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
    purge_source_documents,
)
from rag_core.pii import apply_pii_policy
from rag_core.section_summaries import delete_source_section_summaries, save_source_section_summaries
from rag_core.source_guides import delete_source_guides, get_or_create_source_guide, load_source_guide, load_source_guide_full
from rag_core.text_utils import chunk_document, now_ms
from rag_core.types import Chunk, SourceDocument
from rag_core.versioning import load_current_versions, publish_current_versions, unpublish_current_version

MIN_EMBEDDABLE_IMAGE_SIDE = 8
MIN_EMBEDDABLE_IMAGE_PIXELS = 32
MAX_EMBEDDABLE_IMAGE_ASPECT_RATIO = 20.0


SUPPORTED_FILE_SUFFIXES = {".pdf", ".html", ".htm", ".md", ".txt", ".csv", ".tsv"}
_SOURCE_LIST_CACHE_LOCK = threading.Lock()
_SOURCE_LIST_CACHE: dict[tuple[str, str, str], tuple[float, list["SourceSummary"]]] = {}
_REQUESTED_DOC_VERSION_UNSET = object()


class UploadTooLargeError(ValueError):
    def __init__(self, *, size_bytes: int, limit_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.limit_bytes = limit_bytes
        super().__init__(
            f"Uploaded file is too large: {size_bytes} bytes exceeds RAG_MAX_UPLOAD_BYTES={limit_bytes}"
        )


class SourceTaskNotFoundError(LookupError):
    pass


class SourceTaskNotRetryableError(ValueError):
    pass


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
class QueuedSourceTask:
    tenant_id: str
    source: SourceSummary
    requested_doc_version: int | None


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
    text: str
    blocks: list[dict[str, str]] = field(default_factory=list)
    suggested_title: str = ""


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
    upload_id = uuid.uuid4().hex
    upload_dir = config.object_store_dir / "uploads" / tenant_id / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / safe_name
    try:
        with target.open("wb") as file:
            copy_uploaded_file_limited(content, file, max_bytes=config.max_upload_bytes)
    except Exception:
        target.unlink(missing_ok=True)
        try:
            upload_dir.rmdir()
        except OSError:
            pass
        raise
    if object_store_backend() == "s3":
        relative_path = Path("uploads") / tenant_id / upload_id / safe_name
        upload_file_to_object_store(
            target,
            relative_path,
            content_type=mimetypes.guess_type(safe_name)[0] or "application/octet-stream",
        )
    return target


def copy_uploaded_file_limited(content: BinaryIO, target: BinaryIO, *, max_bytes: int) -> None:
    copied = 0
    chunk_size = 1024 * 1024
    while True:
        chunk = content.read(chunk_size)
        if not chunk:
            return
        copied += len(chunk)
        if copied > max_bytes:
            raise UploadTooLargeError(size_bytes=copied, limit_bytes=max_bytes)
        target.write(chunk)


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
    docs = apply_uploaded_content_identity(docs, path=path, input_dir=input_dir)
    docs = apply_object_store_source_uris(config=config, docs=docs, path=path, input_dir=input_dir)
    if doc_version is None:
        version = next_source_doc_version(config, docs)
        docs = [replace(doc, doc_version=version) for doc in docs]
    return ingest_source_documents(config=config, docs=docs)


def apply_object_store_source_uris(
    *,
    config: RagConfig,
    docs: list[SourceDocument],
    path: Path,
    input_dir: Path,
) -> list[SourceDocument]:
    if object_store_backend() != "s3":
        return docs
    relative_upload_path = relative_upload_object_path(config=config, path=path)
    if relative_upload_path is None:
        return docs
    source_uri = object_uri_for_relative_path(relative_upload_path)
    return [
        replace(
            doc,
            source_uri=source_uri,
            metadata={
                **doc.metadata,
                "source_uri_local_work_path": str(path),
                "display_blocks": rewrite_display_blocks_to_object_store(
                    config=config,
                    input_dir=input_dir,
                    blocks=doc.metadata.get("display_blocks") or [],
                ),
            },
        )
        for doc in docs
    ]


def rewrite_display_blocks_to_object_store(
    *,
    config: RagConfig,
    input_dir: Path,
    blocks: list,
) -> list:
    rewritten: list = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "image":
            rewritten.append(block)
            continue
        image_path = Path(str(block.get("path") or block.get("image_uri") or ""))
        if not image_path.is_file():
            rewritten.append(block)
            continue
        try:
            relative_path = relative_upload_object_path(config=config, path=image_path)
            if relative_path is None:
                relative_path = Path("uploads") / image_path.relative_to(input_dir).as_posix()
        except ValueError:
            rewritten.append(block)
            continue
        media_type = str(block.get("media_type") or mimetypes.guess_type(image_path.name)[0] or "application/octet-stream")
        object_uri = upload_file_to_object_store(image_path, relative_path, content_type=media_type)
        rewritten.append({**block, "path": object_uri, "image_uri": object_uri})
    return rewritten


def relative_upload_object_path(*, config: RagConfig, path: Path) -> Path | None:
    try:
        return path.resolve().relative_to(config.object_store_dir.resolve())
    except ValueError:
        return None


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
        status="queued",
        current=False,
        created_at=timestamp,
        updated_at=timestamp,
        child_doc_ids=[],
    )
    save_source_task_for_tenant(
        config=config,
        tenant_id=tenant_id,
        source=source,
        requested_doc_version=doc_version,
    )
    return source


def save_source_task_for_tenant(
    *,
    config: RagConfig,
    tenant_id: str,
    source: SourceSummary,
    error: str = "",
    requested_doc_version: int | None | object = _REQUESTED_DOC_VERSION_UNSET,
) -> None:
    resolved_requested_version = (
        source.doc_version
        if requested_doc_version is _REQUESTED_DOC_VERSION_UNSET
        else requested_doc_version
    )
    with connect_metadata_db(config) as conn:
        conn.execute(
            """
            INSERT INTO source_tasks(
                id, tenant_id, doc_id, title, source_type, source_uri, doc_version,
                acl_groups, status, error, requested_doc_version, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                error = excluded.error,
                lease_owner = '',
                lease_expires_at = 0,
                next_attempt_at = 0,
                dead_lettered_at = CASE
                    WHEN excluded.status = 'failed' THEN excluded.updated_at
                    ELSE 0
                END,
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
                resolved_requested_version,
                source.created_at or now_ms(),
                source.updated_at or now_ms(),
            ),
        )
    invalidate_source_list_cache(config=config, tenant_id=tenant_id)


def delete_source_task(
    *,
    config: RagConfig,
    tenant_id: str,
    task_id: str,
    lease_owner: str | None = None,
) -> bool:
    with connect_metadata_db(config) as conn:
        if lease_owner:
            cursor = conn.execute(
                """
                DELETE FROM source_tasks
                WHERE tenant_id = ? AND id = ? AND status = 'processing' AND lease_owner = ?
                """,
                (tenant_id, task_id, lease_owner),
            )
        else:
            cursor = conn.execute(
                "DELETE FROM source_tasks WHERE tenant_id = ? AND id = ?",
                (tenant_id, task_id),
            )
    removed = int(cursor.rowcount or 0) > 0
    if removed:
        invalidate_source_list_cache(config=config, tenant_id=tenant_id)
    return removed


def save_source_catalog_for_tenant(*, config: RagConfig, tenant_id: str, sources: list[SourceSummary]) -> None:
    if not sources:
        return
    timestamp = now_ms()
    with connect_metadata_db(config) as conn:
        for source in sources:
            conn.execute(
                """
                INSERT INTO source_catalog(
                    tenant_id, doc_id, title, source_type, source_uri, doc_version,
                    chunk_count, acl_groups, current, created_at, updated_at, child_doc_ids
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, doc_id, doc_version) DO UPDATE SET
                    title = excluded.title,
                    source_type = excluded.source_type,
                    source_uri = excluded.source_uri,
                    chunk_count = excluded.chunk_count,
                    acl_groups = excluded.acl_groups,
                    current = excluded.current,
                    updated_at = excluded.updated_at,
                    child_doc_ids = excluded.child_doc_ids
                """,
                (
                    tenant_id,
                    source.doc_id,
                    source.title,
                    source.source_type,
                    source.source_uri,
                    source.doc_version,
                    source.chunk_count,
                    json.dumps(source.acl_groups, ensure_ascii=False),
                    1 if source.current else 0,
                    source.created_at or timestamp,
                    source.updated_at or timestamp,
                    json.dumps(source.child_doc_ids, ensure_ascii=False),
                ),
            )
    invalidate_source_list_cache(config=config, tenant_id=tenant_id)


def list_source_catalog(*, config: RagConfig, tenant_id: str) -> list[SourceSummary]:
    with connect_metadata_db(config) as conn:
        rows = conn.execute(
            """
            SELECT doc_id, title, source_type, source_uri, doc_version, chunk_count,
                   acl_groups, current, created_at, updated_at, child_doc_ids
            FROM source_catalog
            WHERE tenant_id = ?
            ORDER BY updated_at DESC
            """,
            (tenant_id,),
        ).fetchall()
    title_overrides = load_source_title_overrides(config=config, tenant_id=tenant_id)
    current_versions = load_current_versions(config.object_store_dir, tenant_id=tenant_id, config=config)
    summaries: list[SourceSummary] = []
    for row in rows:
        doc_id = str(row["doc_id"])
        doc_version = int(row["doc_version"])
        child_doc_ids = json.loads(row["child_doc_ids"] or "[]")
        is_current = bool(row["current"])
        if child_doc_ids:
            is_current = all(current_versions.get(str(child_doc_id)) == doc_version for child_doc_id in child_doc_ids)
        elif doc_id in current_versions:
            is_current = current_versions.get(doc_id) == doc_version
        summaries.append(
            SourceSummary(
                doc_id=doc_id,
                title=title_overrides.get((doc_id, doc_version), str(row["title"])),
                source_type=str(row["source_type"]),
                source_uri=str(row["source_uri"]),
                doc_version=doc_version,
                chunk_count=int(row["chunk_count"] or 0),
                acl_groups=json.loads(row["acl_groups"] or "[]"),
                status="ready",
                current=is_current,
                created_at=int(row["created_at"] or 0),
                updated_at=int(row["updated_at"] or 0),
                child_doc_ids=[str(item) for item in child_doc_ids],
            )
        )
    return summaries


def delete_source_catalog(
    *,
    config: RagConfig,
    tenant_id: str,
    doc_id: str,
    doc_version: int | None = None,
    child_doc_ids: list[str] | None = None,
) -> None:
    child_doc_ids = child_doc_ids or []
    with connect_metadata_db(config) as conn:
        if doc_version is None:
            conn.execute("DELETE FROM source_catalog WHERE tenant_id = ? AND doc_id = ?", (tenant_id, doc_id))
        else:
            conn.execute(
                "DELETE FROM source_catalog WHERE tenant_id = ? AND doc_id = ? AND doc_version = ?",
                (tenant_id, doc_id, doc_version),
            )
        for child_doc_id in child_doc_ids:
            if doc_version is None:
                conn.execute(
                    "DELETE FROM source_catalog WHERE tenant_id = ? AND doc_id = ?",
                    (tenant_id, child_doc_id),
                )
            else:
                conn.execute(
                    "DELETE FROM source_catalog WHERE tenant_id = ? AND doc_id = ? AND doc_version = ?",
                    (tenant_id, child_doc_id, doc_version),
                )
    invalidate_source_list_cache(config=config, tenant_id=tenant_id)


def fail_source_task(
    *,
    config: RagConfig,
    tenant_id: str,
    source: SourceSummary,
    error: str,
    lease_owner: str | None = None,
) -> bool:
    failed = replace(source, status="failed", updated_at=now_ms())
    if not lease_owner:
        save_source_task_for_tenant(config=config, tenant_id=tenant_id, source=failed, error=error[:500])
        return True
    with connect_metadata_db(config) as conn:
        cursor = conn.execute(
            """
            UPDATE source_tasks
            SET status = 'failed', error = ?, updated_at = ?, lease_owner = '', lease_expires_at = 0
            WHERE tenant_id = ? AND id = ? AND status = 'processing' AND lease_owner = ?
            """,
            (error[:500], failed.updated_at, tenant_id, source.doc_id, lease_owner),
        )
    updated = int(cursor.rowcount or 0) == 1
    if updated:
        invalidate_source_list_cache(config=config, tenant_id=tenant_id)
    return updated


def retry_or_fail_source_task(
    *,
    config: RagConfig,
    tenant_id: str,
    source: SourceSummary,
    error: str,
    lease_owner: str,
    max_attempts: int,
    backoff_seconds: float,
    backoff_max_seconds: float,
) -> str:
    timestamp = now_ms()
    with connect_metadata_db(config) as conn:
        row = conn.execute(
            """
            SELECT attempt_count
            FROM source_tasks
            WHERE tenant_id = ? AND id = ? AND status = 'processing' AND lease_owner = ?
            """,
            (tenant_id, source.doc_id, lease_owner),
        ).fetchone()
        if row is None:
            return "lost"
        attempt_count = int(row["attempt_count"] or 0)
        if attempt_count < max(1, int(max_attempts)):
            exponent = min(30, max(0, attempt_count - 1))
            delay_seconds = min(
                max(0.1, float(backoff_max_seconds)),
                max(0.1, float(backoff_seconds)) * (2**exponent),
            )
            next_attempt_at = timestamp + max(100, int(delay_seconds * 1000))
            cursor = conn.execute(
                """
                UPDATE source_tasks
                SET status = 'queued', error = ?, updated_at = ?,
                    lease_owner = '', lease_expires_at = 0,
                    next_attempt_at = ?, dead_lettered_at = 0
                WHERE tenant_id = ? AND id = ? AND status = 'processing' AND lease_owner = ?
                """,
                (
                    error[:500],
                    timestamp,
                    next_attempt_at,
                    tenant_id,
                    source.doc_id,
                    lease_owner,
                ),
            )
            outcome = "retried"
        else:
            cursor = conn.execute(
                """
                UPDATE source_tasks
                SET status = 'failed', error = ?, updated_at = ?,
                    lease_owner = '', lease_expires_at = 0,
                    next_attempt_at = 0, dead_lettered_at = ?
                WHERE tenant_id = ? AND id = ? AND status = 'processing' AND lease_owner = ?
                """,
                (
                    error[:500],
                    timestamp,
                    timestamp,
                    tenant_id,
                    source.doc_id,
                    lease_owner,
                ),
            )
            outcome = "failed"
    if int(cursor.rowcount or 0) != 1:
        return "lost"
    invalidate_source_list_cache(config=config, tenant_id=tenant_id)
    return outcome


def retry_failed_source_task(
    *,
    config: RagConfig,
    tenant_id: str,
    task_id: str,
) -> QueuedSourceTask:
    timestamp = now_ms()
    with connect_metadata_db(config) as conn:
        row = conn.execute(
            """
            SELECT tenant_id, doc_id, title, source_type, source_uri, doc_version,
                   acl_groups, status, error, requested_doc_version, created_at, updated_at
            FROM source_tasks
            WHERE tenant_id = ? AND id = ?
            """,
            (tenant_id, task_id),
        ).fetchone()
        if row is None:
            raise SourceTaskNotFoundError("Source task not found")
        if str(row["status"]) != "failed":
            raise SourceTaskNotRetryableError("Only failed source tasks can be retried")
        cursor = conn.execute(
            """
            UPDATE source_tasks
            SET status = 'queued', error = '', updated_at = ?,
                lease_owner = '', lease_expires_at = 0, attempt_count = 0,
                next_attempt_at = 0, dead_lettered_at = 0
            WHERE tenant_id = ? AND id = ? AND status = 'failed'
            """,
            (timestamp, tenant_id, task_id),
        )
        if int(cursor.rowcount or 0) != 1:
            raise SourceTaskNotRetryableError("Source task state changed before retry")
    invalidate_source_list_cache(config=config, tenant_id=tenant_id)
    source = SourceSummary(
        doc_id=str(row["doc_id"]),
        title=str(row["title"]),
        source_type=str(row["source_type"]),
        source_uri=str(row["source_uri"]),
        doc_version=int(row["doc_version"]),
        chunk_count=0,
        acl_groups=json.loads(row["acl_groups"] or "[]"),
        status="queued",
        current=False,
        created_at=int(row["created_at"] or 0),
        updated_at=timestamp,
        child_doc_ids=[],
        error="",
    )
    return QueuedSourceTask(
        tenant_id=tenant_id,
        source=source,
        requested_doc_version=(
            int(row["requested_doc_version"])
            if row["requested_doc_version"] is not None
            else None
        ),
    )


def update_source_task_status(
    *,
    config: RagConfig,
    tenant_id: str,
    source: SourceSummary,
    status: str,
    error: str = "",
) -> SourceSummary:
    updated = replace(source, status=status, updated_at=now_ms())
    save_source_task_for_tenant(config=config, tenant_id=tenant_id, source=updated, error=error[:500])
    return updated


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


def apply_uploaded_content_identity(
    docs: list[SourceDocument],
    *,
    path: Path,
    input_dir: Path,
) -> list[SourceDocument]:
    if not docs:
        return docs
    content_hash = file_sha256(path)
    content_key = f"sha256-{content_hash[:12]}"
    base_id = path.relative_to(input_dir).with_suffix("").as_posix()
    hashed_base_id = f"{base_id}@{content_key}"
    relative_path = path.relative_to(input_dir).as_posix()
    return [
        replace(
            doc,
            doc_id=hashed_doc_id(doc.doc_id, base_id=base_id, hashed_base_id=hashed_base_id),
            metadata={
                **doc.metadata,
                "relative_path": relative_path,
                "content_sha256": content_hash,
                "content_key": content_key,
            },
        )
        for doc in docs
    ]


def hashed_doc_id(doc_id: str, *, base_id: str, hashed_base_id: str) -> str:
    if doc_id == base_id:
        return hashed_base_id
    if doc_id.startswith(f"{base_id}/"):
        return f"{hashed_base_id}/{doc_id[len(base_id) + 1:]}"
    return f"{hashed_base_id}/{doc_id}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    image_chunks = pdf_image_chunks(redacted_docs)
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
    image_docs: list[SourceDocument] = []
    if image_chunks:
        image_model = build_image_embedding_model(config)
        image_dense_vectors = text_model.encode([chunk.text for chunk in image_chunks])
        image_vectors = encode_pdf_image_vectors(config=config, image_model=image_model, chunks=image_chunks)
        entities.extend(
            chunk_to_entity(
                chunk,
                dense_vector=dense_vector,
                image_vector=image_vector,
                embedding_model=text_model.model_name,
                embedding_dim=text_model.dim,
            )
            for chunk, dense_vector, image_vector in zip(
                image_chunks, image_dense_vectors, image_vectors, strict=True
            )
        )
        image_docs = [
            SourceDocument(
                tenant_id=chunk.tenant_id,
                doc_id=chunk.doc_id,
                doc_version=chunk.doc_version,
                source_type=chunk.source_type,
                source_uri=chunk.source_uri,
                title=chunk.title,
                text=chunk.text,
                language=chunk.language,
                acl_groups=chunk.acl_groups,
                metadata=chunk.metadata,
            )
            for chunk in image_chunks
        ]
    canonical_docs = [*redacted_docs, *image_docs]
    all_chunks = [*chunks, *image_chunks]
    sources = summarize_ingested_sources(canonical_docs, all_chunks)
    generate_ingested_source_guides(config=config, sources=sources, docs=redacted_docs)
    upsert_entities(client, collection_name=config.collection_name, entities=entities)
    archive_source_documents(config.object_store_dir, canonical_docs)
    publish_current_versions(config.object_store_dir, canonical_docs, config=config)
    for tenant_id in {doc.tenant_id for doc in canonical_docs}:
        tenant_sources = [
            source
            for source in sources
            if any(doc.tenant_id == tenant_id and doc.doc_id in source.child_doc_ids for doc in canonical_docs)
        ]
        save_source_catalog_for_tenant(config=config, tenant_id=tenant_id, sources=tenant_sources)
    for tenant_id in {doc.tenant_id for doc in canonical_docs}:
        invalidate_source_list_cache(config=config, tenant_id=tenant_id)
    return IngestSummary(
        sources=sources,
        document_count=len(canonical_docs),
        chunk_count=len(all_chunks),
    )


def pdf_image_chunks(docs: list[SourceDocument]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        for image_index, raw_block in enumerate(doc.metadata.get("display_blocks") or [], start=1):
            if not isinstance(raw_block, dict) or raw_block.get("type") != "image":
                continue
            image_path = str(raw_block.get("path") or raw_block.get("image_uri") or "").strip()
            if not image_path:
                continue
            title = str(raw_block.get("title") or f"Image {image_index}")
            page_no = doc.metadata.get("page_no")
            page_label = f"第 {page_no} 页" if page_no else ""
            display_text = str(doc.metadata.get("display_text") or "").strip()
            context = display_text[:1200]
            text = (
                f"标题路径: {doc.title} > {title}\n"
                "来源: image\n"
                f"图片位置: {page_label}\n"
                f"图片描述: PDF 内嵌图片 {title}。\n"
                f"页面上下文:\n{context}"
            )
            block = {
                key: raw_block[key]
                for key in ("type", "title", "path", "image_uri", "media_type")
                if isinstance(raw_block.get(key), str) and str(raw_block.get(key)).strip()
            }
            chunks.append(
                Chunk(
                    tenant_id=doc.tenant_id,
                    doc_id=f"{doc.doc_id}/image-{image_index}",
                    doc_version=doc.doc_version,
                    chunk_index=0,
                    source_type="image",
                    source_uri=image_path,
                    title=f"{doc.title} {title}",
                    text=text,
                    language=doc.language,
                    acl_groups=doc.acl_groups,
                    metadata={
                        **{
                            key: value
                            for key, value in doc.metadata.items()
                            if key not in {"display_text", "display_blocks"}
                        },
                        "page_no": page_no,
                        "page_start": doc.metadata.get("page_start", page_no),
                        "page_end": doc.metadata.get("page_end", page_no),
                        "image_uri": image_path,
                        "linked_doc_id": doc.doc_id,
                        "linked_source_type": doc.source_type,
                        "linked_source_uri": doc.source_uri,
                        "linked_title": doc.title,
                        "derived_from_pdf_image": True,
                        "display_blocks": [block],
                    },
                )
            )
    return chunks


def encode_pdf_image_vectors(*, config: RagConfig, image_model, chunks: list[Chunk]) -> list[list[float]]:
    zero_image = zero_image_vector(config)
    vectors: list[list[float] | None] = []
    embeddable_paths: list[Path] = []
    embeddable_indexes: list[int] = []
    for index, chunk in enumerate(chunks):
        path = Path(chunk.source_uri)
        if pdf_image_is_embeddable(path):
            vectors.append(None)
            embeddable_paths.append(path)
            embeddable_indexes.append(index)
        else:
            vectors.append(zero_image)

    if embeddable_paths:
        try:
            encoded = image_model.encode_images(embeddable_paths)
        except Exception:
            encoded = [
                encode_pdf_image_vector_or_zero(config=config, image_model=image_model, path=path)
                for path in embeddable_paths
            ]
        for index, vector in zip(embeddable_indexes, encoded, strict=True):
            vectors[index] = vector
    return [vector or zero_image for vector in vectors]


def encode_pdf_image_vector_or_zero(*, config: RagConfig, image_model, path: Path) -> list[float]:
    try:
        return image_model.encode_images([path])[0]
    except Exception:
        return zero_image_vector(config)


def pdf_image_is_embeddable(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
    except Exception:
        return False
    if width < MIN_EMBEDDABLE_IMAGE_SIDE or height < MIN_EMBEDDABLE_IMAGE_SIDE:
        return False
    pixels = width * height
    if pixels < MIN_EMBEDDABLE_IMAGE_PIXELS:
        return False
    aspect_ratio = max(width, height) / max(1, min(width, height))
    if aspect_ratio > MAX_EMBEDDABLE_IMAGE_ASPECT_RATIO:
        return False
    return image_bytes_are_informative(path.read_bytes())


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
        doc_source_type = source_summary_source_type(doc.source_type, doc.metadata)
        doc_source_uri = source_summary_source_uri(doc.source_uri, doc.metadata)
        item = grouped.setdefault(
            key,
            {
                "doc_id": document_id,
                "title": title,
                "source_type": doc_source_type,
                "source_uri": doc_source_uri,
                "doc_version": doc.doc_version,
                "acl_groups": set(),
                "child_doc_ids": set(),
                "chunk_keys": set(),
            },
        )
        if not doc.metadata.get("derived_from_pdf_image"):
            item["source_type"] = doc_source_type
            item["source_uri"] = doc_source_uri
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
        save_source_section_summaries(
            config.object_store_dir,
            tenant_id=source_docs[0].tenant_id,
            source_doc_id=source.doc_id,
            doc_version=source.doc_version,
            docs=source_docs,
        )


def list_sources(*, config: RagConfig, tenant_id: str) -> list[SourceSummary]:
    ttl_seconds = max(0.0, float(config.source_list_cache_ttl_seconds))
    if ttl_seconds <= 0:
        return _list_sources_uncached(config=config, tenant_id=tenant_id)
    cache_key = source_list_cache_key(config=config, tenant_id=tenant_id)
    now = time.monotonic()
    with _SOURCE_LIST_CACHE_LOCK:
        cached = _SOURCE_LIST_CACHE.get(cache_key)
        if cached is not None and cached[0] > now:
            return list(cached[1])
    summaries = _list_sources_uncached(config=config, tenant_id=tenant_id)
    with _SOURCE_LIST_CACHE_LOCK:
        _SOURCE_LIST_CACHE[cache_key] = (now + ttl_seconds, list(summaries))
    return summaries


def _list_sources_uncached(*, config: RagConfig, tenant_id: str) -> list[SourceSummary]:
    catalog_sources = list_source_catalog(config=config, tenant_id=tenant_id)
    task_sources = list_source_tasks(config=config, tenant_id=tenant_id)
    if catalog_sources:
        summaries = [*catalog_sources, *task_sources]
        return sorted(summaries, key=lambda item: (item.status == "ready", not item.current, item.title, item.doc_id))
    return _list_sources_from_milvus(config=config, tenant_id=tenant_id, task_sources=task_sources)


def _list_sources_from_milvus(
    *,
    config: RagConfig,
    tenant_id: str,
    task_sources: list[SourceSummary] | None = None,
) -> list[SourceSummary]:
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
    current_versions = load_current_versions(config.object_store_dir, tenant_id=tenant_id, config=config)
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
        row_source_type = source_summary_source_type(str(row["source_type"]), metadata)
        row_source_uri = source_summary_source_uri(str(row["source_uri"]), metadata)
        if "source_type" not in item or not metadata.get("derived_from_pdf_image"):
            item["source_type"] = row_source_type
            item["source_uri"] = row_source_uri
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
    if summaries:
        save_source_catalog_for_tenant(config=config, tenant_id=tenant_id, sources=summaries)
    summaries.extend(task_sources if task_sources is not None else list_source_tasks(config=config, tenant_id=tenant_id))
    return sorted(summaries, key=lambda item: (item.status == "ready", not item.current, item.title, item.doc_id))


def source_list_cache_key(*, config: RagConfig, tenant_id: str) -> tuple[str, str, str]:
    return (str(config.metadata_database_url or config.object_store_dir), config.collection_name, tenant_id)


def invalidate_source_list_cache(*, config: RagConfig, tenant_id: str) -> None:
    cache_key = source_list_cache_key(config=config, tenant_id=tenant_id)
    with _SOURCE_LIST_CACHE_LOCK:
        _SOURCE_LIST_CACHE.pop(cache_key, None)


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


def count_source_tasks_by_status(*, config: RagConfig, tenant_id: str | None = None) -> dict[str, int]:
    params: tuple[object, ...] = ()
    where = ""
    if tenant_id:
        where = "WHERE tenant_id = ?"
        params = (tenant_id,)
    with connect_metadata_db(config) as conn:
        rows = conn.execute(
            f"""
            SELECT status, COUNT(*) AS count
            FROM source_tasks
            {where}
            GROUP BY status
            """,
            params,
        ).fetchall()
    return {str(row["status"]): int(row["count"] or 0) for row in rows}


def source_task_lease_metrics_snapshot(
    *,
    config: RagConfig,
    tenant_id: str | None = None,
) -> dict[str, int]:
    timestamp = now_ms()
    params: list[object] = [timestamp, timestamp]
    where = ""
    if tenant_id:
        where = "WHERE tenant_id = ?"
        params.append(tenant_id)
    with connect_metadata_db(config) as conn:
        row = conn.execute(
            f"""
            SELECT
                SUM(
                    CASE
                        WHEN status = 'processing' AND lease_owner <> '' AND lease_expires_at >= ?
                        THEN 1 ELSE 0
                    END
                ) AS active_leases,
                SUM(
                    CASE
                        WHEN status = 'processing' AND lease_expires_at > 0 AND lease_expires_at < ?
                        THEN 1 ELSE 0
                    END
                ) AS expired_leases,
                SUM(attempt_count) AS attempts_recorded,
                MAX(attempt_count) AS max_attempt_count
            FROM source_tasks
            {where}
            """,
            tuple(params),
        ).fetchone()
    return {
        "active_leases": int(row["active_leases"] or 0) if row is not None else 0,
        "expired_leases": int(row["expired_leases"] or 0) if row is not None else 0,
        "attempts_recorded": int(row["attempts_recorded"] or 0) if row is not None else 0,
        "max_attempt_count": int(row["max_attempt_count"] or 0) if row is not None else 0,
    }


def source_task_recovery_metrics_snapshot(
    *,
    config: RagConfig,
    tenant_id: str | None = None,
) -> dict[str, int]:
    timestamp = now_ms()
    params: list[object] = [timestamp]
    where = ""
    if tenant_id:
        where = "WHERE tenant_id = ?"
        params.append(tenant_id)
    with connect_metadata_db(config) as conn:
        row = conn.execute(
            f"""
            SELECT
                SUM(
                    CASE
                        WHEN status = 'queued' AND next_attempt_at > ?
                        THEN 1 ELSE 0
                    END
                ) AS retry_waiting,
                SUM(
                    CASE
                        WHEN status = 'failed' AND dead_lettered_at > 0
                        THEN 1 ELSE 0
                    END
                ) AS dead_lettered,
                SUM(
                    CASE
                        WHEN attempt_count > 1 THEN attempt_count - 1 ELSE 0
                    END
                ) AS retries_recorded
            FROM source_tasks
            {where}
            """,
            tuple(params),
        ).fetchone()
    return {
        "retry_waiting": int(row["retry_waiting"] or 0) if row is not None else 0,
        "dead_lettered": int(row["dead_lettered"] or 0) if row is not None else 0,
        "retries_recorded": int(row["retries_recorded"] or 0) if row is not None else 0,
    }


def count_active_source_tasks(*, config: RagConfig, tenant_id: str | None = None) -> int:
    params: list[object] = ["queued", "processing", "uploading"]
    where = "WHERE status IN (?, ?, ?)"
    if tenant_id:
        where += " AND tenant_id = ?"
        params.append(tenant_id)
    with connect_metadata_db(config) as conn:
        row = conn.execute(
            f"""
            SELECT COUNT(*) AS count
            FROM source_tasks
            {where}
            """,
            tuple(params),
        ).fetchone()
    return int(row["count"] if row is not None else 0)


def list_queued_source_tasks(*, config: RagConfig, limit: int = 100) -> list[QueuedSourceTask]:
    timestamp = now_ms()
    with connect_metadata_db(config) as conn:
        rows = conn.execute(
            """
            SELECT tenant_id, doc_id, title, source_type, source_uri, doc_version, acl_groups,
                   status, error, requested_doc_version, created_at, updated_at
            FROM source_tasks
            WHERE status = 'queued' AND next_attempt_at <= ?
            ORDER BY next_attempt_at ASC, created_at ASC
            LIMIT ?
            """,
            (timestamp, max(1, int(limit))),
        ).fetchall()
    return [
        QueuedSourceTask(
            tenant_id=str(row["tenant_id"]),
            source=SourceSummary(
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
            ),
            requested_doc_version=(
                int(row["requested_doc_version"])
                if row["requested_doc_version"] is not None
                else None
            ),
        )
        for row in rows
    ]


def requeue_stale_processing_source_tasks(
    *,
    config: RagConfig,
    stale_after_ms: int,
    limit: int = 100,
) -> int:
    timestamp = now_ms()
    cutoff = timestamp - max(1000, int(stale_after_ms))
    with connect_metadata_db(config) as conn:
        rows = conn.execute(
            """
            SELECT id, tenant_id
            FROM source_tasks
            WHERE status = 'processing'
              AND (
                (lease_expires_at > 0 AND lease_expires_at < ?)
                OR (lease_expires_at = 0 AND updated_at < ?)
              )
            ORDER BY updated_at ASC
            LIMIT ?
            """,
            (timestamp, cutoff, max(1, int(limit))),
        ).fetchall()
        task_ids = [str(row["id"]) for row in rows]
        tenant_ids = {str(row["tenant_id"]) for row in rows}
        if not task_ids:
            return 0
        placeholders = ", ".join("?" for _ in task_ids)
        cursor = conn.execute(
            f"""
            UPDATE source_tasks
            SET status = 'queued', updated_at = ?, error = '', lease_owner = '', lease_expires_at = 0,
                next_attempt_at = ?
            WHERE id IN ({placeholders})
              AND status = 'processing'
              AND (
                (lease_expires_at > 0 AND lease_expires_at < ?)
                OR (lease_expires_at = 0 AND updated_at < ?)
              )
            """,
            (timestamp, timestamp, *task_ids, timestamp, cutoff),
        )
        updated_count = int(cursor.rowcount or 0)
    for tenant_id in tenant_ids:
        invalidate_source_list_cache(config=config, tenant_id=tenant_id)
    return updated_count


def claim_source_task_for_processing(
    *,
    config: RagConfig,
    tenant_id: str,
    source: SourceSummary,
    lease_owner: str,
    lease_ms: int,
) -> SourceSummary | None:
    updated = replace(source, status="processing", updated_at=now_ms())
    lease_expires_at = updated.updated_at + max(1000, int(lease_ms))
    with connect_metadata_db(config) as conn:
        cursor = conn.execute(
            """
            UPDATE source_tasks
            SET status = 'processing', updated_at = ?, error = '',
                lease_owner = ?, lease_expires_at = ?, attempt_count = attempt_count + 1,
                next_attempt_at = 0
            WHERE tenant_id = ? AND id = ? AND status = 'queued' AND next_attempt_at <= ?
            """,
            (
                updated.updated_at,
                lease_owner,
                lease_expires_at,
                tenant_id,
                source.doc_id,
                updated.updated_at,
            ),
        )
        if cursor.rowcount != 1:
            return None
    invalidate_source_list_cache(config=config, tenant_id=tenant_id)
    return updated


def renew_source_task_lease(
    *,
    config: RagConfig,
    tenant_id: str,
    task_id: str,
    lease_owner: str,
    lease_ms: int,
) -> bool:
    timestamp = now_ms()
    with connect_metadata_db(config) as conn:
        cursor = conn.execute(
            """
            UPDATE source_tasks
            SET updated_at = ?, lease_expires_at = ?
            WHERE tenant_id = ? AND id = ? AND status = 'processing' AND lease_owner = ?
            """,
            (
                timestamp,
                timestamp + max(1000, int(lease_ms)),
                tenant_id,
                task_id,
                lease_owner,
            ),
        )
    renewed = int(cursor.rowcount or 0) == 1
    return renewed


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
        conn.execute(
            """
            UPDATE source_catalog
            SET title = ?, updated_at = ?
            WHERE tenant_id = ? AND doc_id = ? AND doc_version = ?
            """,
            (clean_title, now_ms(), tenant_id, source.doc_id, source.doc_version),
        )
    invalidate_source_list_cache(config=config, tenant_id=tenant_id)
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
            blocks=[],
        )

    docs = sorted(docs, key=source_document_sort_key)
    display_docs = [doc for doc in docs if not doc.metadata.get("derived_from_pdf_image")]
    text = "\n\n".join(source_document_text_block(doc) for doc in display_docs if doc.text.strip()).strip()
    display_blocks = source_document_display_blocks(config=config, tenant_id=tenant_id, docs=display_docs)
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
        text=source_document_display_text(display_docs) or text,
        blocks=display_blocks,
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
            config=config,
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
    purged = purge_source_documents(
        config.object_store_dir,
        tenant_id=tenant_id,
        doc_ids=target_doc_ids,
        doc_version=effective_version,
    )
    deleted_source_guides = delete_source_guides(
        config.object_store_dir,
        tenant_id=tenant_id,
        source_doc_ids={doc_id},
        doc_version=effective_version,
    )
    deleted_section_summaries = delete_source_section_summaries(
        config.object_store_dir,
        tenant_id=tenant_id,
        source_doc_ids={doc_id},
        doc_version=effective_version,
    )
    delete_source_catalog(
        config=config,
        tenant_id=tenant_id,
        doc_id=source.doc_id if source is not None else doc_id,
        doc_version=effective_version,
        child_doc_ids=target_doc_ids,
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
            conn.execute(
                """
                DELETE FROM source_tasks
                WHERE tenant_id = ? AND (doc_id = ? OR doc_id IN ({placeholders}))
                """.format(placeholders=", ".join("?" for _ in target_doc_ids)),
                (tenant_id, source.doc_id, *target_doc_ids),
            )
    invalidate_source_list_cache(config=config, tenant_id=tenant_id)
    return {
        "filter": filter_expr,
        "milvus": result,
        "target_doc_ids": target_doc_ids,
        "unpublished": unpublished,
        "tombstoned": tombstoned,
        "purged": purged,
        "deleted_source_guides": deleted_source_guides,
        "deleted_section_summaries": deleted_section_summaries,
    }


def next_doc_version(config: RagConfig, *, tenant_id: str, doc_id: str) -> int:
    current = load_current_versions(config.object_store_dir, tenant_id=tenant_id, config=config)
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
        current = load_current_versions(config.object_store_dir, tenant_id=doc.tenant_id, config=config)
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
    content_key = str(metadata.get("content_key") or "").strip()
    if relative_path:
        path = Path(relative_path)
        if content_key:
            return f"{path.with_suffix('').as_posix()}@{content_key}", path.name
        return path.with_suffix("").as_posix(), path.name

    source_name = Path(source_uri).name
    if source_name:
        return Path(source_name).with_suffix("").as_posix(), source_name

    if "/page-" in doc_id:
        return doc_id.rsplit("/page-", 1)[0], title.rsplit(" p", 1)[0]
    return doc_id, title


def source_summary_source_type(source_type: str, metadata: dict[str, Any]) -> str:
    if metadata.get("derived_from_pdf_image"):
        linked_type = str(metadata.get("linked_source_type") or "").strip()
        if linked_type:
            return linked_type
    return source_type


def source_summary_source_uri(source_uri: str, metadata: dict[str, Any]) -> str:
    if metadata.get("derived_from_pdf_image"):
        linked_uri = str(metadata.get("linked_source_uri") or "").strip()
        if linked_uri:
            return linked_uri
    return source_uri


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


def source_document_display_text(docs: list[SourceDocument]) -> str:
    blocks = [
        block["text"]
        for block in source_document_display_blocks(config=None, tenant_id="", docs=docs)
        if block.get("type") == "text" and block.get("text")
    ]
    return "\n\n".join(blocks).strip()


def source_document_display_blocks(*, config: RagConfig | None, tenant_id: str, docs: list[SourceDocument]) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    for doc in docs:
        page_label = localized_page_label(doc)
        display_text = str(doc.metadata.get("display_text") or doc.text).strip()
        if display_text:
            blocks.append(
                {
                    "type": "text",
                    "text": f"{page_label}\n\n{display_text}" if page_label else display_text,
                }
            )
        for raw_block in doc.metadata.get("display_blocks") or []:
            if not isinstance(raw_block, dict):
                continue
            block_type = str(raw_block.get("type") or "")
            if block_type != "image":
                continue
            image_url = image_block_url(config=config, tenant_id=tenant_id or doc.tenant_id, block=raw_block)
            if not image_url:
                continue
            blocks.append(
                {
                    "type": "image",
                    "title": str(raw_block.get("title") or "Image"),
                    "url": image_url,
                    "page": page_label,
                }
            )
    return blocks


def resolve_metadata_display_block_urls(*, config: RagConfig, tenant_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    blocks = metadata.get("display_blocks")
    if not isinstance(blocks, list):
        return metadata
    resolved_blocks: list[dict[str, Any]] = []
    changed = False
    for raw_block in blocks:
        if not isinstance(raw_block, dict):
            resolved_blocks.append(raw_block)
            continue
        block = dict(raw_block)
        if block.get("type") == "image":
            image_url = image_block_url(config=config, tenant_id=tenant_id, block=block)
            if image_url:
                block["url"] = image_url
                changed = True
        resolved_blocks.append(block)
    if not changed:
        return metadata
    return {**metadata, "display_blocks": resolved_blocks}


def image_block_url(*, config: RagConfig | None, tenant_id: str, block: dict[str, Any]) -> str:
    existing_url = str(block.get("url") or "")
    if existing_url.startswith("data:image/") or existing_url.startswith("http://") or existing_url.startswith("https://"):
        return existing_url
    if config is None:
        return ""
    raw_path = str(block.get("path") or block.get("image_uri") or "").strip()
    if not raw_path:
        return ""
    if raw_path.startswith("s3://"):
        return f"/source-assets/__s3__/{quote_object_uri(raw_path)}?tenant_id={quote(tenant_id)}"
    try:
        image_path = Path(raw_path).expanduser().resolve()
        object_store_dir = config.object_store_dir.expanduser().resolve()
        relative_path = image_path.relative_to(object_store_dir)
    except (OSError, ValueError):
        return ""
    if not image_path.is_file():
        return ""
    encoded_path = quote(relative_path.as_posix(), safe="/")
    return f"/source-assets/{encoded_path}?tenant_id={quote(tenant_id)}"


def localized_page_label(doc: SourceDocument) -> str:
    page_no = doc.metadata.get("page_no")
    if page_no is None:
        return ""
    display_text = str(doc.metadata.get("display_text") or doc.text)
    cjk_count = sum(1 for char in display_text if "\u4e00" <= char <= "\u9fff")
    alpha_count = sum(1 for char in display_text if char.isalpha() and char.isascii())
    return f"第 {page_no} 页" if cjk_count > max(8, alpha_count // 3) else f"Page {page_no}"


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
