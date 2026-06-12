from __future__ import annotations

import shutil
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from rag_core.config import RagConfig
from rag_core.embeddings import build_embedding_model, zero_image_vector
from rag_core.io import load_file_documents, load_table_documents
from rag_core.milvus_store import (
    chunk_to_entity,
    connect,
    ensure_collection,
    milvus_string_literal,
    upsert_entities,
)
from rag_core.object_store import archive_delete_tombstone, archive_source_documents
from rag_core.pii import apply_pii_policy
from rag_core.text_utils import chunk_document
from rag_core.types import SourceDocument
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


@dataclass(frozen=True)
class IngestSummary:
    sources: list[SourceSummary]
    document_count: int
    chunk_count: int


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
    version = doc_version or next_doc_version(config, tenant_id=tenant_id, doc_id=path.stem)
    docs = load_documents_for_path(
        path,
        input_dir=input_dir,
        tenant_id=tenant_id,
        doc_version=version,
        acl_groups=acl_groups or ["default"],
        language=language,
    )
    return ingest_source_documents(config=config, docs=docs)


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
    upsert_entities(client, collection_name=config.collection_name, entities=entities)
    archive_source_documents(config.object_store_dir, redacted_docs)
    publish_current_versions(config.object_store_dir, redacted_docs)
    sources = list_sources(config, tenant_id=redacted_docs[0].tenant_id)
    doc_ids = {doc.doc_id for doc in redacted_docs}
    return IngestSummary(
        sources=[source for source in sources if source.doc_id in doc_ids],
        document_count=len(redacted_docs),
        chunk_count=len(chunks),
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
        ],
        limit=10000,
    )
    current_versions = load_current_versions(config.object_store_dir, tenant_id=tenant_id)
    grouped: dict[tuple[str, int], dict] = defaultdict(lambda: {"chunk_indexes": set()})
    for row in rows:
        key = (str(row["doc_id"]), int(row["doc_version"]))
        item = grouped[key]
        item.update(
            {
                "doc_id": str(row["doc_id"]),
                "doc_version": int(row["doc_version"]),
                "title": str(row["title"]),
                "source_type": str(row["source_type"]),
                "source_uri": str(row["source_uri"]),
                "acl_groups": list(row.get("acl_groups") or []),
                "created_at": int(row["created_at"]) if row.get("created_at") else None,
                "updated_at": int(row["updated_at"]) if row.get("updated_at") else None,
            }
        )
        item["chunk_indexes"].add(int(row["chunk_index"]))
    summaries = [
        SourceSummary(
            doc_id=item["doc_id"],
            title=item["title"],
            source_type=item["source_type"],
            source_uri=item["source_uri"],
            doc_version=item["doc_version"],
            chunk_count=len(item["chunk_indexes"]),
            acl_groups=item["acl_groups"],
            status="ready",
            current=current_versions.get(item["doc_id"]) == item["doc_version"],
            created_at=item["created_at"],
            updated_at=item["updated_at"],
        )
        for item in grouped.values()
    ]
    return sorted(summaries, key=lambda item: (not item.current, item.title, item.doc_id))


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
        if source.doc_id == doc_id and (doc_version is None or source.doc_version == doc_version)
    ]
    if not matches:
        return None
    current = [source for source in matches if source.current]
    return (current or matches)[0]


def delete_source(
    *,
    config: RagConfig,
    tenant_id: str,
    doc_id: str,
    doc_version: int | None = None,
) -> dict[str, object]:
    client = connect(config)
    ensure_collection(client, config, reset=False)
    filter_expr = (
        f"tenant_id == {milvus_string_literal(tenant_id)} "
        f"and doc_id == {milvus_string_literal(doc_id)}"
    )
    if doc_version is not None:
        filter_expr += f" and doc_version == {doc_version}"
    result = client.delete(collection_name=config.collection_name, filter=filter_expr)
    unpublished = unpublish_current_version(
        config.object_store_dir,
        tenant_id=tenant_id,
        doc_id=doc_id,
        doc_version=doc_version,
    )
    tombstoned = archive_delete_tombstone(
        config.object_store_dir,
        tenant_id=tenant_id,
        doc_id=doc_id,
        doc_version=doc_version,
        reason="api_delete_source",
    )
    return {
        "filter": filter_expr,
        "milvus": result,
        "unpublished": unpublished,
        "tombstoned": tombstoned,
    }


def next_doc_version(config: RagConfig, *, tenant_id: str, doc_id: str) -> int:
    current = load_current_versions(config.object_store_dir, tenant_id=tenant_id)
    return int(current.get(doc_id, 0)) + 1


def safe_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    cleaned = "".join(char if char.isalnum() or char in ".-_ ()[]" else "_" for char in name)
    return cleaned or "upload.txt"
