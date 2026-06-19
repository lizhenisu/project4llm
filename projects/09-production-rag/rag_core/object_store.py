from __future__ import annotations

import shutil
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from rag_core.io import read_jsonl, write_jsonl
from rag_core.text_utils import now_ms
from rag_core.types import SourceDocument


SOURCE_DOCUMENTS_PATH = Path("canonical/source_documents.jsonl")
DELETE_TOMBSTONES_PATH = Path("canonical/deleted_documents.jsonl")
OBJECT_STORE_INDEX_LOCK = threading.RLock()


def archive_source_documents(
    object_store_dir: Path,
    docs: Iterable[SourceDocument],
    *,
    replace: bool = False,
) -> int:
    path = object_store_dir / SOURCE_DOCUMENTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(doc) for doc in docs]
    if not rows:
        return 0

    with OBJECT_STORE_INDEX_LOCK:
        if replace or not path.exists():
            existing: list[dict] = []
        else:
            existing = read_jsonl(path)

        merged = {
            document_key(row): row
            for row in [*existing, *rows]
        }
        write_jsonl(path, merged.values())
        remove_delete_tombstones_for_rows(object_store_dir, rows)
    return len(rows)


def load_archived_source_documents(
    object_store_dir: Path,
    *,
    include_deleted: bool = False,
) -> list[SourceDocument]:
    path = object_store_dir / SOURCE_DOCUMENTS_PATH
    if not path.exists():
        return []
    rows = read_jsonl(path)
    if not include_deleted:
        tombstones = load_delete_tombstones(object_store_dir)
        rows = [row for row in rows if not is_deleted(row, tombstones)]
    return [SourceDocument(**row) for row in rows]


def purge_source_documents(
    object_store_dir: Path,
    *,
    tenant_id: str,
    doc_ids: Iterable[str],
    doc_version: int | None = None,
) -> dict[str, int]:
    target_doc_ids = {str(doc_id) for doc_id in doc_ids}
    if not target_doc_ids:
        return {"archived_documents": 0, "delete_tombstones": 0, "upload_dirs": 0}

    with OBJECT_STORE_INDEX_LOCK:
        source_path = object_store_dir / SOURCE_DOCUMENTS_PATH
        source_rows = read_jsonl(source_path) if source_path.exists() else []
        kept_rows: list[dict] = []
        removed_rows: list[dict] = []
        for row in source_rows:
            if row_matches_source(row, tenant_id=tenant_id, doc_ids=target_doc_ids, doc_version=doc_version):
                removed_rows.append(row)
            else:
                kept_rows.append(row)
        if removed_rows:
            write_jsonl(source_path, kept_rows)

        tombstone_path = object_store_dir / DELETE_TOMBSTONES_PATH
        tombstone_removed = 0
        if tombstone_path.exists():
            tombstones = read_jsonl(tombstone_path)
            kept_tombstones = [
                row
                for row in tombstones
                if not row_matches_source(row, tenant_id=tenant_id, doc_ids=target_doc_ids, doc_version=doc_version)
            ]
            tombstone_removed = len(tombstones) - len(kept_tombstones)
            if tombstone_removed:
                write_jsonl(tombstone_path, kept_tombstones)

    upload_dirs = purge_upload_dirs_for_rows(
        object_store_dir,
        tenant_id=tenant_id,
        rows=removed_rows,
    )
    return {
        "archived_documents": len(removed_rows),
        "delete_tombstones": tombstone_removed,
        "upload_dirs": upload_dirs,
    }


def archive_delete_tombstone(
    object_store_dir: Path,
    *,
    tenant_id: str,
    doc_id: str,
    doc_version: int | None = None,
    reason: str = "delete_document",
) -> int:
    path = object_store_dir / DELETE_TOMBSTONES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with OBJECT_STORE_INDEX_LOCK:
        rows = read_jsonl(path) if path.exists() else []
        tombstone = {
            "tenant_id": tenant_id,
            "doc_id": doc_id,
            "doc_version": doc_version,
            "reason": reason,
            "deleted_at": now_ms(),
        }
        merged = {
            tombstone_key(row): row
            for row in [*rows, tombstone]
        }
        write_jsonl(path, merged.values())
    return 1


def load_delete_tombstones(object_store_dir: Path) -> list[dict]:
    path = object_store_dir / DELETE_TOMBSTONES_PATH
    if not path.exists():
        return []
    return read_jsonl(path)


def remove_delete_tombstones_for_rows(object_store_dir: Path, rows: list[dict]) -> int:
    path = object_store_dir / DELETE_TOMBSTONES_PATH
    if not path.exists() or not rows:
        return 0
    with OBJECT_STORE_INDEX_LOCK:
        tombstones = read_jsonl(path)
        remaining = [
            tombstone
            for tombstone in tombstones
            if not any(row_restores_tombstone(row, tombstone) for row in rows)
        ]
        removed = len(tombstones) - len(remaining)
        if removed:
            write_jsonl(path, remaining)
        return removed


def is_deleted(row: dict, tombstones: list[dict]) -> bool:
    return any(tombstone_deletes_row(tombstone, row) for tombstone in tombstones)


def tombstone_deletes_row(tombstone: dict, row: dict) -> bool:
    if str(tombstone["tenant_id"]) != str(row["tenant_id"]):
        return False
    if str(tombstone["doc_id"]) != str(row["doc_id"]):
        return False
    version = tombstone.get("doc_version")
    return version is None or int(version) == int(row["doc_version"])


def row_matches_source(
    row: dict,
    *,
    tenant_id: str,
    doc_ids: set[str],
    doc_version: int | None,
) -> bool:
    if str(row.get("tenant_id")) != str(tenant_id):
        return False
    if str(row.get("doc_id")) not in doc_ids:
        return False
    version = row.get("doc_version")
    return doc_version is None or version is None or int(version) == int(doc_version)


def row_restores_tombstone(row: dict, tombstone: dict) -> bool:
    if str(tombstone["tenant_id"]) != str(row["tenant_id"]):
        return False
    if str(tombstone["doc_id"]) != str(row["doc_id"]):
        return False
    version = tombstone.get("doc_version")
    return version is None or int(version) == int(row["doc_version"])


def document_key(row: dict) -> tuple[str, str, int]:
    return (
        str(row["tenant_id"]),
        str(row["doc_id"]),
        int(row["doc_version"]),
    )


def tombstone_key(row: dict) -> tuple[str, str, int | None]:
    version = row.get("doc_version")
    return (
        str(row["tenant_id"]),
        str(row["doc_id"]),
        None if version is None else int(version),
    )


def purge_upload_dirs_for_rows(
    object_store_dir: Path,
    *,
    tenant_id: str,
    rows: list[dict],
) -> int:
    upload_dirs = {
        upload_dir
        for row in rows
        for upload_dir in upload_dirs_for_row(object_store_dir, tenant_id=tenant_id, row=row)
    }
    removed = 0
    for upload_dir in upload_dirs:
        if upload_dir.exists() and upload_dir.is_dir():
            shutil.rmtree(upload_dir)
            removed += 1
    return removed


def upload_dirs_for_row(object_store_dir: Path, *, tenant_id: str, row: dict) -> set[Path]:
    candidates = [
        str(row.get("source_uri") or ""),
        str((row.get("metadata") or {}).get("linked_source_uri") or ""),
        str((row.get("metadata") or {}).get("image_uri") or ""),
    ]
    return {
        upload_dir
        for candidate in candidates
        if candidate
        for upload_dir in upload_dirs_for_path(
            object_store_dir,
            tenant_id=tenant_id,
            path=Path(candidate),
        )
    }


def upload_dirs_for_path(object_store_dir: Path, *, tenant_id: str, path: Path) -> set[Path]:
    uploads_root = (object_store_dir / "uploads" / tenant_id).resolve()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(object_store_dir / path)
    upload_dirs: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.is_relative_to(uploads_root):
            continue
        for parent in [resolved, *resolved.parents]:
            if parent.parent == uploads_root:
                upload_dirs.add(parent)
                break
    return upload_dirs
