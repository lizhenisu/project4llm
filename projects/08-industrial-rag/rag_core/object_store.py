from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from rag_core.io import read_jsonl, write_jsonl
from rag_core.text_utils import now_ms
from rag_core.types import SourceDocument


SOURCE_DOCUMENTS_PATH = Path("canonical/source_documents.jsonl")
DELETE_TOMBSTONES_PATH = Path("canonical/deleted_documents.jsonl")


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


def row_restores_tombstone(row: dict, tombstone: dict) -> bool:
    if str(tombstone["tenant_id"]) != str(row["tenant_id"]):
        return False
    if str(tombstone["doc_id"]) != str(row["doc_id"]):
        return False
    version = tombstone.get("doc_version")
    return version is None or int(version) == int(row["doc_version"])


def document_key(row: dict) -> tuple[str, str, int, str]:
    return (
        str(row["tenant_id"]),
        str(row["doc_id"]),
        int(row["doc_version"]),
        str(row["source_uri"]),
    )


def tombstone_key(row: dict) -> tuple[str, str, int | None]:
    version = row.get("doc_version")
    return (
        str(row["tenant_id"]),
        str(row["doc_id"]),
        None if version is None else int(version),
    )
