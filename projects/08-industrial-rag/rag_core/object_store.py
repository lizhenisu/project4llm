from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from rag_core.io import read_jsonl, write_jsonl
from rag_core.types import SourceDocument


SOURCE_DOCUMENTS_PATH = Path("canonical/source_documents.jsonl")


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
    return len(rows)


def load_archived_source_documents(object_store_dir: Path) -> list[SourceDocument]:
    path = object_store_dir / SOURCE_DOCUMENTS_PATH
    if not path.exists():
        return []
    return [SourceDocument(**row) for row in read_jsonl(path)]


def document_key(row: dict) -> tuple[str, str, int, str]:
    return (
        str(row["tenant_id"]),
        str(row["doc_id"]),
        int(row["doc_version"]),
        str(row["source_uri"]),
    )
