from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from rag_core.types import ImageDocument, SourceDocument


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_source_documents(path: Path) -> list[SourceDocument]:
    return [SourceDocument(**row) for row in read_jsonl(path)]


def load_image_documents(path: Path) -> list[ImageDocument]:
    return [ImageDocument(**row) for row in read_jsonl(path)]

