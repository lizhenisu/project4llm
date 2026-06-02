from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def append_event(runtime_dir: Path, stream: str, payload: dict[str, Any]) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    path = runtime_dir / f"{stream}.jsonl"
    event = {
        "ts": datetime.now(UTC).isoformat(),
        **payload,
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_jsonable(event), ensure_ascii=False) + "\n")


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value

