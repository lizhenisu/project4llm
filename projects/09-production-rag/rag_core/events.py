from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_core.pii import redact_pii
from rag_core.types import SearchHit


def append_event(runtime_dir: Path, stream: str, payload: dict[str, Any]) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    path = runtime_dir / f"{stream}.jsonl"
    event = {
        "ts": datetime.now(UTC).isoformat(),
        **payload,
    }
    event = redact_event(event)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_jsonable(event), ensure_ascii=False) + "\n")


def hit_event_summary(hit: SearchHit) -> dict[str, Any]:
    return {
        "id": hit.id,
        "tenant_id": hit.tenant_id,
        "doc_id": hit.doc_id,
        "title": hit.title,
        "source_uri": hit.source_uri,
        "source_type": hit.source_type,
        "chunk_index": hit.chunk_index,
        "score": hit.score,
        "rerank_score": hit.rerank_score,
        "acl_groups": hit.acl_groups,
        "metadata": hit.metadata,
        "text_preview": hit.text[:240],
    }


def hit_event_summaries(hits: list[SearchHit]) -> list[dict[str, Any]]:
    return [hit_event_summary(hit) for hit in hits]


def redact_event(value):
    if is_dataclass(value):
        return redact_event(asdict(value))
    if isinstance(value, dict):
        return {str(key): redact_event(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [redact_event(item) for item in value]
    if isinstance(value, str):
        return redact_pii(value)
    return value


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value
