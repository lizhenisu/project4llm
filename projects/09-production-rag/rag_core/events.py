from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_core.pii import redact_pii
from rag_core.types import SearchHit


LOGGER = logging.getLogger(__name__)
DEFAULT_EVENT_MAX_JSON_BYTES = 64 * 1024
DEFAULT_EVENT_MAX_STRING_CHARS = 4096
DEFAULT_EVENT_MAX_LIST_ITEMS = 100
DEFAULT_EVENT_MAX_DICT_ITEMS = 200
FALLBACK_EVENT_KEYS = {
    "ts",
    "request_id",
    "query",
    "query_mode",
    "history_len",
    "auth_context",
    "tenant_id",
    "doc_version",
    "doc_ids",
    "source_types",
    "rating",
    "trace",
    "final_context",
    "llm",
}


def append_event(runtime_dir: Path, stream: str, payload: dict[str, Any]) -> None:
    event = {
        "ts": datetime.now(UTC).isoformat(),
        **payload,
    }
    event = redact_event(event)
    try:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        path = runtime_dir / f"{stream}.jsonl"
        with path.open("a", encoding="utf-8") as file:
            file.write(serialize_event(event) + "\n")
    except OSError as exc:
        LOGGER.warning("Failed to append %s event under %s: %s", stream, runtime_dir, exc)


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


def event_log_limits_snapshot() -> dict[str, int]:
    return {
        "max_json_bytes": event_max_json_bytes(),
        "max_string_chars": event_max_string_chars(),
        "max_list_items": event_max_list_items(),
        "max_dict_items": event_max_dict_items(),
    }


def serialize_event(event: dict[str, Any]) -> str:
    max_json_bytes = event_max_json_bytes()
    compacted = compact_event(event)
    text = json.dumps(_jsonable(compacted), ensure_ascii=False)
    encoded_len = len(text.encode("utf-8"))
    if encoded_len <= max_json_bytes:
        return text

    original_json_bytes = len(json.dumps(_jsonable(event), ensure_ascii=False).encode("utf-8"))
    fallback = {
        key: value
        for key, value in event.items()
        if key in FALLBACK_EVENT_KEYS
    }
    fallback["_event_truncated"] = True
    fallback["_original_json_bytes"] = original_json_bytes
    fallback = compact_event(
        fallback,
        max_string_chars=min(event_max_string_chars(), 512),
        max_list_items=min(event_max_list_items(), 20),
        max_dict_items=min(event_max_dict_items(), 50),
    )
    return json.dumps(_jsonable(fallback), ensure_ascii=False)


def compact_event(
    value,
    *,
    max_string_chars: int | None = None,
    max_list_items: int | None = None,
    max_dict_items: int | None = None,
):
    string_limit = event_max_string_chars() if max_string_chars is None else max_string_chars
    list_limit = event_max_list_items() if max_list_items is None else max_list_items
    dict_limit = event_max_dict_items() if max_dict_items is None else max_dict_items

    if is_dataclass(value):
        return compact_event(
            asdict(value),
            max_string_chars=string_limit,
            max_list_items=list_limit,
            max_dict_items=dict_limit,
        )
    if isinstance(value, dict):
        compacted = {}
        items = list(value.items())
        for key, item in items[:dict_limit]:
            compacted[str(key)] = compact_event(
                item,
                max_string_chars=string_limit,
                max_list_items=list_limit,
                max_dict_items=dict_limit,
            )
        if len(items) > dict_limit:
            compacted["_truncated_items"] = len(items) - dict_limit
        return compacted
    if isinstance(value, list | tuple):
        compacted = [
            compact_event(
                item,
                max_string_chars=string_limit,
                max_list_items=list_limit,
                max_dict_items=dict_limit,
            )
            for item in value[:list_limit]
        ]
        if len(value) > list_limit:
            compacted.append({"_truncated_items": len(value) - list_limit})
        return compacted
    if isinstance(value, str) and len(value) > string_limit:
        return f"{value[:string_limit]}...<truncated chars={len(value) - string_limit}>"
    return value


def event_max_json_bytes() -> int:
    return _env_int("RAG_EVENT_MAX_JSON_BYTES", DEFAULT_EVENT_MAX_JSON_BYTES)


def event_max_string_chars() -> int:
    return _env_int("RAG_EVENT_MAX_STRING_CHARS", DEFAULT_EVENT_MAX_STRING_CHARS)


def event_max_list_items() -> int:
    return _env_int("RAG_EVENT_MAX_LIST_ITEMS", DEFAULT_EVENT_MAX_LIST_ITEMS)


def event_max_dict_items() -> int:
    return _env_int("RAG_EVENT_MAX_DICT_ITEMS", DEFAULT_EVENT_MAX_DICT_ITEMS)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        LOGGER.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value
