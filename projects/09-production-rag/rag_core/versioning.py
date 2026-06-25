from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Iterable

from rag_core.config import RagConfig
from rag_core.database import connect_metadata_db
from rag_core.text_utils import now_ms
from rag_core.types import SourceDocument


CURRENT_VERSIONS_PATH = Path("current_versions.json")
CURRENT_VERSIONS_LOCK = threading.Lock()
CURRENT_VERSIONS_MIGRATION_PREFIX = "current_versions_migrated:"


def publish_current_versions(
    object_store_dir: Path,
    docs: Iterable[SourceDocument],
    *,
    config: RagConfig | None = None,
) -> dict[str, int]:
    docs = list(docs)
    if config is not None:
        return publish_current_versions_db(config, docs)
    with CURRENT_VERSIONS_LOCK:
        current = load_all_current_versions(object_store_dir)
        for doc in docs:
            tenant_versions = current.setdefault(doc.tenant_id, {})
            tenant_versions[doc.doc_id] = max(
                int(doc.doc_version),
                int(tenant_versions.get(doc.doc_id, doc.doc_version)),
            )
        save_current_versions(object_store_dir, current)
        return current


def load_current_versions(object_store_dir: Path, *, tenant_id: str, config: RagConfig | None = None) -> dict[str, int]:
    if config is not None:
        return load_current_versions_db(config, tenant_id=tenant_id)
    return load_all_current_versions(object_store_dir).get(tenant_id, {})


def unpublish_current_version(
    object_store_dir: Path,
    *,
    tenant_id: str,
    doc_id: str,
    doc_version: int | None = None,
    config: RagConfig | None = None,
) -> bool:
    if config is not None:
        return unpublish_current_version_db(
            config,
            tenant_id=tenant_id,
            doc_id=doc_id,
            doc_version=doc_version,
        )
    with CURRENT_VERSIONS_LOCK:
        current = load_all_current_versions(object_store_dir)
        tenant_versions = current.get(tenant_id)
        if not tenant_versions or doc_id not in tenant_versions:
            return False
        if doc_version is not None and int(tenant_versions[doc_id]) != int(doc_version):
            return False

        del tenant_versions[doc_id]
        if not tenant_versions:
            del current[tenant_id]
        save_current_versions(object_store_dir, current)
        return True


def load_all_current_versions(object_store_dir: Path) -> dict[str, dict[str, int]]:
    path = object_store_dir / CURRENT_VERSIONS_PATH
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    raw = parse_current_versions_json(text)
    return {
        str(tenant_id): {str(doc_id): int(version) for doc_id, version in versions.items()}
        for tenant_id, versions in raw.items()
    }


def save_current_versions(
    object_store_dir: Path,
    current: dict[str, dict[str, int]],
) -> None:
    object_store_dir.mkdir(parents=True, exist_ok=True)
    path = object_store_dir / CURRENT_VERSIONS_PATH
    payload = json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(payload, encoding="utf-8")
    temp_path.replace(path)


def parse_current_versions_json(text: str) -> dict:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        raw, _ = decoder.raw_decode(text)
    if not isinstance(raw, dict):
        return {}
    return raw


def publish_current_versions_db(config: RagConfig, docs: Iterable[SourceDocument]) -> dict[str, int]:
    docs = list(docs)
    if not docs:
        return {}
    timestamp = now_ms()
    with connect_metadata_db(config) as conn:
        tenant_ids = set()
        for doc in docs:
            tenant_ids.add(doc.tenant_id)
            conn.execute(
                """
                INSERT INTO current_source_versions(tenant_id, doc_id, doc_version, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(tenant_id, doc_id) DO UPDATE SET
                    doc_version = max(current_source_versions.doc_version, excluded.doc_version),
                    updated_at = excluded.updated_at
                """,
                (doc.tenant_id, doc.doc_id, int(doc.doc_version), timestamp),
            )
        for tenant_id in tenant_ids:
            mark_current_versions_migrated(conn, tenant_id=tenant_id)
    first_tenant = docs[0].tenant_id
    return load_current_versions_db(config, tenant_id=first_tenant)


def load_current_versions_db(config: RagConfig, *, tenant_id: str) -> dict[str, int]:
    with connect_metadata_db(config) as conn:
        rows = conn.execute(
            """
            SELECT doc_id, doc_version
            FROM current_source_versions
            WHERE tenant_id = ?
            """,
            (tenant_id,),
        ).fetchall()
        already_migrated = current_versions_migration_is_done(conn, tenant_id=tenant_id)
    current = {str(row["doc_id"]): int(row["doc_version"]) for row in rows}
    if current:
        return current
    if already_migrated:
        return {}

    legacy_current = load_all_current_versions(config.object_store_dir).get(tenant_id, {})
    if not legacy_current:
        return {}

    timestamp = now_ms()
    with connect_metadata_db(config) as conn:
        for doc_id, doc_version in legacy_current.items():
            conn.execute(
                """
                INSERT INTO current_source_versions(tenant_id, doc_id, doc_version, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(tenant_id, doc_id) DO UPDATE SET
                    doc_version = excluded.doc_version,
                    updated_at = excluded.updated_at
                """,
                (tenant_id, doc_id, int(doc_version), timestamp),
            )
        mark_current_versions_migrated(conn, tenant_id=tenant_id)
    return {str(doc_id): int(doc_version) for doc_id, doc_version in legacy_current.items()}


def unpublish_current_version_db(
    config: RagConfig,
    *,
    tenant_id: str,
    doc_id: str,
    doc_version: int | None = None,
) -> bool:
    load_current_versions_db(config, tenant_id=tenant_id)
    with connect_metadata_db(config) as conn:
        if doc_version is None:
            cursor = conn.execute(
                "DELETE FROM current_source_versions WHERE tenant_id = ? AND doc_id = ?",
                (tenant_id, doc_id),
            )
        else:
            cursor = conn.execute(
                """
                DELETE FROM current_source_versions
                WHERE tenant_id = ? AND doc_id = ? AND doc_version = ?
                """,
                (tenant_id, doc_id, int(doc_version)),
            )
        return cursor.rowcount > 0


def current_versions_migration_key(*, tenant_id: str) -> str:
    return f"{CURRENT_VERSIONS_MIGRATION_PREFIX}{tenant_id}"


def current_versions_migration_is_done(conn, *, tenant_id: str) -> bool:
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = ?",
        (current_versions_migration_key(tenant_id=tenant_id),),
    ).fetchone()
    return row is not None and str(row["value"]) == "1"


def mark_current_versions_migrated(conn, *, tenant_id: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, '1')",
        (current_versions_migration_key(tenant_id=tenant_id),),
    )
