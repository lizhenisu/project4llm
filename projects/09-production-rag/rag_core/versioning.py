from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from rag_core.types import SourceDocument


CURRENT_VERSIONS_PATH = Path("current_versions.json")


def publish_current_versions(
    object_store_dir: Path,
    docs: Iterable[SourceDocument],
) -> dict[str, int]:
    current = load_all_current_versions(object_store_dir)
    for doc in docs:
        tenant_versions = current.setdefault(doc.tenant_id, {})
        tenant_versions[doc.doc_id] = max(
            int(doc.doc_version),
            int(tenant_versions.get(doc.doc_id, doc.doc_version)),
        )
    save_current_versions(object_store_dir, current)
    return current


def load_current_versions(object_store_dir: Path, *, tenant_id: str) -> dict[str, int]:
    return load_all_current_versions(object_store_dir).get(tenant_id, {})


def unpublish_current_version(
    object_store_dir: Path,
    *,
    tenant_id: str,
    doc_id: str,
    doc_version: int | None = None,
) -> bool:
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
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
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
    with path.open("w", encoding="utf-8") as file:
        json.dump(current, file, ensure_ascii=False, indent=2, sort_keys=True)
