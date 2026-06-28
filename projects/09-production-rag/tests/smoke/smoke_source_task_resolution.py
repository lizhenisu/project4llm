from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core import config as config_module  # noqa: E402
from rag_core.config import load_config  # noqa: E402
from rag_core.sources import (  # noqa: E402
    SourceSummary,
    delete_source_catalog,
    delete_source_task,
    load_source_task_resolution_aliases,
    list_sources,
    save_source_catalog_for_tenant,
    save_source_task_for_tenant,
    save_source_task_resolutions,
)


def main() -> None:
    old_env_loaded = config_module._ENV_LOADED
    old_runtime = os.environ.get("RAG_RUNTIME_DIR")
    old_object_store = os.environ.get("RAG_OBJECT_STORE_DIR")
    database_url = os.environ.get("RAG_METADATA_DATABASE_URL", "")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        os.environ["RAG_RUNTIME_DIR"] = str(root / "runtime")
        os.environ["RAG_OBJECT_STORE_DIR"] = str(root / "object_store")
        config_module._ENV_LOADED = True
        try:
            run_smoke(load_config(), database_url=database_url)
        finally:
            config_module._ENV_LOADED = old_env_loaded
            restore_env("RAG_RUNTIME_DIR", old_runtime)
            restore_env("RAG_OBJECT_STORE_DIR", old_object_store)
    backend = "postgresql" if database_url else "sqlite"
    print(f"smoke_source_task_resolution=ok backend={backend}")


def run_smoke(config, *, database_url: str) -> None:
    suffix = uuid.uuid4().hex[:10]
    tenant_id = f"task-resolution-{suffix}"
    task_id = f"upload-{suffix}"
    ready = SourceSummary(
        doc_id=f"resolved-{suffix}",
        title="Synthetic refresh-safe.pdf",
        source_type="pdf",
        source_uri=f"synthetic://{suffix}/source.pdf",
        doc_version=3,
        chunk_count=12,
        acl_groups=["engineering"],
        status="ready",
        current=True,
        child_doc_ids=[f"resolved-{suffix}/page-1"],
    )
    pending = SourceSummary(
        doc_id=task_id,
        title=ready.title,
        source_type="pdf",
        source_uri=f"/tmp/{ready.title}",
        doc_version=1,
        chunk_count=0,
        acl_groups=["engineering"],
        status="processing",
        current=False,
    )
    try:
        save_source_task_for_tenant(config=config, tenant_id=tenant_id, source=pending)
        save_source_catalog_for_tenant(config=config, tenant_id=tenant_id, sources=[ready])
        save_source_task_resolutions(
            config=config,
            tenant_id=tenant_id,
            task_id=task_id,
            sources=[ready],
        )
        assert delete_source_task(config=config, tenant_id=tenant_id, task_id=task_id)

        listed = list_sources(config=config, tenant_id=tenant_id)
        assert len(listed) == 1
        assert listed[0].doc_id == ready.doc_id
        assert listed[0].workspace_alias_ids == [task_id]

        delete_source_catalog(
            config=config,
            tenant_id=tenant_id,
            doc_id=ready.doc_id,
            doc_version=ready.doc_version,
            child_doc_ids=ready.child_doc_ids,
        )
        assert load_source_task_resolution_aliases(config=config, tenant_id=tenant_id) == {}
    finally:
        delete_source_task(config=config, tenant_id=tenant_id, task_id=task_id)
        delete_source_catalog(
            config=config,
            tenant_id=tenant_id,
            doc_id=ready.doc_id,
            doc_version=None,
            child_doc_ids=ready.child_doc_ids,
        )


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
