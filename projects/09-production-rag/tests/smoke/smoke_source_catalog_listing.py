from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core import config as config_module  # noqa: E402
from rag_core.config import load_config  # noqa: E402
from rag_core.sources import SourceSummary, list_sources, save_source_catalog_for_tenant  # noqa: E402


def main() -> None:
    old_env_loaded = config_module._ENV_LOADED
    old_env = {
        "RAG_RUNTIME_DIR": os.environ.get("RAG_RUNTIME_DIR"),
        "RAG_OBJECT_STORE_DIR": os.environ.get("RAG_OBJECT_STORE_DIR"),
        "RAG_METADATA_DATABASE_URL": os.environ.get("RAG_METADATA_DATABASE_URL"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        os.environ["RAG_RUNTIME_DIR"] = str(root / "runtime")
        os.environ["RAG_OBJECT_STORE_DIR"] = str(root / "object_store")
        os.environ["RAG_METADATA_DATABASE_URL"] = ""
        config_module._ENV_LOADED = True
        try:
            config = load_config()
            tenant_id = "tenant-source-catalog-smoke"
            sources = [
                SourceSummary(
                    doc_id=f"doc-{index:03d}",
                    title=f"Document {index:03d}.pdf",
                    source_type="pdf",
                    source_uri=f"memory://doc-{index:03d}",
                    doc_version=1,
                    chunk_count=25,
                    acl_groups=["engineering"],
                    status="ready",
                    current=True,
                    created_at=index,
                    updated_at=index,
                    child_doc_ids=[],
                )
                for index in range(100)
            ]
            save_source_catalog_for_tenant(config=config, tenant_id=tenant_id, sources=sources)
            with patch("rag_core.sources.connect", side_effect=AssertionError("Milvus should not be queried")):
                listed = list_sources(config=config, tenant_id=tenant_id)
            assert len(listed) == 100
            assert all(source.status == "ready" for source in listed)
            assert sum(source.chunk_count for source in listed) == 2500
        finally:
            config_module._ENV_LOADED = old_env_loaded
            for name, value in old_env.items():
                restore_env(name, value)
    print("smoke_source_catalog_listing=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
