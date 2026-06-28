from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from rag_core import config as config_module  # noqa: E402
from rag_core.config import load_config  # noqa: E402


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def main() -> None:
    old_env_loaded = config_module._ENV_LOADED
    old_rag_milvus_uri = os.environ.get("RAG_MILVUS_URI")
    old_milvus_uri = os.environ.get("MILVUS_URI")
    try:
        config_module._ENV_LOADED = True
        os.environ.pop("RAG_MILVUS_URI", None)
        os.environ.pop("MILVUS_URI", None)
        config = load_config()
        expected = PROJECT_DIR / "runtime" / "milvus_lite.db"
        assert config.milvus_uri == str(expected), config.milvus_uri
        assert config.milvus_uri.endswith(".db"), config.milvus_uri
        assert not config.milvus_uri.endswith("production_rag.db"), config.milvus_uri

        os.environ["RAG_MILVUS_URI"] = "http://127.0.0.1:19530"
        config = load_config()
        assert config.milvus_uri == "http://127.0.0.1:19530", config.milvus_uri
    finally:
        config_module._ENV_LOADED = old_env_loaded
        restore_env("RAG_MILVUS_URI", old_rag_milvus_uri)
        restore_env("MILVUS_URI", old_milvus_uri)


if __name__ == "__main__":
    main()
