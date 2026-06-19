from __future__ import annotations

from functools import lru_cache

from rag_core.config import PROJECT_DIR


DEFAULT_APP_VERSION = "0.0.0"


@lru_cache(maxsize=1)
def app_version() -> str:
    version_path = PROJECT_DIR / "VERSION"
    try:
        version = version_path.read_text(encoding="utf-8").strip()
    except OSError:
        return DEFAULT_APP_VERSION
    return version or DEFAULT_APP_VERSION
