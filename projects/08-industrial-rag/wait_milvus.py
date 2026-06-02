from __future__ import annotations

import time

from rag_core.config import load_config
from rag_core.milvus_store import connect


def main() -> None:
    config = load_config()
    deadline = time.monotonic() + 120
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            client = connect(config)
            client.list_collections()
            print(f"milvus_ready=ok uri={config.milvus_uri}")
            return
        except Exception as exc:  # pragma: no cover - depends on external Milvus
            last_error = exc
            print(f"milvus_ready=waiting error={exc}")
            time.sleep(3)
    raise SystemExit(f"Milvus not ready after timeout: {last_error}")


if __name__ == "__main__":
    main()

