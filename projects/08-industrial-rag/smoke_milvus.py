from __future__ import annotations

from rag_core.config import load_config
from rag_core.milvus_store import connect, ensure_collection


def main() -> None:
    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=False)
    collections = client.list_collections()
    print(f"smoke_milvus=ok uri={config.milvus_uri}")
    print(f"collection={config.collection_name} exists={config.collection_name in collections}")
    print(f"collections={collections}")


if __name__ == "__main__":
    main()

