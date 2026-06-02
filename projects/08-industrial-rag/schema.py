from __future__ import annotations

import argparse

from rag_core.config import load_config
from rag_core.milvus_store import connect, ensure_collection


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the industrial RAG Milvus schema.")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the collection.")
    args = parser.parse_args()

    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=args.reset or config.reset_collection)
    print(f"Collection ready: {config.collection_name}")
    print(f"Milvus URI: {config.milvus_uri}")
    print(f"Dense dim: {config.embedding_dim}; image dim: {config.image_embedding_dim}")


if __name__ == "__main__":
    main()

