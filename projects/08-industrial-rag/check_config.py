from __future__ import annotations

from dataclasses import asdict

from rag_core.config import load_config
from rag_core.milvus_store import connect


SECRET_KEYS = {"milvus_token", "llm_api_key"}


def main() -> None:
    config = load_config()
    print("RAG config:")
    for key, value in asdict(config).items():
        if key in SECRET_KEYS and value:
            value = "***"
        print(f"- {key}: {value}")

    client = connect(config)
    print(f"Milvus connected: {config.milvus_uri}")
    print(f"Collection exists: {client.has_collection(config.collection_name)}")


if __name__ == "__main__":
    main()

