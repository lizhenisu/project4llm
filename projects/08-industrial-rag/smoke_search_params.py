from __future__ import annotations

import os
from typing import Any

from rag_core.config import load_config
from rag_core.milvus_store import (
    create_index_params,
    dense_search,
    dense_search_params,
    image_search,
    image_search_params,
    sparse_search,
    sparse_search_params,
)


class CapturingClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.calls.append(kwargs)
        return [
            [
                {
                    "id": "smoke-hit",
                    "distance": 1.0,
                    "entity": {
                        "tenant_id": "team_a",
                        "doc_id": "search-param-doc",
                        "doc_version": 1,
                        "chunk_index": 0,
                        "source_type": "md",
                        "source_uri": "memory://search-param-doc",
                        "title": "Search params",
                        "text": "验证 HNSW ef 和 sparse drop ratio 会传入 Milvus search。",
                        "language": "zh",
                        "acl_groups": ["ops"],
                        "created_at": 0,
                        "updated_at": 0,
                        "is_active": True,
                        "embedding_model": "BAAI/bge-m3",
                        "embedding_dim": 4,
                        "content_hash": "content-hash",
                        "metadata": {},
                    },
                }
            ]
        ]


def main() -> None:
    old_values = {
        name: os.environ.get(name)
        for name in [
            "RAG_DENSE_HNSW_M",
            "RAG_DENSE_HNSW_EF_CONSTRUCTION",
            "RAG_DENSE_SEARCH_EF",
            "RAG_IMAGE_HNSW_M",
            "RAG_IMAGE_HNSW_EF_CONSTRUCTION",
            "RAG_IMAGE_SEARCH_EF",
            "RAG_SPARSE_DROP_RATIO_BUILD",
            "RAG_SPARSE_DROP_RATIO_SEARCH",
        ]
    }
    os.environ["RAG_DENSE_HNSW_M"] = "24"
    os.environ["RAG_DENSE_HNSW_EF_CONSTRUCTION"] = "180"
    os.environ["RAG_DENSE_SEARCH_EF"] = "96"
    os.environ["RAG_IMAGE_HNSW_M"] = "12"
    os.environ["RAG_IMAGE_HNSW_EF_CONSTRUCTION"] = "140"
    os.environ["RAG_IMAGE_SEARCH_EF"] = "80"
    os.environ["RAG_SPARSE_DROP_RATIO_BUILD"] = "0.15"
    os.environ["RAG_SPARSE_DROP_RATIO_SEARCH"] = "0.05"
    try:
        run_smoke()
    finally:
        for name, value in old_values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_smoke() -> None:
    config = load_config()
    assert dense_search_params(config) == {
        "metric_type": "COSINE",
        "params": {"ef": 96},
    }
    assert sparse_search_params(config) == {
        "metric_type": "BM25",
        "params": {"drop_ratio_search": 0.05},
    }
    assert image_search_params(config) == {
        "metric_type": "COSINE",
        "params": {"ef": 80},
    }
    create_index_params(config)

    client = CapturingClient()
    dense_search(
        client,  # type: ignore[arg-type]
        collection_name="collection",
        query_vector=[0.1, 0.2, 0.3, 0.4],
        filter_expr='tenant_id == "team_a"',
        limit=3,
    )
    sparse_search(
        client,  # type: ignore[arg-type]
        collection_name="collection",
        query_text="search params",
        filter_expr='tenant_id == "team_a"',
        limit=3,
    )
    image_search(
        client,  # type: ignore[arg-type]
        collection_name="collection",
        image_query_vector=[0.1, 0.2, 0.3, 0.4],
        filter_expr='tenant_id == "team_a"',
        limit=3,
    )

    assert client.calls[0]["search_params"]["params"]["ef"] == 96
    assert client.calls[1]["search_params"]["params"]["drop_ratio_search"] == 0.05
    assert client.calls[2]["search_params"]["params"]["ef"] == 80
    print("smoke_search_params=ok")


if __name__ == "__main__":
    main()
