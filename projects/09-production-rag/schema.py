from __future__ import annotations

import argparse

from rag_core.config import load_config
from rag_core.milvus_store import connect, ensure_collection


FIELD_EXPLANATIONS = [
    ("id", "chunk 主键，唯一定位 tenant/doc/version/chunk。"),
    ("tenant_id", "租户过滤主键之一，避免跨租户召回。"),
    ("doc_id", "文档级聚合、版本控制和 citation 的核心标识。"),
    ("doc_version", "支持灰度发布、历史回放和 current-version 过滤。"),
    ("source_type", "区分 md/pdf/html/image/table 等来源，便于 source filter。"),
    ("title", "把标题路径带进 chunk，提升召回和可解释性。"),
    ("text", "原始检索文本字段，启用 analyzer 供 BM25/sparse 检索。"),
    ("acl_groups", "检索阶段直接做 ACL 过滤，而不是召回后再删。"),
    ("text_dense_vector", "dense 语义召回向量。"),
    ("bm25_sparse_vector", "由 Milvus BM25 function 从 text 自动生成的关键词召回向量。"),
    ("image_dense_vector", "多模态扩展字段，为 image search 预留。"),
    ("metadata", "页码、heading_path、bbox、row_range 等结构化定位信息。"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the industrial RAG Milvus schema.")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the collection.")
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print the teaching-oriented schema and index rationale.",
    )
    args = parser.parse_args()

    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=args.reset or config.reset_collection)
    print(f"Collection ready: {config.collection_name}")
    print(f"Milvus URI: {config.milvus_uri}")
    print(f"Dense dim: {config.embedding_dim}; image dim: {config.image_embedding_dim}")
    if args.explain:
        print("\nCore fields:")
        for field, explanation in FIELD_EXPLANATIONS:
            print(f"- {field}: {explanation}")
        print("\nIndexes:")
        print(
            "- text_dense_vector -> HNSW/COSINE "
            f"(M={config.dense_hnsw_m}, efConstruction={config.dense_hnsw_ef_construction})"
        )
        print(
            "- bm25_sparse_vector -> SPARSE_INVERTED_INDEX/BM25 "
            f"(drop_ratio_build={config.sparse_drop_ratio_build})"
        )
        print(
            "- image_dense_vector -> HNSW/COSINE "
            f"(M={config.image_hnsw_m}, efConstruction={config.image_hnsw_ef_construction})"
        )


if __name__ == "__main__":
    main()
