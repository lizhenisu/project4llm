from __future__ import annotations

import argparse

from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model
from rag_core.milvus_store import build_filter_expr, connect, dense_search, ensure_collection


def main() -> None:
    parser = argparse.ArgumentParser(description="Run metadata-filtered dense search.")
    parser.add_argument("query", help="User query.")
    parser.add_argument("--tenant-id", default="team_a")
    parser.add_argument(
        "--acl-group",
        action="append",
        default=[],
        help="Allowed ACL group. Repeat to allow multiple groups.",
    )
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=False)
    model = build_embedding_model(config)
    query_vector = model.encode([args.query])[0]
    filter_expr = build_filter_expr(
        tenant_id=args.tenant_id,
        allowed_acl_groups=args.acl_group or None,
    )
    hits = dense_search(
        client,
        collection_name=config.collection_name,
        query_vector=query_vector,
        filter_expr=filter_expr,
        limit=args.limit,
    )

    print(f"filter: {filter_expr}")
    for rank, hit in enumerate(hits, start=1):
        print(
            f"{rank}. score={hit.score:.4f} doc={hit.doc_id} "
            f"chunk={hit.chunk_index} title={hit.title}"
        )
        print(hit.text[:240].replace("\n", " "))


if __name__ == "__main__":
    main()
