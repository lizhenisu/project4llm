from __future__ import annotations

import argparse

from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model
from rag_core.milvus_store import build_filter_expr, connect, ensure_collection, hybrid_search
from rag_core.text_utils import sparse_embedding


def run_hybrid(
    query: str,
    *,
    tenant_id: str,
    limit: int,
    acl_groups: list[str] | None = None,
    doc_version: int | None = None,
    source_types: list[str] | None = None,
):
    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=False)
    model = build_embedding_model(config)
    query_vector = model.encode([query])[0]
    query_sparse = sparse_embedding(query)
    filter_expr = build_filter_expr(
        tenant_id=tenant_id,
        allowed_acl_groups=acl_groups,
        doc_version=doc_version,
        embedding_model=model.model_name,
        source_types=source_types,
    )
    return hybrid_search(
        client,
        collection_name=config.collection_name,
        query_vector=query_vector,
        query_sparse=query_sparse,
        filter_expr=filter_expr,
        limit=limit,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Milvus hybrid dense+sparse search.")
    parser.add_argument("query", help="User query.")
    parser.add_argument("--tenant-id", default="team_a")
    parser.add_argument(
        "--acl-group",
        action="append",
        default=[],
        help="Allowed ACL group. Repeat to allow multiple groups.",
    )
    parser.add_argument("--doc-version", type=int)
    parser.add_argument(
        "--source-type",
        action="append",
        default=[],
        help="Restrict retrieval to a source type. Repeat for multiple types.",
    )
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    hits = run_hybrid(
        args.query,
        tenant_id=args.tenant_id,
        limit=args.limit,
        acl_groups=args.acl_group or None,
        doc_version=args.doc_version,
        source_types=args.source_type or None,
    )
    for rank, hit in enumerate(hits, start=1):
        print(
            f"{rank}. score={hit.score:.4f} doc={hit.doc_id} "
            f"chunk={hit.chunk_index} title={hit.title}"
        )
        print(hit.text[:240].replace("\n", " "))


if __name__ == "__main__":
    main()
