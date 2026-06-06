from __future__ import annotations

import argparse
from pathlib import Path

from rag_core.config import load_config
from rag_core.embeddings import build_image_embedding_model
from rag_core.milvus_store import build_filter_expr, connect, ensure_collection, image_search


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run image-vector search against image_dense_vector."
    )
    parser.add_argument(
        "image_query",
        help=(
            "Image path or text query. CLIP encodes image bytes for existing files and "
            "text features otherwise."
        ),
    )
    parser.add_argument("--tenant-id", default="team_a")
    parser.add_argument(
        "--acl-group",
        action="append",
        default=[],
        help="Allowed ACL group. Repeat to allow multiple groups.",
    )
    parser.add_argument("--doc-version", type=int)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=False)
    model = build_image_embedding_model(config)
    query_path = Path(args.image_query)
    if query_path.exists() and query_path.is_file():
        query_vector = model.encode_images([query_path])[0]
    else:
        query_vector = model.encode([args.image_query])[0]
    filter_expr = build_filter_expr(
        tenant_id=args.tenant_id,
        allowed_acl_groups=args.acl_group or None,
        source_types=["image"],
        doc_version=args.doc_version,
    )
    hits = image_search(
        client,
        collection_name=config.collection_name,
        image_query_vector=query_vector,
        filter_expr=filter_expr,
        limit=args.limit,
    )

    print(f"filter: {filter_expr}")
    for rank, hit in enumerate(hits, start=1):
        print(
            f"{rank}. score={hit.score:.4f} doc={hit.doc_id} "
            f"chunk={hit.chunk_index} title={hit.title}"
        )
        print(hit.text[:260].replace("\n", " "))


if __name__ == "__main__":
    main()
