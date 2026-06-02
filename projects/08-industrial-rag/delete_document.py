from __future__ import annotations

import argparse

from rag_core.config import load_config
from rag_core.milvus_store import connect, ensure_collection


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete all chunks for a document.")
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--doc-version", type=int)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deletion. Required to avoid accidental deletes.",
    )
    args = parser.parse_args()

    if not args.yes:
        raise SystemExit("Refusing to delete without --yes")

    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=False)

    filter_expr = f'tenant_id == "{args.tenant_id}" and doc_id == "{args.doc_id}"'
    if args.doc_version is not None:
        filter_expr += f" and doc_version == {args.doc_version}"

    result = client.delete(
        collection_name=config.collection_name,
        filter=filter_expr,
    )
    print(f"delete_filter: {filter_expr}")
    print(result)


if __name__ == "__main__":
    main()

