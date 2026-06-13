from __future__ import annotations

import argparse

from rag_core.config import load_config
from rag_core.milvus_store import connect, ensure_collection
from rag_core.object_store import archive_delete_tombstone
from rag_core.versioning import unpublish_current_version


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
    parser.add_argument(
        "--keep-current-version",
        action="store_true",
        help="Do not update current_versions.json after deleting chunks.",
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
    unpublished = False
    if not args.keep_current_version:
        unpublished = unpublish_current_version(
            config.object_store_dir,
            tenant_id=args.tenant_id,
            doc_id=args.doc_id,
            doc_version=args.doc_version,
        )
    tombstoned = 0
    tombstoned = archive_delete_tombstone(
        config.object_store_dir,
        tenant_id=args.tenant_id,
        doc_id=args.doc_id,
        doc_version=args.doc_version,
    )
    print(f"delete_filter: {filter_expr}")
    print(f"unpublished_current_version: {unpublished}")
    print(f"archived_delete_tombstones: {tombstoned}")
    print(result)


if __name__ == "__main__":
    main()
