from __future__ import annotations

import argparse
from collections import defaultdict

from rag_core.config import load_config
from rag_core.milvus_store import connect, ensure_collection
from rag_core.versioning import load_current_versions


def main() -> None:
    parser = argparse.ArgumentParser(description="List documents currently indexed in Milvus.")
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--doc-version", type=int)
    args = parser.parse_args()

    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=False)
    current_versions = load_current_versions(config.object_store_dir, tenant_id=args.tenant_id)

    filter_expr = f'tenant_id == "{args.tenant_id}" and is_active == true'
    if args.doc_version is not None:
        filter_expr += f" and doc_version == {args.doc_version}"

    rows = client.query(
        collection_name=config.collection_name,
        filter=filter_expr,
        output_fields=[
            "doc_id",
            "doc_version",
            "title",
            "source_type",
            "source_uri",
            "chunk_index",
            "acl_groups",
        ],
        limit=10_000,
    )
    docs = defaultdict(lambda: {"chunks": 0, "chunk_indexes": set()})
    for row in rows:
        key = (row["doc_id"], row["doc_version"])
        docs[key].update(
            {
                "doc_id": row["doc_id"],
                "doc_version": row["doc_version"],
                "title": row["title"],
                "source_type": row["source_type"],
                "source_uri": row["source_uri"],
                "acl_groups": row.get("acl_groups") or [],
            }
        )
        docs[key]["chunks"] += 1
        docs[key]["chunk_indexes"].add(row["chunk_index"])

    print(f"filter: {filter_expr}")
    for item in sorted(docs.values(), key=lambda doc: (doc["doc_id"], doc["doc_version"])):
        print(
            f"doc_id={item['doc_id']} version={item['doc_version']} "
            f"current={current_versions.get(item['doc_id']) == item['doc_version']} "
            f"chunks={item['chunks']} source={item['source_type']} "
            f"acl={','.join(item['acl_groups'])} title={item['title']}"
        )


if __name__ == "__main__":
    main()
