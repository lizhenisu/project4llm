from __future__ import annotations

import argparse
from collections import Counter

from rag_core.config import load_config
from rag_core.milvus_store import connect, ensure_collection


def main() -> None:
    parser = argparse.ArgumentParser(description="Print collection-level RAG stats.")
    parser.add_argument("--tenant-id")
    args = parser.parse_args()

    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=False)

    filter_expr = ""
    if args.tenant_id:
        filter_expr = f'tenant_id == "{args.tenant_id}"'

    rows = client.query(
        collection_name=config.collection_name,
        filter=filter_expr,
        output_fields=["tenant_id", "doc_id", "doc_version", "source_type", "is_active"],
        limit=10_000,
    )

    docs = {(row["tenant_id"], row["doc_id"], row["doc_version"]) for row in rows}
    source_counts = Counter(row["source_type"] for row in rows)
    tenant_counts = Counter(row["tenant_id"] for row in rows)
    active_count = sum(1 for row in rows if row.get("is_active"))

    print(f"collection={config.collection_name}")
    print(f"filter={filter_expr or '<none>'}")
    print(f"chunks={len(rows)}")
    print(f"active_chunks={active_count}")
    print(f"documents={len(docs)}")
    print(f"tenants={dict(sorted(tenant_counts.items()))}")
    print(f"sources={dict(sorted(source_counts.items()))}")


if __name__ == "__main__":
    main()

