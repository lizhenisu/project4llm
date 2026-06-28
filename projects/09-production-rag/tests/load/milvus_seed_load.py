from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Iterator
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.config import load_config
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.types import Chunk


def main() -> None:
    args = parse_args()
    config = load_config()
    client = connect(config)
    if args.action == "drop":
        existed = client.has_collection(config.collection_name)
        if existed:
            client.drop_collection(config.collection_name)
        print(json.dumps({"collection": config.collection_name, "dropped": existed}))
        return

    started = time.perf_counter()
    ensure_collection(client, config, reset=args.reset)
    inserted = 0
    batch: list[dict] = []
    zero_image_vector = [0.0] * config.image_embedding_dim
    for ordinal, chunk in enumerate(iter_chunks(args)):
        batch.append(
            chunk_to_entity(
                chunk,
                dense_vector=deterministic_vector(ordinal, config.embedding_dim),
                image_vector=zero_image_vector,
                embedding_model=config.embedding_model,
                embedding_dim=config.embedding_dim,
            )
        )
        if len(batch) >= args.batch_size:
            inserted += upsert_entities(
                client,
                collection_name=config.collection_name,
                entities=batch,
                batch_size=args.batch_size,
            )
            batch = []
    if batch:
        inserted += upsert_entities(
            client,
            collection_name=config.collection_name,
            entities=batch,
            batch_size=args.batch_size,
        )
    client.flush(config.collection_name)
    elapsed_s = round(time.perf_counter() - started, 3)
    expected = expected_chunk_count(args)
    if inserted != expected:
        raise RuntimeError(f"Expected {expected} seeded chunks, inserted {inserted}")
    print(
        json.dumps(
            {
                "collection": config.collection_name,
                "tenants": args.tenant_count,
                "documents_per_tenant": args.documents_per_tenant,
                "chunks_per_document": args.chunks_per_document,
                "chunks": inserted,
                "elapsed_s": elapsed_s,
                "chunks_per_second": round(inserted / max(0.001, elapsed_s), 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed or drop an isolated synthetic Milvus collection for /search load tests."
    )
    parser.add_argument("action", choices=("seed", "drop"))
    parser.add_argument("--tenant-id", default="milvus-capacity")
    parser.add_argument("--tenant-count", type=positive_int, default=10)
    parser.add_argument("--documents-per-tenant", type=positive_int, default=100)
    parser.add_argument("--chunks-per-document", type=positive_int, default=10)
    parser.add_argument("--batch-size", type=positive_int, default=256)
    parser.add_argument("--acl-group", default="engineering")
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def iter_chunks(args: argparse.Namespace) -> Iterator[Chunk]:
    for tenant_index in range(args.tenant_count):
        tenant_id = tenant_for_index(args.tenant_id, args.tenant_count, tenant_index)
        for document_index in range(args.documents_per_tenant):
            doc_id = f"synthetic-doc-{document_index:06d}"
            for chunk_index in range(args.chunks_per_document):
                ordinal = (
                    tenant_index * args.documents_per_tenant * args.chunks_per_document
                    + document_index * args.chunks_per_document
                    + chunk_index
                )
                yield Chunk(
                    tenant_id=tenant_id,
                    doc_id=doc_id,
                    doc_version=1,
                    chunk_index=chunk_index,
                    source_type="txt",
                    source_uri=f"synthetic://{tenant_id}/{doc_id}",
                    title=f"Synthetic capacity document {document_index}",
                    text=(
                        f"Synthetic RAG capacity evidence {ordinal}. "
                        "Hybrid dense sparse retrieval embedding reranking context. "
                        f"Tenant {tenant_index} document {document_index} chunk {chunk_index}."
                    ),
                    language="en",
                    acl_groups=[args.acl_group],
                    metadata={
                        "synthetic": True,
                        "tenant_index": tenant_index,
                        "document_index": document_index,
                    },
                )


def deterministic_vector(ordinal: int, dim: int) -> list[float]:
    vector = [0.0] * dim
    for offset, weight in enumerate((1.0, 0.75, 0.5, 0.25)):
        vector[(ordinal * 17 + offset * 193) % dim] = weight
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector]


def tenant_for_index(base_tenant_id: str, tenant_count: int, index: int) -> str:
    if tenant_count <= 1:
        return base_tenant_id
    return f"{base_tenant_id}-{index:04d}"


def expected_chunk_count(args: argparse.Namespace) -> int:
    return args.tenant_count * args.documents_per_tenant * args.chunks_per_document


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


if __name__ == "__main__":
    main()
