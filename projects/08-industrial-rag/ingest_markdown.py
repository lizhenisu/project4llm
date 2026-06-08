from __future__ import annotations

import argparse
from pathlib import Path

from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, zero_image_vector
from rag_core.io import load_markdown_section_documents
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.object_store import archive_source_documents
from rag_core.pii import apply_pii_policy
from rag_core.text_utils import chunk_document
from rag_core.types import SourceDocument
from rag_core.versioning import publish_current_versions


def load_markdown_docs(
    input_dir: Path,
    *,
    tenant_id: str,
    doc_version: int,
    acl_groups: list[str],
) -> list[SourceDocument]:
    docs: list[SourceDocument] = []
    for path in sorted(input_dir.rglob("*.md")):
        docs.extend(
            load_markdown_section_documents(
                path,
                input_dir=input_dir,
                tenant_id=tenant_id,
                doc_version=doc_version,
                acl_groups=acl_groups,
                language="zh",
            )
        )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Markdown files from a directory.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--doc-version", type=int, default=1)
    parser.add_argument(
        "--no-publish-current",
        action="store_true",
        help="Archive and index documents without changing current-version registry.",
    )
    parser.add_argument(
        "--acl-group",
        action="append",
        default=[],
        help="ACL group allowed to retrieve these docs. Repeat for multiple groups.",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")

    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=False)

    docs = load_markdown_docs(
        args.input_dir,
        tenant_id=args.tenant_id,
        doc_version=args.doc_version,
        acl_groups=args.acl_group or ["default"],
    )
    docs = [
        SourceDocument(
            **{
                **doc.__dict__,
                "text": apply_pii_policy(
                    doc.text,
                    policy=config.pii_policy,
                    label=f"{doc.doc_id}:text",
                ),
            }
        )
        for doc in docs
    ]
    text_model = build_embedding_model(config)
    chunks = [
        chunk
        for doc in docs
        for chunk in chunk_document(
            doc,
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
            token_counter=text_model.count_tokens,
        )
    ]

    dense_vectors = text_model.encode([chunk.text for chunk in chunks])
    zero_image = zero_image_vector(config)
    entities = [
        chunk_to_entity(
            chunk,
            dense_vector=dense_vector,
            image_vector=zero_image,
            embedding_model=text_model.model_name,
            embedding_dim=text_model.dim,
        )
        for chunk, dense_vector in zip(chunks, dense_vectors, strict=True)
    ]
    upserted = upsert_entities(
        client,
        collection_name=config.collection_name,
        entities=entities,
    )
    archived = archive_source_documents(config.object_store_dir, docs)
    current_versions = (
        {}
        if args.no_publish_current
        else publish_current_versions(config.object_store_dir, docs)
    )
    print(f"Loaded markdown docs: {len(docs)}")
    print(f"Upserted chunks: {upserted}")
    print(f"Archived canonical docs: {archived}")
    if not args.no_publish_current:
        print(f"Published current versions: {current_versions}")


if __name__ == "__main__":
    main()
