from __future__ import annotations

import argparse
from pathlib import Path

from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, build_image_embedding_model
from rag_core.io import load_file_documents
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.object_store import archive_source_documents
from rag_core.pii import apply_pii_policy
from rag_core.text_utils import chunk_document
from rag_core.types import SourceDocument
from rag_core.versioning import publish_current_versions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest PDF/HTML/Markdown/TXT files from a directory."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--doc-version", type=int, default=1)
    parser.add_argument(
        "--no-publish-current",
        action="store_true",
        help="Archive and index documents without changing current-version registry.",
    )
    parser.add_argument("--language", default="zh")
    parser.add_argument(
        "--acl-group",
        action="append",
        default=[],
        help="ACL group allowed to retrieve these docs. Repeat for multiple groups.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only load files directly under input-dir.",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")

    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=False)

    docs = load_file_documents(
        args.input_dir,
        tenant_id=args.tenant_id,
        doc_version=args.doc_version,
        acl_groups=args.acl_group or ["default"],
        language=args.language,
        recursive=not args.no_recursive,
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
    chunks = [
        chunk
        for doc in docs
        for chunk in chunk_document(
            doc,
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
        )
    ]

    text_model = build_embedding_model(config)
    image_model = build_image_embedding_model(config)
    dense_vectors = text_model.encode([chunk.text for chunk in chunks])
    zero_image = image_model.encode(["no image"])[0]
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
    print(f"Loaded file docs: {len(docs)}")
    print(f"Upserted chunks: {upserted}")
    print(f"Archived canonical docs: {archived}")
    if not args.no_publish_current:
        print(f"Published current versions: {current_versions}")


if __name__ == "__main__":
    main()
