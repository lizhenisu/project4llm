from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from rag_core.config import DATA_DIR, load_config
from rag_core.embeddings import build_embedding_model, build_image_embedding_model
from rag_core.io import load_source_documents
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.object_store import archive_source_documents
from rag_core.pii import apply_pii_policy
from rag_core.text_utils import chunk_document, tokenize
from rag_core.types import SourceDocument
from rag_core.versioning import publish_current_versions


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest text documents into Milvus.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "sample_docs.jsonl",
        help="JSONL file of SourceDocument rows.",
    )
    parser.add_argument(
        "--no-publish-current",
        action="store_true",
        help="Archive and index documents without changing current-version registry.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print how documents become chunks before upsert.",
    )
    parser.add_argument(
        "--preview-chunks",
        type=int,
        default=3,
        help="How many chunk previews to print when --explain is enabled.",
    )
    parser.add_argument("--reset", action="store_true", help="Drop and recreate collection first.")
    args = parser.parse_args()

    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=args.reset or config.reset_collection)

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
        for doc in load_source_documents(args.input)
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
    if args.explain:
        chunk_counts = Counter(chunk.doc_id for chunk in chunks)
        print("Chunk plan:")
        for doc in docs:
            print(
                f"- doc_id={doc.doc_id} title={doc.title} "
                f"tokens={len(tokenize(doc.text))} chunks={chunk_counts.get(doc.doc_id, 0)} "
                f"acl={','.join(doc.acl_groups)}"
            )
        for chunk in chunks[: args.preview_chunks]:
            preview = chunk.text[:220].replace("\n", " ")
            print(
                f"  chunk doc={chunk.doc_id} idx={chunk.chunk_index} "
                f"tokens={len(tokenize(chunk.text))} preview={preview}"
            )

    embedding_model = build_embedding_model(config)
    image_embedding_model = build_image_embedding_model(config)
    dense_vectors = embedding_model.encode([chunk.text for chunk in chunks])
    zero_image = image_embedding_model.encode(["no image"])[0]

    entities = [
        chunk_to_entity(
            chunk,
            dense_vector=dense_vector,
            image_vector=zero_image,
            embedding_model=embedding_model.model_name,
            embedding_dim=embedding_model.dim,
        )
        for chunk, dense_vector in zip(chunks, dense_vectors, strict=True)
    ]

    count = upsert_entities(
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
    print(f"Loaded documents: {len(docs)}")
    print(f"Upserted chunks: {count}")
    print(f"Archived canonical docs: {archived}")
    if not args.no_publish_current:
        print(f"Published current versions: {current_versions}")
    print(f"Collection: {config.collection_name}")


if __name__ == "__main__":
    main()
