from __future__ import annotations

import argparse
from pathlib import Path

from rag_core.config import DATA_DIR, load_config
from rag_core.embeddings import build_embedding_model, build_image_embedding_model
from rag_core.io import load_image_documents
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.object_store import archive_source_documents
from rag_core.pii import apply_pii_policy
from rag_core.types import Chunk, SourceDocument
from rag_core.versioning import publish_current_versions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest image metadata through OCR/caption text plus image vectors."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "sample_images.jsonl",
        help="JSONL file of ImageDocument rows.",
    )
    parser.add_argument(
        "--no-publish-current",
        action="store_true",
        help="Archive and index documents without changing current-version registry.",
    )
    args = parser.parse_args()

    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=False)

    image_docs = load_image_documents(args.input)
    chunks = [
        Chunk(
            tenant_id=doc.tenant_id,
            doc_id=doc.doc_id,
            doc_version=doc.doc_version,
            chunk_index=0,
            source_type="image",
            source_uri=doc.source_uri,
            title=doc.title,
            text=apply_pii_policy(
                f"标题路径: {doc.title}\n来源: image\nOCR:\n{doc.ocr_text}\n"
                f"图片描述:\n{doc.caption}",
                policy=config.pii_policy,
                label=f"{doc.doc_id}:image_text",
            ),
            language=doc.language,
            acl_groups=doc.acl_groups,
            metadata=normalize_image_metadata(doc),
        )
        for doc in image_docs
    ]

    text_model = build_embedding_model(config)
    image_model = build_image_embedding_model(config)
    dense_vectors = text_model.encode([chunk.text for chunk in chunks])
    image_vectors = image_model.encode(
        [f"{chunk.source_uri}\n{chunk.title}\n{chunk.text}" for chunk in chunks]
    )

    entities = [
        chunk_to_entity(
            chunk,
            dense_vector=dense_vector,
            image_vector=image_vector,
            embedding_model=text_model.model_name,
            embedding_dim=text_model.dim,
        )
        for chunk, dense_vector, image_vector in zip(
            chunks, dense_vectors, image_vectors, strict=True
        )
    ]
    count = upsert_entities(client, collection_name=config.collection_name, entities=entities)
    canonical_docs = [
        SourceDocument(
            tenant_id=chunk.tenant_id,
            doc_id=chunk.doc_id,
            doc_version=chunk.doc_version,
            source_type=chunk.source_type,
            source_uri=chunk.source_uri,
            title=chunk.title,
            text=chunk.text,
            language=chunk.language,
            acl_groups=chunk.acl_groups,
            metadata=chunk.metadata,
        )
        for chunk in chunks
    ]
    archived = archive_source_documents(config.object_store_dir, canonical_docs)
    current_versions = (
        {}
        if args.no_publish_current
        else publish_current_versions(config.object_store_dir, canonical_docs)
    )
    print(f"Loaded image docs: {len(image_docs)}")
    print(f"Upserted image chunks: {count}")
    print(f"Archived canonical docs: {archived}")
    if not args.no_publish_current:
        print(f"Published current versions: {current_versions}")


def normalize_image_metadata(doc) -> dict:
    metadata = {
        **doc.metadata,
        "image_uri": doc.metadata.get("image_uri") or doc.source_uri,
        "caption": doc.caption,
        "ocr_text": doc.ocr_text,
        "bbox": doc.metadata.get("bbox", []),
        "linked_doc_id": doc.metadata.get("linked_doc_id", ""),
    }
    return metadata


if __name__ == "__main__":
    main()
