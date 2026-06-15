from __future__ import annotations

import argparse
from pathlib import Path

from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, build_image_embedding_model, zero_image_vector
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.object_store import load_archived_source_documents
from rag_core.sources import pdf_image_chunks
from rag_core.text_utils import chunk_document
from rag_core.types import Chunk, SourceDocument
from rag_core.versioning import publish_current_versions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild Milvus index from archived canonical SourceDocument rows."
    )
    parser.add_argument("--reset", action="store_true", help="Drop collection before rebuild.")
    args = parser.parse_args()

    config = load_config()
    docs = load_archived_source_documents(config.object_store_dir)
    if not docs:
        raise SystemExit(f"No archived documents under {config.object_store_dir}")

    client = connect(config)
    ensure_collection(client, config, reset=args.reset)
    text_model = build_embedding_model(config)
    chunks = chunks_for_rebuild(config=config, docs=docs, text_model=text_model)
    dense_vectors = text_model.encode([chunk.text for chunk in chunks])
    image_vectors = image_vectors_for_rebuild(config=config, chunks=chunks)
    entities = [
        chunk_to_entity(
            chunk,
            dense_vector=dense_vector,
            image_vector=image_vector,
            embedding_model=text_model.model_name,
            embedding_dim=text_model.dim,
        )
        for chunk, dense_vector, image_vector in zip(chunks, dense_vectors, image_vectors, strict=True)
    ]
    upserted = upsert_entities(
        client,
        collection_name=config.collection_name,
        entities=entities,
    )
    current_versions = publish_current_versions(config.object_store_dir, docs)
    print(f"Archived documents: {len(docs)}")
    print(f"Rebuilt chunks: {upserted}")
    print(f"Published current versions: {current_versions}")
    print(f"Collection: {config.collection_name}")


def chunks_for_rebuild(*, config, docs: list[SourceDocument], text_model) -> list[Chunk]:
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
    existing_doc_ids = {doc.doc_id for doc in docs}
    extra_image_chunks = [
        chunk
        for chunk in pdf_image_chunks(docs)
        if chunk.doc_id not in existing_doc_ids
    ]
    return [*chunks, *extra_image_chunks]


def image_vectors_for_rebuild(*, config, chunks: list[Chunk]) -> list[list[float]]:
    zero_image = zero_image_vector(config)
    image_model = None
    vectors: list[list[float]] = []
    for chunk in chunks:
        if chunk.source_type != "image":
            vectors.append(zero_image)
            continue
        image_path = Path(chunk.source_uri)
        if not image_path.exists() or not image_path.is_file():
            vectors.append(zero_image)
            continue
        if image_model is None:
            image_model = build_image_embedding_model(config)
        vectors.extend(image_model.encode_images([image_path]))
    return vectors


if __name__ == "__main__":
    main()
