from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from answer_multimodal import answer_multimodal_query
from eval_answer import evaluate_answers
from eval_retrieval import evaluate_retrieval
from ingest_image import normalize_image_metadata
from rag_core.config import FIXTURE_DATA_DIR, load_config
from rag_core.embeddings import build_embedding_model, build_image_embedding_model
from rag_core.io import load_image_documents
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.object_store import archive_source_documents
from rag_core.pii import apply_pii_policy
from rag_core.types import Chunk, SourceDocument
from rag_core.versioning import publish_current_versions
from search_multimodal import retrieve_multimodal


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk through the multimodal RAG path for teaching."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=FIXTURE_DATA_DIR / "sample_images.jsonl",
        help="JSONL file of ImageDocument rows.",
    )
    parser.add_argument(
        "--eval-input",
        type=Path,
        default=FIXTURE_DATA_DIR / "multimodal_eval_queries.jsonl",
        help="JSONL multimodal eval set.",
    )
    parser.add_argument(
        "--query",
        default="RAG Dashboard latency recall",
        help="Text query to inspect step by step.",
    )
    parser.add_argument("--tenant-id", default="team_a")
    parser.add_argument(
        "--acl-group",
        action="append",
        default=["support"],
        help="Allowed ACL group. Repeat for multiple groups.",
    )
    parser.add_argument("--candidate-limit", type=int, default=5)
    parser.add_argument("--context-limit", type=int, default=3)
    parser.add_argument("--eval-limit", type=int, default=3)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        env_overrides = {
            "RAG_MILVUS_URI": str(Path(tmp) / "walkthrough_multimodal.db"),
            "RAG_OBJECT_STORE_DIR": str(Path(tmp) / "object_store"),
            "RAG_RUNTIME_DIR": str(Path(tmp) / "runtime"),
            "RAG_COLLECTION": "rag_walkthrough_multimodal",
        }
        with temporary_env(env_overrides):
            run_walkthrough(args)


def run_walkthrough(args: argparse.Namespace) -> None:
    config = load_config()
    client = connect(config)
    try:
        ensure_collection(client, config, reset=True)
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
                    f"标题路径: {doc.title}\n来源: image\nOCR:\n{doc.ocr_text}\n图片描述:\n{doc.caption}",
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
        upserted = upsert_entities(
            client,
            collection_name=config.collection_name,
            entities=[
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
            ],
        )
        archive_source_documents(config.object_store_dir, canonical_docs)
        publish_current_versions(config.object_store_dir, canonical_docs)

        print_section("1. Multimodal Docs")
        print(f"loaded_image_docs={len(image_docs)} from={args.input}")
        for doc in image_docs:
            print(
                f"doc_id={doc.doc_id} source_uri={doc.source_uri} "
                f"ocr_chars={len(doc.ocr_text)} caption_chars={len(doc.caption)} "
                f"acl={','.join(doc.acl_groups)}"
            )

        print_section("2. Ingest")
        print(f"text_embedding_backend={config.embedding_backend} model={text_model.model_name}")
        print(
            f"image_embedding_backend={config.image_embedding_backend} "
            f"model={image_model.model_name}"
        )
        print(f"upserted_image_chunks={upserted}")

        print_section("3. Multimodal Retrieval")
        retrieval = retrieve_multimodal(
            args.query,
            tenant_id=args.tenant_id,
            candidate_limit=args.candidate_limit,
            context_limit=args.context_limit,
            acl_groups=args.acl_group or None,
            source_types=["image"],
        )
        print(f"rewritten_query={retrieval.trace.rewritten_query}")
        print(f"retrieval_mode={retrieval.trace.retrieval_mode}")
        print(f"stage_latency_ms={retrieval.trace.stage_latency_ms}")
        for rank, hit in enumerate(retrieval.hits, start=1):
            fusion = (hit.metadata or {}).get("fusion") or {}
            print(
                f"{rank}. doc={hit.doc_id} source={hit.source_type} "
                f"channels={fusion.get('channels', {})}"
            )
            print(hit.text[:220].replace("\n", " "))

        print_section("4. Multimodal Answer")
        answer = answer_multimodal_query(
            args.query,
            tenant_id=args.tenant_id,
            candidate_limit=args.candidate_limit,
            context_limit=args.context_limit,
            acl_groups=args.acl_group or None,
            source_types=["image"],
        )
        print(f"request_id={answer.request_id}")
        print(f"answer_preview={answer.answer[:500]}")
        print(f"citation_doc_ids={[hit.doc_id for hit in answer.hits]}")

        print_section("5. Multimodal Eval")
        retrieval_metrics = evaluate_retrieval(
            input_path=args.eval_input,
            limit=args.eval_limit,
            mode="multimodal",
        )
        answer_metrics = evaluate_answers(
            input_path=args.eval_input,
            candidate_limit=args.candidate_limit,
            context_limit=args.context_limit,
            mode="multimodal",
        )
        print(
            "retrieval_metrics="
            f"recall@{args.eval_limit}={retrieval_metrics['recall']:.3f} "
            f"mrr@{args.eval_limit}={retrieval_metrics['mrr']:.3f} "
            f"ndcg@{args.eval_limit}={retrieval_metrics['ndcg']:.3f}"
        )
        print(
            "answer_metrics="
            f"citation_accuracy={answer_metrics['citation_accuracy']:.3f} "
            f"evidence_hit_rate={answer_metrics['evidence_hit_rate']:.3f} "
            f"faithfulness={answer_metrics['faithfulness']:.3f}"
        )
    finally:
        client.close()


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


@contextmanager
def temporary_env(overrides: dict[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    main()
