from __future__ import annotations

import argparse
import os
import tempfile
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from answer import answer_query
from eval_answer import evaluate_answers
from eval_retrieval import evaluate_retrieval
from rag_core.answering import build_prompt, generate_answer
from rag_core.config import DATA_DIR, load_config
from rag_core.embeddings import build_embedding_model, build_image_embedding_model
from rag_core.io import load_source_documents
from rag_core.milvus_store import (
    build_filter_expr,
    chunk_to_entity,
    connect,
    dense_search,
    ensure_collection,
    hybrid_search,
    sparse_search,
    upsert_entities,
)
from rag_core.object_store import archive_source_documents
from rag_core.pii import apply_pii_policy
from rag_core.context import explain_context_packing
from rag_core.rerankers import build_reranker
from rag_core.rewrite import rewrite_query
from rag_core.text_utils import chunk_document, sparse_embedding, tokenize
from rag_core.types import SearchHit, SourceDocument
from rag_core.versioning import load_current_versions, publish_current_versions


KEY_SCHEMA_FIELDS = [
    "id",
    "tenant_id",
    "doc_id",
    "doc_version",
    "source_type",
    "title",
    "text",
    "acl_groups",
    "text_dense_vector",
    "bm25_sparse_vector",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk through the core industrial RAG path for teaching."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "sample_docs.jsonl",
        help="JSONL file of SourceDocument rows.",
    )
    parser.add_argument(
        "--eval-input",
        type=Path,
        default=DATA_DIR / "eval_queries.jsonl",
        help="JSONL eval set used at the end of the walkthrough.",
    )
    parser.add_argument(
        "--query",
        default="RAG 检索变慢时应该排查哪些环节？",
        help="Query to inspect step by step.",
    )
    parser.add_argument("--tenant-id", default="team_a")
    parser.add_argument(
        "--acl-group",
        action="append",
        default=["support"],
        help="Allowed ACL group. Repeat to allow multiple groups.",
    )
    parser.add_argument(
        "--history",
        action="append",
        default=[],
        help="Optional chat history. Repeat for multiple turns.",
    )
    parser.add_argument(
        "--rewrite-backend",
        choices=["none", "heuristic", "llm"],
        default="none",
        help="Query rewrite backend to demonstrate.",
    )
    parser.add_argument("--candidate-limit", type=int, default=5)
    parser.add_argument("--context-limit", type=int, default=3)
    parser.add_argument("--eval-limit", type=int, default=3)
    parser.add_argument(
        "--prompt-preview-chars",
        type=int,
        default=800,
        help="How many characters of the final prompt to print.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        env_overrides = {
            "RAG_MILVUS_URI": str(Path(tmp) / "walkthrough.db"),
            "RAG_OBJECT_STORE_DIR": str(Path(tmp) / "object_store"),
            "RAG_RUNTIME_DIR": str(Path(tmp) / "runtime"),
            "RAG_COLLECTION": "rag_walkthrough_core",
            "RAG_QUERY_REWRITE_BACKEND": args.rewrite_backend,
        }
        with temporary_env(env_overrides):
            run_walkthrough(args)


def run_walkthrough(args: argparse.Namespace) -> None:
    config = load_config()
    client = connect(config)
    try:
        ensure_collection(client, config, reset=True)
        docs = load_docs(args.input, config)
        chunks = [
            chunk
            for doc in docs
            for chunk in chunk_document(
                doc,
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap,
            )
        ]

        print_section("1. Schema")
        print("Milvus collection uses explicit fields instead of dynamic JSON.")
        print(f"collection={config.collection_name}")
        print(f"milvus_uri={config.milvus_uri}")
        print(f"key_fields={', '.join(KEY_SCHEMA_FIELDS)}")
        print(f"dense_dim={config.embedding_dim} image_dim={config.image_embedding_dim}")

        print_section("2. Documents -> Chunks")
        print(f"loaded_docs={len(docs)} from={args.input}")
        print_doc_summary(docs)
        print_chunk_summary(chunks)

        print_section("3. Ingest")
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
        upserted = upsert_entities(
            client,
            collection_name=config.collection_name,
            entities=entities,
        )
        archived = archive_source_documents(config.object_store_dir, docs)
        published = publish_current_versions(config.object_store_dir, docs)
        print(f"embedding_backend={config.embedding_backend} model={embedding_model.model_name}")
        print(f"upserted_chunks={upserted}")
        print(f"archived_canonical_docs={archived}")
        print(f"published_current_versions={published}")

        print_section("4. Query -> Dense/Sparse/Hybrid")
        rewrite = rewrite_query(args.query, history=args.history, config=config)
        current_versions = load_current_versions(config.object_store_dir, tenant_id=args.tenant_id)
        filter_expr = build_filter_expr(
            tenant_id=args.tenant_id,
            allowed_acl_groups=args.acl_group or None,
            current_doc_versions=current_versions,
            embedding_model=embedding_model.model_name,
        )
        query_sparse = sparse_embedding(rewrite.rewritten_query)
        query_vector = embedding_model.encode([rewrite.rewritten_query])[0]
        print(f"original_query={rewrite.original_query}")
        print(f"rewritten_query={rewrite.rewritten_query}")
        print(f"rewrite_backend={rewrite.backend}")
        print(f"query_tokens={tokenize(rewrite.rewritten_query)}")
        print(f"sparse_nonzero_buckets={len(query_sparse)}")
        print(f"filter_expr={filter_expr}")

        dense_hits = dense_search(
            client,
            collection_name=config.collection_name,
            query_vector=query_vector,
            filter_expr=filter_expr,
            limit=args.candidate_limit,
        )
        sparse_hits = sparse_search(
            client,
            collection_name=config.collection_name,
            query_sparse=query_sparse,
            filter_expr=filter_expr,
            limit=args.candidate_limit,
        )
        hybrid_hits = hybrid_search(
            client,
            collection_name=config.collection_name,
            query_vector=query_vector,
            query_sparse=query_sparse,
            filter_expr=filter_expr,
            limit=args.candidate_limit,
        )
        print_hits("dense_hits", dense_hits)
        print_hits("sparse_hits", sparse_hits)
        print_hits("hybrid_hits", hybrid_hits)

        print_section("5. Rerank -> Context")
        reranker = build_reranker(config)
        reranked = reranker.rerank(
            rewrite.rewritten_query,
            hybrid_hits,
            limit=len(hybrid_hits),
        )
        selected, packing_stats, decisions = explain_context_packing(
            reranked,
            max_selected=args.context_limit,
            max_chars=config.max_context_chars,
            max_chunks_per_doc=config.max_chunks_per_doc,
            min_rerank_score=config.min_rerank_score,
        )
        print_hits("reranked_hits", reranked, include_rerank=True)
        print(
            "context_stats="
            f"selected={packing_stats.selected_count} "
            f"dropped_by_score={packing_stats.dropped_by_score} "
            f"dropped_by_doc_limit={packing_stats.dropped_by_doc_limit} "
            f"dropped_by_budget={packing_stats.dropped_by_budget}"
        )
        for decision in decisions:
            print(
                "packing_decision "
                f"doc={decision.doc_id} chunk={decision.chunk_index} "
                f"decision={decision.decision} reason={decision.reason} "
                f"rerank={format_optional(decision.rerank_score)}"
            )

        print_section("6. Prompt -> Answer")
        prompt = build_prompt(rewrite.rewritten_query, selected)
        prompt_preview = prompt[: args.prompt_preview_chars]
        print("prompt_preview:")
        print(prompt_preview)
        if len(prompt) > len(prompt_preview):
            print("... (prompt truncated)")
        generation = generate_answer(config, rewrite.rewritten_query, selected)
        print(f"llm_backend={generation.llm_backend} llm_model={generation.llm_model}")
        print("answer_preview:")
        print(generation.answer[:600])

        print_section("7. Production Pipeline Trace")
        answer_result = answer_query(
            args.query,
            tenant_id=args.tenant_id,
            candidate_limit=max(args.candidate_limit, 5),
            context_limit=args.context_limit,
            acl_groups=args.acl_group or None,
            history=args.history or None,
        )
        print(f"request_id={answer_result.request_id}")
        print(f"retrieval_mode={answer_result.trace.retrieval_mode}")
        print(f"stage_latency_ms={answer_result.trace.stage_latency_ms}")
        print(f"final_context_doc_ids={[hit.doc_id for hit in answer_result.hits]}")

        print_section("8. Eval")
        retrieval_metrics = evaluate_retrieval(
            input_path=args.eval_input,
            limit=args.eval_limit,
            mode="rerank",
        )
        answer_metrics = evaluate_answers(
            input_path=args.eval_input,
            candidate_limit=max(args.candidate_limit, 5),
            context_limit=args.context_limit,
            mode="text",
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


def load_docs(input_path: Path, config) -> list[SourceDocument]:
    return [
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
        for doc in load_source_documents(input_path)
    ]


def print_doc_summary(docs: list[SourceDocument]) -> None:
    for doc in docs:
        print(
            f"doc_id={doc.doc_id} tenant={doc.tenant_id} "
            f"acl={','.join(doc.acl_groups)} tokens={len(tokenize(doc.text))} "
            f"title={doc.title}"
        )


def print_chunk_summary(chunks) -> None:
    counts = Counter(chunk.doc_id for chunk in chunks)
    print(f"total_chunks={len(chunks)} per_doc={dict(counts)}")
    for chunk in chunks[:3]:
        preview = chunk.text[:180].replace("\n", " ")
        print(
            f"chunk doc={chunk.doc_id} idx={chunk.chunk_index} "
            f"chars={len(chunk.text)} preview={preview}"
        )


def print_hits(name: str, hits: list[SearchHit], *, include_rerank: bool = False) -> None:
    print(f"{name}:")
    if not hits:
        print("  (no hits)")
        return
    for rank, hit in enumerate(hits, start=1):
        rerank = (
            f" rerank={hit.rerank_score:.4f}"
            if include_rerank and hit.rerank_score is not None
            else ""
        )
        preview = hit.text[:120].replace("\n", " ")
        print(
            f"  {rank}. score={hit.score:.4f}{rerank} "
            f"doc={hit.doc_id} chunk={hit.chunk_index} preview={preview}"
        )


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


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
