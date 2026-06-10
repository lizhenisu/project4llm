from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from eval_retrieval import evaluate_retrieval
from rag_core.config import DATA_DIR, load_config
from rag_core.embeddings import build_embedding_model, zero_image_vector
from rag_core.io import load_source_documents, write_jsonl
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.text_utils import chunk_document
from rag_core.versioning import publish_current_versions


@dataclass(frozen=True)
class ChunkSpec:
    chunk_size: int
    overlap: int


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep chunk_size/overlap and evaluate retrieval on temporary collections."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "sample_docs.jsonl",
        help="JSONL SourceDocument input.",
    )
    parser.add_argument(
        "--eval-input",
        type=Path,
        default=DATA_DIR / "eval_queries.jsonl",
        help="Eval JSONL input.",
    )
    parser.add_argument(
        "--spec",
        action="append",
        default=[],
        help="Chunk spec formatted as chunk_size:overlap. Repeat for multiple specs.",
    )
    parser.add_argument(
        "--mode",
        choices=["dense", "sparse", "hybrid", "rerank"],
        default="hybrid",
        help="Retrieval mode used for each eval run.",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--json-output", type=Path, help="Write sweep rows as JSONL.")
    args = parser.parse_args()

    specs = parse_specs(args.spec) or [
        ChunkSpec(chunk_size=400, overlap=80),
        ChunkSpec(chunk_size=700, overlap=100),
        ChunkSpec(chunk_size=1000, overlap=150),
    ]
    rows = sweep_chunking(
        input_path=args.input,
        eval_input_path=args.eval_input,
        specs=specs,
        mode=args.mode,
        limit=args.limit,
    )
    print_rows(rows)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.json_output, rows)


def sweep_chunking(
    *,
    input_path: Path,
    eval_input_path: Path,
    specs: list[ChunkSpec],
    mode: str,
    limit: int,
) -> list[dict[str, object]]:
    old_env = {
        "RAG_COLLECTION": os.environ.get("RAG_COLLECTION"),
        "RAG_OBJECT_STORE_DIR": os.environ.get("RAG_OBJECT_STORE_DIR"),
    }
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as object_store:
        try:
            for spec in specs:
                collection_name = (
                    f"rag_chunk_sweep_{spec.chunk_size}_{spec.overlap}_{uuid4().hex[:8]}"
                )
                os.environ["RAG_COLLECTION"] = collection_name
                os.environ["RAG_OBJECT_STORE_DIR"] = object_store
                row = {
                    "temporary_collection": collection_name,
                    "cleanup": "not_started",
                }
                try:
                    row.update(
                        run_one_spec(
                            input_path=input_path,
                            eval_input_path=eval_input_path,
                            spec=spec,
                            mode=mode,
                            limit=limit,
                        )
                    )
                finally:
                    row["cleanup"] = drop_collection_if_exists(collection_name)
                rows.append(row)
        finally:
            for name, value in old_env.items():
                restore_env(name, value)
    return rows


def run_one_spec(
    *,
    input_path: Path,
    eval_input_path: Path,
    spec: ChunkSpec,
    mode: str,
    limit: int,
) -> dict[str, object]:
    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=True)
    docs = load_source_documents(input_path)
    text_model = build_embedding_model(config)
    chunks = [
        chunk
        for doc in docs
        for chunk in chunk_document(
            doc,
            chunk_size=spec.chunk_size,
            overlap=spec.overlap,
            token_counter=text_model.count_tokens,
        )
    ]
    dense_vectors = text_model.encode([chunk.text for chunk in chunks])
    zero_image = zero_image_vector(config)
    upserted = upsert_entities(
        client,
        collection_name=config.collection_name,
        entities=[
            chunk_to_entity(
                chunk,
                dense_vector=dense_vector,
                image_vector=zero_image,
                embedding_model=text_model.model_name,
                embedding_dim=text_model.dim,
            )
            for chunk, dense_vector in zip(chunks, dense_vectors, strict=True)
        ],
    )
    publish_current_versions(config.object_store_dir, docs)
    metrics = evaluate_retrieval(
        input_path=eval_input_path,
        limit=limit,
        mode=mode,
    )
    return {
        "chunk_size": spec.chunk_size,
        "overlap": spec.overlap,
        "doc_count": len(docs),
        "chunk_count": len(chunks),
        "avg_chunk_tokens": avg([text_model.count_tokens(chunk.text) for chunk in chunks]),
        "max_chunk_tokens": max(
            [text_model.count_tokens(chunk.text) for chunk in chunks],
            default=0,
        ),
        "upserted": upserted,
        "mode": mode,
        "limit": limit,
        "recall": metrics["recall"],
        "mrr": metrics["mrr"],
        "ndcg": metrics["ndcg"],
        "p95_latency_ms": metrics["p95_latency_ms"],
        "stage_p95_latency_ms": metrics["stage_p95_latency_ms"],
        "permission_leakage_failures": metrics["permission_leakage_failures"],
    }


def parse_specs(values: list[str]) -> list[ChunkSpec]:
    specs: list[ChunkSpec] = []
    for value in values:
        try:
            chunk_size, overlap = value.split(":", maxsplit=1)
            spec = ChunkSpec(chunk_size=int(chunk_size), overlap=int(overlap))
        except ValueError as exc:
            raise ValueError(f"Invalid --spec {value!r}; expected chunk_size:overlap") from exc
        if spec.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if spec.overlap < 0:
            raise ValueError("overlap must be non-negative")
        if spec.overlap >= spec.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        specs.append(spec)
    return specs


def print_rows(rows: list[dict[str, object]]) -> None:
    for row in rows:
        print(
            "chunk_size={chunk_size} overlap={overlap} chunks={chunk_count} "
            "avg_tokens={avg_chunk_tokens:.1f} recall={recall:.3f} "
            "mrr={mrr:.3f} ndcg={ndcg:.3f} p95={p95_latency_ms:.2f}ms".format(
                **row
            )
        )


def avg(values: list[int]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def drop_collection_if_exists(collection_name: str) -> str:
    try:
        config = load_config()
        client = connect(config)
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
            return "dropped"
        return "already_absent"
    except Exception as exc:
        return f"cleanup_failed:{type(exc).__name__}"


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
