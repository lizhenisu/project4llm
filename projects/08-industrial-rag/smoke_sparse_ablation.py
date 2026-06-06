from __future__ import annotations

import os
import tempfile
from pathlib import Path

from eval_retrieval import evaluate_retrieval
from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, zero_image_vector
from rag_core.io import write_jsonl
from rag_core.milvus_store import (
    build_filter_expr,
    chunk_to_entity,
    connect,
    ensure_collection,
    sparse_search,
    upsert_entities,
)
from rag_core.text_utils import chunk_document
from rag_core.types import SourceDocument


def main() -> None:
    old_milvus_uri = os.environ.get("MILVUS_URI")
    old_collection = os.environ.get("RAG_COLLECTION")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MILVUS_URI"] = str(Path(tmp) / "sparse_ablation.db")
        os.environ["RAG_COLLECTION"] = "rag_smoke_sparse_ablation"
        try:
            run_smoke()
        finally:
            restore_env("MILVUS_URI", old_milvus_uri)
            restore_env("RAG_COLLECTION", old_collection)


def run_smoke() -> None:
    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=True)
    doc = SourceDocument(
        tenant_id="team_a",
        doc_id="error-runbook",
        doc_version=1,
        source_type="md",
        source_uri="memory://error-runbook",
        title="错误码排障",
        text="错误码 ECOM_7741 支付回调失败时，先检查 webhook 签名和重试队列。",
        acl_groups=["ops"],
    )
    chunks = chunk_document(doc, chunk_size=config.chunk_size, overlap=config.chunk_overlap)
    text_model = build_embedding_model(config)
    dense_vectors = text_model.encode([chunk.text for chunk in chunks])
    zero_image = zero_image_vector(config)
    upsert_entities(
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
    query = "ECOM_7741 webhook 签名"
    hits = sparse_search(
        client,
        collection_name=config.collection_name,
        query_text=query,
        filter_expr=build_filter_expr(
            tenant_id="team_a",
            allowed_acl_groups=["ops"],
            embedding_model=text_model.model_name,
        ),
        limit=3,
    )
    assert hits
    assert hits[0].doc_id == "error-runbook"

    with tempfile.TemporaryDirectory() as tmp:
        eval_path = Path(tmp) / "eval.jsonl"
        write_jsonl(
            eval_path,
            [
                {
                    "query": query,
                    "tenant_id": "team_a",
                    "acl_groups": ["ops"],
                    "expected_doc_ids": ["error-runbook"],
                    "answerable": True,
                    "query_type": "sparse_ablation",
                }
            ],
        )
        metrics = evaluate_retrieval(input_path=eval_path, limit=3, mode="sparse")

    assert metrics["recall"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["stage_p95_latency_ms"]["milvus_search"] >= 0.0
    print("smoke_sparse_ablation=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
