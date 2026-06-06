from __future__ import annotations

import os
import tempfile
from pathlib import Path

from eval_answer import evaluate_answers
from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, zero_image_vector
from rag_core.io import write_jsonl
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.text_utils import chunk_document
from rag_core.types import SourceDocument
from rag_core.versioning import publish_current_versions


def main() -> None:
    old_milvus_uri = os.environ.get("MILVUS_URI")
    old_collection = os.environ.get("RAG_COLLECTION")
    old_object_store = os.environ.get("RAG_OBJECT_STORE_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MILVUS_URI"] = str(Path(tmp) / "answer_quality.db")
        os.environ["RAG_COLLECTION"] = "rag_smoke_answer_quality"
        os.environ["RAG_OBJECT_STORE_DIR"] = str(Path(tmp) / "object_store")
        try:
            run_smoke()
        finally:
            restore_env("MILVUS_URI", old_milvus_uri)
            restore_env("RAG_COLLECTION", old_collection)
            restore_env("RAG_OBJECT_STORE_DIR", old_object_store)


def run_smoke() -> None:
    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=True)
    doc = SourceDocument(
        tenant_id="team_a",
        doc_id="answer-quality-runbook",
        doc_version=1,
        source_type="md",
        source_uri="memory://answer-quality-runbook",
        title="答案质量 Runbook",
        text="RAG 检索变慢时，需要检查 embedding、Milvus search、rerank 和 LLM 生成。",
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
    publish_current_versions(config.object_store_dir, [doc])

    with tempfile.TemporaryDirectory() as tmp:
        eval_path = Path(tmp) / "answer_eval.jsonl"
        write_jsonl(
            eval_path,
            [
                {
                    "query": "RAG 检索变慢时应该排查什么？",
                    "tenant_id": "team_a",
                    "acl_groups": ["ops"],
                    "expected_doc_ids": ["answer-quality-runbook"],
                    "answerable": True,
                    "expected_answer_terms": ["embedding", "Milvus search", "rerank"],
                    "unsupported_answer_terms": ["Redis", "Kafka"],
                }
            ],
        )
        metrics = evaluate_answers(
            input_path=eval_path,
            candidate_limit=5,
            context_limit=3,
        )

    assert metrics["evidence_hit_rate"] == 1.0
    assert metrics["answer_correctness"] == 1.0
    assert metrics["faithfulness"] == 1.0
    print("smoke_answer_quality_eval=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
