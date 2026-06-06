from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from diagnose_retrieval import diagnose_retrieval, main as diagnose_main
from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, zero_image_vector
from rag_core.io import read_jsonl
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.text_utils import chunk_document
from rag_core.types import SourceDocument


def main() -> None:
    old_milvus_uri = os.environ.get("MILVUS_URI")
    old_collection = os.environ.get("RAG_COLLECTION")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MILVUS_URI"] = str(Path(tmp) / "retrieval_diagnosis.db")
        os.environ["RAG_COLLECTION"] = "rag_smoke_retrieval_diagnosis"
        try:
            run_smoke()
        finally:
            restore_env("MILVUS_URI", old_milvus_uri)
            restore_env("RAG_COLLECTION", old_collection)


def run_smoke() -> None:
    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=True)
    docs = [
        SourceDocument(
            tenant_id="team_a",
            doc_id="diagnose-runbook",
            doc_version=1,
            source_type="md",
            source_uri="memory://diagnose-runbook",
            title="诊断 Runbook",
            text="RAG 检索变慢时检查 embedding batch、Milvus hybrid search 和 rerank 输入。",
            acl_groups=["ops"],
        ),
        SourceDocument(
            tenant_id="team_a",
            doc_id="diagnose-refund",
            doc_version=1,
            source_type="md",
            source_uri="memory://diagnose-refund",
            title="退款规则",
            text="退款需要订单号、付款凭证和问题截图。",
            acl_groups=["ops"],
        ),
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

    rows = diagnose_retrieval(
        "RAG 检索变慢 rerank 输入",
        tenant_id="team_a",
        acl_groups=["ops"],
        candidate_limit=5,
        limit=3,
    )
    assert rows
    assert rows[0].doc_id == "diagnose-runbook"
    assert rows[0].dense_rank is not None
    assert rows[0].sparse_rank is not None
    assert rows[0].hybrid_rank is not None
    assert rows[0].rerank_rank == 1
    assert rows[0].lexical_overlap > 0

    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "diagnosis.jsonl"
        old_argv = sys.argv
        sys.argv = [
            "diagnose_retrieval.py",
            "RAG 检索变慢 rerank 输入",
            "--tenant-id",
            "team_a",
            "--acl-group",
            "ops",
            "--json-output",
            str(output),
        ]
        try:
            diagnose_main()
        finally:
            sys.argv = old_argv
        exported = read_jsonl(output)
    assert exported
    assert exported[0]["doc_id"] == "diagnose-runbook"
    print("smoke_retrieval_diagnosis=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
