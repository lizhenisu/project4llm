from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from diagnose_context import diagnose_context, main as diagnose_main
from rag_core.config import load_config
from rag_core.context import explain_context_packing
from rag_core.embeddings import build_embedding_model, zero_image_vector
from rag_core.io import read_jsonl
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.text_utils import chunk_document
from rag_core.types import SearchHit, SourceDocument
from rag_core.versioning import publish_current_versions


def main() -> None:
    test_decision_reasons()
    old_milvus_uri = os.environ.get("MILVUS_URI")
    old_collection = os.environ.get("RAG_COLLECTION")
    old_object_store = os.environ.get("RAG_OBJECT_STORE_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MILVUS_URI"] = str(Path(tmp) / "context_diagnosis.db")
        os.environ["RAG_COLLECTION"] = "rag_smoke_context_diagnosis"
        os.environ["RAG_OBJECT_STORE_DIR"] = str(Path(tmp) / "object_store")
        try:
            run_pipeline_smoke()
        finally:
            restore_env("MILVUS_URI", old_milvus_uri)
            restore_env("RAG_COLLECTION", old_collection)
            restore_env("RAG_OBJECT_STORE_DIR", old_object_store)
    print("smoke_context_diagnosis=ok")


def test_decision_reasons() -> None:
    hits = [
        make_hit("doc-a", 0, "a" * 40, 0.9),
        make_hit("doc-a", 1, "b" * 40, 0.8),
        make_hit("doc-b", 0, "c" * 80, 0.7),
        make_hit("doc-c", 0, "d" * 20, 0.1),
    ]
    selected, stats, decisions = explain_context_packing(
        hits,
        max_chars=70,
        max_chunks_per_doc=1,
        min_rerank_score=0.2,
        text_unit_counter=len,
    )
    assert [hit.doc_id for hit in selected] == ["doc-a"]
    assert stats.dropped_by_doc_limit == 1
    assert stats.dropped_by_budget == 1
    assert stats.dropped_by_score == 1
    assert [decision.reason for decision in decisions] == [
        "fits_budget",
        "max_chunks_per_doc",
        "context_char_budget",
        "below_min_rerank_score",
    ]


def run_pipeline_smoke() -> None:
    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=True)
    docs = [
        SourceDocument(
            tenant_id="team_a",
            doc_id="context-pack-a",
            doc_version=1,
            source_type="md",
            source_uri="memory://context-pack-a",
            title="Context A",
            text="RAG context packing 需要控制同文档 chunk 数量，并保留 citation。",
            acl_groups=["ops"],
        ),
        SourceDocument(
            tenant_id="team_a",
            doc_id="context-pack-b",
            doc_version=1,
            source_type="md",
            source_uri="memory://context-pack-b",
            title="Context B",
            text="RAG context packing 还要控制总字符预算，避免 prompt 被低价值证据塞满。",
            acl_groups=["ops"],
        ),
    ]
    chunks = [
        replace(chunk, text=chunk.text + " " + ("extra " * 20))
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
    publish_current_versions(config.object_store_dir, docs)
    rows = diagnose_context(
        "RAG context packing citation 字符预算",
        tenant_id="team_a",
        acl_groups=["ops"],
        candidate_limit=5,
        context_limit=5,
        max_context_chars=40,
        max_chunks_per_doc=1,
    )
    assert rows
    assert any(row["decision"] == "select" for row in rows)
    assert any(row["reason"] == "context_char_budget" for row in rows)

    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "context.jsonl"
        old_argv = sys.argv
        sys.argv = [
            "diagnose_context.py",
            "RAG context packing citation 字符预算",
            "--tenant-id",
            "team_a",
            "--acl-group",
            "ops",
            "--max-context-chars",
            "40",
            "--json-output",
            str(output),
        ]
        try:
            diagnose_main()
        finally:
            sys.argv = old_argv
        exported = read_jsonl(output)
    assert exported
    assert "packing_stats" in exported[0]


def make_hit(doc_id: str, chunk_index: int, text: str, rerank_score: float) -> SearchHit:
    return SearchHit(
        id=f"{doc_id}:{chunk_index}",
        score=1.0,
        text=text,
        doc_id=doc_id,
        title=doc_id,
        source_uri=f"memory://{doc_id}",
        source_type="md",
        chunk_index=chunk_index,
        tenant_id="team_a",
        acl_groups=["ops"],
        metadata={},
        rerank_score=rerank_score,
    )


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
