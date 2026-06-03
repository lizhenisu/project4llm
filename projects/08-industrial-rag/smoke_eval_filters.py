from __future__ import annotations

import os
import tempfile
from pathlib import Path

from eval_answer import evaluate_answers
from eval_retrieval import evaluate_retrieval
from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, build_image_embedding_model
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
        os.environ["MILVUS_URI"] = str(Path(tmp) / "eval_filters.db")
        os.environ["RAG_COLLECTION"] = "rag_smoke_eval_filters"
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

    docs = [
        SourceDocument(
            tenant_id="team_a",
            doc_id="eval-filter-md-v1",
            doc_version=1,
            source_type="md",
            source_uri="memory://eval-filter-md-v1",
            title="Old Markdown",
            text="eval filter exact alpha target old markdown version",
            acl_groups=["ops"],
        ),
        SourceDocument(
            tenant_id="team_a",
            doc_id="eval-filter-csv",
            doc_version=2,
            source_type="csv",
            source_uri="memory://eval-filter-csv",
            title="Current CSV",
            text="eval filter exact alpha target current csv version",
            acl_groups=["ops"],
        ),
        SourceDocument(
            tenant_id="team_a",
            doc_id="eval-filter-html",
            doc_version=2,
            source_type="html",
            source_uri="memory://eval-filter-html",
            title="Wrong ACL HTML",
            text="eval filter exact alpha target current html support only",
            acl_groups=["support"],
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
    image_model = build_image_embedding_model(config)
    dense_vectors = text_model.encode([chunk.text for chunk in chunks])
    zero_image = image_model.encode(["no image"])[0]
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

    with tempfile.TemporaryDirectory() as tmp:
        eval_path = Path(tmp) / "eval.jsonl"
        write_jsonl(
            eval_path,
            [
                {
                    "query": "eval filter exact alpha target",
                    "tenant_id": "team_a",
                    "acl_groups": ["ops"],
                    "doc_version": 2,
                    "source_types": ["csv"],
                    "expected_doc_ids": ["eval-filter-csv"],
                    "expected_chunk_ids": ["eval-filter-csv:0"],
                    "answerable": True,
                    "query_type": "eval_filter",
                }
            ],
        )
        retrieval = evaluate_retrieval(input_path=eval_path, limit=3, mode="rerank")
        answer = evaluate_answers(
            input_path=eval_path,
            candidate_limit=10,
            context_limit=3,
        )

    assert retrieval["query_count"] == 1
    assert retrieval["recall"] == 1.0
    assert retrieval["mrr"] == 1.0
    assert retrieval["ndcg"] == 1.0
    assert answer["evidence_hit_rate"] == 1.0
    assert answer["citation_accuracy"] == 1.0
    print("smoke_eval_filters=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
