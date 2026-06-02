from __future__ import annotations

import os

from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, build_image_embedding_model
from rag_core.rerankers import build_reranker
from rag_core.types import SearchHit


def main() -> None:
    if os.environ.get("RAG_RUN_MODEL_SMOKE") != "1":
        print("smoke_models=skipped; set RAG_RUN_MODEL_SMOKE=1 to load model backends")
        return

    config = load_config()
    embedding_model = build_embedding_model(config)
    vectors = embedding_model.encode(["Milvus hybrid search smoke"])
    assert vectors and len(vectors[0]) == embedding_model.dim
    print(f"embedding=ok backend={config.embedding_backend} dim={embedding_model.dim}")

    image_model = build_image_embedding_model(config)
    image_vectors = image_model.encode(["RAG dashboard screenshot"])
    assert image_vectors and len(image_vectors[0]) == image_model.dim
    print(
        f"image_embedding=ok backend={config.image_embedding_backend} "
        f"dim={image_model.dim}"
    )

    reranker = build_reranker(config)
    reranked = reranker.rerank(
        "RAG 检索变慢排查什么",
        [
            SearchHit(
                id="1",
                score=0.5,
                text="RAG 检索延迟升高时检查 embedding、Milvus search 和 rerank。",
                doc_id="rag-runbook",
                title="RAG runbook",
                source_uri="docs/rag-runbook.md",
                source_type="md",
                chunk_index=0,
                tenant_id="team_a",
                acl_groups=["ops"],
                metadata={},
            )
        ],
        limit=1,
    )
    assert reranked
    print(f"reranker=ok backend={config.rerank_backend}")


if __name__ == "__main__":
    main()

