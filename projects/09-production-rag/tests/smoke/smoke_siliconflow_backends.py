from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import rag_core.embeddings as embeddings
import rag_core.rerankers as rerankers
from rag_core.config import RagConfig
from rag_core.embeddings import build_embedding_model
from rag_core.rerankers import build_reranker
from rag_core.types import SearchHit


def main() -> None:
    config = RagConfig(
        milvus_uri="memory://unused",
        milvus_token=None,
        collection_name="unused",
        embedding_model="BAAI/bge-m3",
        embedding_backend="siliconflow",
        embedding_dim=3,
        embedding_batch_size=2,
        embedding_max_length=8192,
        rerank_model="BAAI/bge-reranker-v2-m3",
        rerank_backend="siliconflow",
        rerank_batch_size=8,
        rerank_max_length=1024,
        image_embedding_backend="clip",
        image_embedding_model="openai/clip-vit-base-patch32",
        image_embedding_dim=512,
        image_embedding_batch_size=8,
        model_device="auto",
        model_dtype="auto",
        llm_base_url=None,
        llm_api_key=None,
        llm_model="test-llm",
        siliconflow_base_url="https://api.siliconflow.cn",
        siliconflow_api_key="test-key",
        answer_backend="extractive",
        chunk_size=700,
        chunk_overlap=100,
        reset_collection=False,
        runtime_dir=Path("/tmp/runtime"),
        object_store_dir=Path("/tmp/object_store"),
        pii_policy="warn",
        max_context_chars=6000,
        max_chunks_per_doc=2,
        min_rerank_score=None,
        query_rewrite_backend="none",
        query_rewrite_history_turns=6,
        query_rewrite_max_tokens=256,
        require_auth_context=False,
        api_token=None,
        dense_hnsw_m=16,
        dense_hnsw_ef_construction=100,
        dense_search_ef=128,
        image_hnsw_m=16,
        image_hnsw_ef_construction=100,
        image_search_ef=128,
        sparse_drop_ratio_build=0.2,
        sparse_drop_ratio_search=0.0,
    )

    old_embedding_post_json = embeddings.post_json
    old_reranker_post_json = rerankers.post_json
    try:
        embeddings.post_json = fake_embedding_post_json
        model = build_embedding_model(config)
        vectors = model.encode(["alpha", "beta"])
        assert vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        rerankers.post_json = fake_rerank_post_json
        hits = [
            make_hit("doc-a", "alpha evidence", 0.2),
            make_hit("doc-b", "beta evidence", 0.1),
        ]
        reranked = build_reranker(config).rerank("beta", hits, limit=2)
        assert [hit.doc_id for hit in reranked] == ["doc-b", "doc-a"]
        assert [hit.rerank_score for hit in reranked] == [0.91, 0.12]

        try:
            build_embedding_model(replace(config, siliconflow_api_key=None)).encode(["x"])
        except RuntimeError as exc:
            assert "SILICONFLOW_API_KEY" in str(exc)
        else:
            raise AssertionError("missing SiliconFlow API key should fail")
    finally:
        embeddings.post_json = old_embedding_post_json
        rerankers.post_json = old_reranker_post_json

    print("smoke_siliconflow_backends=ok")


def fake_embedding_post_json(url: str, *, api_key: str, payload: dict) -> dict:
    assert url == "https://api.siliconflow.cn/v1/embeddings"
    assert api_key == "test-key"
    assert payload["model"] == "BAAI/bge-m3"
    assert payload["input"] == ["alpha", "beta"]
    return {
        "data": [
            {"index": 1, "embedding": [0.0, 1.0, 0.0]},
            {"index": 0, "embedding": [1.0, 0.0, 0.0]},
        ]
    }


def fake_rerank_post_json(url: str, *, api_key: str, payload: dict) -> dict:
    assert url == "https://api.siliconflow.cn/v1/rerank"
    assert api_key == "test-key"
    assert payload["model"] == "BAAI/bge-reranker-v2-m3"
    assert payload["query"] == "beta"
    assert payload["documents"] == ["alpha evidence", "beta evidence"]
    return {
        "results": [
            {"index": 1, "relevance_score": 0.91},
            {"index": 0, "relevance_score": 0.12},
        ]
    }


def make_hit(doc_id: str, text: str, score: float) -> SearchHit:
    return SearchHit(
        id=doc_id,
        score=score,
        text=text,
        doc_id=doc_id,
        title=doc_id,
        source_uri=f"memory://{doc_id}",
        source_type="txt",
        chunk_index=0,
        tenant_id="team_a",
        acl_groups=["engineering"],
        metadata={},
    )


if __name__ == "__main__":
    main()
