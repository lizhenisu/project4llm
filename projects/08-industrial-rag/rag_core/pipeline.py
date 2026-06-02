from __future__ import annotations

import uuid
from dataclasses import dataclass
from time import perf_counter

from rag_core.config import load_config
from rag_core.context import pack_context
from rag_core.embeddings import build_embedding_model
from rag_core.guards import mentions_other_tenant
from rag_core.milvus_store import build_filter_expr, connect, ensure_collection, hybrid_search
from rag_core.versioning import load_current_versions
from rag_core.rerankers import build_reranker
from rag_core.rewrite import rewrite_query
from rag_core.text_utils import sparse_embedding
from rag_core.types import SearchHit, TraceInfo


@dataclass(frozen=True)
class RetrievalResult:
    request_id: str
    hits: list[SearchHit]
    candidates: list[SearchHit]
    reranked: list[SearchHit]
    trace: TraceInfo


def retrieve_and_rerank(
    query: str,
    *,
    tenant_id: str,
    candidate_limit: int,
    context_limit: int,
    acl_groups: list[str] | None = None,
    doc_version: int | None = None,
    source_types: list[str] | None = None,
    history: list[str] | None = None,
    request_id: str | None = None,
) -> RetrievalResult:
    config = load_config()
    resolved_request_id = request_id or str(uuid.uuid4())
    client = connect(config)
    ensure_collection(client, config, reset=False)
    embedding_model = build_embedding_model(config)
    rewrite_start = perf_counter()
    rewrite = rewrite_query(query, history=history, config=config)
    rewrite_ms = elapsed_ms(rewrite_start)
    if mentions_other_tenant(rewrite.rewritten_query, tenant_id):
        trace = TraceInfo(
            request_id=resolved_request_id,
            original_query=rewrite.original_query,
            rewritten_query=rewrite.rewritten_query,
            rewrite_backend=rewrite.backend,
            tenant_id=tenant_id,
            acl_groups=acl_groups or [],
            doc_version=doc_version,
            current_versions={},
            embedding_model=embedding_model.model_name,
            source_types=source_types or [],
            filter_expr=f'tenant_id == "{tenant_id}" and blocked_other_tenant == true',
            retrieval_mode="blocked_cross_tenant_query",
            candidate_count=0,
            reranked_count=0,
            context_count=0,
            dropped_by_score=0,
            dropped_by_doc_limit=0,
            dropped_by_budget=0,
            stage_latency_ms={"rewrite": rewrite_ms},
        )
        return RetrievalResult(
            request_id=resolved_request_id,
            hits=[],
            candidates=[],
            reranked=[],
            trace=trace,
        )
    embedding_start = perf_counter()
    query_vector = embedding_model.encode([rewrite.rewritten_query])[0]
    embedding_ms = elapsed_ms(embedding_start)
    current_versions = (
        {}
        if doc_version is not None
        else load_current_versions(config.object_store_dir, tenant_id=tenant_id)
    )
    filter_expr = build_filter_expr(
        tenant_id=tenant_id,
        allowed_acl_groups=acl_groups,
        doc_version=doc_version,
        current_doc_versions=current_versions,
        embedding_model=embedding_model.model_name,
        source_types=source_types,
    )
    search_start = perf_counter()
    candidates = hybrid_search(
        client,
        collection_name=config.collection_name,
        query_vector=query_vector,
        query_sparse=sparse_embedding(rewrite.rewritten_query),
        filter_expr=filter_expr,
        limit=candidate_limit,
    )
    search_ms = elapsed_ms(search_start)
    rerank_start = perf_counter()
    reranked = build_reranker(config).rerank(
        rewrite.rewritten_query,
        candidates,
        limit=context_limit,
    )
    rerank_ms = elapsed_ms(rerank_start)
    packing_start = perf_counter()
    hits, packing_stats = pack_context(
        reranked,
        max_chars=config.max_context_chars,
        max_chunks_per_doc=config.max_chunks_per_doc,
        min_rerank_score=config.min_rerank_score,
    )
    packing_ms = elapsed_ms(packing_start)
    trace = TraceInfo(
        request_id=resolved_request_id,
        original_query=rewrite.original_query,
        rewritten_query=rewrite.rewritten_query,
        rewrite_backend=rewrite.backend,
        tenant_id=tenant_id,
        acl_groups=acl_groups or [],
        doc_version=doc_version,
        current_versions=current_versions,
        embedding_model=embedding_model.model_name,
        source_types=source_types or [],
        filter_expr=filter_expr,
        retrieval_mode="hybrid_dense_sparse_rerank",
        candidate_count=len(candidates),
        reranked_count=len(reranked),
        context_count=len(hits),
        dropped_by_score=packing_stats.dropped_by_score,
        dropped_by_doc_limit=packing_stats.dropped_by_doc_limit,
        dropped_by_budget=packing_stats.dropped_by_budget,
        stage_latency_ms={
            "rewrite": rewrite_ms,
            "embedding": embedding_ms,
            "milvus_search": search_ms,
            "rerank": rerank_ms,
            "context_pack": packing_ms,
        },
    )
    return RetrievalResult(
        request_id=resolved_request_id,
        hits=hits,
        candidates=candidates,
        reranked=reranked,
        trace=trace,
    )


def elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 2)
