from __future__ import annotations

import uuid
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from rag_core.config import load_config
from rag_core.context import pack_context
from rag_core.embeddings import build_embedding_model
from rag_core.guards import mentions_other_tenant
from rag_core.milvus_store import build_filter_expr, connect, ensure_collection, hybrid_search
from rag_core.versioning import load_current_versions
from rag_core.rerankers import build_reranker
from rag_core.retrieval_scope import (
    annotate_retrieval_source,
    group_selected_doc_ids,
    per_source_candidate_limit,
    round_robin_hit_groups,
    should_fan_out_source_retrieval,
)
from rag_core.rewrite import rewrite_query
from rag_core.source_guides import load_source_guides_for_rewrite
from rag_core.types import SearchHit, TraceInfo


@dataclass(frozen=True)
class RetrievalResult:
    request_id: str
    hits: list[SearchHit]
    candidates: list[SearchHit]
    reranked: list[SearchHit]
    trace: TraceInfo


StageCallback = Callable[[dict[str, object]], None]


def retrieve_and_rerank(
    query: str,
    *,
    tenant_id: str,
    candidate_limit: int,
    context_limit: int,
    acl_groups: list[str] | None = None,
    doc_version: int | None = None,
    doc_ids: list[str] | None = None,
    source_types: list[str] | None = None,
    include_all_sources: bool = False,
    history: list[str] | None = None,
    request_id: str | None = None,
    stage_callback: StageCallback | None = None,
) -> RetrievalResult:
    config = load_config()
    resolved_request_id = request_id or str(uuid.uuid4())
    client = connect(config)
    ensure_collection(client, config, reset=False)
    embedding_model = build_embedding_model(config)
    current_versions = None if include_all_sources else (
        {}
        if doc_version is not None
        else load_current_versions(config.object_store_dir, tenant_id=tenant_id, config=config)
    )
    emit_stage(
        stage_callback,
        "source_guides",
        "active",
        "读取知识库摘要",
        "正在加载来源摘要，用于让查询改写更贴近当前文档。",
    )
    source_summaries = load_source_guides_for_rewrite(
        config.object_store_dir,
        tenant_id=tenant_id,
        doc_ids=doc_ids,
        doc_version=doc_version,
        current_doc_versions=current_versions,
    )
    emit_stage(
        stage_callback,
        "source_guides",
        "done",
        "读取知识库摘要",
        f"已加载 {len(source_summaries)} 条来源摘要。",
    )
    emit_stage(
        stage_callback,
        "rewrite",
        "active",
        "查询重写",
        "正在结合对话历史和来源摘要改写问题。",
    )
    rewrite_start = perf_counter()
    rewrite = rewrite_query(query, history=history, source_summaries=source_summaries, config=config)
    rewrite_ms = elapsed_ms(rewrite_start)
    emit_stage(
        stage_callback,
        "rewrite",
        "done",
        "查询重写",
        "已得到检索查询。",
        latency_ms=rewrite_ms,
        rewritten_query=rewrite.rewritten_query,
        backend=rewrite.backend,
    )
    if mentions_other_tenant(rewrite.rewritten_query, tenant_id):
        trace = TraceInfo(
            request_id=resolved_request_id,
            original_query=rewrite.original_query,
            rewritten_query=rewrite.rewritten_query,
            rewrite_backend=rewrite.backend,
            tenant_id=tenant_id,
            acl_groups=acl_groups or [],
            doc_version=doc_version,
            current_versions=current_versions or {},
            embedding_model=embedding_model.model_name,
            source_types=source_types or [],
            doc_ids=doc_ids or [],
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
    emit_stage(
        stage_callback,
        "embedding",
        "active",
        "向量编码",
        "正在把查询转换为向量表示。",
    )
    embedding_start = perf_counter()
    query_vector = embedding_model.encode([rewrite.rewritten_query])[0]
    embedding_ms = elapsed_ms(embedding_start)
    emit_stage(
        stage_callback,
        "embedding",
        "done",
        "向量编码",
        f"已使用 {embedding_model.model_name} 完成编码。",
        latency_ms=embedding_ms,
        embedding_model=embedding_model.model_name,
    )
    filter_expr = build_filter_expr(
        tenant_id=tenant_id,
        allowed_acl_groups=acl_groups,
        doc_version=doc_version,
        current_doc_versions=current_versions,
        embedding_model=embedding_model.model_name,
        doc_ids=doc_ids,
        source_types=source_types,
    )
    emit_stage(
        stage_callback,
        "search",
        "active",
        "向量检索",
        "正在 Milvus 中执行 hybrid dense + sparse 检索。",
    )
    search_start = perf_counter()
    selected_source_groups = group_selected_doc_ids(doc_ids or [])
    if should_fan_out_source_retrieval(selected_source_groups):
        per_source_limit = per_source_candidate_limit(candidate_limit, len(selected_source_groups))
        candidate_groups = [
            annotate_retrieval_source(
                hybrid_search(
                    client,
                    collection_name=config.collection_name,
                    query_vector=query_vector,
                    query_text=rewrite.rewritten_query,
                    filter_expr=build_filter_expr(
                        tenant_id=tenant_id,
                        allowed_acl_groups=acl_groups,
                        doc_version=doc_version,
                        current_doc_versions=current_versions,
                        embedding_model=embedding_model.model_name,
                        doc_ids=group_doc_ids,
                        source_types=source_types,
                    ),
                    limit=per_source_limit,
                ),
                source_id,
            )
            for source_id, group_doc_ids in selected_source_groups
        ]
        candidates = round_robin_hit_groups(candidate_groups)
    else:
        candidates = hybrid_search(
            client,
            collection_name=config.collection_name,
            query_vector=query_vector,
            query_text=rewrite.rewritten_query,
            filter_expr=filter_expr,
            limit=candidate_limit,
        )
    search_ms = elapsed_ms(search_start)
    emit_stage(
        stage_callback,
        "search",
        "done",
        "向量检索",
        f"已召回 {len(candidates)} 个候选片段。",
        latency_ms=search_ms,
        candidate_count=len(candidates),
        source_group_count=len(selected_source_groups),
        source_fanout=should_fan_out_source_retrieval(selected_source_groups),
    )
    emit_stage(
        stage_callback,
        "rerank",
        "active",
        "重排序",
        "正在按问题相关性重排候选片段。",
    )
    rerank_start = perf_counter()
    reranked = build_reranker(config).rerank(
        rewrite.rewritten_query,
        candidates,
        limit=len(candidates),
    )
    rerank_ms = elapsed_ms(rerank_start)
    emit_stage(
        stage_callback,
        "rerank",
        "done",
        "重排序",
        f"已完成 {len(reranked)} 个候选片段重排。",
        latency_ms=rerank_ms,
        reranked_count=len(reranked),
    )
    emit_stage(
        stage_callback,
        "context",
        "active",
        "上下文组装",
        "正在选择最终进入大模型提示词的证据片段。",
    )
    packing_start = perf_counter()
    hits, packing_stats = pack_context(
        reranked,
        max_selected=context_limit,
        max_chars=config.max_context_chars,
        max_chunks_per_doc=config.max_chunks_per_doc,
        min_rerank_score=config.min_rerank_score,
        text_unit_counter=getattr(embedding_model, "count_tokens", None),
    )
    packing_ms = elapsed_ms(packing_start)
    emit_stage(
        stage_callback,
        "context",
        "done",
        "上下文组装",
        f"已选出 {len(hits)} 个证据片段进入回答上下文。",
        latency_ms=packing_ms,
        context_count=len(hits),
        dropped_by_score=packing_stats.dropped_by_score,
        dropped_by_doc_limit=packing_stats.dropped_by_doc_limit,
        dropped_by_budget=packing_stats.dropped_by_budget,
    )
    trace = TraceInfo(
        request_id=resolved_request_id,
        original_query=rewrite.original_query,
        rewritten_query=rewrite.rewritten_query,
        rewrite_backend=rewrite.backend,
        tenant_id=tenant_id,
        acl_groups=acl_groups or [],
        doc_version=doc_version,
        current_versions=current_versions or {},
        embedding_model=embedding_model.model_name,
        source_types=source_types or [],
        doc_ids=doc_ids or [],
        filter_expr=filter_expr,
        retrieval_mode=(
            "hybrid_dense_sparse_source_fanout_rerank"
            if should_fan_out_source_retrieval(selected_source_groups)
            else "hybrid_dense_sparse_rerank"
        ),
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


def emit_stage(
    callback: StageCallback | None,
    stage: str,
    status: str,
    label: str,
    detail: str,
    **payload: object,
) -> None:
    if callback is None:
        return
    callback(
        {
            "stage": stage,
            "status": status,
            "label": label,
            "detail": detail,
            **payload,
        }
    )
