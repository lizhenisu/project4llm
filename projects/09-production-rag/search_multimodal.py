from __future__ import annotations

import argparse
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

from rag_core.config import load_config
from rag_core.context import pack_context
from rag_core.embeddings import build_embedding_model, build_image_embedding_model
from rag_core.guards import mentions_other_tenant
from rag_core.milvus_store import (
    build_filter_expr,
    connect,
    ensure_collection,
    hybrid_search,
    image_search,
)
from rag_core.multimodal import reciprocal_rank_fusion
from rag_core.pipeline import StageCallback, elapsed_ms, emit_stage
from rag_core.rerankers import build_reranker
from rag_core.retrieval_scope import (
    annotate_retrieval_source,
    group_selected_doc_ids,
    per_source_candidate_limit,
    round_robin_hit_groups,
    should_fan_out_source_retrieval,
)
from rag_core.rewrite import rewrite_query
from rag_core.types import SearchHit, TraceInfo
from rag_core.versioning import load_current_versions


@dataclass(frozen=True)
class MultimodalRetrievalResult:
    request_id: str
    hits: list[SearchHit]
    candidates: list[SearchHit]
    reranked: list[SearchHit]
    trace: TraceInfo


def retrieve_multimodal(
    query: str | None = None,
    *,
    text_query: str | None = None,
    image_query_path: str | Path | None = None,
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
) -> MultimodalRetrievalResult:
    config = load_config()
    resolved_request_id = request_id or str(uuid.uuid4())
    client = connect(config)
    ensure_collection(client, config, reset=False)
    text_model = build_embedding_model(config)
    image_model = build_image_embedding_model(config)
    query_path = Path(image_query_path) if image_query_path else Path(query or "")
    has_image_file_query = query_path.exists() and query_path.is_file()
    query_text = (
        text_query
        if text_query is not None
        else ""
        if has_image_file_query
        else query or ""
    ).strip()
    has_text_query = bool(query_text)
    if has_image_file_query and not has_text_query:
        rewritten_query = str(query_path)
        rewrite_backend = "file-path"
        rewrite_ms = 0.0
    else:
        if not has_text_query:
            raise ValueError("text_query or image_query_path is required")
        emit_stage(
            stage_callback,
            "rewrite",
            "active",
            "查询重写",
            "正在结合对话历史改写多模态问题。",
        )
        rewrite_start = perf_counter()
        rewrite = rewrite_query(query_text, history=history, config=config)
        rewrite_ms = elapsed_ms(rewrite_start)
        rewritten_query = rewrite.rewritten_query
        rewrite_backend = rewrite.backend
        emit_stage(
            stage_callback,
            "rewrite",
            "done",
            "查询重写",
            "已得到多模态检索查询。",
            latency_ms=rewrite_ms,
            rewritten_query=rewritten_query,
            backend=rewrite_backend,
        )
        if mentions_other_tenant(rewritten_query, tenant_id):
            trace = TraceInfo(
                request_id=resolved_request_id,
                original_query=query_text,
                rewritten_query=rewritten_query,
                rewrite_backend=rewrite_backend,
                tenant_id=tenant_id,
                acl_groups=acl_groups or [],
                doc_version=doc_version,
                current_versions={},
                embedding_model=text_model.model_name,
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
            return MultimodalRetrievalResult(
                request_id=resolved_request_id,
                hits=[],
                candidates=[],
                reranked=[],
                trace=trace,
            )

    text_source_types = source_types
    image_source_types = source_types or ["image"]
    trace_source_types = source_types or []
    current_versions = None if include_all_sources else (
        {}
        if doc_version is not None
        else load_current_versions(config.object_store_dir, tenant_id=tenant_id, config=config)
    )
    text_filter_expr = build_filter_expr(
        tenant_id=tenant_id,
        allowed_acl_groups=acl_groups,
        doc_version=doc_version,
        current_doc_versions=current_versions,
        embedding_model=text_model.model_name,
        doc_ids=doc_ids,
        source_types=text_source_types,
    )
    image_filter_expr = build_filter_expr(
        tenant_id=tenant_id,
        allowed_acl_groups=acl_groups,
        doc_version=doc_version,
        current_doc_versions=current_versions,
        embedding_model=text_model.model_name,
        doc_ids=doc_ids,
        source_types=image_source_types,
    )
    selected_source_groups = group_selected_doc_ids(doc_ids or [])
    use_source_fanout = should_fan_out_source_retrieval(selected_source_groups)
    per_source_limit = per_source_candidate_limit(candidate_limit, len(selected_source_groups))
    text_embedding_ms = 0.0
    text_search_ms = 0.0
    text_hits: list[SearchHit] = []
    if has_text_query:
        emit_stage(
            stage_callback,
            "embedding",
            "active",
            "文本向量编码",
            "正在把问题转换为文本向量。",
        )
        text_embedding_start = perf_counter()
        text_query_vector = text_model.encode([rewritten_query])[0]
        text_embedding_ms = elapsed_ms(text_embedding_start)
        emit_stage(
            stage_callback,
            "embedding",
            "done",
            "文本向量编码",
            f"已使用 {text_model.model_name} 完成文本编码。",
            latency_ms=text_embedding_ms,
        )
        emit_stage(
            stage_callback,
            "search",
            "active",
            "文本检索",
            "正在检索 OCR、标题和文本证据。",
        )
        text_search_start = perf_counter()
        if use_source_fanout:
            text_hit_groups = [
                annotate_retrieval_source(
                    hybrid_search(
                        client,
                        collection_name=config.collection_name,
                        query_vector=text_query_vector,
                        query_text=rewritten_query,
                        filter_expr=build_filter_expr(
                            tenant_id=tenant_id,
                            allowed_acl_groups=acl_groups,
                            doc_version=doc_version,
                            current_doc_versions=current_versions,
                            embedding_model=text_model.model_name,
                            doc_ids=group_doc_ids,
                            source_types=text_source_types,
                        ),
                        limit=per_source_limit,
                    ),
                    source_id,
                )
                for source_id, group_doc_ids in selected_source_groups
            ]
            text_hits = round_robin_hit_groups(text_hit_groups)
        else:
            text_hits = hybrid_search(
                client,
                collection_name=config.collection_name,
                query_vector=text_query_vector,
                query_text=rewritten_query,
                filter_expr=text_filter_expr,
                limit=max(candidate_limit, 10),
            )
        text_search_ms = elapsed_ms(text_search_start)
        emit_stage(
            stage_callback,
            "search",
            "done",
            "文本检索",
            f"已召回 {len(text_hits)} 个文本候选。",
            latency_ms=text_search_ms,
            candidate_count=len(text_hits),
            source_group_count=len(selected_source_groups),
            source_fanout=use_source_fanout,
        )
    emit_stage(
        stage_callback,
        "image_embedding",
        "active",
        "图片向量编码",
        "正在生成图片/多模态向量。",
    )
    image_embedding_start = perf_counter()
    if has_image_file_query:
        image_query_vector = image_model.encode_images([query_path])[0]
    else:
        image_query_vector = image_model.encode([rewritten_query])[0]
    image_embedding_ms = elapsed_ms(image_embedding_start)
    emit_stage(
        stage_callback,
        "image_embedding",
        "done",
        "图片向量编码",
        "图片/多模态向量已生成。",
        latency_ms=image_embedding_ms,
    )
    emit_stage(
        stage_callback,
        "image_search",
        "active",
        "图片向量检索",
        "正在检索相似图片和多模态证据。",
    )
    image_search_start = perf_counter()
    if use_source_fanout:
        image_hit_groups = [
            annotate_retrieval_source(
                image_search(
                    client,
                    collection_name=config.collection_name,
                    image_query_vector=image_query_vector,
                    filter_expr=build_filter_expr(
                        tenant_id=tenant_id,
                        allowed_acl_groups=acl_groups,
                        doc_version=doc_version,
                        current_doc_versions=current_versions,
                        embedding_model=text_model.model_name,
                        doc_ids=group_doc_ids,
                        source_types=image_source_types,
                    ),
                    limit=per_source_limit,
                ),
                source_id,
            )
            for source_id, group_doc_ids in selected_source_groups
        ]
        image_hits = round_robin_hit_groups(image_hit_groups)
        image_anchor_hits = (
            image_search(
                client,
                collection_name=config.collection_name,
                image_query_vector=image_query_vector,
                filter_expr=image_filter_expr,
                limit=1,
            )
            if has_image_file_query
            else image_hits
        )
    else:
        image_hits = image_search(
            client,
            collection_name=config.collection_name,
            image_query_vector=image_query_vector,
            filter_expr=image_filter_expr,
            limit=max(candidate_limit, 10),
        )
        image_anchor_hits = image_hits
    image_search_ms = elapsed_ms(image_search_start)
    emit_stage(
        stage_callback,
        "image_search",
        "done",
        "图片向量检索",
        f"已召回 {len(image_hits)} 个图片候选。",
        latency_ms=image_search_ms,
        candidate_count=len(image_hits),
        source_group_count=len(selected_source_groups),
        source_fanout=use_source_fanout,
    )
    emit_stage(
        stage_callback,
        "fusion",
        "active",
        "多模态融合检索",
        "正在融合文本与图片检索结果。",
    )
    fusion_start = perf_counter()
    if has_image_file_query and not has_text_query:
        candidates = image_only_candidates(image_hits, limit=candidate_limit)
    else:
        candidates = reciprocal_rank_fusion(
            [
                ("text_hybrid", text_hits),
                ("image_vector", image_hits),
            ],
            limit=candidate_limit,
        )
    fusion_ms = elapsed_ms(fusion_start)
    emit_stage(
        stage_callback,
        "fusion",
        "done",
        "多模态融合检索",
        f"已融合得到 {len(candidates)} 个候选证据。",
        latency_ms=fusion_ms,
        candidate_count=len(candidates),
    )
    rerank_ms = 0.0
    if has_text_query and candidates:
        emit_stage(
            stage_callback,
            "rerank",
            "active",
            "多模态重排序",
            "正在按文字问题重排融合后的候选证据。",
        )
        rerank_start = perf_counter()
        reranked = build_reranker(config).rerank(
            rewritten_query,
            candidates,
            limit=len(candidates),
        )
        rerank_ms = elapsed_ms(rerank_start)
        emit_stage(
            stage_callback,
            "rerank",
            "done",
            "多模态重排序",
            f"已完成 {len(reranked)} 个候选证据重排。",
            latency_ms=rerank_ms,
            reranked_count=len(reranked),
        )
    else:
        reranked = candidates
    reranked = anchor_query_image_evidence(
        reranked,
        image_hits=image_anchor_hits,
        has_image_file_query=has_image_file_query,
    )
    emit_stage(
        stage_callback,
        "context",
        "active",
        "上下文组装",
        "正在选择最终进入大模型提示词的多模态证据。",
    )
    packing_start = perf_counter()
    hits, packing_stats = pack_context(
        reranked,
        max_selected=context_limit,
        max_chars=config.max_context_chars,
        max_chunks_per_doc=config.max_chunks_per_doc,
        min_rerank_score=config.min_rerank_score,
        text_unit_counter=getattr(text_model, "count_tokens", None),
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
    )
    trace = TraceInfo(
        request_id=resolved_request_id,
        original_query=query_text or str(query_path),
        rewritten_query=rewritten_query,
        rewrite_backend=rewrite_backend,
        tenant_id=tenant_id,
        acl_groups=acl_groups or [],
        doc_version=doc_version,
        current_versions=current_versions or {},
        embedding_model=text_model.model_name,
        source_types=trace_source_types,
        doc_ids=doc_ids or [],
        filter_expr=text_filter_expr if has_text_query else image_filter_expr,
        retrieval_mode=multimodal_retrieval_mode(
            has_text_query=has_text_query,
            has_image_file_query=has_image_file_query,
            source_fanout=use_source_fanout,
        ),
        candidate_count=len(candidates),
        reranked_count=len(reranked),
        context_count=len(hits),
        dropped_by_score=packing_stats.dropped_by_score,
        dropped_by_doc_limit=packing_stats.dropped_by_doc_limit,
        dropped_by_budget=packing_stats.dropped_by_budget,
        stage_latency_ms={
            "rewrite": rewrite_ms,
            "text_embedding": text_embedding_ms,
            "text_search": text_search_ms,
            "image_embedding": image_embedding_ms,
            "image_search": image_search_ms,
            "fusion": fusion_ms,
            "rerank": rerank_ms,
            "context_pack": packing_ms,
        },
    )
    return MultimodalRetrievalResult(
        request_id=resolved_request_id,
        hits=hits,
        candidates=candidates,
        reranked=reranked,
        trace=trace,
    )


def anchor_query_image_evidence(
    reranked: list[SearchHit],
    *,
    image_hits: list[SearchHit],
    has_image_file_query: bool,
) -> list[SearchHit]:
    """Keep the nearest indexed image in context for an uploaded-image query.

    Text-only rerankers can otherwise demote the actual visual match in favor of
    chunks whose wording happens to resemble the question.
    """
    if not has_image_file_query or not image_hits:
        return reranked
    anchor = replace(image_hits[0], rerank_score=None)
    return [anchor, *(hit for hit in reranked if hit.id != anchor.id)]


def image_only_candidates(image_hits: list[SearchHit], *, limit: int) -> list[SearchHit]:
    candidates: list[SearchHit] = []
    for rank, hit in enumerate(image_hits[:limit], start=1):
        metadata = {
            **hit.metadata,
            "fusion": {
                "mode": "image_only",
                "channels": {"image_vector": rank},
                "channel_scores": {"image_vector": hit.score},
            },
        }
        candidates.append(replace_search_hit_metadata(hit, metadata))
    return candidates


def multimodal_retrieval_mode(
    *,
    has_text_query: bool,
    has_image_file_query: bool,
    source_fanout: bool = False,
) -> str:
    if has_text_query and has_image_file_query:
        mode = "multimodal_text_image_file_fusion_rerank"
    elif has_text_query:
        mode = "multimodal_text_image_fusion_rerank"
    else:
        mode = "image_vector_file_query"
    return f"{mode}_source_fanout" if source_fanout else mode


def replace_search_hit_metadata(hit: SearchHit, metadata: dict) -> SearchHit:
    return replace(hit, metadata=metadata)


def run_multimodal_search(
    query: str,
    *,
    tenant_id: str,
    limit: int,
    acl_groups: list[str] | None = None,
    doc_version: int | None = None,
    doc_ids: list[str] | None = None,
    source_types: list[str] | None = None,
    include_all_sources: bool = False,
    history: list[str] | None = None,
) -> list[SearchHit]:
    result = retrieve_multimodal(
        query,
        tenant_id=tenant_id,
        candidate_limit=limit,
        context_limit=limit,
        acl_groups=acl_groups,
        doc_version=doc_version,
        doc_ids=doc_ids,
        source_types=source_types,
        include_all_sources=include_all_sources,
        history=history,
    )
    return result.hits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multimodal image retrieval with OCR/caption text hybrid + image vector fusion."
    )
    parser.add_argument(
        "query",
        help="Text query or image path. Text queries search OCR/caption and image vectors.",
    )
    parser.add_argument("--tenant-id", default="team_a")
    parser.add_argument(
        "--acl-group",
        action="append",
        default=[],
        help="Allowed ACL group. Repeat to allow multiple groups.",
    )
    parser.add_argument("--doc-version", type=int)
    parser.add_argument(
        "--source-type",
        action="append",
        default=[],
        help="Restrict retrieval to a source type. Defaults to image.",
    )
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    result = retrieve_multimodal(
        args.query,
        tenant_id=args.tenant_id,
        candidate_limit=args.limit,
        context_limit=args.limit,
        acl_groups=args.acl_group or None,
        doc_version=args.doc_version,
        source_types=args.source_type or None,
    )
    for rank, hit in enumerate(result.hits, start=1):
        fusion = hit.metadata.get("fusion") or {}
        channels = fusion.get("channels") or {}
        print(
            f"{rank}. score={hit.score:.6f} doc={hit.doc_id} "
            f"chunk={hit.chunk_index} source={hit.source_type} channels={channels}"
        )
        print(hit.text[:260].replace("\n", " "))


if __name__ == "__main__":
    main()
