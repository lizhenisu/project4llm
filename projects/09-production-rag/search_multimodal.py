from __future__ import annotations

import argparse
import uuid
from dataclasses import dataclass
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
from rag_core.pipeline import elapsed_ms
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
    query: str,
    *,
    tenant_id: str,
    candidate_limit: int,
    context_limit: int,
    acl_groups: list[str] | None = None,
    doc_version: int | None = None,
    doc_ids: list[str] | None = None,
    source_types: list[str] | None = None,
    history: list[str] | None = None,
    request_id: str | None = None,
) -> MultimodalRetrievalResult:
    config = load_config()
    resolved_request_id = request_id or str(uuid.uuid4())
    client = connect(config)
    ensure_collection(client, config, reset=False)
    text_model = build_embedding_model(config)
    image_model = build_image_embedding_model(config)
    query_path = Path(query)
    if query_path.exists() and query_path.is_file():
        rewritten_query = str(query_path)
        rewrite_backend = "file-path"
        rewrite_ms = 0.0
    else:
        rewrite_start = perf_counter()
        rewrite = rewrite_query(query, history=history, config=config)
        rewrite_ms = elapsed_ms(rewrite_start)
        rewritten_query = rewrite.rewritten_query
        rewrite_backend = rewrite.backend
        if mentions_other_tenant(rewritten_query, tenant_id):
            trace = TraceInfo(
                request_id=resolved_request_id,
                original_query=query,
                rewritten_query=rewritten_query,
                rewrite_backend=rewrite_backend,
                tenant_id=tenant_id,
                acl_groups=acl_groups or [],
                doc_version=doc_version,
                current_versions={},
                embedding_model=text_model.model_name,
                source_types=source_types or ["image"],
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

    resolved_source_types = source_types or ["image"]
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
        embedding_model=text_model.model_name,
        doc_ids=doc_ids,
        source_types=resolved_source_types,
    )
    text_embedding_start = perf_counter()
    text_query_vector = text_model.encode([rewritten_query])[0]
    text_embedding_ms = elapsed_ms(text_embedding_start)
    text_search_start = perf_counter()
    text_hits = hybrid_search(
        client,
        collection_name=config.collection_name,
        query_vector=text_query_vector,
        query_text=rewritten_query,
        filter_expr=filter_expr,
        limit=max(candidate_limit, 10),
    )
    text_search_ms = elapsed_ms(text_search_start)

    image_embedding_start = perf_counter()
    if query_path.exists() and query_path.is_file():
        image_query_vector = image_model.encode_images([query_path])[0]
    else:
        image_query_vector = image_model.encode([rewritten_query])[0]
    image_embedding_ms = elapsed_ms(image_embedding_start)
    image_search_start = perf_counter()
    image_hits = image_search(
        client,
        collection_name=config.collection_name,
        image_query_vector=image_query_vector,
        filter_expr=filter_expr,
        limit=max(candidate_limit, 10),
    )
    image_search_ms = elapsed_ms(image_search_start)
    fusion_start = perf_counter()
    candidates = reciprocal_rank_fusion(
        [
            ("text_hybrid", text_hits),
            ("image_vector", image_hits),
        ],
        limit=candidate_limit,
    )
    fusion_ms = elapsed_ms(fusion_start)
    packing_start = perf_counter()
    hits, packing_stats = pack_context(
        candidates,
        max_selected=context_limit,
        max_chars=config.max_context_chars,
        max_chunks_per_doc=config.max_chunks_per_doc,
        min_rerank_score=config.min_rerank_score,
        text_unit_counter=getattr(text_model, "count_tokens", None),
    )
    packing_ms = elapsed_ms(packing_start)
    trace = TraceInfo(
        request_id=resolved_request_id,
        original_query=query,
        rewritten_query=rewritten_query,
        rewrite_backend=rewrite_backend,
        tenant_id=tenant_id,
        acl_groups=acl_groups or [],
        doc_version=doc_version,
        current_versions=current_versions,
        embedding_model=text_model.model_name,
        source_types=resolved_source_types,
        doc_ids=doc_ids or [],
        filter_expr=filter_expr,
        retrieval_mode="multimodal_text_image_fusion",
        candidate_count=len(candidates),
        reranked_count=len(candidates),
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
            "context_pack": packing_ms,
        },
    )
    return MultimodalRetrievalResult(
        request_id=resolved_request_id,
        hits=hits,
        candidates=candidates,
        reranked=candidates,
        trace=trace,
    )


def run_multimodal_search(
    query: str,
    *,
    tenant_id: str,
    limit: int,
    acl_groups: list[str] | None = None,
    doc_version: int | None = None,
    doc_ids: list[str] | None = None,
    source_types: list[str] | None = None,
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
