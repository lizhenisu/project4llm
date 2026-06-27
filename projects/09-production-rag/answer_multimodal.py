from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from rag_core.answering import generate_answer
from rag_core.config import load_config
from rag_core.document_scope import answer_document_scope, build_scope_plan
from rag_core.io import PdfImageCaptioner, detect_text_language
from rag_core.pipeline import StageCallback, emit_stage
from rag_core.types import SearchHit, TraceInfo
from search_multimodal import retrieve_multimodal


@dataclass(frozen=True)
class MultimodalAnswerResult:
    request_id: str
    answer: str
    hits: list[SearchHit]
    candidates: list[SearchHit]
    reranked: list[SearchHit]
    trace: TraceInfo
    generation: object


def answer_multimodal_query(
    query: str | None = None,
    *,
    text_query: str | None = None,
    image_query_path: str | None = None,
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
    answer_query: str | None = None,
    stage_callback: StageCallback | None = None,
) -> MultimodalAnswerResult:
    resolved_text_query = text_query if text_query is not None else query
    config = load_config()
    has_doc_filter = bool(doc_ids or source_types or doc_version or include_all_sources)
    if has_doc_filter and resolved_text_query:
        scope_plan = build_scope_plan(
            config=config,
            tenant_id=tenant_id,
            query=resolved_text_query,
            doc_ids=doc_ids,
            doc_version=doc_version,
            include_all_sources=include_all_sources,
        )
        emit_stage(
            stage_callback,
            "intent_router",
            "done",
            "意图与范围识别",
            scope_plan.route.reason,
            **scope_plan.route.as_dict(),
        )
        emit_stage(
            stage_callback,
            "scope_resolution",
            "done",
            "文档范围解析",
            f"已解析 {len(scope_plan.resolved_doc_ids)} 个文档，覆盖要求为 {scope_plan.route.coverage_required}。",
            selected_doc_ids=scope_plan.selected_doc_ids,
            resolved_doc_ids=scope_plan.resolved_doc_ids,
            missing_or_skipped_doc_ids=scope_plan.missing_doc_ids,
            coverage_required=scope_plan.route.coverage_required,
        )
        if scope_plan.should_use_document_pipeline:
            return answer_document_scope(
                config=config,
                query=resolved_text_query,
                tenant_id=tenant_id,
                acl_groups=acl_groups,
                plan=scope_plan,
                request_id=request_id,
                stage_callback=stage_callback,
            )
        if scope_plan.route.explicit_doc_refs:
            doc_ids = scope_plan.resolved_doc_ids
    retrieval = retrieve_multimodal(
        query,
        text_query=text_query,
        image_query_path=image_query_path,
        tenant_id=tenant_id,
        candidate_limit=candidate_limit,
        context_limit=context_limit,
        acl_groups=acl_groups,
        doc_version=doc_version,
        doc_ids=doc_ids,
        source_types=source_types,
        include_all_sources=include_all_sources,
        history=history,
        request_id=request_id,
        stage_callback=stage_callback,
    )
    final_answer_query = answer_query or resolved_text_query or retrieval.trace.rewritten_query
    query_image_caption = describe_query_image_for_answer(
        image_query_path=image_query_path,
        query=final_answer_query,
    )
    if query_image_caption:
        final_answer_query = multimodal_answer_query_with_image_description(
            query=final_answer_query,
            image_description=query_image_caption,
        )
    emit_stage(
        stage_callback,
        "answer",
        "active",
        "大模型最终输出",
        "正在基于多模态证据生成最终回答。",
    )
    generation = generate_answer(
        config,
        multimodal_answer_query(final_answer_query),
        retrieval.hits,
    )
    emit_stage(
        stage_callback,
        "answer",
        "done",
        "大模型最终输出",
        "最终回答已生成。",
        latency_ms=generation.latency_ms,
        llm_model=generation.llm_model,
    )
    return MultimodalAnswerResult(
        request_id=retrieval.request_id,
        answer=generation.answer,
        hits=retrieval.hits,
        candidates=retrieval.candidates,
        reranked=retrieval.reranked,
        trace=retrieval.trace,
        generation=generation,
    )


def multimodal_answer_query(query: str) -> str:
    return (
        "这是一次图片/多模态检索问答。用户上传的图片已经被系统用于向量检索，"
        "下面的证据是与该图片或问题最相关的相似图片/文档片段。"
        "请基于证据回答用户，不要说你无法查看图片；如果证据不足，就说明相似图片证据不足。"
        f"\n\n用户问题: {query}"
    )


def multimodal_answer_query_with_image_description(*, query: str, image_description: str) -> str:
    return (
        f"{query}\n\n"
        "用户上传图片的文字化描述:\n"
        f"{image_description}"
    )


def describe_query_image_for_answer(*, image_query_path: str | None, query: str) -> str:
    if not image_query_path:
        return ""
    captioner = PdfImageCaptioner.from_query_env()
    if captioner is None:
        return ""
    return captioner.caption_image_path(
        Path(image_query_path),
        query=query,
        language_hint=detect_text_language(query),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multimodal retrieval -> context packing -> answer flow."
    )
    parser.add_argument("query")
    parser.add_argument("--tenant-id", default="team_a")
    parser.add_argument(
        "--acl-group",
        action="append",
        default=[],
        help="Allowed ACL group. Repeat to allow multiple groups.",
    )
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--context-limit", type=int, default=5)
    parser.add_argument("--doc-version", type=int)
    parser.add_argument(
        "--source-type",
        action="append",
        default=[],
        help="Restrict retrieval to a source type. Defaults to image.",
    )
    args = parser.parse_args()

    result = answer_multimodal_query(
        args.query,
        tenant_id=args.tenant_id,
        candidate_limit=args.candidate_limit,
        context_limit=args.context_limit,
        acl_groups=args.acl_group or None,
        doc_version=args.doc_version,
        source_types=args.source_type or None,
    )
    print(f"request_id: {result.request_id}\n")
    print(result.answer)
    print("\nCitations:")
    for index, hit in enumerate(result.hits, start=1):
        fusion = hit.metadata.get("fusion") or {}
        print(
            f"[{index}] doc={hit.doc_id} chunk={hit.chunk_index} "
            f"source={hit.source_type} channels={fusion.get('channels', {})}"
        )


if __name__ == "__main__":
    main()
