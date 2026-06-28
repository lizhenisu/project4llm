from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from rag_core.answering import build_prompt, generate_answer, generate_chat
from rag_core.config import load_config
from rag_core.document_scope import OPEN_CHAT, answer_document_scope, build_scope_plan
from rag_core.pipeline import StageCallback, emit_stage, retrieve_and_rerank
from rag_core.types import SearchHit, TraceInfo


@dataclass(frozen=True)
class AnswerResult:
    request_id: str
    answer: str
    hits: list[SearchHit]
    candidates: list[SearchHit]
    reranked: list[SearchHit]
    trace: object
    generation: object


def answer_query(
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
) -> AnswerResult:
    config = load_config()
    has_doc_filter = bool(doc_ids or source_types or doc_version or include_all_sources)
    if not has_doc_filter:
        # Pure LLM chat mode — no Milvus retrieval
        return answer_query_without_retrieval(
            query,
            history=history,
            request_id=request_id,
            stage_callback=stage_callback,
        )
    scope_plan = build_scope_plan(
        config=config,
        tenant_id=tenant_id,
        query=query,
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
        (
            f"已解析 {len(scope_plan.resolved_doc_ids)} 个文档；"
            f"{'需要逐份覆盖全部文档' if scope_plan.route.coverage_required else '无需逐份覆盖全部文档'}。"
        ),
        selected_doc_ids=scope_plan.selected_doc_ids,
        resolved_doc_ids=scope_plan.resolved_doc_ids,
        missing_or_skipped_doc_ids=scope_plan.missing_doc_ids,
        coverage_required=scope_plan.route.coverage_required,
    )
    if scope_plan.route.intent == OPEN_CHAT and not scope_plan.resolved_doc_ids:
        return answer_query_without_retrieval(
            query,
            history=history,
            request_id=request_id,
            stage_callback=stage_callback,
        )
    if scope_plan.should_use_document_pipeline:
        return answer_document_scope(
            config=config,
            query=query,
            tenant_id=tenant_id,
            acl_groups=acl_groups,
            plan=scope_plan,
            request_id=request_id,
            stage_callback=stage_callback,
        )
    if scope_plan.route.explicit_doc_refs:
        doc_ids = scope_plan.resolved_doc_ids
    retrieval = retrieve_and_rerank(
        query,
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
    emit_stage(
        stage_callback,
        "answer",
        "active",
        "大模型最终输出",
        "正在基于证据片段生成最终回答。",
    )
    generation = generate_answer(config, retrieval.trace.rewritten_query, retrieval.hits)
    emit_stage(
        stage_callback,
        "answer",
        "done",
        "大模型最终输出",
        "最终回答已生成。",
        latency_ms=generation.latency_ms,
        llm_model=generation.llm_model,
    )
    return AnswerResult(
        request_id=retrieval.request_id,
        answer=generation.answer,
        hits=retrieval.hits,
        candidates=retrieval.candidates,
        reranked=retrieval.reranked,
        trace=retrieval.trace,
        generation=generation,
    )


def answer_query_without_retrieval(
    query: str,
    *,
    history: list[str] | None = None,
    request_id: str | None = None,
    stage_callback: StageCallback | None = None,
) -> AnswerResult:
    """Pure LLM chat mode — bypasses Milvus retrieval entirely. Passes conversation history directly to LLM."""
    import uuid as _uuid
    config = load_config()
    resolved_request_id = request_id or str(_uuid.uuid4())
    # Build messages from history + current query — no system prompt, no RAG context
    chat_messages: list[dict[str, str]] = []
    if history:
        for msg in history:
            prefix, _, content = msg.partition(": ")
            role = "assistant" if prefix == "assistant" else "user"
            chat_messages.append({"role": role, "content": content})
    chat_messages.append({"role": "user", "content": query})
    emit_stage(
        stage_callback,
        "answer",
        "active",
        "大模型直接回答",
        "当前未选择文档，正在直接调用大模型。",
    )
    generation = generate_chat(config, chat_messages)
    emit_stage(
        stage_callback,
        "answer",
        "done",
        "大模型直接回答",
        "回答已生成。",
        latency_ms=generation.latency_ms,
        llm_model=generation.llm_model,
    )
    trace = TraceInfo(
        request_id=resolved_request_id,
        original_query=query,
        rewritten_query=query,
        rewrite_backend="none",
        tenant_id="",
        acl_groups=[],
        doc_version=None,
        current_versions={},
        embedding_model="none",
        source_types=[],
        doc_ids=[],
        filter_expr="",
        retrieval_mode="direct_llm_no_retrieval",
        candidate_count=0,
        reranked_count=0,
        context_count=0,
        dropped_by_score=0,
        dropped_by_doc_limit=0,
        dropped_by_budget=0,
        stage_latency_ms={"rewrite": 0},
    )
    return AnswerResult(
        request_id=resolved_request_id,
        answer=generation.answer,
        hits=[],
        candidates=[],
        reranked=[],
        trace=trace,
        generation=generation,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full retrieval -> rerank -> answer flow.")
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
        help="Restrict retrieval to a source type. Repeat for multiple types.",
    )
    parser.add_argument(
        "--show-trace",
        action="store_true",
        help="Print rewritten query, filter, and stage latency for teaching.",
    )
    parser.add_argument(
        "--show-prompt-chars",
        type=int,
        default=0,
        help="Print the first N prompt characters before the answer.",
    )
    args = parser.parse_args()

    result = answer_query(
        args.query,
        tenant_id=args.tenant_id,
        candidate_limit=args.candidate_limit,
        context_limit=args.context_limit,
        acl_groups=args.acl_group or None,
        doc_version=args.doc_version,
        source_types=args.source_type or None,
    )
    if args.show_trace:
        trace_payload = {
            "request_id": result.trace.request_id,
            "original_query": result.trace.original_query,
            "rewritten_query": result.trace.rewritten_query,
            "filter_expr": result.trace.filter_expr,
            "retrieval_mode": result.trace.retrieval_mode,
            "candidate_count": result.trace.candidate_count,
            "reranked_count": result.trace.reranked_count,
            "context_count": result.trace.context_count,
            "stage_latency_ms": result.trace.stage_latency_ms,
        }
        print("trace:")
        print(json.dumps(trace_payload, ensure_ascii=False, indent=2))
        print()
    if args.show_prompt_chars > 0:
        prompt = build_prompt(result.trace.rewritten_query, result.hits)
        preview = prompt[: args.show_prompt_chars]
        print("prompt_preview:")
        print(preview)
        if len(prompt) > len(preview):
            print("... (prompt truncated)")
        print()
    print(f"request_id: {result.request_id}\n")
    print(result.answer)
    print("\nCitations:")
    for index, hit in enumerate(result.hits, start=1):
        print(
            f"[{index}] doc={hit.doc_id} chunk={hit.chunk_index} "
            f"source={hit.source_type} acl={','.join(hit.acl_groups)}"
        )


if __name__ == "__main__":
    main()
