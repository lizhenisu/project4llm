from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterable

from rag_core.answering import AnswerGeneration, generate_answer
from rag_core.config import RagConfig
from rag_core.jsonl_store import object_exists, read_object_jsonl
from rag_core.object_store import load_archived_source_documents
from rag_core.pipeline import StageCallback, emit_stage
from rag_core.section_summaries import SourceSectionSummary, load_source_section_summaries
from rag_core.source_guides import SOURCE_GUIDES_PATH, current_source_guide_version
from rag_core.types import SearchHit
from rag_core.types import SourceDocument
from rag_core.versioning import load_current_versions


LOCAL_QA = "local_qa"
SELECTED_DOC_SUMMARY = "selected_doc_summary"
SELECTED_DOC_COMPARE = "selected_doc_compare"
SELECTED_DOC_SYNTHESIS = "selected_doc_synthesis"
SELECTED_DOC_EXTRACT = "selected_doc_extract"
REPORT_GENERATION = "report_generation"
OPEN_CHAT = "open_chat"

SELECTED_DOCUMENTS = "selected_documents"
EXPLICIT_NAMED_DOCUMENTS = "explicit_named_documents"
ALL_VISIBLE_DOCUMENTS = "all_visible_documents"
RETRIEVAL_DISCOVERED_DOCUMENTS = "retrieval_discovered_documents"

TOP_K = "top_k"
ALL_DOCS_SUMMARY = "all_docs_summary"
PER_DOC_MAP_REDUCE = "per_doc_map_reduce"
EXPLICIT_DOCS_ONLY = "explicit_docs_only"


@dataclass(frozen=True)
class SourceGuideRecord:
    tenant_id: str
    source_doc_id: str
    doc_version: int
    title: str
    guide: str


@dataclass(frozen=True)
class DocumentRoute:
    intent: str
    scope: str
    coverage_required: bool
    explicit_doc_refs: list[str]
    reason: str
    confidence: float

    def as_dict(self) -> dict[str, object]:
        return {
            "intent": self.intent,
            "scope": self.scope,
            "coverage_required": self.coverage_required,
            "explicit_doc_refs": self.explicit_doc_refs,
            "reason": self.reason,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ScopePlan:
    route: DocumentRoute
    selected_doc_ids: list[str]
    resolved_doc_ids: list[str]
    guides: list[SourceGuideRecord]
    missing_doc_ids: list[str]
    current_versions: dict[str, int]
    coverage_mode: str
    section_summaries: list[SourceSectionSummary] = field(default_factory=list)

    @property
    def should_use_document_pipeline(self) -> bool:
        return self.route.coverage_required

    def coverage(self) -> dict[str, object]:
        return {
            "resolved_scope_count": len(self.resolved_doc_ids),
            "covered_doc_count": len(self.guides),
            "covered_doc_ids": [guide.source_doc_id for guide in self.guides],
            "missing_or_skipped_doc_ids": self.missing_doc_ids,
            "coverage_mode": self.coverage_mode,
        }


@dataclass(frozen=True)
class DocumentScopeTrace:
    request_id: str
    original_query: str
    rewritten_query: str
    rewrite_backend: str
    tenant_id: str
    acl_groups: list[str]
    doc_version: int | None
    current_versions: dict[str, int]
    embedding_model: str
    source_types: list[str]
    doc_ids: list[str]
    filter_expr: str
    retrieval_mode: str
    candidate_count: int
    reranked_count: int
    context_count: int
    dropped_by_score: int
    dropped_by_doc_limit: int
    dropped_by_budget: int
    stage_latency_ms: dict[str, float]
    intent_router: dict[str, object] = field(default_factory=dict)
    scope_resolution: dict[str, object] = field(default_factory=dict)
    coverage_plan: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentAnswerResult:
    request_id: str
    answer: str
    hits: list[SearchHit]
    candidates: list[SearchHit]
    reranked: list[SearchHit]
    trace: DocumentScopeTrace
    generation: AnswerGeneration


def build_scope_plan(
    *,
    config: RagConfig,
    tenant_id: str,
    query: str,
    doc_ids: list[str] | None,
    doc_version: int | None,
    include_all_sources: bool,
) -> ScopePlan:
    selected_doc_ids = stable_unique(doc_ids or [])
    current_versions = (
        {}
        if doc_version is not None
        else load_current_versions(config.object_store_dir, tenant_id=tenant_id, config=config)
    )
    available_guides = load_available_source_guides(
        config=config,
        tenant_id=tenant_id,
        doc_version=doc_version,
        current_versions=None if include_all_sources else current_versions,
    )
    selected_doc_ids = collapse_selected_doc_ids(
        selected_doc_ids,
        available_guides=available_guides,
    )
    explicit_guides = match_explicit_guides(query=query, guides=available_guides)
    route = classify_document_route(
        query=query,
        selected_doc_ids=selected_doc_ids,
        include_all_sources=include_all_sources,
        explicit_guides=explicit_guides,
    )
    resolved_doc_ids = resolve_doc_ids(
        route=route,
        selected_doc_ids=selected_doc_ids,
        explicit_guides=explicit_guides,
        available_guides=available_guides,
        include_all_sources=include_all_sources,
    )
    guides = filter_guides_for_scope(
        available_guides,
        resolved_doc_ids=resolved_doc_ids,
        explicit_guides=explicit_guides,
        route=route,
    )
    if route.coverage_required:
        guides = add_archived_doc_fallback_guides(
            config=config,
            tenant_id=tenant_id,
            guides=guides,
            resolved_doc_ids=resolved_doc_ids,
            doc_version=doc_version,
            current_versions=None if include_all_sources else current_versions,
        )
    covered_ids = {guide.source_doc_id for guide in guides}
    missing_doc_ids = [
        doc_id
        for doc_id in resolved_doc_ids
        if not source_guide_matches_any_doc_id(doc_id, covered_ids)
    ]
    coverage_mode = coverage_mode_for(route)
    section_summaries = (
        load_source_section_summaries(
            config.object_store_dir,
            tenant_id=tenant_id,
            source_keys={(guide.source_doc_id, guide.doc_version) for guide in guides},
        )
        if route.coverage_required and route.intent in detailed_section_intents()
        else []
    )
    return ScopePlan(
        route=route,
        selected_doc_ids=selected_doc_ids,
        resolved_doc_ids=resolved_doc_ids,
        guides=guides,
        missing_doc_ids=missing_doc_ids,
        current_versions=current_versions,
        coverage_mode=coverage_mode,
        section_summaries=section_summaries,
    )


def classify_document_route(
    *,
    query: str,
    selected_doc_ids: list[str],
    include_all_sources: bool,
    explicit_guides: list[SourceGuideRecord],
) -> DocumentRoute:
    normalized = normalize_text(query)
    has_scope = bool(selected_doc_ids or include_all_sources or explicit_guides)
    if not has_scope:
        return DocumentRoute(
            intent=OPEN_CHAT,
            scope=RETRIEVAL_DISCOVERED_DOCUMENTS,
            coverage_required=False,
            explicit_doc_refs=[],
            reason="当前没有选中或明确提及的文档，按普通对话处理。",
            confidence=0.9,
        )
    explicit_refs = [guide.source_doc_id for guide in explicit_guides]
    scope = EXPLICIT_NAMED_DOCUMENTS if explicit_guides else (
        ALL_VISIBLE_DOCUMENTS if include_all_sources and not selected_doc_ids else SELECTED_DOCUMENTS
    )
    intent = route_intent(normalized)
    if intent == LOCAL_QA:
        return DocumentRoute(
            intent=LOCAL_QA,
            scope=scope,
            coverage_required=False,
            explicit_doc_refs=explicit_refs,
            reason="该问题可由少量相关证据回答，无需逐份覆盖范围内的全部文档。",
            confidence=0.72 if not explicit_guides else 0.82,
        )
    coverage_required = True
    reason = "该任务需要对解析范围内的文档进行总结、综合、比较、抽取或报告生成。"
    if intent == LOCAL_QA and is_ambiguous_scope_query(normalized) and len(selected_doc_ids) >= 2:
        intent = SELECTED_DOC_SYNTHESIS
        reason = "问题表述较宽泛且选中了多个文档，采用覆盖式综合以避免遗漏。"
    return DocumentRoute(
        intent=intent,
        scope=scope,
        coverage_required=coverage_required,
        explicit_doc_refs=explicit_refs,
        reason=reason,
        confidence=0.84 if explicit_guides else 0.76,
    )


def route_intent(normalized_query: str) -> str:
    if contains_any(normalized_query, ["汇报", "提纲", "报告", "综述", "写一份", "生成"]):
        return REPORT_GENERATION
    if contains_any(normalized_query, ["对比", "比较", "区别", "差异", "共同点", "不同点", "异同"]):
        return SELECTED_DOC_COMPARE
    if contains_any(normalized_query, ["提取", "抽取", "列出", "整理成表", "条款", "字段", "指标"]):
        return SELECTED_DOC_EXTRACT
    if contains_any(normalized_query, ["总结", "概括", "梳理", "整理一下", "主要内容", "讲了什么", "核心内容"]):
        return SELECTED_DOC_SUMMARY
    if contains_any(normalized_query, ["风险", "问题", "结论", "趋势", "归纳", "分析", "反映", "观点", "主题"]):
        return SELECTED_DOC_SYNTHESIS
    return LOCAL_QA


def is_ambiguous_scope_query(normalized_query: str) -> bool:
    return contains_any(normalized_query, ["看看", "看一下", "这里面", "这批", "这些", "这几份", "这几个"])


def answer_document_scope(
    *,
    config: RagConfig,
    query: str,
    tenant_id: str,
    acl_groups: list[str] | None,
    plan: ScopePlan,
    request_id: str | None,
    stage_callback: StageCallback | None,
) -> DocumentAnswerResult:
    resolved_request_id = request_id or str(uuid.uuid4())
    emit_stage(
        stage_callback,
        "coverage_plan",
        "active",
        "覆盖范围规划",
        f"正在按“{document_intent_label(plan.route.intent)}”处理 {len(plan.resolved_doc_ids)} 个文档。",
        **plan.route.as_dict(),
        **plan.coverage(),
    )
    emit_stage(
        stage_callback,
        "search",
        "active",
        "文档摘要检索",
        "正在读取解析范围内的文档摘要和归档正文。",
    )
    guide_hits = source_guide_hits(
        plan.guides,
        section_summaries=plan.section_summaries,
        tenant_id=tenant_id,
        acl_groups=acl_groups or [],
    )
    emit_stage(
        stage_callback,
        "search",
        "done",
        "文档摘要检索",
        f"已读取 {len(guide_hits)} 份文档证据。",
        candidate_count=len(guide_hits),
    )
    emit_stage(
        stage_callback,
        "context",
        "active",
        "上下文组装",
        "正在根据上下文预算组织文档证据。",
    )
    max_chars = max(1000, config.max_context_chars)
    start = perf_counter()
    if total_hit_chars(guide_hits) <= max_chars:
        coverage_mode = ALL_DOCS_SUMMARY if plan.route.scope != EXPLICIT_NAMED_DOCUMENTS else EXPLICIT_DOCS_ONLY
        final_hits = guide_hits
        emit_stage(
            stage_callback,
            "context",
            "done",
            "上下文组装",
            f"已选择 {len(final_hits)} 份文档证据进入回答上下文。",
            context_count=len(final_hits),
        )
        emit_stage(
            stage_callback,
            "answer",
            "active",
            "大模型最终输出",
            "正在基于完整文档范围生成最终回答。",
        )
        generation = generate_answer(
            config,
            build_document_scope_query(query=query, plan=plan, map_reduce=False),
            final_hits,
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
        document_map_count = 0
    else:
        coverage_mode = PER_DOC_MAP_REDUCE
        partial_hits: list[SearchHit] = []
        for index, batch in enumerate(batch_hits_by_chars(guide_hits, max_chars=max_chars), start=1):
            emit_stage(
                stage_callback,
                "document_map",
                "active",
                "文档批次摘要",
                f"正在压缩第 {index} 批文档摘要。",
                batch_index=index,
                batch_doc_ids=[hit.doc_id for hit in batch],
            )
            partial = generate_answer(
                config,
                build_document_scope_query(query=query, plan=plan, map_reduce=True),
                batch,
            )
            partial_hits.append(
                SearchHit(
                    id=f"partial-summary-{index}",
                    score=1.0,
                    text=partial.answer,
                    doc_id=f"partial-summary-{index}",
                    title=f"文档批次摘要 {index}",
                    source_uri="memory://document-scope-map-reduce",
                    source_type="source_summary",
                    chunk_index=index - 1,
                    tenant_id=tenant_id,
                    acl_groups=acl_groups or [],
                    metadata={
                        "covered_doc_ids": [hit.doc_id for hit in batch],
                        "coverage_mode": PER_DOC_MAP_REDUCE,
                    },
                )
            )
            emit_stage(
                stage_callback,
                "document_map",
                "done",
                "文档批次摘要",
                f"第 {index} 批文档摘要已完成。",
                latency_ms=partial.latency_ms,
            )
        emit_stage(
            stage_callback,
            "document_reduce",
            "active",
            "文档综合归纳",
            "正在合并批次摘要生成最终回答。",
        )
        final_hits = partial_hits
        emit_stage(
            stage_callback,
            "context",
            "done",
            "上下文组装",
            f"已将 {len(final_hits)} 个批次摘要组装为最终回答上下文。",
            context_count=len(final_hits),
        )
        emit_stage(
            stage_callback,
            "answer",
            "active",
            "大模型最终输出",
            "正在综合全部批次摘要生成最终回答。",
        )
        generation = generate_answer(
            config,
            build_document_scope_query(query=query, plan=plan, map_reduce=False),
            final_hits,
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
        emit_stage(
            stage_callback,
            "document_reduce",
            "done",
            "文档综合归纳",
            "批次摘要已合并，最终回答已生成。",
            latency_ms=generation.latency_ms,
        )
        document_map_count = len(partial_hits)
    latency_ms = elapsed_ms(start)
    emit_stage(
        stage_callback,
        "coverage_plan",
        "done",
        "覆盖范围规划",
        f"已覆盖 {len(plan.guides)}/{len(plan.resolved_doc_ids)} 个文档。",
        latency_ms=latency_ms,
        coverage_mode=coverage_mode,
    )
    trace = DocumentScopeTrace(
        request_id=resolved_request_id,
        original_query=query,
        rewritten_query=query,
        rewrite_backend="document_scope_router",
        tenant_id=tenant_id,
        acl_groups=acl_groups or [],
        doc_version=None,
        current_versions=plan.current_versions,
        embedding_model="source_guides",
        source_types=[
            "source_summary",
            *(["section_summary"] if plan.section_summaries else []),
        ],
        doc_ids=plan.resolved_doc_ids,
        filter_expr=document_scope_filter_expr(tenant_id=tenant_id, doc_ids=plan.resolved_doc_ids),
        retrieval_mode="document_scope_coverage",
        candidate_count=len(guide_hits),
        reranked_count=len(guide_hits),
        context_count=len(final_hits),
        dropped_by_score=0,
        dropped_by_doc_limit=0,
        dropped_by_budget=max(0, len(guide_hits) - len(final_hits)),
        stage_latency_ms={
            "document_scope": latency_ms,
            "answer": generation.latency_ms,
        },
        intent_router=plan.route.as_dict(),
        scope_resolution={
            "selected_doc_ids": plan.selected_doc_ids,
            "resolved_doc_ids": plan.resolved_doc_ids,
            "explicit_doc_refs": plan.route.explicit_doc_refs,
            "missing_or_skipped_doc_ids": plan.missing_doc_ids,
        },
        coverage_plan={
            **plan.coverage(),
            "coverage_mode": coverage_mode,
            "document_map_batches": document_map_count,
        },
    )
    return DocumentAnswerResult(
        request_id=resolved_request_id,
        answer=generation.answer,
        hits=final_hits,
        candidates=guide_hits,
        reranked=guide_hits,
        trace=trace,
        generation=generation,
    )


def load_available_source_guides(
    *,
    config: RagConfig,
    tenant_id: str,
    doc_version: int | None,
    current_versions: dict[str, int] | None,
) -> list[SourceGuideRecord]:
    if not object_exists(config.object_store_dir, SOURCE_GUIDES_PATH):
        return []
    guides_by_source: dict[str, SourceGuideRecord] = {}
    for row in read_object_jsonl(config.object_store_dir, SOURCE_GUIDES_PATH):
        if str(row.get("tenant_id")) != tenant_id:
            continue
        source_doc_id = str(row.get("source_doc_id") or "")
        guide = str(row.get("guide") or "").strip()
        if not source_doc_id or not guide:
            continue
        row_version = int(row.get("doc_version", 0))
        if doc_version is not None:
            if row_version != int(doc_version):
                continue
        elif current_versions is not None:
            current = current_source_guide_version(source_doc_id, current_doc_versions=current_versions)
            if current != row_version:
                continue
        guides_by_source[source_doc_id] = SourceGuideRecord(
            tenant_id=tenant_id,
            source_doc_id=source_doc_id,
            doc_version=row_version,
            title=str(row.get("title") or source_doc_id).strip(),
            guide=guide,
        )
    return list(guides_by_source.values())


def collapse_selected_doc_ids(
    selected_doc_ids: list[str],
    *,
    available_guides: list[SourceGuideRecord],
) -> list[str]:
    """Convert page/image child IDs into logical uploaded-document IDs."""
    collapsed: list[str] = []
    for doc_id in selected_doc_ids:
        matching_source_ids = [
            guide.source_doc_id
            for guide in available_guides
            if source_guide_matches_any_doc_id(guide.source_doc_id, {doc_id})
        ]
        collapsed.extend(matching_source_ids or [doc_id])
    return stable_unique(collapsed)


def match_explicit_guides(*, query: str, guides: list[SourceGuideRecord]) -> list[SourceGuideRecord]:
    normalized_query = normalize_text(query)
    matched: list[SourceGuideRecord] = []
    for guide in guides:
        aliases = doc_aliases(guide)
        if any(alias and alias in normalized_query for alias in aliases):
            matched.append(guide)
    return matched


def resolve_doc_ids(
    *,
    route: DocumentRoute,
    selected_doc_ids: list[str],
    explicit_guides: list[SourceGuideRecord],
    available_guides: list[SourceGuideRecord],
    include_all_sources: bool,
) -> list[str]:
    if explicit_guides:
        return [guide.source_doc_id for guide in explicit_guides]
    if selected_doc_ids:
        return selected_doc_ids
    if include_all_sources:
        return [guide.source_doc_id for guide in available_guides]
    return []


def filter_guides_for_scope(
    guides: list[SourceGuideRecord],
    *,
    resolved_doc_ids: list[str],
    explicit_guides: list[SourceGuideRecord],
    route: DocumentRoute,
) -> list[SourceGuideRecord]:
    if explicit_guides:
        return explicit_guides
    if not resolved_doc_ids:
        return []
    allowed = set(resolved_doc_ids)
    return [
        guide
        for guide in guides
        if source_guide_matches_any_doc_id(guide.source_doc_id, allowed)
    ]


def add_archived_doc_fallback_guides(
    *,
    config: RagConfig,
    tenant_id: str,
    guides: list[SourceGuideRecord],
    resolved_doc_ids: list[str],
    doc_version: int | None,
    current_versions: dict[str, int] | None,
) -> list[SourceGuideRecord]:
    covered_ids = {guide.source_doc_id for guide in guides}
    missing = [
        doc_id
        for doc_id in resolved_doc_ids
        if not source_guide_matches_any_doc_id(doc_id, covered_ids)
    ]
    if not missing:
        return guides
    docs = [
        doc
        for doc in load_archived_source_documents(config.object_store_dir)
        if doc.tenant_id == tenant_id
        and archived_doc_version_matches(doc, doc_version=doc_version, current_versions=current_versions)
    ]
    fallback_guides: list[SourceGuideRecord] = []
    for doc_id in missing:
        matched_docs = [
            doc
            for doc in docs
            if source_guide_matches_any_doc_id(doc.doc_id, {doc_id})
        ]
        if not matched_docs:
            continue
        fallback_guides.append(source_docs_to_fallback_guide(doc_id=doc_id, docs=matched_docs))
    return [*guides, *fallback_guides]


def archived_doc_version_matches(
    doc: SourceDocument,
    *,
    doc_version: int | None,
    current_versions: dict[str, int] | None,
) -> bool:
    if doc_version is not None:
        return int(doc.doc_version) == int(doc_version)
    if current_versions is None:
        return True
    current = current_versions.get(doc.doc_id)
    if current is None:
        return False
    return int(current) == int(doc.doc_version)


def source_docs_to_fallback_guide(*, doc_id: str, docs: list[SourceDocument]) -> SourceGuideRecord:
    sorted_docs = sorted(docs, key=lambda doc: (doc.doc_version, doc.doc_id, str(doc.metadata.get("page_no") or "")))
    title = sorted_docs[0].title or doc_id
    doc_version = max(int(doc.doc_version) for doc in sorted_docs)
    blocks = []
    for doc in sorted_docs[:40]:
        text = doc.text.strip()
        if not text:
            continue
        page_no = doc.metadata.get("page_no")
        location = f"第 {page_no} 页" if page_no is not None else doc.title
        blocks.append(f"[{location}]\n{text[:1200]}")
    guide = "\n\n".join(blocks).strip() or "该文档暂无可用解析正文。"
    return SourceGuideRecord(
        tenant_id=sorted_docs[0].tenant_id,
        source_doc_id=doc_id,
        doc_version=doc_version,
        title=title,
        guide=guide,
    )


def source_guide_hits(
    guides: list[SourceGuideRecord],
    *,
    section_summaries: list[SourceSectionSummary] | None = None,
    tenant_id: str,
    acl_groups: list[str],
) -> list[SearchHit]:
    guide_hits = [
        SearchHit(
            id=f"source-guide-{guide.source_doc_id}-{guide.doc_version}",
            score=1.0,
            text=f"文档标题: {guide.title}\n文档摘要:\n{guide.guide}",
            doc_id=guide.source_doc_id,
            title=guide.title,
            source_uri=f"source-guide://{guide.source_doc_id}",
            source_type="source_summary",
            chunk_index=0,
            tenant_id=tenant_id,
            acl_groups=acl_groups,
            metadata={
                "doc_version": guide.doc_version,
                "coverage_mode": "document_scope",
                "source_doc_id": guide.source_doc_id,
            },
        )
        for guide in guides
    ]
    section_hits = [
        SearchHit(
            id=(
                f"section-summary-{section.source_doc_id}-"
                f"{section.doc_version}-{section.section_index}"
            ),
            score=1.0,
            text=f"文档标题: {section.title}\n章节提取摘要:\n{section.summary}",
            doc_id=section.source_doc_id,
            title=section.title,
            source_uri=f"section-summary://{section.source_doc_id}/{section.section_index}",
            source_type="section_summary",
            chunk_index=section.section_index,
            tenant_id=tenant_id,
            acl_groups=acl_groups,
            metadata={
                "doc_version": section.doc_version,
                "coverage_mode": "document_scope_section",
                "source_doc_id": section.source_doc_id,
                "section_index": section.section_index,
            },
        )
        for section in (section_summaries or [])
    ]
    return [*guide_hits, *section_hits]


def detailed_section_intents() -> set[str]:
    return {
        SELECTED_DOC_COMPARE,
        SELECTED_DOC_SYNTHESIS,
        SELECTED_DOC_EXTRACT,
        REPORT_GENERATION,
    }


def build_document_scope_query(*, query: str, plan: ScopePlan, map_reduce: bool) -> str:
    task_hint = {
        SELECTED_DOC_SUMMARY: "总结每个文档的主要内容，并综合归纳整体主题。",
        SELECTED_DOC_COMPARE: "对比这些文档的共同点、差异点和关键结论。",
        SELECTED_DOC_SYNTHESIS: "归纳这些文档反映的问题、主题、风险、趋势或结论。",
        SELECTED_DOC_EXTRACT: "从每个文档中抽取用户要求的信息，并合并为结构化结果。",
        REPORT_GENERATION: "基于这些文档生成汇报、提纲、报告或综述。",
    }.get(plan.route.intent, "回答用户问题。")
    mode = "这是 map 阶段，请只处理当前批次证据。" if map_reduce else "这是最终回答阶段，请综合所有给定证据。"
    return (
        f"{query}\n\n"
        f"任务类型: {plan.route.intent}\n"
        f"覆盖要求: 必须覆盖解析范围内的文档。\n"
        f"解析范围文档数: {len(plan.resolved_doc_ids)}\n"
        f"已加载文档摘要数: {len(plan.guides)}\n"
        f"缺失或跳过文档: {', '.join(plan.missing_doc_ids) if plan.missing_doc_ids else '无'}\n"
        f"处理阶段: {mode}\n"
        f"任务说明: {task_hint}\n"
        "回答时请优先按文档范围给出整体结论，并在有证据支持的句子后标注引用编号。"
    )


def coverage_mode_for(route: DocumentRoute) -> str:
    if not route.coverage_required:
        return TOP_K
    if route.scope == EXPLICIT_NAMED_DOCUMENTS:
        return EXPLICIT_DOCS_ONLY
    return ALL_DOCS_SUMMARY


def document_intent_label(intent: str) -> str:
    return {
        LOCAL_QA: "局部问答",
        SELECTED_DOC_SUMMARY: "选中文档总结",
        SELECTED_DOC_COMPARE: "选中文档比较",
        SELECTED_DOC_SYNTHESIS: "选中文档综合",
        SELECTED_DOC_EXTRACT: "选中文档信息抽取",
        REPORT_GENERATION: "报告生成",
        OPEN_CHAT: "普通对话",
    }.get(intent, "文档任务")


def document_scope_filter_expr(*, tenant_id: str, doc_ids: list[str]) -> str:
    if not doc_ids:
        return f'tenant_id == "{tenant_id}"'
    quoted = ", ".join(f'"{doc_id}"' for doc_id in doc_ids)
    return f'tenant_id == "{tenant_id}" and doc_id in [{quoted}]'


def total_hit_chars(hits: Iterable[SearchHit]) -> int:
    return sum(len(hit.text) for hit in hits)


def batch_hits_by_chars(hits: list[SearchHit], *, max_chars: int) -> list[list[SearchHit]]:
    batches: list[list[SearchHit]] = []
    current: list[SearchHit] = []
    current_chars = 0
    for hit in hits:
        hit_chars = len(hit.text)
        if current and current_chars + hit_chars > max_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(hit)
        current_chars += hit_chars
    if current:
        batches.append(current)
    return batches


def stable_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def doc_aliases(guide: SourceGuideRecord) -> list[str]:
    raw_aliases = {
        guide.source_doc_id,
        guide.title,
        guide.source_doc_id.rsplit("/", 1)[-1],
        guide.title.rsplit("/", 1)[-1],
    }
    aliases = [normalize_text(alias) for alias in raw_aliases if alias]
    return [alias for alias in aliases if len(alias) >= 2]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value.lower())


def contains_any(value: str, keywords: list[str]) -> bool:
    return any(normalize_text(keyword) in value for keyword in keywords)


def source_guide_matches_any_doc_id(source_doc_id: str, doc_ids: set[str]) -> bool:
    if source_doc_id in doc_ids:
        return True
    child_prefix = f"{source_doc_id}/"
    if any(doc_id.startswith(child_prefix) for doc_id in doc_ids):
        return True
    return any(source_doc_id.startswith(f"{doc_id}/") for doc_id in doc_ids)


def elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 2)
