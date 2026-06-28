from __future__ import annotations

import json
from datetime import date
from typing import Any

from rag_core.types import SearchHit


ANSWER_SYSTEM_PROMPT = (
    "你是企业知识库问答助手。对于知识库事实问题，你必须严格依据检索证据回答；"
    "证据不足时自然说明边界，不要用外部知识补齐事实或因果解释。"
    "只有用户明确询问通用知识时，才可以补充并清楚标明它不来自知识库。"
)

QUERY_REWRITE_SYSTEM_PROMPT = (
    "你是 RAG 查询改写器。不要回答问题。"
    "保留专有名词、公式、数字和缩写；当问题或资料含中英技术术语时，"
    "用简洁的中文概念加对应英文关键词扩展查询，以便检索英文原文。"
    "只返回 JSON 对象：{\"query\":\"主检索查询\",\"english_keywords\":\"英文关键词\"}。"
    "不要添加用户没有表达的权限、租户或事实。"
)

SOURCE_GUIDE_SYSTEM_PROMPT = "你是企业知识库来源指南助手。你只依据给定原文，为读者生成简洁准确的中文摘要。"

PARTIAL_MINDMAP_SYSTEM_PROMPT = "你只输出合法 JSON。你擅长把长文整理为层级清晰、节点简洁的中文思维导图。"

MERGE_MINDMAP_SYSTEM_PROMPT = "你只输出合法 JSON。你负责合并、去重和压缩思维导图节点。"

DATA_TABLE_SYSTEM_PROMPT = "你只输出合法 JSON。你擅长从文档中抽取结构化表格。"


def build_answer_prompt(
    query: str,
    hits: list[SearchHit],
    *,
    current_date: date | None = None,
) -> str:
    system_date = current_date or date.today()
    evidence = []
    for index, hit in enumerate(hits, start=1):
        evidence.append(
            "\n".join(
                [
                    format_evidence_header(index, hit),
                    hit.text,
                ]
            )
        )
    evidence_text = "\n\n".join(evidence) if evidence else "无"
    image_rule = (
        "\n- 图片证据来自 OCR/caption 或图片向量召回，可能不完整；回答时必须把它当作图片派生证据。"
        if any(hit.source_type == "image" for hit in hits)
        else ""
    )
    return f"""问题:
{query}

当前系统日期:
{system_date.isoformat()}

证据:
{evidence_text}

回答策略:
- 优先回答用户真正想问的问题，不要机械拒答。
- 证据中能支持的事实、数字、岗位、要求、流程、结论，必须基于证据回答，并在对应句子后标注引用编号。
- 如果用户问到的某个点在证据中没有出现，要明确说“来源中未提及/没有具体说明”，但仍然可以整理证据中相关的可用信息。
- 如果问题完全不是知识库事实问题，例如日期、星期、简单计算、语言解释或通用常识，可以直接回答；这类内容不要添加引用，并说明它不是来自知识库证据。
- 对知识库事实问题，不要为了让解释更完整而补充证据未出现的理论动机、因果机制或训练效果。
- 当问题要求列举某篇资料中的若干项时，以同一章节的原文段落为准，原文优先于批次摘要或综合摘要。
- 不要为了凑足数量而跨章节补入相关设置；原文分别列出的应用位置应分别保留，不要先合并再用其他实验设置补位。
- 如果原文声明有若干类、但后续只有较少的命名小标题，应把原文分别描述的应用对象或位置逐项计数，不要猜测一个原文未列出的标题。
- 例如原文说“对 A 应用某机制，另外对 B 也应用该机制”，随后列出 C；若问题要求三项，就分别列为“A 上的机制、B 上的机制、C”，不要把 A/B 合并后留下空缺。
- 可以给出合理的下一步建议，例如建议查看官方渠道、联系人、制度文档或在面试/沟通中确认；建议不要伪装成证据事实。
- 不要编造公司关系、政策、待遇、时间、人物、法规、价格等需要来源支撑的事实。
- 当证据和通用知识冲突时，以证据为准，并指出证据范围。
- 回答要自然、具体、易读；优先使用简短段落或项目符号。
{image_rule}
"""


def build_query_rewrite_prompt(*, source_summary_text: str = "", history_text: str, query: str) -> str:
    return f"""资料摘要:
{source_summary_text or '无'}

对话历史:
{history_text or '无'}

当前问题:
{query}

检索改写要求:
- 只输出指定 JSON，不要回答。
- 保留问题原意，并补充原文中可能出现的英文技术关键词。
- 中英混合资料必须同时保留中文概念和 English terms。"""


def build_source_guide_prompt(*, title: str, source_text: str) -> str:
    return f"""请为下面这个知识库来源生成“来源指南”摘要。

要求:
- 只依据原文，不编造。
- 输出 2-4 句中文自然语言摘要。
- 说明这份资料主要讲什么、包含哪些关键信息、适合用来回答什么类型的问题。
- 不要输出标题、Markdown、编号列表或引用标记。
- 不要直接复制原文长句。

来源标题: {title}

原文:
{source_text}
"""


def build_partial_mindmap_prompt(*, title: str, batch_text: str, index: int, total: int) -> str:
    return f"""你是知识库思维导图专家。请把下面第 {index}/{total} 批原文块整理为局部思维导图。

要求:
- 只依据原文，不编造。
- 输出严格 JSON，不要 Markdown。
- JSON schema: {{"label": "局部主题", "children": [{{"label": "二级主题", "children": [{{"label": "三级要点", "children": []}}]}}]}}
- 二级主题 3-6 个，每个二级主题下三级要点 2-5 个。
- 标签要短，像思维导图节点，不要长段落。

总标题: {title}

原文块批次:
{batch_text}
"""


def build_merge_mindmap_prompt(*, title: str, partial_roots: list[dict[str, Any]]) -> str:
    return f"""请把多个局部思维导图合并成一个最终思维导图。

要求:
- 输出严格 JSON，不要 Markdown。
- JSON schema: {{"label": "总主题", "children": [{{"label": "二级主题", "children": [{{"label": "三级要点", "children": []}}]}}]}}
- 合并同义或重复节点。
- 最终二级主题控制在 4-8 个。
- 每个二级主题下三级要点控制在 2-6 个。
- 节点标签简洁、准确、适合前端思维导图展示。

总标题: {title}

局部思维导图 JSON:
{json.dumps(partial_roots, ensure_ascii=False)}
"""


def build_data_table_prompt(*, title: str, source_text: str) -> str:
    return f"""你是知识库数据表格专家。请把下面原文整理成一个适合阅读和比较的数据表格。

要求:
- 只依据原文，不编造。
- 输出严格 JSON，不要 Markdown。
- JSON schema: {{"title": "表格标题", "columns": ["列名1", "列名2"], "rows": [["单元格1", "单元格2"]], "summary": "一句话说明表格用途"}}
- 根据资料内容选择最有价值的列，例如标题、作者、主题、主要发现、关键引文、城市、最佳时间、景点、费用、岗位、职责、要求等。
- 列数 3-8 列，行数 1-24 行，只保留至少一个单元格含有实质信息的行。
- 单元格要简洁；个别没有证据的单元格可填“未提及”，但整行均为“未提及/未说明/暂无”时必须省略该行。

表格任务: {title}

原文:
{source_text}
"""


def format_location(metadata: dict) -> str:
    if not metadata:
        return ""
    if "page_start" in metadata and "page_end" in metadata:
        start = metadata["page_start"]
        end = metadata["page_end"]
        return f", page={start}" if start == end else f", pages={start}-{end}"
    if "page_no" in metadata:
        return f", page={metadata['page_no']}"
    if "row_start" in metadata and "row_end" in metadata:
        return f", rows={metadata['row_start']}-{metadata['row_end']}"
    if "bbox" in metadata and metadata["bbox"]:
        return f", bbox={metadata['bbox']}"
    return ""


def format_evidence_header(index: int, hit: SearchHit) -> str:
    parts = [
        f"[{index}] doc_id={hit.doc_id}",
        f"title={hit.title}",
        f"source_type={hit.source_type}",
        f"source_uri={hit.source_uri}",
        f"chunk_index={hit.chunk_index}",
    ]
    location = format_location(hit.metadata)
    if location:
        parts.append(location.lstrip(", "))
    if hit.source_type == "image":
        image_uri = hit.metadata.get("image_uri") or hit.source_uri
        parts.append(f"image_uri={image_uri}")
        if hit.metadata.get("linked_doc_id"):
            parts.append(f"linked_doc_id={hit.metadata['linked_doc_id']}")
    return ", ".join(parts)
