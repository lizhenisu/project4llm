from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rag_core.config import RagConfig
from rag_core.database import connect_metadata_db
from rag_core.object_store import load_archived_source_documents
from rag_core.text_utils import now_ms


ARTIFACTS_DIR = Path("artifacts")
DEFAULT_MINDMAP_BATCH_CHUNK_COUNT = 5
MINDMAP_MAX_CHILDREN = 8
MINDMAP_MAX_GRANDCHILDREN = 6
TABLE_CHUNK_CHARS = 9000
TABLE_MAX_ROWS = 24
TABLE_MAX_COLUMNS = 8


@dataclass(frozen=True)
class MindMapArtifact:
    id: str
    title: str
    status: str
    tenant_id: str
    source_doc_ids: list[str]
    created_at: int
    updated_at: int
    artifact_type: str = "mindmap"
    root: dict[str, Any] | None = None
    table: dict[str, Any] | None = None
    error: str = ""


def list_artifacts(config: RagConfig, *, tenant_id: str) -> list[MindMapArtifact]:
    artifact_dir = config.object_store_dir / ARTIFACTS_DIR / tenant_id
    if not artifact_dir.exists():
        return []
    artifacts = [
        artifact_from_row(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(artifact_dir.glob("*.json"))
    ]
    return sorted(artifacts, key=lambda item: item.updated_at, reverse=True)


def load_artifact(
    config: RagConfig,
    *,
    tenant_id: str,
    artifact_id: str,
) -> MindMapArtifact | None:
    path = artifact_path(config, tenant_id=tenant_id, artifact_id=artifact_id)
    if not path.exists():
        return None
    return artifact_from_row(json.loads(path.read_text(encoding="utf-8")))


def delete_artifact(config: RagConfig, *, tenant_id: str, artifact_id: str) -> bool:
    path = artifact_path(config, tenant_id=tenant_id, artifact_id=artifact_id)
    if not path.exists():
        return False
    path.unlink()
    return True


def list_metadata_artifacts(config: RagConfig, *, tenant_id: str) -> list[MindMapArtifact]:
    with connect_metadata_db(config) as conn:
        rows = conn.execute(
            """
            SELECT * FROM artifacts
            WHERE tenant_id = ?
            ORDER BY updated_at DESC
            """,
            (tenant_id,),
        ).fetchall()
    return [artifact_from_metadata_row(row) for row in rows]


def load_metadata_artifact(config: RagConfig, *, tenant_id: str, artifact_id: str) -> MindMapArtifact | None:
    with connect_metadata_db(config) as conn:
        row = conn.execute(
            "SELECT * FROM artifacts WHERE tenant_id = ? AND id = ?",
            (tenant_id, artifact_id),
        ).fetchone()
    return artifact_from_metadata_row(row) if row is not None else None


def save_metadata_artifact(config: RagConfig, artifact: MindMapArtifact) -> None:
    with connect_metadata_db(config) as conn:
        conn.execute(
            """
            INSERT INTO artifacts(
                id, tenant_id, title, status, artifact_type, source_doc_ids,
                root, table_json, error, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                tenant_id = excluded.tenant_id,
                title = excluded.title,
                status = excluded.status,
                artifact_type = excluded.artifact_type,
                source_doc_ids = excluded.source_doc_ids,
                root = excluded.root,
                table_json = excluded.table_json,
                error = excluded.error,
                updated_at = excluded.updated_at
            """,
            (
                artifact.id,
                artifact.tenant_id,
                artifact.title,
                artifact.status,
                artifact.artifact_type,
                json.dumps(artifact.source_doc_ids, ensure_ascii=False),
                json.dumps(artifact.root, ensure_ascii=False) if artifact.root is not None else None,
                json.dumps(artifact.table, ensure_ascii=False) if artifact.table is not None else None,
                artifact.error,
                artifact.created_at,
                artifact.updated_at,
            ),
        )


def delete_metadata_artifact(config: RagConfig, *, tenant_id: str, artifact_id: str) -> bool:
    with connect_metadata_db(config) as conn:
        cursor = conn.execute(
            "DELETE FROM artifacts WHERE tenant_id = ? AND id = ?",
            (tenant_id, artifact_id),
        )
        return cursor.rowcount > 0


def fail_metadata_artifact(config: RagConfig, artifact: MindMapArtifact, error: str) -> None:
    save_metadata_artifact(
        config,
        replace(
            artifact,
            status="failed",
            error=error,
            updated_at=now_ms(),
        ),
    )


def create_mindmap_artifact(
    config: RagConfig,
    *,
    title: str,
    tenant_id: str,
    source_doc_ids: list[str],
    acl_groups: list[str] | None,
    doc_version: int | None = None,
    batch_chunk_count: int = DEFAULT_MINDMAP_BATCH_CHUNK_COUNT,
) -> MindMapArtifact:
    artifact_id = f"mindmap-{uuid.uuid4().hex[:12]}"
    timestamp = now_ms()
    root = build_mindmap_root(
        title=title or infer_title(source_doc_ids),
        config=config,
        tenant_id=tenant_id,
        source_doc_ids=source_doc_ids,
        batch_chunk_count=batch_chunk_count,
    )
    artifact = MindMapArtifact(
        id=artifact_id,
        title=title or root["label"],
        status="ready",
        tenant_id=tenant_id,
        source_doc_ids=source_doc_ids,
        created_at=timestamp,
        updated_at=timestamp,
        artifact_type="mindmap",
        root=root,
    )
    save_artifact(config, artifact)
    return artifact


def create_table_artifact(
    config: RagConfig,
    *,
    title: str,
    tenant_id: str,
    source_doc_ids: list[str],
    acl_groups: list[str] | None,
    doc_version: int | None = None,
) -> MindMapArtifact:
    artifact_id = f"table-{uuid.uuid4().hex[:12]}"
    timestamp = now_ms()
    table = build_llm_table(
        title=title or infer_table_title(source_doc_ids),
        config=config,
        tenant_id=tenant_id,
        source_doc_ids=source_doc_ids,
    )
    artifact = MindMapArtifact(
        id=artifact_id,
        title=title or table.get("title") or infer_table_title(source_doc_ids),
        status="ready",
        tenant_id=tenant_id,
        source_doc_ids=source_doc_ids,
        created_at=timestamp,
        updated_at=timestamp,
        artifact_type="table",
        table=table,
    )
    save_artifact(config, artifact)
    return artifact


def build_mindmap_root(
    *,
    title: str,
    config: RagConfig | None = None,
    tenant_id: str | None = None,
    source_doc_ids: list[str] | None = None,
    batch_chunk_count: int = DEFAULT_MINDMAP_BATCH_CHUNK_COUNT,
) -> dict[str, Any]:
    return build_llm_outline(
        title=title,
        config=config,
        tenant_id=tenant_id,
        source_doc_ids=source_doc_ids or [],
        batch_chunk_count=batch_chunk_count,
    )


def build_llm_outline(
    *,
    title: str,
    config: RagConfig | None,
    tenant_id: str | None,
    source_doc_ids: list[str],
    batch_chunk_count: int = DEFAULT_MINDMAP_BATCH_CHUNK_COUNT,
) -> dict[str, Any] | None:
    if config is None or tenant_id is None or not source_doc_ids:
        raise ValueError("tenant_id and source_doc_ids are required for mind map generation.")
    if not config.llm_base_url or not config.llm_api_key:
        raise RuntimeError("NEW_API_URL/NEW_API_KEY must be configured for LLM mind map generation.")

    docs = load_selected_archived_docs(config=config, tenant_id=tenant_id, source_doc_ids=source_doc_ids)
    if not docs:
        raise RuntimeError("No archived source documents found for LLM mind map generation.")

    batches = batch_mindmap_docs(docs, batch_chunk_count=batch_chunk_count)
    if not batches:
        raise RuntimeError("Archived source documents are empty; cannot generate mind map.")

    partial_roots = [
        generate_partial_mindmap_with_llm(
            config=config,
            title=title,
            batch_text=batch_text,
            index=index,
            total=len(batches),
        )
        for index, batch_text in enumerate(batches, start=1)
    ]
    root = merge_mindmaps_with_llm(config=config, title=title, partial_roots=partial_roots)
    return normalize_mindmap_node(root, default_label=title or infer_title(source_doc_ids), node_id="root")


def build_llm_table(
    *,
    title: str,
    config: RagConfig | None,
    tenant_id: str | None,
    source_doc_ids: list[str],
) -> dict[str, Any]:
    if config is None or tenant_id is None or not source_doc_ids:
        raise ValueError("tenant_id and source_doc_ids are required for table generation.")
    if not config.llm_base_url or not config.llm_api_key:
        raise RuntimeError("NEW_API_URL/NEW_API_KEY must be configured for LLM table generation.")

    docs = load_selected_archived_docs(config=config, tenant_id=tenant_id, source_doc_ids=source_doc_ids)
    if not docs:
        raise RuntimeError("No archived source documents found for LLM table generation.")

    text = "\n\n".join(format_doc_for_mindmap(doc) for doc in docs if doc.text.strip()).strip()
    chunks = split_mindmap_text(text, chunk_chars=TABLE_CHUNK_CHARS, max_chunks=1)
    if not chunks:
        raise RuntimeError("Archived source documents are empty; cannot generate table.")

    raw_table = generate_table_with_llm(config=config, title=title, source_text=chunks[0])
    return normalize_table(raw_table, default_title=title)


def load_selected_archived_docs(*, config: RagConfig, tenant_id: str, source_doc_ids: list[str]) -> list:
    selected_doc_ids = set(source_doc_ids)
    docs = [
        doc
        for doc in load_archived_source_documents(config.object_store_dir)
        if doc.tenant_id == tenant_id and doc.doc_id in selected_doc_ids
    ]
    return sorted(docs, key=lambda doc: (str(doc.metadata.get("relative_path") or ""), source_page_no(doc), doc.doc_id))


def format_doc_for_mindmap(doc) -> str:
    location = f"第 {source_page_no(doc)} 页" if source_page_no(doc) else doc.title
    return f"[{location}]\n{doc.text.strip()}"


def batch_mindmap_docs(docs: list, *, batch_chunk_count: int) -> list[str]:
    batch_size = max(1, batch_chunk_count)
    formatted_docs = [format_doc_for_mindmap(doc) for doc in docs if doc.text.strip()]
    return [
        "\n\n".join(formatted_docs[start : start + batch_size])
        for start in range(0, len(formatted_docs), batch_size)
    ]


def split_mindmap_text(text: str, *, chunk_chars: int, max_chunks: int) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in text.splitlines() if paragraph.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        if current and current_len + len(paragraph) > chunk_chars:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
            if len(chunks) >= max_chunks:
                break
        current.append(paragraph)
        current_len += len(paragraph)
    if current and len(chunks) < max_chunks:
        chunks.append("\n".join(current))
    return chunks


def generate_partial_mindmap_with_llm(
    *,
    config: RagConfig,
    title: str,
    batch_text: str,
    index: int,
    total: int,
) -> dict[str, Any]:
    prompt = f"""你是知识库思维导图专家。请把下面第 {index}/{total} 批原文块整理为局部思维导图。

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
    return call_mindmap_llm(
        config=config,
        system_prompt="你只输出合法 JSON。你擅长把长文整理为层级清晰、节点简洁的中文思维导图。",
        user_prompt=prompt,
    )


def merge_mindmaps_with_llm(*, config: RagConfig, title: str, partial_roots: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = f"""请把多个局部思维导图合并成一个最终思维导图。

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
    return call_mindmap_llm(
        config=config,
        system_prompt="你只输出合法 JSON。你负责合并、去重和压缩思维导图节点。",
        user_prompt=prompt,
    )


def generate_table_with_llm(*, config: RagConfig, title: str, source_text: str) -> dict[str, Any]:
    prompt = f"""你是知识库数据表格专家。请把下面原文整理成一个适合阅读和比较的数据表格。

要求:
- 只依据原文，不编造。
- 输出严格 JSON，不要 Markdown。
- JSON schema: {{"title": "表格标题", "columns": ["列名1", "列名2"], "rows": [["单元格1", "单元格2"]], "summary": "一句话说明表格用途"}}
- 根据资料内容选择最有价值的列，例如标题、作者、主题、主要发现、关键引文、城市、最佳时间、景点、费用、岗位、职责、要求等。
- 列数 3-8 列，行数 3-24 行。
- 单元格要简洁；没有证据的单元格填“未提及”。

表格任务: {title}

原文:
{source_text}
"""
    return call_json_llm(
        config=config,
        system_prompt="你只输出合法 JSON。你擅长从文档中抽取结构化表格。",
        user_prompt=prompt,
    )


def call_mindmap_llm(*, config: RagConfig, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    return call_json_llm(config=config, system_prompt=system_prompt, user_prompt=user_prompt)


def call_json_llm(*, config: RagConfig, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)
    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content or ""
    return parse_mindmap_json(content)


def parse_mindmap_json(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Mind map LLM response must be a JSON object.")
    return parsed


def normalize_mindmap_node(raw: dict[str, Any], *, default_label: str, node_id: str) -> dict[str, Any]:
    label = clean_label(str(raw.get("label") or default_label or "思维导图"))
    raw_children = raw.get("children") if isinstance(raw.get("children"), list) else []
    children = [
        normalize_mindmap_child(child, parent_id=node_id, index=index)
        for index, child in enumerate(raw_children[:MINDMAP_MAX_CHILDREN])
        if isinstance(child, dict)
    ]
    return {
        "id": node_id,
        "label": label,
        "children": children,
        "citationIds": [],
    }


def normalize_mindmap_child(raw: dict[str, Any], *, parent_id: str, index: int) -> dict[str, Any]:
    node_id = f"{parent_id}-{index}"
    label = clean_label(str(raw.get("label") or f"主题 {index + 1}"))
    raw_children = raw.get("children") if isinstance(raw.get("children"), list) else []
    children = [
        {
            "id": f"{node_id}-{child_index}",
            "label": clean_label(str(child.get("label") or f"要点 {child_index + 1}")),
            "children": [],
            "citationIds": [],
        }
        for child_index, child in enumerate(raw_children[:MINDMAP_MAX_GRANDCHILDREN])
        if isinstance(child, dict)
    ]
    return {
        "id": node_id,
        "label": label,
        "children": children,
        "citationIds": [],
    }


def clean_label(value: str) -> str:
    return value.strip().strip("：:，,。；;")[:80]


def normalize_table(raw: dict[str, Any], *, default_title: str) -> dict[str, Any]:
    columns = raw.get("columns") if isinstance(raw.get("columns"), list) else []
    clean_columns = [clean_table_cell(column)[:40] for column in columns if clean_table_cell(column)]
    clean_columns = clean_columns[:TABLE_MAX_COLUMNS]
    if not clean_columns:
        clean_columns = ["主题", "要点", "证据"]

    rows = raw.get("rows") if isinstance(raw.get("rows"), list) else []
    clean_rows: list[list[str]] = []
    for row in rows[:TABLE_MAX_ROWS]:
        values = row if isinstance(row, list) else []
        clean_row = [clean_table_cell(value) for value in values[: len(clean_columns)]]
        if len(clean_row) < len(clean_columns):
            clean_row.extend(["未提及"] * (len(clean_columns) - len(clean_row)))
        if any(cell and cell != "未提及" for cell in clean_row):
            clean_rows.append(clean_row)
    return {
        "title": clean_table_cell(raw.get("title")) or default_title or "数据表格",
        "columns": clean_columns,
        "rows": clean_rows,
        "summary": clean_table_cell(raw.get("summary")),
    }


def clean_table_cell(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()[:260]


def source_page_no(doc) -> int:
    page_no = doc.metadata.get("page_no")
    if isinstance(page_no, int):
        return page_no
    if isinstance(page_no, str) and page_no.isdigit():
        return int(page_no)
    return 0


def infer_title(source_doc_ids: list[str]) -> str:
    if source_doc_ids:
        return source_doc_ids[0]
    return "思维导图"


def infer_table_title(source_doc_ids: list[str]) -> str:
    if source_doc_ids:
        return f"{source_doc_ids[0]} 数据表格"
    return "数据表格"


def save_artifact(config: RagConfig, artifact: MindMapArtifact) -> None:
    path = artifact_path(config, tenant_id=artifact.tenant_id, artifact_id=artifact.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact.__dict__, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def artifact_path(config: RagConfig, *, tenant_id: str, artifact_id: str) -> Path:
    return config.object_store_dir / ARTIFACTS_DIR / tenant_id / f"{artifact_id}.json"


def artifact_from_row(row: dict[str, Any]) -> MindMapArtifact:
    return MindMapArtifact(
        id=str(row["id"]),
        title=str(row["title"]),
        status=str(row.get("status", "ready")),
        tenant_id=str(row["tenant_id"]),
        source_doc_ids=list(row.get("source_doc_ids") or []),
        created_at=int(row.get("created_at") or 0),
        updated_at=int(row.get("updated_at") or 0),
        artifact_type=str(row.get("artifact_type") or "mindmap"),
        root=row.get("root"),
        table=row.get("table"),
        error=str(row.get("error") or ""),
    )


def artifact_from_metadata_row(row) -> MindMapArtifact:
    return MindMapArtifact(
        id=str(row["id"]),
        title=str(row["title"]),
        status=str(row["status"]),
        tenant_id=str(row["tenant_id"]),
        source_doc_ids=json.loads(row["source_doc_ids"] or "[]"),
        created_at=int(row["created_at"] or 0),
        updated_at=int(row["updated_at"] or 0),
        artifact_type=str(row["artifact_type"] or "mindmap"),
        root=json.loads(row["root"]) if row["root"] else None,
        table=json.loads(row["table_json"]) if row["table_json"] else None,
        error=str(row["error"] or ""),
    )
