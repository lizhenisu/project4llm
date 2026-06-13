from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_core.config import RagConfig
from rag_core.object_store import load_archived_source_documents
from rag_core.text_utils import now_ms


ARTIFACTS_DIR = Path("artifacts")
MINDMAP_CHUNK_CHARS = 3500
MINDMAP_MAX_CHUNKS = 8
MINDMAP_MAX_CHILDREN = 8
MINDMAP_MAX_GRANDCHILDREN = 6


@dataclass(frozen=True)
class MindMapArtifact:
    id: str
    title: str
    status: str
    tenant_id: str
    source_doc_ids: list[str]
    created_at: int
    updated_at: int
    root: dict[str, Any] | None = None
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


def create_mindmap_artifact(
    config: RagConfig,
    *,
    title: str,
    tenant_id: str,
    source_doc_ids: list[str],
    acl_groups: list[str] | None,
    doc_version: int | None = None,
) -> MindMapArtifact:
    artifact_id = f"mindmap-{uuid.uuid4().hex[:12]}"
    timestamp = now_ms()
    root = build_mindmap_root(
        title=title or infer_title(source_doc_ids),
        config=config,
        tenant_id=tenant_id,
        source_doc_ids=source_doc_ids,
    )
    artifact = MindMapArtifact(
        id=artifact_id,
        title=title or root["label"],
        status="ready",
        tenant_id=tenant_id,
        source_doc_ids=source_doc_ids,
        created_at=timestamp,
        updated_at=timestamp,
        root=root,
    )
    save_artifact(config, artifact)
    return artifact


def build_mindmap_root(
    *,
    title: str,
    config: RagConfig | None = None,
    tenant_id: str | None = None,
    source_doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    return build_llm_outline(
        title=title,
        config=config,
        tenant_id=tenant_id,
        source_doc_ids=source_doc_ids or [],
    )


def build_llm_outline(
    *,
    title: str,
    config: RagConfig | None,
    tenant_id: str | None,
    source_doc_ids: list[str],
) -> dict[str, Any] | None:
    if config is None or tenant_id is None or not source_doc_ids:
        raise ValueError("tenant_id and source_doc_ids are required for mind map generation.")
    if not config.llm_base_url or not config.llm_api_key:
        raise RuntimeError("NEW_API_URL/NEW_API_KEY must be configured for LLM mind map generation.")

    docs = load_selected_archived_docs(config=config, tenant_id=tenant_id, source_doc_ids=source_doc_ids)
    if not docs:
        raise RuntimeError("No archived source documents found for LLM mind map generation.")

    text = "\n\n".join(format_doc_for_mindmap(doc) for doc in docs if doc.text.strip()).strip()
    chunks = split_mindmap_text(text, chunk_chars=MINDMAP_CHUNK_CHARS, max_chunks=MINDMAP_MAX_CHUNKS)
    if not chunks:
        raise RuntimeError("Archived source documents are empty; cannot generate mind map.")

    partial_roots = [
        generate_partial_mindmap_with_llm(config=config, title=title, chunk=chunk, index=index)
        for index, chunk in enumerate(chunks, start=1)
    ]
    root = merge_mindmaps_with_llm(config=config, title=title, partial_roots=partial_roots)
    return normalize_mindmap_node(root, default_label=title or infer_title(source_doc_ids), node_id="root")


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


def generate_partial_mindmap_with_llm(*, config: RagConfig, title: str, chunk: str, index: int) -> dict[str, Any]:
    prompt = f"""你是知识库思维导图专家。请把下面第 {index} 个原文块整理为局部思维导图。

要求:
- 只依据原文，不编造。
- 输出严格 JSON，不要 Markdown。
- JSON schema: {{"label": "局部主题", "children": [{{"label": "二级主题", "children": [{{"label": "三级要点", "children": []}}]}}]}}
- 二级主题 3-6 个，每个二级主题下三级要点 2-5 个。
- 标签要短，像思维导图节点，不要长段落。

总标题: {title}

原文块:
{chunk}
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


def call_mindmap_llm(*, config: RagConfig, system_prompt: str, user_prompt: str) -> dict[str, Any]:
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
        root=row.get("root"),
        error=str(row.get("error") or ""),
    )
