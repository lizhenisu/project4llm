from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_core.config import RagConfig
from rag_core.pipeline import retrieve_and_rerank
from rag_core.text_utils import now_ms


ARTIFACTS_DIR = Path("artifacts")


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
    query = f"生成 {title or '当前来源'} 的思维导图"
    retrieval = retrieve_and_rerank(
        query,
        tenant_id=tenant_id,
        candidate_limit=30,
        context_limit=12,
        acl_groups=acl_groups,
        doc_version=doc_version,
        doc_ids=source_doc_ids or None,
        request_id=artifact_id,
    )
    root = build_mindmap_root(
        title=title or infer_title(source_doc_ids, retrieval.hits),
        hits=retrieval.hits,
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


def build_mindmap_root(*, title: str, hits) -> dict[str, Any]:
    grouped: dict[str, list] = {}
    for hit in hits:
        grouped.setdefault(hit.doc_id, []).append(hit)
    children: list[dict[str, Any]] = []
    for doc_id, doc_hits in grouped.items():
        first = doc_hits[0]
        doc_node = {
            "id": f"doc-{doc_id}",
            "label": first.title[:80],
            "citationIds": [hit.id for hit in doc_hits[:5]],
            "children": [
                {
                    "id": f"{hit.id}-point",
                    "label": summarize_hit(hit.text),
                    "citationIds": [hit.id],
                }
                for hit in doc_hits[:5]
            ],
        }
        children.append(doc_node)
    if not children:
        children = [
            {
                "id": "empty",
                "label": "未检索到足够证据",
                "children": [],
                "citationIds": [],
            }
        ]
    return {
        "id": "root",
        "label": title[:80] or "思维导图",
        "children": children,
    }


def summarize_hit(text: str) -> str:
    normalized = " ".join(text.strip().split())
    return normalized[:80] + ("..." if len(normalized) > 80 else "")


def infer_title(source_doc_ids: list[str], hits) -> str:
    if hits:
        return hits[0].title
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
