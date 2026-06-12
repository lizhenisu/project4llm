from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from rag_core.config import RagConfig
from rag_core.text_utils import now_ms


CONVERSATIONS_DIR = Path("conversations")


@dataclass(frozen=True)
class ConversationMessage:
    id: str
    role: Literal["user", "assistant"]
    content: str
    status: Literal["sending", "done", "failed"] = "done"
    request_id: str | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    created_at: int | None = None


@dataclass(frozen=True)
class Conversation:
    id: str
    tenant_id: str
    title: str
    messages: list[ConversationMessage]
    source_doc_ids: list[str]
    created_at: int
    updated_at: int


def list_conversations(config: RagConfig, *, tenant_id: str) -> list[Conversation]:
    root = tenant_conversation_dir(config, tenant_id=tenant_id)
    if not root.exists():
        return []
    conversations = [
        conversation_from_row(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(root.glob("*.json"))
    ]
    return sorted(conversations, key=lambda item: item.updated_at, reverse=True)


def load_conversation(
    config: RagConfig,
    *,
    tenant_id: str,
    conversation_id: str,
) -> Conversation | None:
    path = conversation_path(config, tenant_id=tenant_id, conversation_id=conversation_id)
    if not path.exists():
        return None
    return conversation_from_row(json.loads(path.read_text(encoding="utf-8")))


def save_conversation(
    config: RagConfig,
    *,
    tenant_id: str,
    title: str,
    messages: list[ConversationMessage],
    source_doc_ids: list[str],
    conversation_id: str | None = None,
) -> Conversation:
    timestamp = now_ms()
    resolved_id = conversation_id or f"conv-{uuid.uuid4().hex[:12]}"
    existing = load_conversation(config, tenant_id=tenant_id, conversation_id=resolved_id)
    conversation = Conversation(
        id=resolved_id,
        tenant_id=tenant_id,
        title=title or infer_title(messages),
        messages=messages,
        source_doc_ids=source_doc_ids,
        created_at=existing.created_at if existing else timestamp,
        updated_at=timestamp,
    )
    path = conversation_path(config, tenant_id=tenant_id, conversation_id=resolved_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(conversation), ensure_ascii=False, indent=2), encoding="utf-8")
    return conversation


def delete_conversation(
    config: RagConfig,
    *,
    tenant_id: str,
    conversation_id: str,
) -> bool:
    path = conversation_path(config, tenant_id=tenant_id, conversation_id=conversation_id)
    if not path.exists():
        return False
    path.unlink()
    return True


def tenant_conversation_dir(config: RagConfig, *, tenant_id: str) -> Path:
    return config.runtime_dir / CONVERSATIONS_DIR / tenant_id


def conversation_path(config: RagConfig, *, tenant_id: str, conversation_id: str) -> Path:
    safe_id = conversation_id.replace("/", "_")
    return tenant_conversation_dir(config, tenant_id=tenant_id) / f"{safe_id}.json"


def conversation_from_row(row: dict[str, Any]) -> Conversation:
    return Conversation(
        id=str(row["id"]),
        tenant_id=str(row["tenant_id"]),
        title=str(row.get("title") or "未命名对话"),
        messages=[
            ConversationMessage(
                id=str(message["id"]),
                role=message["role"],
                content=str(message.get("content") or ""),
                status=message.get("status") or "done",
                request_id=message.get("request_id"),
                citations=list(message.get("citations") or []),
                created_at=message.get("created_at"),
            )
            for message in row.get("messages", [])
        ],
        source_doc_ids=list(row.get("source_doc_ids") or []),
        created_at=int(row.get("created_at") or now_ms()),
        updated_at=int(row.get("updated_at") or now_ms()),
    )


def infer_title(messages: list[ConversationMessage]) -> str:
    first_user = next((message.content.strip() for message in messages if message.role == "user"), "")
    if not first_user:
        return "未命名对话"
    return first_user[:40]
