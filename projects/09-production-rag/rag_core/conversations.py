from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from rag_core.config import RagConfig
from rag_core.database import connect_metadata_db
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
    feedback_rating: int | None = None


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
    migrate_legacy_conversations(config, tenant_id=tenant_id)
    with connect_metadata_db(config) as conn:
        rows = conn.execute(
            """
            SELECT id, tenant_id, title, source_doc_ids, created_at, updated_at
            FROM conversations
            WHERE tenant_id = ?
            ORDER BY updated_at DESC
            """,
            (tenant_id,),
        ).fetchall()
    return [load_conversation(config, tenant_id=tenant_id, conversation_id=str(row["id"])) for row in rows if row]


def load_conversation(
    config: RagConfig,
    *,
    tenant_id: str,
    conversation_id: str,
) -> Conversation | None:
    legacy: Conversation | None = None
    with connect_metadata_db(config) as conn:
        row = conn.execute(
            """
            SELECT id, tenant_id, title, source_doc_ids, created_at, updated_at
            FROM conversations
            WHERE tenant_id = ? AND id = ?
            """,
            (tenant_id, conversation_id),
        ).fetchone()
        if row is None:
            legacy = load_legacy_conversation(config, tenant_id=tenant_id, conversation_id=conversation_id)
            message_rows = []
        else:
            message_rows = conn.execute(
                """
                SELECT id, role, content, status, request_id, citations, created_at, feedback_rating
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            ).fetchall()
    if legacy is not None:
        save_conversation_row(config, legacy)
        return legacy
    if row is None:
        return None
    return Conversation(
        id=str(row["id"]),
        tenant_id=str(row["tenant_id"]),
        title=str(row["title"] or "未命名对话"),
        messages=[
            ConversationMessage(
                id=str(message["id"]),
                role=message["role"],
                content=str(message["content"] or ""),
                status=message["status"] or "done",
                request_id=message["request_id"],
                citations=json.loads(message["citations"] or "[]"),
                created_at=int(message["created_at"] or now_ms()),
                feedback_rating=message["feedback_rating"],
            )
            for message in message_rows
        ],
        source_doc_ids=json.loads(row["source_doc_ids"] or "[]"),
        created_at=int(row["created_at"] or now_ms()),
        updated_at=int(row["updated_at"] or now_ms()),
    )


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
    save_conversation_row(config, conversation)
    return conversation


def delete_conversation(
    config: RagConfig,
    *,
    tenant_id: str,
    conversation_id: str,
) -> bool:
    with connect_metadata_db(config) as conn:
        cursor = conn.execute(
            "DELETE FROM conversations WHERE tenant_id = ? AND id = ?",
            (tenant_id, conversation_id),
        )
    path = conversation_path(config, tenant_id=tenant_id, conversation_id=conversation_id)
    if path.exists():
        path.unlink()
        return True
    return cursor.rowcount > 0


def save_conversation_row(config: RagConfig, conversation: Conversation) -> None:
    with connect_metadata_db(config) as conn:
        conn.execute(
            """
            INSERT INTO conversations(id, tenant_id, title, source_doc_ids, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                tenant_id = excluded.tenant_id,
                title = excluded.title,
                source_doc_ids = excluded.source_doc_ids,
                updated_at = excluded.updated_at
            """,
            (
                conversation.id,
                conversation.tenant_id,
                conversation.title,
                json.dumps(conversation.source_doc_ids, ensure_ascii=False),
                conversation.created_at,
                conversation.updated_at,
            ),
        )
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation.id,))
        for index, message in enumerate(conversation.messages):
            conn.execute(
                """
                INSERT INTO messages(id, conversation_id, role, content, status, request_id, citations, created_at, feedback_rating)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    conversation.id,
                    message.role,
                    message.content,
                    message.status,
                    message.request_id,
                    json.dumps(message.citations, ensure_ascii=False),
                    message.created_at or conversation.created_at + index,
                    message.feedback_rating,
                ),
            )


def migrate_legacy_conversations(config: RagConfig, *, tenant_id: str) -> None:
    root = tenant_conversation_dir(config, tenant_id=tenant_id)
    if not root.exists():
        return
    for path in sorted(root.glob("*.json")):
        conversation = conversation_from_row(json.loads(path.read_text(encoding="utf-8")))
        if not conversation_exists(config, tenant_id=tenant_id, conversation_id=conversation.id):
            save_conversation_row(config, conversation)


def conversation_exists(config: RagConfig, *, tenant_id: str, conversation_id: str) -> bool:
    with connect_metadata_db(config) as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE tenant_id = ? AND id = ?",
            (tenant_id, conversation_id),
        ).fetchone()
    return row is not None


def load_legacy_conversation(
    config: RagConfig,
    *,
    tenant_id: str,
    conversation_id: str,
) -> Conversation | None:
    path = conversation_path(config, tenant_id=tenant_id, conversation_id=conversation_id)
    if not path.exists():
        return None
    return conversation_from_row(json.loads(path.read_text(encoding="utf-8")))


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
                feedback_rating=message.get("feedback_rating"),
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
