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


class ConversationTenantConflictError(ValueError):
    """Raised when a tenant attempts to reuse another tenant's conversation ID."""


@dataclass(frozen=True)
class ConversationMessage:
    id: str
    role: Literal["user", "assistant"]
    content: str
    status: Literal["sending", "done", "failed"] = "done"
    request_id: str | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    image_data_url: str | None = None
    created_at: int | None = None
    feedback_rating: int | None = None
    rag_progress: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Conversation:
    id: str
    tenant_id: str
    workspace_id: str
    title: str
    messages: list[ConversationMessage]
    source_doc_ids: list[str]
    created_at: int
    updated_at: int


@dataclass(frozen=True)
class ConversationListItem:
    id: str
    tenant_id: str
    workspace_id: str
    title: str
    message_count: int
    source_doc_ids: list[str]
    created_at: int
    updated_at: int


def list_conversations(
    config: RagConfig,
    *,
    tenant_id: str,
    workspace_id: str | None = None,
) -> list[Conversation]:
    migrate_legacy_conversations(config, tenant_id=tenant_id)
    with connect_metadata_db(config) as conn:
        workspace_clause = "" if workspace_id is None else " AND workspace_id = ?"
        params = (tenant_id,) if workspace_id is None else (tenant_id, workspace_id)
        rows = conn.execute(
            f"""
            SELECT id, tenant_id, workspace_id, title, source_doc_ids, created_at, updated_at
            FROM conversations
            WHERE tenant_id = ?{workspace_clause}
            ORDER BY updated_at DESC
            """,
            params,
        ).fetchall()
    return [
        conversation
        for row in rows
        if row
        and (
            conversation := load_conversation(
                config,
                tenant_id=tenant_id,
                conversation_id=str(row["id"]),
                workspace_id=workspace_id,
            )
        )
    ]


def list_conversation_items(
    config: RagConfig,
    *,
    tenant_id: str,
    workspace_id: str | None = None,
) -> list[ConversationListItem]:
    migrate_legacy_conversations(config, tenant_id=tenant_id)
    with connect_metadata_db(config) as conn:
        workspace_clause = "" if workspace_id is None else " AND c.workspace_id = ?"
        params = (tenant_id,) if workspace_id is None else (tenant_id, workspace_id)
        rows = conn.execute(
            f"""
            SELECT c.id, c.tenant_id, c.workspace_id, c.title, c.source_doc_ids, c.created_at, c.updated_at,
                   COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.tenant_id = ?{workspace_clause}
            GROUP BY c.id, c.tenant_id, c.workspace_id, c.title, c.source_doc_ids, c.created_at, c.updated_at
            ORDER BY c.updated_at DESC
            """,
            params,
        ).fetchall()
    return [
        ConversationListItem(
            id=str(row["id"]),
            tenant_id=str(row["tenant_id"]),
            workspace_id=str(row["workspace_id"]),
            title=str(row["title"] or "未命名对话"),
            message_count=int(row["message_count"] or 0),
            source_doc_ids=json.loads(row["source_doc_ids"] or "[]"),
            created_at=int(row["created_at"] or now_ms()),
            updated_at=int(row["updated_at"] or now_ms()),
        )
        for row in rows
    ]


def load_conversation_metadata(
    config: RagConfig,
    *,
    tenant_id: str,
    conversation_id: str,
    workspace_id: str | None = None,
) -> ConversationListItem | None:
    with connect_metadata_db(config) as conn:
        workspace_clause = "" if workspace_id is None else " AND c.workspace_id = ?"
        params = (
            (tenant_id, conversation_id)
            if workspace_id is None
            else (tenant_id, conversation_id, workspace_id)
        )
        row = conn.execute(
            f"""
            SELECT c.id, c.tenant_id, c.workspace_id, c.title, c.source_doc_ids, c.created_at, c.updated_at,
                   COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.tenant_id = ? AND c.id = ?{workspace_clause}
            GROUP BY c.id, c.tenant_id, c.workspace_id, c.title, c.source_doc_ids, c.created_at, c.updated_at
            """,
            params,
        ).fetchone()
    if row is None:
        if workspace_id not in (None, "default-workspace"):
            return None
        legacy = load_legacy_conversation(config, tenant_id=tenant_id, conversation_id=conversation_id)
        if legacy is None:
            return None
        save_conversation_row(config, legacy)
        return ConversationListItem(
            id=legacy.id,
            tenant_id=legacy.tenant_id,
            workspace_id=legacy.workspace_id,
            title=legacy.title,
            message_count=len(legacy.messages),
            source_doc_ids=legacy.source_doc_ids,
            created_at=legacy.created_at,
            updated_at=legacy.updated_at,
        )
    return ConversationListItem(
        id=str(row["id"]),
        tenant_id=str(row["tenant_id"]),
        workspace_id=str(row["workspace_id"]),
        title=str(row["title"] or "未命名对话"),
        message_count=int(row["message_count"] or 0),
        source_doc_ids=json.loads(row["source_doc_ids"] or "[]"),
        created_at=int(row["created_at"] or now_ms()),
        updated_at=int(row["updated_at"] or now_ms()),
    )


def load_conversation(
    config: RagConfig,
    *,
    tenant_id: str,
    conversation_id: str,
    workspace_id: str | None = None,
) -> Conversation | None:
    legacy: Conversation | None = None
    with connect_metadata_db(config) as conn:
        workspace_clause = "" if workspace_id is None else " AND workspace_id = ?"
        params = (
            (tenant_id, conversation_id)
            if workspace_id is None
            else (tenant_id, conversation_id, workspace_id)
        )
        row = conn.execute(
            f"""
            SELECT id, tenant_id, workspace_id, title, source_doc_ids, created_at, updated_at
            FROM conversations
            WHERE tenant_id = ? AND id = ?{workspace_clause}
            """,
            params,
        ).fetchone()
        if row is None:
            legacy = (
                load_legacy_conversation(config, tenant_id=tenant_id, conversation_id=conversation_id)
                if workspace_id in (None, "default-workspace")
                else None
            )
            message_rows = []
        else:
            message_rows = conn.execute(
                """
                SELECT id, role, content, status, request_id, citations, image_data_url, created_at, feedback_rating, rag_progress
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
        workspace_id=str(row["workspace_id"]),
        title=str(row["title"] or "未命名对话"),
        messages=[
            ConversationMessage(
                id=str(message["id"]),
                role=message["role"],
                content=str(message["content"] or ""),
                status=message["status"] or "done",
                request_id=message["request_id"],
                citations=json.loads(message["citations"] or "[]"),
                image_data_url=message["image_data_url"],
                created_at=int(message["created_at"] or now_ms()),
                feedback_rating=message["feedback_rating"],
                rag_progress=json.loads(message["rag_progress"] or "[]"),
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
    workspace_id: str = "default-workspace",
) -> Conversation:
    timestamp = now_ms()
    resolved_id = conversation_id or f"conv-{uuid.uuid4().hex[:12]}"
    existing = load_conversation_metadata(config, tenant_id=tenant_id, conversation_id=resolved_id)
    if existing is not None and existing.workspace_id != workspace_id:
        raise ConversationTenantConflictError("Conversation ID belongs to another workspace")
    conversation = Conversation(
        id=resolved_id,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
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
    workspace_id: str | None = None,
) -> bool:
    with connect_metadata_db(config) as conn:
        workspace_clause = "" if workspace_id is None else " AND workspace_id = ?"
        params = (
            (tenant_id, conversation_id)
            if workspace_id is None
            else (tenant_id, conversation_id, workspace_id)
        )
        cursor = conn.execute(
            f"DELETE FROM conversations WHERE tenant_id = ? AND id = ?{workspace_clause}",
            params,
        )
    path = conversation_path(config, tenant_id=tenant_id, conversation_id=conversation_id)
    if path.exists() and (cursor.rowcount > 0 or workspace_id is None):
        path.unlink()
        return True
    return cursor.rowcount > 0


def rename_conversation(
    config: RagConfig,
    *,
    tenant_id: str,
    conversation_id: str,
    title: str,
    workspace_id: str | None = None,
) -> ConversationListItem | None:
    existing = load_conversation_metadata(
        config,
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        workspace_id=workspace_id,
    )
    if existing is None:
        return None
    updated_at = now_ms()
    normalized_title = title.strip()
    with connect_metadata_db(config) as conn:
        workspace_clause = "" if workspace_id is None else " AND workspace_id = ?"
        params = (
            (normalized_title, updated_at, tenant_id, conversation_id)
            if workspace_id is None
            else (normalized_title, updated_at, tenant_id, conversation_id, workspace_id)
        )
        cursor = conn.execute(
            f"""
            UPDATE conversations
            SET title = ?, updated_at = ?
            WHERE tenant_id = ? AND id = ?{workspace_clause}
            """,
            params,
        )
    if cursor.rowcount == 0:
        return None
    return ConversationListItem(
        id=existing.id,
        tenant_id=existing.tenant_id,
        workspace_id=existing.workspace_id,
        title=normalized_title,
        message_count=existing.message_count,
        source_doc_ids=existing.source_doc_ids,
        created_at=existing.created_at,
        updated_at=updated_at,
    )


def save_conversation_row(config: RagConfig, conversation: Conversation) -> None:
    with connect_metadata_db(config) as conn:
        existing = conn.execute(
            "SELECT tenant_id, workspace_id FROM conversations WHERE id = ?",
            (conversation.id,),
        ).fetchone()
        if existing is not None and str(existing["tenant_id"]) != conversation.tenant_id:
            raise ConversationTenantConflictError("Conversation ID belongs to another tenant")
        if existing is not None and str(existing["workspace_id"]) != conversation.workspace_id:
            raise ConversationTenantConflictError("Conversation ID belongs to another workspace")
        cursor = conn.execute(
            """
            INSERT INTO conversations(id, tenant_id, workspace_id, title, source_doc_ids, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                source_doc_ids = excluded.source_doc_ids,
                updated_at = excluded.updated_at
            WHERE conversations.tenant_id = excluded.tenant_id
            """,
            (
                conversation.id,
                conversation.tenant_id,
                conversation.workspace_id,
                conversation.title,
                json.dumps(conversation.source_doc_ids, ensure_ascii=False),
                conversation.created_at,
                conversation.updated_at,
            ),
        )
        if cursor.rowcount == 0:
            raise ConversationTenantConflictError("Conversation ID belongs to another tenant")
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation.id,))
        for index, message in enumerate(conversation.messages):
            conn.execute(
                """
                INSERT INTO messages(id, conversation_id, role, content, status, request_id, citations, image_data_url, created_at, feedback_rating, rag_progress)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    conversation.id,
                    message.role,
                    message.content,
                    message.status,
                    message.request_id,
                    json.dumps(message.citations, ensure_ascii=False),
                    message.image_data_url,
                    message.created_at or conversation.created_at + index,
                    message.feedback_rating,
                    json.dumps(message.rag_progress, ensure_ascii=False),
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
        workspace_id=str(row.get("workspace_id") or "default-workspace"),
        title=str(row.get("title") or "未命名对话"),
        messages=[
            ConversationMessage(
                id=str(message["id"]),
                role=message["role"],
                content=str(message.get("content") or ""),
                status=message.get("status") or "done",
                request_id=message.get("request_id"),
                citations=list(message.get("citations") or []),
                image_data_url=message.get("image_data_url"),
                created_at=message.get("created_at"),
                feedback_rating=message.get("feedback_rating"),
                rag_progress=list(message.get("rag_progress") or []),
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
