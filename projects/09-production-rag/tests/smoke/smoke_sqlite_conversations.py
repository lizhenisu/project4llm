from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.config import load_config
from rag_core.conversations import (
    ConversationMessage,
    delete_conversation,
    list_conversations,
    load_conversation,
    rename_conversation,
    save_conversation,
)


def main() -> None:
    old_runtime = os.environ.get("RAG_RUNTIME_DIR")
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["RAG_RUNTIME_DIR"] = str(Path(temp_dir) / "runtime")
        try:
            run_smoke()
        finally:
            restore_env("RAG_RUNTIME_DIR", old_runtime)


def run_smoke() -> None:
    config = load_config()
    legacy_db = config.runtime_dir / "db" / "metadata.db"
    legacy_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(legacy_db) as conn:
        conn.execute(
            """
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                title TEXT NOT NULL,
                source_doc_ids TEXT NOT NULL DEFAULT '[]',
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO conversations(id, tenant_id, title, source_doc_ids, created_at, updated_at)
            VALUES ('legacy-conversation', 'legacy-tenant', '旧会话', '[]', 1, 1)
            """
        )
    legacy_rows = list_conversations(
        config,
        tenant_id="legacy-tenant",
        workspace_id="default-workspace",
    )
    assert len(legacy_rows) == 1
    assert legacy_rows[0].workspace_id == "default-workspace"

    saved = save_conversation(
        config,
        tenant_id="tenant-a",
        title="SQLite 会话",
        messages=[
            ConversationMessage(id="m1", role="user", content="问题", created_at=1),
            ConversationMessage(id="m2", role="assistant", content="回答", created_at=2, feedback_rating=1),
        ],
        source_doc_ids=["doc-1"],
        workspace_id="workspace-a",
    )
    save_conversation(
        config,
        tenant_id="tenant-a",
        title="另一个知识库",
        messages=[ConversationMessage(id="m3", role="user", content="隔离问题", created_at=3)],
        source_doc_ids=["doc-2"],
        workspace_id="workspace-b",
    )
    rows = list_conversations(config, tenant_id="tenant-a", workspace_id="workspace-a")
    assert len(rows) == 1
    assert rows[0].id == saved.id
    assert len(list_conversations(config, tenant_id="tenant-a")) == 2
    assert (
        load_conversation(
            config,
            tenant_id="tenant-a",
            conversation_id=saved.id,
            workspace_id="workspace-b",
        )
        is None
    )
    loaded = load_conversation(
        config,
        tenant_id="tenant-a",
        conversation_id=saved.id,
        workspace_id="workspace-a",
    )
    assert loaded is not None
    assert [message.content for message in loaded.messages] == ["问题", "回答"]
    assert loaded.messages[1].feedback_rating == 1
    assert loaded.source_doc_ids == ["doc-1"]
    renamed = rename_conversation(
        config,
        tenant_id="tenant-a",
        conversation_id=saved.id,
        title="重命名后的会话",
        workspace_id="workspace-a",
    )
    assert renamed is not None
    assert renamed.title == "重命名后的会话"
    assert load_conversation(config, tenant_id="tenant-a", conversation_id=saved.id).title == "重命名后的会话"
    assert rename_conversation(
        config,
        tenant_id="tenant-b",
        conversation_id=saved.id,
        title="越权重命名",
    ) is None
    assert load_conversation(config, tenant_id="tenant-a", conversation_id=saved.id).title == "重命名后的会话"
    assert not delete_conversation(
        config,
        tenant_id="tenant-a",
        conversation_id=saved.id,
        workspace_id="workspace-b",
    )
    assert delete_conversation(
        config,
        tenant_id="tenant-a",
        conversation_id=saved.id,
        workspace_id="workspace-a",
    )
    assert list_conversations(config, tenant_id="tenant-a", workspace_id="workspace-a") == []
    print("sqlite conversations smoke passed")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
