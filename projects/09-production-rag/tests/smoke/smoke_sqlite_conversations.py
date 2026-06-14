from __future__ import annotations

import os
import tempfile
from pathlib import Path

from rag_core.config import load_config
from rag_core.conversations import ConversationMessage, delete_conversation, list_conversations, load_conversation, save_conversation


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
    saved = save_conversation(
        config,
        tenant_id="tenant-a",
        title="SQLite 会话",
        messages=[
            ConversationMessage(id="m1", role="user", content="问题", created_at=1),
            ConversationMessage(id="m2", role="assistant", content="回答", created_at=2, feedback_rating=1),
        ],
        source_doc_ids=["doc-1"],
    )
    rows = list_conversations(config, tenant_id="tenant-a")
    assert len(rows) == 1
    assert rows[0].id == saved.id
    loaded = load_conversation(config, tenant_id="tenant-a", conversation_id=saved.id)
    assert loaded is not None
    assert [message.content for message in loaded.messages] == ["问题", "回答"]
    assert loaded.messages[1].feedback_rating == 1
    assert loaded.source_doc_ids == ["doc-1"]
    assert delete_conversation(config, tenant_id="tenant-a", conversation_id=saved.id)
    assert list_conversations(config, tenant_id="tenant-a") == []
    print("sqlite conversations smoke passed")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
