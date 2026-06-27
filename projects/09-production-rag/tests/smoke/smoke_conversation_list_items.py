from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core import config as config_module  # noqa: E402
from rag_core.config import load_config  # noqa: E402
from rag_core.conversations import (  # noqa: E402
    ConversationMessage,
    list_conversation_items,
    load_conversation,
    save_conversation,
)


def main() -> None:
    old_env_loaded = config_module._ENV_LOADED
    old_env = {
        "RAG_RUNTIME_DIR": os.environ.get("RAG_RUNTIME_DIR"),
        "RAG_OBJECT_STORE_DIR": os.environ.get("RAG_OBJECT_STORE_DIR"),
        "RAG_METADATA_DATABASE_URL": os.environ.get("RAG_METADATA_DATABASE_URL"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        os.environ["RAG_RUNTIME_DIR"] = str(root / "runtime")
        os.environ["RAG_OBJECT_STORE_DIR"] = str(root / "object_store")
        os.environ["RAG_METADATA_DATABASE_URL"] = ""
        config_module._ENV_LOADED = True
        try:
            config = load_config()
            tenant_id = "tenant-conversation-list-smoke"
            for conversation_index in range(20):
                messages = [
                    ConversationMessage(
                        id=f"msg-{conversation_index:03d}-{message_index:03d}",
                        role="user" if message_index % 2 == 0 else "assistant",
                        content=f"message {conversation_index}-{message_index}",
                        created_at=conversation_index * 1000 + message_index,
                    )
                    for message_index in range(30)
                ]
                save_conversation(
                    config,
                    tenant_id=tenant_id,
                    title=f"Conversation {conversation_index:03d}",
                    messages=messages,
                    source_doc_ids=[f"doc-{conversation_index:03d}"],
                    conversation_id=f"conv-{conversation_index:03d}",
                )

            with patch(
                "rag_core.conversations.load_conversation",
                side_effect=AssertionError("list_conversation_items should not load full messages"),
            ):
                items = list_conversation_items(config, tenant_id=tenant_id)
            assert len(items) == 20
            assert all(item.message_count == 30 for item in items)
            assert items[0].updated_at >= items[-1].updated_at
            assert items[0].source_doc_ids

            existing = load_conversation(config, tenant_id=tenant_id, conversation_id="conv-000")
            assert existing is not None
            with patch(
                "rag_core.conversations.load_conversation",
                side_effect=AssertionError("save_conversation should only load metadata for existing rows"),
            ):
                updated = save_conversation(
                    config,
                    tenant_id=tenant_id,
                    title="Conversation 000 updated",
                    messages=[
                        ConversationMessage(id="updated-user", role="user", content="updated", created_at=9001),
                        ConversationMessage(id="updated-assistant", role="assistant", content="done", created_at=9002),
                    ],
                    source_doc_ids=["doc-000"],
                    conversation_id="conv-000",
                )
            assert updated.created_at == existing.created_at
            assert len(load_conversation(config, tenant_id=tenant_id, conversation_id="conv-000").messages) == 2
        finally:
            config_module._ENV_LOADED = old_env_loaded
            for name, value in old_env.items():
                restore_env(name, value)
    print("smoke_conversation_list_items=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
