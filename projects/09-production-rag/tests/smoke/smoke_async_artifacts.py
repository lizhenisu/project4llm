from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from rag_core.object_store import archive_source_documents
from rag_core.types import SourceDocument
from serve import create_app


class FakeOpenAI:
    def __init__(self, *, base_url: str, api_key: str) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, *, model: str, messages: list[dict], temperature: float):
        content = {
            "label": "异步思维导图",
            "children": [{"label": "后台任务", "children": [{"label": "状态更新", "children": []}]}],
        }
        if "合并" not in messages[-1]["content"]:
            content = {"label": "局部主题", "children": [{"label": "后台任务", "children": []}]}
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(content, ensure_ascii=False)))]
        )


def main() -> None:
    old_runtime = os.environ.get("RAG_RUNTIME_DIR")
    old_object_store = os.environ.get("RAG_OBJECT_STORE_DIR")
    old_api_url = os.environ.get("NEW_API_URL")
    old_api_key = os.environ.get("NEW_API_KEY")
    old_model = os.environ.get("LLM_MODEL")
    old_token = os.environ.get("RAG_API_TOKEN")
    old_openai = sys.modules.get("openai")
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["RAG_RUNTIME_DIR"] = str(Path(temp_dir) / "runtime")
        os.environ["RAG_OBJECT_STORE_DIR"] = str(Path(temp_dir) / "object_store")
        os.environ["NEW_API_URL"] = "https://llm.example"
        os.environ["NEW_API_KEY"] = "test-key"
        os.environ["LLM_MODEL"] = "test-llm"
        os.environ["RAG_API_TOKEN"] = "smoke-token"
        sys.modules["openai"] = SimpleNamespace(OpenAI=FakeOpenAI)
        try:
            archive_source_documents(
                Path(os.environ["RAG_OBJECT_STORE_DIR"]),
                [
                    SourceDocument(
                        tenant_id="team_a",
                        doc_id="async-doc/page-1",
                        doc_version=1,
                        source_type="pdf",
                        source_uri="/tmp/async.pdf",
                        title="异步资料",
                        text="后台任务应该先返回 generating，然后更新为 ready。",
                        metadata={"relative_path": "async.pdf", "page_no": 1},
                    )
                ],
            )
            run_smoke()
        finally:
            restore_env("RAG_RUNTIME_DIR", old_runtime)
            restore_env("RAG_OBJECT_STORE_DIR", old_object_store)
            restore_env("NEW_API_URL", old_api_url)
            restore_env("NEW_API_KEY", old_api_key)
            restore_env("LLM_MODEL", old_model)
            restore_env("RAG_API_TOKEN", old_token)
            if old_openai is None:
                sys.modules.pop("openai", None)
            else:
                sys.modules["openai"] = old_openai


def run_smoke() -> None:
    api = TestClient(create_app())
    headers = {
        "Authorization": "Bearer smoke-token",
        "X-RAG-Tenant-ID": "team_a",
        "X-RAG-ACL-Groups": "engineering",
    }
    created = api.post(
        "/artifacts/mindmap",
        headers=headers,
        json={
            "title": "异步思维导图",
            "tenant_id": "team_a",
            "acl_groups": ["engineering"],
            "source_doc_ids": ["async-doc/page-1"],
            "context_limit": 5,
        },
    )
    assert created.status_code == 200, created.text
    created_body = created.json()
    assert created_body["status"] == "generating"
    artifact_id = created_body["id"]

    loaded = api.get(f"/artifacts/{artifact_id}?tenant_id=team_a", headers=headers)
    assert loaded.status_code == 200, loaded.text
    loaded_body = loaded.json()
    assert loaded_body["status"] == "ready"
    assert loaded_body["root"]["label"] == "异步思维导图"
    print("async artifacts smoke passed")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
