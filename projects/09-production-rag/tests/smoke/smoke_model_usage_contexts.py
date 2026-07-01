from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import serve  # noqa: E402
from rag_core.auth import AuthContext  # noqa: E402
from rag_core.config import load_config  # noqa: E402


@dataclass(frozen=True)
class FakeTrace:
    request_id: str
    retrieval_mode: str = "synthetic"


def main() -> None:
    old_shared_admission = os.environ.get("RAG_QUERY_SHARED_ADMISSION")
    os.environ["RAG_QUERY_SHARED_ADMISSION"] = "0"
    with tempfile.TemporaryDirectory(prefix="rag-model-context-") as tmp:
        config = replace(
            load_config(),
            metadata_database_url=None,
            runtime_dir=Path(tmp) / "runtime",
            object_store_dir=Path(tmp) / "object_store",
        )
        try:
            test_query_search_and_stream_contexts(config)
        finally:
            restore_env("RAG_QUERY_SHARED_ADMISSION", old_shared_admission)
    print("smoke_model_usage_contexts=ok")


def test_query_search_and_stream_contexts(config) -> None:
    auth_context = AuthContext(
        tenant_id="model-context-tenant",
        acl_groups=["engineering"],
        source="smoke",
        user_id="model-context-user",
    )
    captured: list[dict[str, str]] = []
    capture_lock = threading.Lock()

    @contextmanager
    def capture_context(**kwargs):
        with capture_lock:
            captured.append(
                {
                    "tenant_id": kwargs["tenant_id"],
                    "principal_key": kwargs["principal_key"],
                    "workload": kwargs["workload"],
                }
            )
        yield

    answer_result = SimpleNamespace(
        request_id="model-context-answer",
        answer="synthetic answer",
        hits=[],
        candidates=[],
        reranked=[],
        trace=FakeTrace("model-context-answer"),
        generation={"token_usage": {"total_tokens": 1}},
    )
    search_result = SimpleNamespace(
        request_id="model-context-search",
        hits=[],
        candidates=[],
        reranked=[],
        trace=FakeTrace("model-context-search"),
    )
    with (
        patch("serve.load_config", return_value=config),
        patch("serve.resolve_auth_context", return_value=auth_context),
        patch("serve.resolve_answer_result", return_value=answer_result),
        patch("serve.resolve_search_result", return_value=search_result),
        patch("serve.model_usage_context", side_effect=capture_context),
    ):
        api = TestClient(serve.create_app())
        payload = {
            "query": "synthetic",
            "tenant_id": auth_context.tenant_id,
            "acl_groups": auth_context.acl_groups,
        }
        query = api.post("/query", json=payload)
        assert query.status_code == 200, query.text
        search = api.post("/search", json=payload)
        assert search.status_code == 200, search.text
        stream = api.post("/query/stream", json=payload)
        assert stream.status_code == 200, stream.text
        events = [json.loads(line) for line in stream.text.splitlines() if line.strip()]
        assert events[-1]["answer"] == "synthetic answer"

    assert [item["workload"] for item in captured] == ["query", "search", "query"]
    assert all(item["tenant_id"] == auth_context.tenant_id for item in captured)
    assert all(
        item["principal_key"] == f"user:{auth_context.user_id}"
        for item in captured
    )


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
