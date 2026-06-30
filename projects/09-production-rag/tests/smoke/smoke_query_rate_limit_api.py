from __future__ import annotations

import json
import os
import sys
import tempfile
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
    retrieval_mode: str


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="rag-query-rate-api-") as tmp:
        config = replace(
            load_config(),
            metadata_database_url=None,
            runtime_dir=Path(tmp) / "runtime",
            object_store_dir=Path(tmp) / "object_store",
        )
        auth_context = AuthContext(
            "rate-limit-tenant",
            ["engineering"],
            "smoke",
            user_id="rate-limit-user",
        )
        calls = {"count": 0}

        def resolve_once(request, _auth_context, stage_callback=None):
            calls["count"] += 1
            if stage_callback is not None:
                stage_callback(
                    {
                        "stage": "answer",
                        "status": "done",
                        "label": "大模型最终输出",
                        "detail": "限流测试回答完成。",
                    }
                )
            return SimpleNamespace(
                request_id=request.request_id,
                answer="rate limit smoke answer",
                hits=[],
                candidates=[],
                reranked=[],
                trace=FakeTrace(
                    request_id=request.request_id,
                    retrieval_mode="synthetic-rate-limit",
                ),
                generation={"model": "synthetic"},
            )

        with (
            rate_limit_environment(),
            patch("serve.load_config", return_value=config),
            patch("serve.resolve_auth_context", return_value=auth_context),
            patch("serve.resolve_answer_result", side_effect=resolve_once),
        ):
            api = TestClient(serve.create_app())
            payload = {
                "query": "one logical query",
                "tenant_id": auth_context.tenant_id,
                "acl_groups": auth_context.acl_groups,
                "request_id": "rate-limit-request-one",
            }
            first = api.post("/query/stream", json=payload)
            assert first.status_code == 200, first.text
            assert parse_events(first)[-1]["answer"] == "rate limit smoke answer"

            replay = api.post("/query/stream", json=payload)
            assert replay.status_code == 200, replay.text
            replay_events = parse_events(replay)
            assert replay_events[-1]["answer"] == "rate limit smoke answer"
            assert any(event.get("stage") == "resume" for event in replay_events)
            assert calls["count"] == 1

            rejected = api.post(
                "/query/stream",
                json={**payload, "request_id": "rate-limit-request-two"},
            )
            assert rejected.status_code == 429, rejected.text
            assert int(rejected.headers["retry-after"]) >= 1
            assert rejected.json() == {
                "detail": "Query request rate limit exceeded. Please retry later."
            }
            assert auth_context.tenant_id not in rejected.text
            assert auth_context.user_id not in rejected.text
            assert calls["count"] == 1

            metrics = api.get("/runtime-metrics")
            assert metrics.status_code == 200, metrics.text
            rate_metrics = metrics.json()["query_rate_limit"]
            assert rate_metrics["enabled"] is True
            assert rate_metrics["limits"] == {"global": 10, "tenant": 10, "user": 1}
            assert rate_metrics["requests"] == {"global": 1, "tenant": 1, "user": 1}
            assert rate_metrics["active_keys"] == {"global": 1, "tenant": 1, "user": 1}
            assert rate_metrics["events"]["accepted_total"] >= 1
            assert rate_metrics["events"]["replay_bypassed_total"] >= 1
            assert rate_metrics["events"]["rejected_user_total"] >= 1
            assert auth_context.tenant_id not in metrics.text
            assert auth_context.user_id not in metrics.text
    print("smoke_query_rate_limit_api=ok")


def parse_events(response) -> list[dict]:
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


@contextmanager
def rate_limit_environment():
    names = {
        "RAG_QUERY_SHARED_ADMISSION": "0",
        "RAG_QUERY_RATE_LIMIT_GLOBAL": "10",
        "RAG_QUERY_RATE_LIMIT_TENANT": "10",
        "RAG_QUERY_RATE_LIMIT_USER": "1",
        "RAG_QUERY_RATE_LIMIT_WINDOW_SECONDS": "300",
    }
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update(names)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    main()
