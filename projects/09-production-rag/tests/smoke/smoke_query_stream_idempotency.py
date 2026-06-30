from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
from rag_core.database import connect_metadata_db  # noqa: E402


@dataclass(frozen=True)
class FakeTrace:
    request_id: str
    retrieval_mode: str


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="rag-query-stream-idempotency-") as tmp:
        config = replace(
            load_config(),
            metadata_database_url=None,
            runtime_dir=Path(tmp) / "runtime",
            object_store_dir=Path(tmp) / "object_store",
        )
        auth_context = AuthContext(
            "idempotency-tenant",
            ["engineering"],
            "smoke",
            user_id="idempotency-user",
        )
        calls = {"count": 0}
        started = threading.Event()
        release = threading.Event()

        def resolve_once(request, _auth_context, stage_callback=None):
            calls["count"] += 1
            started.set()
            if stage_callback is not None:
                stage_callback(
                    {
                        "stage": "search",
                        "status": "active",
                        "label": "向量检索",
                        "detail": "正在检索。",
                    }
                )
            assert release.wait(timeout=5)
            if stage_callback is not None:
                stage_callback(
                    {
                        "stage": "search",
                        "status": "done",
                        "label": "向量检索",
                        "detail": "检索完成。",
                    }
                )
            return SimpleNamespace(
                request_id=request.request_id,
                answer="generated exactly once",
                hits=[],
                candidates=[],
                reranked=[],
                trace=FakeTrace(
                    request_id=request.request_id,
                    retrieval_mode="synthetic-idempotency",
                ),
                generation={"model": "synthetic"},
            )

        with (
            patch("serve.load_config", return_value=config),
            patch("serve.resolve_auth_context", return_value=auth_context),
            patch("serve.resolve_answer_result", side_effect=resolve_once),
        ):
            app = serve.create_app()
            api = TestClient(app)
            retry_api = TestClient(app)
            payload = {
                "query": "same browser retry",
                "tenant_id": auth_context.tenant_id,
                "acl_groups": auth_context.acl_groups,
                "request_id": "query-stream-idempotency-request",
            }
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(api.post, "/query/stream", json=payload)
                assert started.wait(timeout=5)
                second = executor.submit(retry_api.post, "/query/stream", json=payload)
                time.sleep(0.2)
                release.set()
                responses = [first.result(timeout=10), second.result(timeout=10)]

            assert calls["count"] == 1
            parsed = [parse_events(response) for response in responses]
            assert all(events[-1]["type"] == "result" for events in parsed), parsed
            assert all(events[-1]["answer"] == "generated exactly once" for events in parsed)
            assert any(any(event.get("stage") == "resume" for event in events) for events in parsed)
            replayed = next(events for events in parsed if any(event.get("stage") == "resume" for event in events))
            assert [event["status"] for event in replayed if event.get("stage") == "search"] == [
                "active",
                "done",
            ]
            assert [event["sequence"] for event in replayed if event.get("stage") == "search"] == [1, 2]

            replay = api.post("/query/stream", json=payload)
            replay_events = parse_events(replay)
            assert replay_events[-1]["answer"] == "generated exactly once"
            assert [event["sequence"] for event in replay_events if event.get("stage") == "search"] == [1, 2]
            assert calls["count"] == 1

            conflict = api.post(
                "/query/stream",
                json={**payload, "query": "different query"},
            )
            conflict_events = parse_events(conflict)
            assert conflict_events[-1]["type"] == "error"
            assert calls["count"] == 1

        with connect_metadata_db(config) as conn:
            row = conn.execute(
                """
                SELECT status, response_json
                FROM query_result_cache
                WHERE tenant_id = ? AND request_id = ?
                """,
                (auth_context.tenant_id, payload["request_id"]),
            ).fetchone()
        assert row is not None
        assert row["status"] == "completed"
        assert json.loads(row["response_json"])["answer"] == "generated exactly once"
    print("smoke_query_stream_idempotency=ok")


def parse_events(response) -> list[dict]:
    assert response.status_code == 200, response.text
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert events
    return events


if __name__ == "__main__":
    main()
