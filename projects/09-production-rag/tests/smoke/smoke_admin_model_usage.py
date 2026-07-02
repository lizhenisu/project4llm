from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import serve  # noqa: E402
from rag_core.config import load_config  # noqa: E402
from rag_core.database import connect_metadata_db  # noqa: E402
from rag_core.model_usage import record_model_usage, usage_date  # noqa: E402
from rag_core.text_utils import now_ms  # noqa: E402


PREFIX = f"admin-model-usage-{uuid.uuid4().hex[:10]}"


def main() -> None:
    old_env = {
        name: os.environ.get(name)
        for name in (
            "RAG_RUNTIME_DIR",
            "RAG_OBJECT_STORE_DIR",
            "RAG_METADATA_DATABASE_URL",
        )
    }
    with tempfile.TemporaryDirectory(prefix="rag-admin-model-usage-") as tmp:
        os.environ["RAG_RUNTIME_DIR"] = str(Path(tmp) / "runtime")
        os.environ["RAG_OBJECT_STORE_DIR"] = str(Path(tmp) / "object_store")
        os.environ["RAG_METADATA_DATABASE_URL"] = (
            os.environ.get("SMOKE_METADATA_DATABASE_URL") or ""
        )
        mocked_admin = bool(os.environ["RAG_METADATA_DATABASE_URL"])
        try:
            if mocked_admin:
                with patch(
                    "serve.require_admin",
                    return_value=SimpleNamespace(id=f"{PREFIX}-admin"),
                ):
                    test_admin_usage_api(mocked_admin=True)
            else:
                test_admin_usage_api(mocked_admin=False)
        finally:
            cleanup(load_config(), delete_users=not mocked_admin)
            for name, value in old_env.items():
                restore_env(name, value)
    print("smoke_admin_model_usage=ok")


def test_admin_usage_api(*, mocked_admin: bool) -> None:
    api = TestClient(serve.create_app())
    if mocked_admin:
        headers = {"Authorization": "Bearer synthetic-model-usage-admin"}
    else:
        admin = api.post(
            "/auth/login",
            json={
                "username": "admin",
                "password": "admin",
            },
        )
        assert admin.status_code == 200, admin.text
        assert admin.json()["user"]["role"] == "admin"
        headers = {"Authorization": f"Bearer {admin.json()['token']}"}
        user = api.post(
            "/auth/register",
            json={
                "username": f"usage_user_{uuid.uuid4().hex[:6]}",
                "password": "strong-password",
                "display_name": "Usage User",
            },
        )
        assert user.status_code == 200, user.text
        assert api.get("/admin/model-usage").status_code == 401
        assert api.get(
            "/admin/model-usage",
            headers={"Authorization": f"Bearer {user.json()['token']}"},
        ).status_code == 403

    config = load_config()
    timestamp = now_ms()
    today = usage_date(timestamp)
    seed_usage(config, f"{PREFIX}-tenant-a", "answer_generation", 10, 5, timestamp)
    seed_usage(config, f"{PREFIX}-tenant-a", "query_rewrite", 4, 2, timestamp)
    seed_usage(config, f"{PREFIX}-tenant-b", "source_guide", 20, 8, timestamp)

    response = api.get(
        "/admin/model-usage",
        headers=headers,
        params={
            "tenant_id": f"{PREFIX}-tenant-a",
            "start_date": today,
            "end_date": today,
            "limit": 1,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["tenant_id"] == f"{PREFIX}-tenant-a"
    assert body["start_date"] == today
    assert body["end_date"] == today
    assert body["total"] == 2
    assert len(body["rows"]) == 1
    assert body["totals"] == {
        "request_count": 2,
        "prompt_tokens": 14,
        "completion_tokens": 7,
        "total_tokens": 21,
    }
    second_page = api.get(
        "/admin/model-usage",
        headers=headers,
        params={
            "tenant_id": f"{PREFIX}-tenant-a",
            "start_date": today,
            "end_date": today,
            "limit": 1,
            "offset": 1,
        },
    )
    assert second_page.status_code == 200, second_page.text
    assert len(second_page.json()["rows"]) == 1
    assert second_page.json()["rows"][0] != body["rows"][0]

    invalid = api.get(
        "/admin/model-usage",
        headers=headers,
        params={"start_date": "2026-07-02", "end_date": "2026-07-01"},
    )
    assert invalid.status_code == 400
    invalid_calendar_date = api.get(
        "/admin/model-usage",
        headers=headers,
        params={"start_date": "2026-99-01"},
    )
    assert invalid_calendar_date.status_code == 400

    runtime = api.get("/runtime-metrics")
    assert runtime.status_code == 200, runtime.text
    query_usage = runtime.json()["model_usage"]["workloads"]["query"]
    assert query_usage == {
        "request_count": 2,
        "prompt_tokens": 14,
        "completion_tokens": 7,
        "total_tokens": 21,
    }
    assert f"{PREFIX}-tenant-a" not in runtime.text
    prometheus = api.get("/metrics")
    assert prometheus.status_code == 200, prometheus.text
    assert 'rag_model_usage_daily{workload="query",kind="total_tokens"} 21' in prometheus.text
    assert f"{PREFIX}-tenant-a" not in prometheus.text


def seed_usage(
    config,
    tenant_id: str,
    operation: str,
    prompt_tokens: int,
    completion_tokens: int,
    timestamp: int,
) -> None:
    record_model_usage(
        config=config,
        tenant_id=tenant_id,
        principal_key=f"user:{PREFIX}",
        workload="query" if tenant_id.endswith("tenant-a") else "ingestion",
        provider="synthetic-provider",
        model="synthetic-model",
        operation=operation,
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        timestamp_ms=timestamp,
    )


def cleanup(config, *, delete_users: bool) -> None:
    with connect_metadata_db(config) as conn:
        conn.execute(
            "DELETE FROM model_usage_daily WHERE tenant_id LIKE ?",
            (f"{PREFIX}%",),
        )
        if delete_users:
            conn.execute(
                "DELETE FROM sessions WHERE user_id IN "
                "(SELECT id FROM users WHERE username LIKE ?)",
                ("usage_admin_%",),
            )
            conn.execute(
                "DELETE FROM sessions WHERE user_id IN "
                "(SELECT id FROM users WHERE username LIKE ?)",
                ("usage_user_%",),
            )
            conn.execute("DELETE FROM users WHERE username LIKE ?", ("usage_admin_%",))
            conn.execute("DELETE FROM users WHERE username LIKE ?", ("usage_user_%",))


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
