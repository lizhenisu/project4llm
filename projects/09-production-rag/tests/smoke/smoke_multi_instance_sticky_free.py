from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import replace
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.config import load_config  # noqa: E402
from rag_core.database import connect_metadata_db  # noqa: E402
from rag_core.user_auth import is_registration_enabled, set_registration_enabled  # noqa: E402


PREFIX = f"sticky-free-{uuid.uuid4().hex[:10]}"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        metadata_url = os.environ.get("SMOKE_METADATA_DATABASE_URL") or ""
        previous_registration_enabled = None
        if metadata_url:
            config = smoke_config(metadata_url, root)
            previous_registration_enabled = is_registration_enabled(config)
            set_registration_enabled(config, enabled=True)
        environment = server_environment(root, metadata_url=metadata_url)
        first_port = free_port()
        second_port = free_port()
        servers: list[tuple[subprocess.Popen, object, Path]] = []
        try:
            first = start_server(first_port, environment, root / "first.log")
            servers.append(first)
            wait_for_health(first_port, first)
            second = start_server(second_port, environment, root / "second.log")
            servers.append(second)
            wait_for_health(second_port, second)
            run_cross_instance_flow(first_port, second_port)
        finally:
            for process, _handle, _path in servers:
                process.terminate()
            for process, handle, _path in servers:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                handle.close()
            if metadata_url:
                cleanup_postgres(metadata_url, root)
                set_registration_enabled(
                    smoke_config(metadata_url, root),
                    enabled=bool(previous_registration_enabled),
                )
    print("smoke_multi_instance_sticky_free=ok")


def run_cross_instance_flow(first_port: int, second_port: int) -> None:
    username = f"{PREFIX}_user"
    password = "synthetic-strong-password"
    registered = request_json(
        first_port,
        "POST",
        "/auth/register",
        payload={
            "username": username,
            "password": password,
            "display_name": "Synthetic Multi Instance User",
        },
        expected_status=200,
    )
    token = registered["token"]
    tenant_id = registered["user"]["tenant_id"]
    user_id = registered["user"]["id"]

    me_on_second = request_json(
        second_port,
        "GET",
        "/auth/me",
        token=token,
        expected_status=200,
    )
    assert me_on_second["id"] == user_id
    assert me_on_second["tenant_id"] == tenant_id

    refreshed = request_json(
        second_port,
        "POST",
        "/auth/token/refresh",
        token=token,
        expected_status=200,
    )
    refreshed_token = refreshed["token"]
    assert refreshed_token != token
    me_on_first = request_json(
        first_port,
        "GET",
        "/auth/me",
        token=refreshed_token,
        expected_status=200,
    )
    assert me_on_first["id"] == user_id
    time.sleep(0.2)
    request_json(
        first_port,
        "GET",
        "/auth/me",
        token=token,
        expected_status=401,
    )

    conversation_id = f"{PREFIX}-conversation"
    first_payload = conversation_payload(
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        title="created on first",
        content="message created on first process",
    )
    created = request_json(
        first_port,
        "POST",
        "/conversations",
        payload=first_payload,
        token=refreshed_token,
        expected_status=200,
    )
    assert created["title"] == "created on first"

    loaded_on_second = request_json(
        second_port,
        "GET",
        f"/conversations/{conversation_id}?tenant_id={tenant_id}",
        token=refreshed_token,
        expected_status=200,
    )
    assert loaded_on_second["messages"][0]["content"] == "message created on first process"

    second_payload = conversation_payload(
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        title="updated on second",
        content="message updated on second process",
    )
    request_json(
        second_port,
        "POST",
        "/conversations",
        payload=second_payload,
        token=refreshed_token,
        expected_status=200,
    )
    listed_on_first = request_json(
        first_port,
        "GET",
        f"/conversations?tenant_id={tenant_id}",
        token=refreshed_token,
        expected_status=200,
    )
    listed = next(item for item in listed_on_first["conversations"] if item["id"] == conversation_id)
    assert listed["title"] == "updated on second"
    assert listed["message_count"] == 1

    deleted = request_json(
        first_port,
        "DELETE",
        f"/conversations/{conversation_id}?tenant_id={tenant_id}",
        token=refreshed_token,
        expected_status=200,
    )
    assert deleted["status"] == "deleted"
    request_json(
        second_port,
        "GET",
        f"/conversations/{conversation_id}?tenant_id={tenant_id}",
        token=refreshed_token,
        expected_status=404,
    )
    request_json(
        second_port,
        "POST",
        "/auth/logout",
        token=refreshed_token,
        expected_status=200,
    )
    time.sleep(0.2)
    request_json(
        first_port,
        "GET",
        "/auth/me",
        token=refreshed_token,
        expected_status=401,
    )


def conversation_payload(
    *,
    conversation_id: str,
    tenant_id: str,
    title: str,
    content: str,
) -> dict:
    return {
        "id": conversation_id,
        "tenant_id": tenant_id,
        "title": title,
        "messages": [
            {
                "id": f"{conversation_id}-message",
                "role": "user",
                "content": content,
                "status": "done",
            }
        ],
        "source_doc_ids": [],
    }


def server_environment(root: Path, *, metadata_url: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(PROJECT_DIR),
            "RAG_RUNTIME_DIR": str(root / "runtime"),
            "RAG_OBJECT_STORE_DIR": str(root / "object_store"),
            "RAG_OBJECT_STORE_BACKEND": "local",
            "RAG_METADATA_DATABASE_URL": metadata_url,
            "RAG_FIXED_TEST_LOGIN_TOKEN": f"{PREFIX}-fixed-test-token",
            "RAG_AUTH_TOKEN_CACHE_TTL_SECONDS": "0.1",
            "RAG_QUERY_SHARED_ADMISSION": "1",
        }
    )
    return environment


def start_server(port: int, environment: dict[str, str], log_path: Path):
    handle = log_path.open("w+", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "serve:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=PROJECT_DIR,
        env=environment,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, handle, log_path


def wait_for_health(port: int, server, *, timeout: float = 20.0) -> None:
    process, handle, log_path = server
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"server exited early: {read_log(handle, log_path)}")
        try:
            request_json(port, "GET", "/health", expected_status=200)
            return
        except (OSError, AssertionError):
            time.sleep(0.1)
    raise AssertionError(f"server did not become healthy: {read_log(handle, log_path)}")


def request_json(
    port: int,
    method: str,
    path: str,
    *,
    payload: dict | None = None,
    token: str | None = None,
    expected_status: int,
) -> dict:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            status = response.status
            response_body = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        response_body = exc.read()
    assert status == expected_status, (status, expected_status, response_body.decode("utf-8", errors="replace"))
    return json.loads(response_body or b"{}")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def read_log(handle, path: Path) -> str:
    handle.flush()
    return path.read_text(encoding="utf-8")[-4000:]


def cleanup_postgres(metadata_url: str, root: Path) -> None:
    config = smoke_config(metadata_url, root)
    with connect_metadata_db(config) as conn:
        row = conn.execute("SELECT id FROM users WHERE username = ?", (f"{PREFIX}_user",)).fetchone()
        if row is not None:
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (row["id"],))
            conn.execute("DELETE FROM users WHERE id = ?", (row["id"],))


def smoke_config(metadata_url: str, root: Path):
    return replace(
        load_config(),
        metadata_database_url=metadata_url,
        runtime_dir=root / "runtime",
        object_store_dir=root / "object_store",
    )


if __name__ == "__main__":
    main()
