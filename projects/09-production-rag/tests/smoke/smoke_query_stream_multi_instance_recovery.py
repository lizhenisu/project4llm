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
from typing import BinaryIO


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.config import load_config  # noqa: E402
from rag_core.database import connect_metadata_db  # noqa: E402


REQUEST_ID = f"multi-instance-recovery-{uuid.uuid4().hex}"
TENANT_ID = "tenant-fixed-test"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="rag-query-multi-instance-") as tmp:
        root = Path(tmp)
        metadata_url = os.environ.get("SMOKE_METADATA_DATABASE_URL") or ""
        environment = server_environment(root, metadata_url=metadata_url)
        first_port = free_port()
        second_port = free_port()
        servers: list[tuple[subprocess.Popen, object, Path]] = []
        first_stream: BinaryIO | None = None
        try:
            first = start_server(first_port, environment, root / "first.log")
            servers.append(first)
            wait_for_health(first_port, first)
            second = start_server(second_port, environment, root / "second.log")
            servers.append(second)
            wait_for_health(second_port, second)

            first_stream = open_stream(first_port, query_payload())
            first_events = read_until(
                first_stream,
                lambda event: event.get("stage") == "search" and event.get("status") == "active",
            )
            first_search = next(event for event in first_events if event.get("stage") == "search")
            assert first_search["sequence"] == 1, first_events
            first_stream.close()
            first_stream = None
            wait_for_path(root / "coordination" / "started")

            recovery_stream = open_stream(second_port, query_payload())
            recovery_prefix = read_until(
                recovery_stream,
                lambda event: event.get("stage") == "search" and event.get("status") == "active",
            )
            assert any(
                event.get("stage") == "resume" and event.get("status") == "active"
                for event in recovery_prefix
            ), recovery_prefix
            replayed_active = next(event for event in recovery_prefix if event.get("stage") == "search")
            assert replayed_active["sequence"] == 1, recovery_prefix

            (root / "coordination" / "release").touch()
            recovery_events = recovery_prefix + read_to_end(recovery_stream)
            assert_search_sequence(recovery_events)
            assert recovery_events[-1]["type"] == "result", recovery_events
            assert recovery_events[-1]["answer"] == "跨实例恢复只执行了一次模型工作。"
            assert any(
                event.get("stage") == "resume" and event.get("status") == "done"
                for event in recovery_events
            ), recovery_events

            cached_events = request_stream(first_port, query_payload())
            assert_search_sequence(cached_events)
            assert cached_events[-1]["type"] == "result", cached_events
            assert cached_events[-1]["answer"] == "跨实例恢复只执行了一次模型工作。"

            conflict_events = request_stream(
                second_port,
                {**query_payload(), "query": "conflicting retry payload"},
            )
            assert conflict_events[-1]["type"] == "error", conflict_events
            assert "already associated with a different query" in conflict_events[-1]["detail"]

            call_marker = root / "coordination" / "model-call"
            assert call_marker.exists()
            assert call_marker.read_text(encoding="utf-8").strip().isdigit()
            assert_query_result_completed(metadata_url, root)
        finally:
            if first_stream is not None:
                first_stream.close()
            for process, _handle, _path in servers:
                process.terminate()
            for process, handle, _path in servers:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                handle.close()
            cleanup_query_result(metadata_url, root)
    print("smoke_query_stream_multi_instance_recovery=ok")


def query_payload() -> dict[str, object]:
    return {
        "query": "recover this request on another API instance",
        "tenant_id": TENANT_ID,
        "acl_groups": ["engineering"],
        "request_id": REQUEST_ID,
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
            "RAG_FIXED_TEST_LOGIN_TOKEN": "multi-instance-recovery-fixed-token",
            "RAG_QUERY_RECOVERY_COORDINATION_DIR": str(root / "coordination"),
            "RAG_QUERY_RESULT_POLL_SECONDS": "0.02",
            "RAG_QUERY_RESULT_POLL_MAX_SECONDS": "0.05",
            "RAG_QUERY_RESULT_WAIT_SECONDS": "20",
            "RAG_QUERY_SHARED_ADMISSION": "1",
            "RAG_QUERY_STREAM_QUEUE_LIMIT": "8",
            "RAG_QUERY_STREAM_TENANT_QUEUE_LIMIT": "8",
            "RAG_QUERY_STREAM_USER_QUEUE_LIMIT": "8",
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
            "tests.smoke.query_recovery_test_app:app",
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
            request_json(port, "/health", expected_status=200)
            return
        except (OSError, AssertionError):
            time.sleep(0.1)
    raise AssertionError(f"server did not become healthy: {read_log(handle, log_path)}")


def open_stream(port: int, payload: dict[str, object]) -> BinaryIO:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/query/stream",
        data=body,
        headers={
            "Authorization": "Bearer multi-instance-recovery-fixed-token",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=25)


def request_stream(port: int, payload: dict[str, object]) -> list[dict]:
    with open_stream(port, payload) as response:
        return read_to_end(response)


def read_until(response: BinaryIO, predicate) -> list[dict]:
    events: list[dict] = []
    while True:
        line = response.readline()
        if not line:
            raise AssertionError(f"stream ended before expected event: {events}")
        event = json.loads(line)
        events.append(event)
        if predicate(event):
            return events


def read_to_end(response: BinaryIO) -> list[dict]:
    return [json.loads(line) for line in response if line.strip()]


def assert_search_sequence(events: list[dict]) -> None:
    search_events = [event for event in events if event.get("stage") == "search"]
    assert [event["status"] for event in search_events] == ["active", "done"], events
    assert [event["sequence"] for event in search_events] == [1, 2], events


def request_json(port: int, path: str, *, expected_status: int) -> dict:
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            status = response.status
            body = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = exc.read()
    assert status == expected_status, (status, expected_status, body)
    return json.loads(body or b"{}")


def wait_for_path(path: Path, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise AssertionError(f"path did not appear: {path}")


def assert_query_result_completed(metadata_url: str, root: Path) -> None:
    config = smoke_config(metadata_url, root)
    with connect_metadata_db(config) as conn:
        row = conn.execute(
            """
            SELECT status, response_json
            FROM query_result_cache
            WHERE tenant_id = ? AND request_id = ?
            """,
            (TENANT_ID, REQUEST_ID),
        ).fetchone()
        events = conn.execute(
            """
            SELECT sequence
            FROM query_result_events
            WHERE tenant_id = ? AND request_id = ?
            ORDER BY sequence
            """,
            (TENANT_ID, REQUEST_ID),
        ).fetchall()
    assert row is not None
    assert row["status"] == "completed"
    assert json.loads(row["response_json"])["answer"] == "跨实例恢复只执行了一次模型工作。"
    assert [event["sequence"] for event in events] == [1, 2]


def cleanup_query_result(metadata_url: str, root: Path) -> None:
    config = smoke_config(metadata_url, root)
    try:
        with connect_metadata_db(config) as conn:
            conn.execute(
                "DELETE FROM query_result_cache WHERE tenant_id = ? AND request_id = ?",
                (TENANT_ID, REQUEST_ID),
            )
    except Exception:
        pass


def smoke_config(metadata_url: str, root: Path):
    return replace(
        load_config(),
        metadata_database_url=metadata_url or None,
        runtime_dir=root / "runtime",
        object_store_dir=root / "object_store",
    )


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def read_log(handle, path: Path) -> str:
    handle.flush()
    return path.read_text(encoding="utf-8")[-4000:]


if __name__ == "__main__":
    main()
