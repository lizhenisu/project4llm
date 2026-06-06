from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, zero_image_vector
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.text_utils import chunk_document
from rag_core.types import SourceDocument
from rag_core.versioning import publish_current_versions


PROJECT_DIR = Path(__file__).resolve().parent


def main() -> None:
    args = parse_args()
    if args.serve_local:
        serve_local_process(port=args.port)
        return

    configured_base_url = os.environ.get("RAG_API_URL")
    tenant_id = os.environ.get("RAG_DEPLOY_TENANT_ID", "team_a")
    acl_groups = parse_csv(os.environ.get("RAG_DEPLOY_ACL_GROUPS")) or ["ops"]

    if configured_base_url:
        run_smoke(
            base_url=configured_base_url.rstrip("/"),
            tenant_id=tenant_id,
            acl_groups=acl_groups,
        )
        return

    with tempfile.TemporaryDirectory() as tmp:
        port = reserve_local_port()
        runtime_dir = Path(tmp) / "runtime"
        env_overrides = {
            "RAG_MILVUS_URI": str(Path(tmp) / "deploy.db"),
            "RAG_COLLECTION": "rag_smoke_deploy",
            "RAG_OBJECT_STORE_DIR": str(Path(tmp) / "object_store"),
            "RAG_RUNTIME_DIR": str(runtime_dir),
            "RAG_REQUIRE_AUTH_CONTEXT": os.environ.get("RAG_REQUIRE_AUTH_CONTEXT", "1"),
            "RAG_API_TOKEN": os.environ.get("RAG_API_TOKEN", "dev-only-token"),
            "RAG_DEPLOY_TENANT_ID": tenant_id,
            "RAG_DEPLOY_ACL_GROUPS": ",".join(acl_groups),
        }
        with temporary_env(env_overrides):
            process = start_local_server(port=port, env=server_env(env_overrides))
            try:
                base_url = f"http://127.0.0.1:{port}"
                wait_for_server(base_url, process=process)
                run_smoke(
                    base_url=base_url,
                    tenant_id=tenant_id,
                    acl_groups=acl_groups,
                )
            finally:
                stop_local_server(process)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve-local", action="store_true")
    parser.add_argument("--port", type=int, default=8008)
    return parser.parse_args()


def run_smoke(*, base_url: str, tenant_id: str, acl_groups: list[str]) -> None:
    headers = auth_headers(tenant_id=tenant_id, acl_groups=acl_groups)

    health = request("GET", f"{base_url}/health")
    assert health["status"] == "ok"

    ready = request("GET", f"{base_url}/ready", headers=headers)
    assert ready["status"] == "ok"

    search = request(
        "POST",
        f"{base_url}/search",
        {
            "query": "RAG 检索变慢时应该排查什么",
            "tenant_id": tenant_id,
            "acl_groups": acl_groups,
            "candidate_limit": 5,
            "context_limit": 3,
            "request_id": "smoke-deploy-search",
        },
        headers=headers,
    )
    assert search["request_id"] == "smoke-deploy-search"
    assert search["hits"]
    assert search["trace"]["filter_expr"]

    query = request(
        "POST",
        f"{base_url}/query",
        {
            "query": "RAG 检索变慢时应该排查什么",
            "tenant_id": tenant_id,
            "acl_groups": acl_groups,
            "candidate_limit": 5,
            "context_limit": 3,
            "request_id": "smoke-deploy-query",
        },
        headers=headers,
    )
    assert query["request_id"] == "smoke-deploy-query"
    assert query["answer"]
    assert query["citations"]

    feedback = request(
        "POST",
        f"{base_url}/feedback",
        {
            "request_id": query["request_id"],
            "rating": 1,
            "comment": "deploy smoke ok",
            "selected_doc_ids": selected_doc_ids(query),
        },
        headers=headers,
    )
    assert feedback["status"] == "accepted"
    assert feedback["request_id"] == query["request_id"]

    print(
        f"smoke_deploy=ok base_url={base_url} "
        f"ready={ready['status']} hits={len(search['hits'])} "
        f"citations={len(query['citations'])}"
    )


def seed_local_deploy_data() -> None:
    config = load_config()
    client = connect(config)
    try:
        ensure_collection(client, config, reset=True)
        doc = SourceDocument(
            tenant_id=os.environ.get("RAG_DEPLOY_TENANT_ID", "team_a"),
            doc_id="deploy-runbook",
            doc_version=1,
            source_type="md",
            source_uri="memory://deploy-runbook",
            title="Deploy Runbook",
            text="RAG 检索变慢时应该排查 rewrite、Milvus search、rerank 和 context packing。",
            acl_groups=parse_csv(os.environ.get("RAG_DEPLOY_ACL_GROUPS")) or ["ops"],
        )
        chunks = chunk_document(doc, chunk_size=config.chunk_size, overlap=config.chunk_overlap)
        text_model = build_embedding_model(config)
        dense_vectors = text_model.encode([chunk.text for chunk in chunks])
        zero_image = zero_image_vector(config)
        upsert_entities(
            client,
            collection_name=config.collection_name,
            entities=[
                chunk_to_entity(
                    chunk,
                    dense_vector=dense_vector,
                    image_vector=zero_image,
                    embedding_model=text_model.model_name,
                    embedding_dim=text_model.dim,
                )
                for chunk, dense_vector in zip(chunks, dense_vectors, strict=True)
            ],
        )
        publish_current_versions(config.object_store_dir, [doc])
    finally:
        client.close()


def serve_local_process(*, port: int) -> None:
    seed_local_deploy_data()

    import uvicorn

    uvicorn.run(
        "serve:app",
        host="127.0.0.1",
        port=port,
        reload=False,
        app_dir=str(PROJECT_DIR),
    )


def start_local_server(*, port: int, env: dict[str, str]) -> subprocess.Popen[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--serve-local",
        "--port",
        str(port),
    ]
    return subprocess.Popen(
        command,
        cwd=PROJECT_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def stop_local_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def wait_for_server(
    base_url: str,
    *,
    process: subprocess.Popen[str],
    timeout_seconds: float = 10.0,
) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                "local deploy smoke server exited before ready: "
                f"returncode={process.returncode}\nstdout={stdout}\nstderr={stderr}"
            )
        try:
            health = request("GET", f"{base_url}/health")
            if health.get("status") == "ok":
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.2)
    if last_error is not None:
        raise RuntimeError(f"local deploy smoke server did not become ready: {last_error}")
    raise RuntimeError("local deploy smoke server did not become ready")


def reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        return int(sock.getsockname()[1])


def server_env(overrides: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(overrides)
    # Milvus Lite uses a local file path, which pymilvus cannot parse from the
    # global MILVUS_URI import-time setting used by the ORM layer.
    if "RAG_MILVUS_URI" in overrides:
        env.pop("MILVUS_URI", None)
    pythonpath = env.get("PYTHONPATH", "")
    project_path = str(PROJECT_DIR)
    env["PYTHONPATH"] = (
        project_path
        if not pythonpath
        else f"{project_path}{os.pathsep}{pythonpath}"
    )
    return env


@contextmanager
def temporary_env(overrides: dict[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def request(
    method: str,
    url: str,
    payload: dict | None = None,
    *,
    headers: dict[str, str] | None = None,
) -> dict:
    data = None
    request_headers = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: {exc.code} {body}") from exc


def auth_headers(*, tenant_id: str, acl_groups: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    api_token = os.environ.get("RAG_API_TOKEN")
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    if os.environ.get("RAG_REQUIRE_AUTH_CONTEXT", "").lower() in {"1", "true", "yes", "on"}:
        headers["X-RAG-Tenant-ID"] = tenant_id
        headers["X-RAG-ACL-Groups"] = ",".join(acl_groups)
    return headers


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def selected_doc_ids(query_response: dict) -> list[str]:
    citations = query_response.get("citations", [])
    if not citations:
        return []
    return [str(citations[0]["doc_id"])]


if __name__ == "__main__":
    main()
