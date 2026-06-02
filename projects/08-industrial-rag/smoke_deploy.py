from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def main() -> None:
    base_url = os.environ.get("RAG_API_URL", "http://127.0.0.1:8008").rstrip("/")
    health = request("GET", f"{base_url}/health")
    assert health["status"] == "ok"

    search = request(
        "POST",
        f"{base_url}/search",
        {
            "query": "RAG 检索变慢时应该排查什么",
            "tenant_id": "team_a",
            "acl_groups": ["ops"],
            "candidate_limit": 5,
            "context_limit": 3,
            "request_id": "smoke-deploy-search",
        },
    )
    assert search["request_id"] == "smoke-deploy-search"
    assert "hits" in search
    print(f"smoke_deploy=ok base_url={base_url} hits={len(search['hits'])}")


def request(method: str, url: str, payload: dict | None = None) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: {exc.code} {body}") from exc


if __name__ == "__main__":
    main()

