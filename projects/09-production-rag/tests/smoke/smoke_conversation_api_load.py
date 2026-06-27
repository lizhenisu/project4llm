from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[2]
LOAD_DIR = PROJECT_DIR / "tests" / "load"
if str(LOAD_DIR) not in sys.path:
    sys.path.insert(0, str(LOAD_DIR))

import conversation_api_load as load  # noqa: E402


def main() -> None:
    args = argparse.Namespace(
        base_urls="http://api-a.test,http://api-b.test",
        token="synthetic-load-token",
        tenant_prefix="synthetic-load-tenant",
        users=12,
        concurrency=4,
        timeout=1.0,
        output="",
    )
    calls: list[tuple[str, str, str]] = []
    conversation_ids: dict[str, str] = {}

    def fake_request(
        _args: argparse.Namespace,
        origin: str,
        path: str,
        tenant_id: str,
        *,
        method: str = "GET",
        payload=None,
    ):
        calls.append((origin, method, path))
        if path == "/conversations":
            conversation_ids[tenant_id] = payload["id"]
            return payload
        if path.startswith("/conversations?"):
            return {"conversations": [{"id": conversation_ids[tenant_id]}]}
        if method == "DELETE":
            return {"status": "deleted"}
        return {
            "title": "updated",
            "messages": [{"id": "one"}, {"id": "two"}],
        }

    with patch.object(load, "request_json", side_effect=fake_request):
        samples = load.run_load(args)
    summary = load.build_summary(args, samples, wall_ms=1000.0)

    assert summary["success"] == 12
    assert summary["failed"] == 0
    assert summary["request_throughput_rps"] == 60.0
    assert set(origin for origin, _method, _path in calls) == {
        "http://api-a.test",
        "http://api-b.test",
    }
    assert sum(1 for _origin, method, _path in calls if method == "DELETE") == 12
    assert load.parse_base_urls("http://a.test, https://b.test/") == [
        "http://a.test",
        "https://b.test",
    ]
    print("smoke_conversation_api_load=ok")


if __name__ == "__main__":
    main()
