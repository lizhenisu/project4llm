from __future__ import annotations

import argparse
import sys
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[2]
LOAD_DIR = PROJECT_DIR / "tests" / "load"
if str(LOAD_DIR) not in sys.path:
    sys.path.insert(0, str(LOAD_DIR))

import ingestion_upload_load as load  # noqa: E402


class FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    @staticmethod
    def read() -> bytes:
        return b'{"status":"deleted"}'


def main() -> None:
    args = argparse.Namespace(
        base_url="http://api.test",
        token="synthetic-load-token",
        tenant_id="unused-fixed-tenant",
        tenant_prefix="synthetic-tenant",
        acl_groups="engineering",
        uploads=4,
        users=2,
        docs_per_user=2,
        concurrency=4,
        timeout=1.0,
        sample_limit=10,
        wait=True,
        wait_timeout=2.0,
        poll_interval=0.001,
        output="",
        cleanup=True,
    )
    samples = [
        load.UploadSample(
            index=index,
            tenant_id=f"synthetic-tenant-{index // 2:04d}",
            ok=True,
            status_code=200,
            latency_ms=10.0,
            detail="accepted",
            source_title=f"synthetic-{index}.txt",
        )
        for index in range(4)
    ]
    calls: dict[str, int] = {}

    def staged_sources(_args, *, tenant_id: str):
        calls[tenant_id] = calls.get(tenant_id, 0) + 1
        user_index = int(tenant_id.rsplit("-", 1)[1])
        if tenant_id.endswith("0000") and calls[tenant_id] == 1:
            raise HTTPError(
                "http://api.test/sources",
                500,
                "synthetic transient error",
                {},
                BytesIO(b'{"detail":"synthetic transient error"}'),
            )
        status = "queued" if calls[tenant_id] == 1 else "ready"
        return [
            {
                "doc_id": f"doc-{user_index * 2 + offset}",
                "title": f"synthetic-{user_index * 2 + offset}.txt",
                "status": status,
            }
            for offset in range(2)
        ]

    with patch.object(load, "fetch_sources", side_effect=staged_sources):
        result = load.wait_for_ingestion_completion(args, samples)
    assert result["completed"] is True
    assert result["all_ready"] is True
    assert result["expected"] == 4
    assert result["matched"] == 4
    assert result["missing"] == 0
    assert result["status_counts"] == {"ready": 4}
    assert result["poll_errors"] == 1
    assert result["recent_poll_errors"] == [
        'tenant=synthetic-tenant-0000 status=500 detail={"detail":"synthetic transient error"}'
    ]
    assert set(calls) == {"synthetic-tenant-0000", "synthetic-tenant-0001"}

    def ready_sources(_args, *, tenant_id: str):
        user_index = int(tenant_id.rsplit("-", 1)[1])
        return [
            {
                "doc_id": f"doc-{user_index * 2 + offset}",
                "title": f"synthetic-{user_index * 2 + offset}.txt",
                "status": "ready",
            }
            for offset in range(2)
        ]

    requests = []
    discovery_calls: dict[str, int] = {}

    def transient_ready_sources(_args, *, tenant_id: str):
        discovery_calls[tenant_id] = discovery_calls.get(tenant_id, 0) + 1
        if tenant_id.endswith("0001") and discovery_calls[tenant_id] == 1:
            raise HTTPError(
                "http://api.test/sources",
                503,
                "synthetic transient error",
                {},
                BytesIO(b'{"detail":"retry cleanup discovery"}'),
            )
        return ready_sources(_args, tenant_id=tenant_id)

    def fake_urlopen(request, *, timeout: float):
        requests.append((request, timeout))
        return FakeResponse()

    with (
        patch.object(load, "fetch_sources", side_effect=transient_ready_sources),
        patch.object(load, "urlopen", side_effect=fake_urlopen),
    ):
        cleanup = load.cleanup_ingested_sources(args, samples)
    assert cleanup == {"targets": 4, "deleted": 4, "failed": 0, "failures": []}
    assert len(requests) == 4
    assert discovery_calls["synthetic-tenant-0001"] == 2
    assert all(request.method == "DELETE" for request, _timeout in requests)
    assert all(request.get_header("Authorization") == "Bearer synthetic-load-token" for request, _timeout in requests)
    assert load.preferred_source_status([{"status": "ready"}, {"status": "processing"}]) == "processing"
    print("smoke_ingestion_upload_load=ok")


if __name__ == "__main__":
    main()
