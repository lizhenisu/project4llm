from __future__ import annotations

import os
import sys
import tempfile
import threading
import uuid
from dataclasses import replace
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.config import load_config  # noqa: E402
from rag_core.database import connect_metadata_db  # noqa: E402
from rag_core.query_results import (  # noqa: E402
    QueryResultConflictError,
    QueryResultLeaseGuard,
    append_query_result_event,
    claim_query_result,
    complete_query_result,
    list_query_result_events,
    query_result_cache_snapshot,
    query_result_fingerprint,
    wait_for_query_result,
)
from rag_core.text_utils import now_ms  # noqa: E402


PREFIX = f"query-result-smoke-{uuid.uuid4().hex[:10]}"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="rag-query-result-") as tmp:
        config = replace(
            load_config(),
            metadata_database_url=os.environ.get("SMOKE_METADATA_DATABASE_URL") or None,
            runtime_dir=Path(tmp) / "runtime",
            object_store_dir=Path(tmp) / "object_store",
        )
        try:
            test_owner_waiter_and_cached_replay(config)
            test_conflicting_payload_is_rejected(config)
            test_expired_owner_is_replaced(config)
        finally:
            with connect_metadata_db(config) as conn:
                conn.execute(
                    "DELETE FROM query_result_cache WHERE request_id LIKE ?",
                    (f"{PREFIX}%",),
                )
    print("smoke_query_result_cache=ok")


def test_owner_waiter_and_cached_replay(config) -> None:
    tenant_id = f"{PREFIX}-tenant"
    request_id = f"{PREFIX}-shared"
    fingerprint = query_result_fingerprint({"query": "same query", "doc_ids": ["doc-a"]})
    first = claim_query_result(
        config=config,
        tenant_id=tenant_id,
        request_id=request_id,
        fingerprint=fingerprint,
        lease_ms=1000,
        ttl_ms=60_000,
        owner=f"{PREFIX}-owner",
    )
    assert first.mode == "owner"
    waiting = claim_query_result(
        config=config,
        tenant_id=tenant_id,
        request_id=request_id,
        fingerprint=fingerprint,
        lease_ms=1000,
        ttl_ms=60_000,
        owner=f"{PREFIX}-waiter",
    )
    assert waiting.mode == "waiting"

    guard = QueryResultLeaseGuard(
        config=config,
        tenant_id=tenant_id,
        request_id=request_id,
        owner=first.owner,
        lease_ms=1000,
    )
    guard.start()
    response = {
        "request_id": request_id,
        "answer": "generated once",
        "citations": [],
        "trace": {"retrieval_mode": "synthetic"},
    }
    waiter_result: list = []
    replayed_events: list[dict] = []
    first_sequence = append_query_result_event(
        config=config,
        tenant_id=tenant_id,
        request_id=request_id,
        owner=first.owner,
        event={"type": "stage", "stage": "search", "status": "active"},
    )
    second_sequence = append_query_result_event(
        config=config,
        tenant_id=tenant_id,
        request_id=request_id,
        owner=first.owner,
        event={"type": "stage", "stage": "search", "status": "done"},
    )
    assert (first_sequence, second_sequence) == (1, 2)

    def wait() -> None:
        waiter_result.append(
            wait_for_query_result(
                config=config,
                tenant_id=tenant_id,
                request_id=request_id,
                fingerprint=fingerprint,
                lease_ms=1000,
                ttl_ms=60_000,
                owner=waiting.owner,
                timeout_seconds=5,
                on_event=replayed_events.append,
            )
        )

    thread = threading.Thread(target=wait)
    thread.start()
    try:
        assert complete_query_result(
            config=config,
            tenant_id=tenant_id,
            request_id=request_id,
            owner=first.owner,
            response=response,
            ttl_ms=60_000,
        )
    finally:
        guard.close()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(waiter_result) == 1
    assert waiter_result[0].mode == "cached"
    assert waiter_result[0].response == response
    assert [event["sequence"] for event in replayed_events] == [1, 2]
    assert [event["status"] for event in replayed_events] == ["active", "done"]
    assert [sequence for sequence, _event in list_query_result_events(
        config=config,
        tenant_id=tenant_id,
        request_id=request_id,
    )] == [1, 2]

    replay = claim_query_result(
        config=config,
        tenant_id=tenant_id,
        request_id=request_id,
        fingerprint=fingerprint,
        lease_ms=1000,
        ttl_ms=60_000,
    )
    assert replay.mode == "cached"
    assert replay.response == response
    snapshot = query_result_cache_snapshot(config=config)
    assert snapshot["completed"] >= 1
    assert snapshot["events"] >= 2


def test_conflicting_payload_is_rejected(config) -> None:
    try:
        claim_query_result(
            config=config,
            tenant_id=f"{PREFIX}-tenant",
            request_id=f"{PREFIX}-shared",
            fingerprint=query_result_fingerprint({"query": "different query"}),
            lease_ms=1000,
            ttl_ms=60_000,
        )
    except QueryResultConflictError:
        return
    raise AssertionError("request_id reuse with a different payload was not rejected")


def test_expired_owner_is_replaced(config) -> None:
    tenant_id = f"{PREFIX}-expiry-tenant"
    request_id = f"{PREFIX}-expiry"
    fingerprint = query_result_fingerprint({"query": "recover after crash"})
    old = claim_query_result(
        config=config,
        tenant_id=tenant_id,
        request_id=request_id,
        fingerprint=fingerprint,
        lease_ms=60_000,
        ttl_ms=60_000,
        owner=f"{PREFIX}-expired",
    )
    assert old.mode == "owner"
    with connect_metadata_db(config) as conn:
        conn.execute(
            """
            UPDATE query_result_cache
            SET lease_expires_at = ?
            WHERE tenant_id = ? AND request_id = ?
            """,
            (now_ms() - 1, tenant_id, request_id),
        )
    replacement = claim_query_result(
        config=config,
        tenant_id=tenant_id,
        request_id=request_id,
        fingerprint=fingerprint,
        lease_ms=60_000,
        ttl_ms=60_000,
        owner=f"{PREFIX}-replacement",
    )
    assert replacement.mode == "owner"
    assert replacement.owner != old.owner


if __name__ == "__main__":
    main()
