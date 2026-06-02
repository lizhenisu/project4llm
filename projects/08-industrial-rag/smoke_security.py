from __future__ import annotations

from fastapi.testclient import TestClient

from rag_core.pii import apply_pii_policy, detect_pii
from serve import app


def main() -> None:
    assert detect_pii("联系 test@example.com 或 13800138000")
    redacted = apply_pii_policy(
        "联系 test@example.com 或 13800138000",
        policy="redact",
        label="smoke",
    )
    assert "test@example.com" not in redacted
    assert "13800138000" not in redacted

    client = TestClient(app)
    response = client.post(
        "/search",
        json={
            "query": "team_b 报销规则",
            "tenant_id": "team_a",
            "acl_groups": ["finance"],
            "candidate_limit": 10,
            "context_limit": 5,
            "request_id": "smoke-security",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    doc_ids = [hit["doc_id"] for hit in body["hits"]]
    assert "finance-private" not in doc_ids, doc_ids
    assert not doc_ids
    assert body["trace"]["retrieval_mode"] == "blocked_cross_tenant_query"
    print("smoke_security=ok")


if __name__ == "__main__":
    main()
