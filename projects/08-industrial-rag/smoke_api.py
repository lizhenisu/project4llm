from __future__ import annotations

from fastapi.testclient import TestClient

from serve import app


def main() -> None:
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    search = client.post(
        "/search",
        json={
            "query": "RAG 检索变慢时应该排查什么",
            "tenant_id": "team_a",
            "acl_groups": ["ops"],
            "candidate_limit": 5,
            "context_limit": 3,
            "request_id": "smoke-api-search",
        },
    )
    assert search.status_code == 200, search.text
    search_body = search.json()
    assert search_body["request_id"] == "smoke-api-search"
    assert search_body["hits"]
    assert search_body["trace"]["filter_expr"]

    query = client.post(
        "/query",
        json={
            "query": "RAG 检索变慢时应该排查什么",
            "tenant_id": "team_a",
            "acl_groups": ["ops"],
            "candidate_limit": 5,
            "context_limit": 3,
        },
    )
    assert query.status_code == 200, query.text
    query_body = query.json()
    assert query_body["request_id"]
    assert query_body["answer"]
    assert query_body["citations"]

    feedback = client.post(
        "/feedback",
        json={
            "request_id": query_body["request_id"],
            "rating": 1,
            "comment": "smoke ok",
            "selected_doc_ids": [query_body["citations"][0]["doc_id"]],
        },
    )
    assert feedback.status_code == 200, feedback.text
    assert feedback.json()["status"] == "accepted"

    print("smoke_api=ok")


if __name__ == "__main__":
    main()

