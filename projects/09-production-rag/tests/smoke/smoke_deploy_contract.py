from __future__ import annotations

import os

from smoke_deploy import auth_headers, selected_doc_ids


def main() -> None:
    old_env = {
        "RAG_API_TOKEN": os.environ.get("RAG_API_TOKEN"),
        "RAG_REQUIRE_AUTH_CONTEXT": os.environ.get("RAG_REQUIRE_AUTH_CONTEXT"),
    }
    try:
        os.environ["RAG_API_TOKEN"] = "dev-token"
        os.environ["RAG_REQUIRE_AUTH_CONTEXT"] = "1"
        headers = auth_headers(
            tenant_id="team_a",
            acl_groups=["ops", "support"],
        )
        assert headers == {
            "Authorization": "Bearer dev-token",
            "X-RAG-Tenant-ID": "team_a",
            "X-RAG-ACL-Groups": "ops,support",
        }
        assert selected_doc_ids({"citations": [{"doc_id": "rag-runbook"}]}) == [
            "rag-runbook"
        ]
        assert selected_doc_ids({"citations": []}) == []
    finally:
        for name, value in old_env.items():
            restore_env(name, value)
    print("smoke_deploy_contract=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
