from __future__ import annotations

import os

from rag_core.pipeline import retrieve_and_rerank


def main() -> None:
    previous = os.environ.get("RAG_QUERY_REWRITE_BACKEND")
    os.environ["RAG_QUERY_REWRITE_BACKEND"] = "heuristic"
    try:
        result = retrieve_and_rerank(
            "怎么排查？",
            history=["用户在问 RAG 检索延迟升高的问题"],
            tenant_id="team_a",
            acl_groups=["ops"],
            candidate_limit=5,
            context_limit=3,
            request_id="smoke-rewrite",
        )
        assert result.trace.original_query == "怎么排查？"
        assert "RAG" in result.trace.rewritten_query
        assert result.trace.rewrite_backend == "heuristic"
        assert result.hits
    finally:
        if previous is None:
            os.environ.pop("RAG_QUERY_REWRITE_BACKEND", None)
        else:
            os.environ["RAG_QUERY_REWRITE_BACKEND"] = previous

    print("smoke_rewrite=ok")


if __name__ == "__main__":
    main()

