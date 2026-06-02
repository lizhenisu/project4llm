from __future__ import annotations

import os

from answer import answer_query
from rag_core.pipeline import retrieve_and_rerank


def main() -> None:
    previous = os.environ.get("RAG_MIN_RERANK_SCORE")
    os.environ["RAG_MIN_RERANK_SCORE"] = "999"
    try:
        retrieval = retrieve_and_rerank(
            "RAG 检索变慢时应该排查什么",
            tenant_id="team_a",
            acl_groups=["ops"],
            candidate_limit=5,
            context_limit=3,
            request_id="smoke-context",
        )
        assert retrieval.trace.dropped_by_score > 0
        assert retrieval.trace.context_count == 0

        result = answer_query(
            "RAG 检索变慢时应该排查什么",
            tenant_id="team_a",
            acl_groups=["ops"],
            candidate_limit=5,
            context_limit=3,
        )
        assert result.answer == "当前知识库没有足够证据。"
    finally:
        if previous is None:
            os.environ.pop("RAG_MIN_RERANK_SCORE", None)
        else:
            os.environ["RAG_MIN_RERANK_SCORE"] = previous

    print("smoke_context=ok")


if __name__ == "__main__":
    main()

