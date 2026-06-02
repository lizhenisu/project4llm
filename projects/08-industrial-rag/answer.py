from __future__ import annotations

import argparse
from dataclasses import dataclass

from rag_core.answering import generate_answer
from rag_core.config import load_config
from rag_core.pipeline import retrieve_and_rerank
from rag_core.types import SearchHit


@dataclass(frozen=True)
class AnswerResult:
    request_id: str
    answer: str
    hits: list[SearchHit]
    candidates: list[SearchHit]
    reranked: list[SearchHit]
    trace: object
    generation: object


def answer_query(
    query: str,
    *,
    tenant_id: str,
    candidate_limit: int,
    context_limit: int,
    acl_groups: list[str] | None = None,
    doc_version: int | None = None,
    source_types: list[str] | None = None,
    history: list[str] | None = None,
    request_id: str | None = None,
) -> AnswerResult:
    config = load_config()
    retrieval = retrieve_and_rerank(
        query,
        tenant_id=tenant_id,
        candidate_limit=candidate_limit,
        context_limit=context_limit,
        acl_groups=acl_groups,
        doc_version=doc_version,
        source_types=source_types,
        history=history,
        request_id=request_id,
    )
    generation = generate_answer(config, retrieval.trace.rewritten_query, retrieval.hits)
    return AnswerResult(
        request_id=retrieval.request_id,
        answer=generation.answer,
        hits=retrieval.hits,
        candidates=retrieval.candidates,
        reranked=retrieval.reranked,
        trace=retrieval.trace,
        generation=generation,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full retrieval -> rerank -> answer flow.")
    parser.add_argument("query")
    parser.add_argument("--tenant-id", default="team_a")
    parser.add_argument(
        "--acl-group",
        action="append",
        default=[],
        help="Allowed ACL group. Repeat to allow multiple groups.",
    )
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--context-limit", type=int, default=5)
    parser.add_argument("--doc-version", type=int)
    parser.add_argument(
        "--source-type",
        action="append",
        default=[],
        help="Restrict retrieval to a source type. Repeat for multiple types.",
    )
    args = parser.parse_args()

    result = answer_query(
        args.query,
        tenant_id=args.tenant_id,
        candidate_limit=args.candidate_limit,
        context_limit=args.context_limit,
        acl_groups=args.acl_group or None,
        doc_version=args.doc_version,
        source_types=args.source_type or None,
    )
    print(f"request_id: {result.request_id}\n")
    print(result.answer)
    print("\nCitations:")
    for index, hit in enumerate(result.hits, start=1):
        print(
            f"[{index}] doc={hit.doc_id} chunk={hit.chunk_index} "
            f"source={hit.source_type} acl={','.join(hit.acl_groups)}"
        )


if __name__ == "__main__":
    main()
