from __future__ import annotations

import argparse

from rag_core.config import load_config
from rag_core.rerankers import build_reranker
from search_hybrid import run_hybrid


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid search followed by rerank.")
    parser.add_argument("query")
    parser.add_argument("--tenant-id", default="team_a")
    parser.add_argument(
        "--acl-group",
        action="append",
        default=[],
        help="Allowed ACL group. Repeat to allow multiple groups.",
    )
    parser.add_argument("--doc-version", type=int)
    parser.add_argument(
        "--source-type",
        action="append",
        default=[],
        help="Restrict retrieval to a source type. Repeat for multiple types.",
    )
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    config = load_config()
    hits = run_hybrid(
        args.query,
        tenant_id=args.tenant_id,
        limit=args.candidate_limit,
        acl_groups=args.acl_group or None,
        doc_version=args.doc_version,
        source_types=args.source_type or None,
    )
    reranker = build_reranker(config)
    reranked = reranker.rerank(args.query, hits, limit=args.limit)
    for rank, hit in enumerate(reranked, start=1):
        print(
            f"{rank}. rerank={hit.rerank_score:.4f} search={hit.score:.4f} "
            f"doc={hit.doc_id} title={hit.title}"
        )
        print(hit.text[:260].replace("\n", " "))


if __name__ == "__main__":
    main()
