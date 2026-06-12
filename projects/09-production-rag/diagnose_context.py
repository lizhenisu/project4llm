from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from rag_core.config import load_config
from rag_core.context import explain_context_packing
from rag_core.embeddings import build_embedding_model
from rag_core.io import write_jsonl
from rag_core.pipeline import retrieve_and_rerank


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain why reranked candidates are selected or dropped from final context."
    )
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
    parser.add_argument("--context-limit", type=int, default=8)
    parser.add_argument("--max-context-chars", type=int)
    parser.add_argument("--max-chunks-per-doc", type=int)
    parser.add_argument("--min-rerank-score", type=float)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    rows = diagnose_context(
        args.query,
        tenant_id=args.tenant_id,
        acl_groups=args.acl_group or None,
        doc_version=args.doc_version,
        source_types=args.source_type or None,
        candidate_limit=args.candidate_limit,
        context_limit=args.context_limit,
        max_context_chars=args.max_context_chars,
        max_chunks_per_doc=args.max_chunks_per_doc,
        min_rerank_score=args.min_rerank_score,
    )
    print_decisions(rows)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.json_output, rows)


def diagnose_context(
    query: str,
    *,
    tenant_id: str,
    acl_groups: list[str] | None = None,
    doc_version: int | None = None,
    source_types: list[str] | None = None,
    candidate_limit: int = 20,
    context_limit: int = 8,
    max_context_chars: int | None = None,
    max_chunks_per_doc: int | None = None,
    min_rerank_score: float | None = None,
) -> list[dict[str, object]]:
    config = load_config()
    embedding_model = build_embedding_model(config)
    retrieval = retrieve_and_rerank(
        query,
        tenant_id=tenant_id,
        acl_groups=acl_groups,
        doc_version=doc_version,
        source_types=source_types,
        candidate_limit=candidate_limit,
        context_limit=context_limit,
    )
    _, stats, decisions = explain_context_packing(
        retrieval.reranked,
        max_selected=context_limit,
        max_chars=max_context_chars or config.max_context_chars,
        max_chunks_per_doc=max_chunks_per_doc or config.max_chunks_per_doc,
        min_rerank_score=(
            config.min_rerank_score if min_rerank_score is None else min_rerank_score
        ),
        text_unit_counter=embedding_model.count_tokens,
    )
    rows: list[dict[str, object]] = []
    for rank, decision in enumerate(decisions, start=1):
        rows.append(
            {
                "rerank_rank": rank,
                **asdict(decision),
                "packing_stats": asdict(stats),
            }
        )
    return rows


def print_decisions(rows: list[dict[str, object]]) -> None:
    for row in rows:
        print(
            f"rank={row['rerank_rank']} decision={row['decision']} "
            f"reason={row['reason']} doc={row['doc_id']} chunk={row['chunk_index']} "
            f"rerank={row['rerank_score']} text_units={row['text_chars']} "
            f"used_before={row['used_chars_before']}"
        )


if __name__ == "__main__":
    main()
