from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from answer import answer_query
from answer_multimodal import answer_multimodal_query
from rag_core.citations import citation_accuracy, faithfulness_score, is_refusal, term_coverage
from rag_core.config import DATA_DIR
from rag_core.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate answer citation accuracy, evidence hit, and refusal quality."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "eval_queries.jsonl",
        help="JSONL eval set.",
    )
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--context-limit", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=["text", "multimodal"],
        default="text",
        help="Answer pipeline to evaluate.",
    )
    parser.add_argument(
        "--force-refusal-threshold",
        action="store_true",
        help="Temporarily set a high rerank threshold to exercise refusal behavior.",
    )
    parser.add_argument("--json-output", type=Path, help="Write metrics as JSON.")
    args = parser.parse_args()

    previous_threshold = os.environ.get("RAG_MIN_RERANK_SCORE")
    if args.force_refusal_threshold:
        os.environ["RAG_MIN_RERANK_SCORE"] = "999"

    try:
        metrics = evaluate_answers(
            input_path=args.input,
            candidate_limit=args.candidate_limit,
            context_limit=args.context_limit,
            mode=args.mode,
        )
        print(f"citation_accuracy: {metrics['citation_accuracy']:.3f}")
        print(f"evidence_hit_rate: {metrics['evidence_hit_rate']:.3f}")
        print(f"refusal_quality: {metrics['refusal_quality']:.3f}")
        print(f"answer_correctness: {metrics['answer_correctness']:.3f}")
        print(f"faithfulness: {metrics['faithfulness']:.3f}")
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    finally:
        if args.force_refusal_threshold:
            if previous_threshold is None:
                os.environ.pop("RAG_MIN_RERANK_SCORE", None)
            else:
                os.environ["RAG_MIN_RERANK_SCORE"] = previous_threshold


def evaluate_answers(
    *,
    input_path: Path,
    candidate_limit: int,
    context_limit: int,
    mode: str = "text",
) -> dict[str, float | int]:
    rows = read_jsonl(input_path)

    # 下面这些变量都是“逐条 query 先记分，最后统一求平均/比例”的中间量。
    #
    # citation_scores:
    #   每条回答的引用合法率。回答里形如 [1]、[2] 的引用编号必须落在
    #   1..len(result.hits) 范围内，表示引用的是本次实际提供给 LLM 的证据。
    # evidence_hits:
    #   对可回答问题，统计检索上下文里是否至少包含 1 个期望文档/片段。
    # refusal_correct:
    #   对不可回答问题，统计模型是否正确拒答，或检索阶段没有返回任何证据。
    # correctness_scores:
    #   每条回答覆盖 expected_answer_terms 的比例，用关键词覆盖近似衡量答案正确性。
    # faithfulness_scores:
    #   每条回答相对证据的忠实度；如果回答包含证据里没有的 unsupported terms，会扣分。
    citation_scores: list[float] = []
    evidence_hits = 0
    refusal_correct = 0
    answerable_count = 0
    unanswerable_count = 0
    correctness_scores: list[float] = []
    faithfulness_scores: list[float] = []

    for row in rows:
        result = run_answer_eval_query(
            row,
            candidate_limit=candidate_limit,
            context_limit=context_limit,
            mode=mode,
        )

        # expected / returned 的粒度由评测集决定：
        # - 有 expected_chunk_ids 时，按 chunk 级别判断证据是否命中。
        # - 否则按 expected_doc_ids 判断文档是否命中。
        # 这里返回的是 set，因此 answer evaluation 只关心“有没有命中”，不关心排序。
        expected, returned = eval_targets(row, result.hits)

        # answerable 表示这条样本是否应该能从知识库回答。
        # 如果评测集显式写了 answerable，就尊重该字段；
        # 否则只要有 expected doc/chunk，就默认它是可回答问题。
        answerable = bool(row.get("answerable", bool(expected)))

        # refusal 的判断是教学版规则：回答中包含固定拒答文案
        # “当前知识库没有足够证据” 就认为模型拒答。
        refused = is_refusal(result.answer)

        # Citation Accuracy:
        #   citation_accuracy = 合法引用编号数 / 回答中出现的引用编号总数。
        # 例子：本次给了 3 条 evidence，回答引用 [1][4]，只有 [1] 合法，得分 1/2。
        # 如果没有任何引用：有 evidence 时得 0；没有 evidence 时得 1。
        citation_scores.append(citation_accuracy(result.answer, len(result.hits)))

        # 后面的 faithfulness 会把所有检索证据拼成一个 evidence_text，
        # 用来判断回答里的某些“高风险词”是否能在证据中找到支撑。
        evidence_text = "\n".join(hit.text for hit in result.hits)

        # Answer Correctness:
        #   correctness = 回答中命中的 expected_answer_terms 数 / expected_answer_terms 总数。
        # 这是一个轻量关键词覆盖指标，不等价于真正的语义正确性；
        # 它适合教学 smoke test，因为可解释、稳定、无需再调用 judge LLM。
        correctness_scores.append(
            term_coverage(result.answer, list(row.get("expected_answer_terms", [])))
        )

        # Faithfulness:
        #   unsupported_term_rate =
        #       “出现在回答中、但没有出现在证据中”的 unsupported terms 数
        #       / unsupported terms 总数
        #   faithfulness = 1 - unsupported_term_rate。
        # 直觉：如果回答说出了证据没有支撑的关键结论，就降低忠实度。
        faithfulness_scores.append(
            faithfulness_score(
                result.answer,
                evidence_text,
                list(row.get("unsupported_answer_terms", [])),
            )
        )

        if answerable:
            # Evidence Hit Rate 的分子：
            # 对可回答问题，只要返回证据集合 returned 和期望集合 expected 有交集，
            # 就说明回答链路至少拿到了一个正确证据。
            answerable_count += 1
            if returned & expected:
                evidence_hits += 1
        else:
            # Refusal Quality 的分子：
            # 对不可回答问题，正确行为是拒答；如果没有返回任何 evidence，
            # 也视为没有强行编造答案的安全结果。
            unanswerable_count += 1
            if refused or not returned:
                refusal_correct += 1

        print(
            f"query={row['query']} answerable={answerable} refused={refused} "
            f"expected={sorted(expected)} returned={sorted(returned)}"
        )

    # 最终指标汇总：
    # - citation_accuracy / answer_correctness / faithfulness 都是逐 query 分数的平均值。
    # - evidence_hit_rate 只在可回答问题上计算，分母是 answerable_count。
    # - refusal_quality 只在不可回答问题上计算，分母是 unanswerable_count。
    #   如果评测集里没有不可回答问题，则默认 refusal_quality=1.0，表示该维度未暴露失败。
    return {
        "mode": mode,
        "query_count": len(rows),
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
        "citation_accuracy": avg(citation_scores),
        "evidence_hit_rate": evidence_hits / answerable_count if answerable_count else 0.0,
        "refusal_quality": refusal_correct / unanswerable_count if unanswerable_count else 1.0,
        "answer_correctness": avg(correctness_scores),
        "faithfulness": avg(faithfulness_scores),
    }


def run_answer_eval_query(
    row: dict,
    *,
    candidate_limit: int,
    context_limit: int,
    mode: str,
):
    if mode == "multimodal":
        return answer_multimodal_query(
            row["query"],
            tenant_id=row["tenant_id"],
            candidate_limit=candidate_limit,
            context_limit=context_limit,
            acl_groups=row.get("acl_groups") or None,
            doc_version=row.get("doc_version"),
            source_types=row.get("source_types") or None,
            history=row.get("history") or None,
        )
    return answer_query(
        row["query"],
        tenant_id=row["tenant_id"],
        candidate_limit=candidate_limit,
        context_limit=context_limit,
        acl_groups=row.get("acl_groups") or None,
        doc_version=row.get("doc_version"),
        source_types=row.get("source_types") or None,
        history=row.get("history") or None,
    )


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def eval_targets(row: dict, hits) -> tuple[set[str], set[str]]:
    expected_chunk_ids = set(row.get("expected_chunk_ids", []))
    if expected_chunk_ids:
        return expected_chunk_ids, {hit_eval_chunk_id(hit, expected_chunk_ids) for hit in hits}
    return set(row.get("expected_doc_ids", [])), {hit.doc_id for hit in hits}


def hit_eval_chunk_id(hit, expected: set[str]) -> str:
    if hit.id in expected:
        return hit.id
    metadata_chunk_id = str((hit.metadata or {}).get("chunk_id", ""))
    if metadata_chunk_id in expected:
        return metadata_chunk_id
    return f"{hit.doc_id}:{hit.chunk_index}"


if __name__ == "__main__":
    main()
