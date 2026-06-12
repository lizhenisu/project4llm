from __future__ import annotations

from rag_core.citations import (
    citation_accuracy,
    extract_citation_numbers,
    faithfulness_score,
    is_refusal,
    term_coverage,
    unsupported_term_rate,
)


def main() -> None:
    assert extract_citation_numbers("结论来自 [1] 和 [2]。") == [1, 2]
    assert citation_accuracy("结论来自 [1] 和 [3]。", evidence_count=2) == 0.5
    assert citation_accuracy("没有引用。", evidence_count=2) == 0.0
    assert is_refusal("当前知识库没有足够证据。")
    assert term_coverage("需要订单号和付款凭证。", ["订单号", "付款凭证"]) == 1.0
    assert term_coverage("只提到了订单号。", ["订单号", "付款凭证"]) == 0.5
    assert unsupported_term_rate(
        "答案声称需要检查 Redis。",
        "证据只说 SLA 是 15 分钟。",
        ["Redis"],
    ) == 1.0
    assert faithfulness_score(
        "答案声称 SLA 是 15 分钟。",
        "证据说 SLA 是 15 分钟。",
        ["15 分钟"],
    ) == 1.0
    print("smoke_answer_eval=ok")


if __name__ == "__main__":
    main()
