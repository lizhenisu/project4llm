from __future__ import annotations

from rag_core.citations import citation_accuracy, extract_citation_numbers, is_refusal


def main() -> None:
    assert extract_citation_numbers("结论来自 [1] 和 [2]。") == [1, 2]
    assert citation_accuracy("结论来自 [1] 和 [3]。", evidence_count=2) == 0.5
    assert citation_accuracy("没有引用。", evidence_count=2) == 0.0
    assert is_refusal("当前知识库没有足够证据。")
    print("smoke_answer_eval=ok")


if __name__ == "__main__":
    main()

