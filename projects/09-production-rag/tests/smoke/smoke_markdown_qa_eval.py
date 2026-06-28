from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from eval_markdown_qa import (  # noqa: E402
    MarkdownQaCase,
    QaEvaluation,
    bounded_score,
    judge_answer,
    parse_json_object,
    parse_markdown_qa,
    summarize_evaluations,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic-qa.md"
        path.write_text(
            """Synthetic introduction.

### 1. **What is synthetic alpha?**

* **Answer**: Alpha is a placeholder.

### 2. How does synthetic beta work?

* Beta remains deterministic.
""",
            encoding="utf-8",
        )
        cases = parse_markdown_qa(path)
    assert cases == [
        MarkdownQaCase(1, "What is synthetic alpha?", "Answer: Alpha is a placeholder."),
        MarkdownQaCase(2, "How does synthetic beta work?", "Beta remains deterministic."),
    ]
    assert parse_json_object('prefix {"answer_correctness": 0.9} suffix') == {
        "answer_correctness": 0.9
    }
    assert bounded_score(1.5) == 1.0
    judge_client = FakeJudgeClient([
        "",
        '{"answer_correctness":1,"groundedness":0.9,'
        '"retrieval_sufficiency":0.8,"issue":""}',
    ])
    judgment = judge_answer(
        judge_client,
        model="synthetic",
        case=MarkdownQaCase(3, "q3", "a3"),
        answer="a3",
        citations=[{"text": "a3"}],
    )
    assert judgment == {
        "answer_correctness": 1.0,
        "groundedness": 0.9,
        "retrieval_sufficiency": 0.8,
        "issue": "",
    }
    assert judge_client.calls == 2
    summary = summarize_evaluations(
        [
            QaEvaluation(1, "q1", "a1", 2, 100.0, 1.0, 0.9, 0.8, ""),
            QaEvaluation(2, "q2", "a2", 0, 300.0, 0.5, 0.6, 0.7, "synthetic issue"),
        ]
    )
    assert summary["answer_correctness"] == 0.75
    assert summary["citation_count"] == {"avg": 1.0, "zero_count": 1}
    assert summary["latency_ms"] == {"avg": 200.0, "p95": 300.0}
    assert [item["number"] for item in summary["weak_cases"]] == [2]
    print("smoke_markdown_qa_eval=ok")


class FakeJudgeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls = 0
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create),
        )

    def create(self, **_kwargs):
        self.calls += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=next(self.responses)))]
        )


if __name__ == "__main__":
    main()
