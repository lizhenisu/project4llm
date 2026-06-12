from __future__ import annotations

import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import release_gate
from release_gate import gate_failures


def main() -> None:
    test_gate_failures()
    test_release_gate_main_pass()
    test_release_gate_main_fail()
    print("smoke_release_gate=ok")


def test_gate_failures() -> None:
    retrieval = {
        "recall": 1.0,
        "mrr": 1.0,
        "ndcg": 1.0,
        "permission_leakage_failures": 0,
        "p95_latency_ms": 10.0,
        "stage_p95_latency_ms": {"rerank": 20.0},
    }
    answer = {
        "citation_accuracy": 1.0,
        "evidence_hit_rate": 1.0,
        "refusal_quality": 1.0,
        "answer_correctness": 1.0,
        "faithfulness": 1.0,
    }
    args = Args()
    assert gate_failures(args, retrieval, answer) == []
    multimodal = {
        "recall": 1.0,
        "mrr": 1.0,
        "ndcg": 1.0,
        "p95_latency_ms": 12.0,
    }
    multimodal_answer = {
        "citation_accuracy": 1.0,
        "evidence_hit_rate": 1.0,
        "answer_correctness": 1.0,
        "faithfulness": 1.0,
    }
    assert gate_failures(args, retrieval, answer, multimodal) == []
    assert gate_failures(args, retrieval, answer, multimodal, multimodal_answer) == []
    failing = dict(retrieval, recall=0.5, permission_leakage_failures=1)
    failures = gate_failures(args, failing, answer)
    assert any(item.startswith("recall=") for item in failures)
    assert any(item.startswith("permission_leakage_failures=") for item in failures)
    slow = dict(retrieval, stage_p95_latency_ms={"rerank": 2000.0})
    failures = gate_failures(args, slow, answer)
    assert any(item.startswith("p95_rerank_ms=") for item in failures)
    bad_answer = dict(answer, answer_correctness=0.5, faithfulness=0.5)
    failures = gate_failures(args, retrieval, bad_answer)
    assert any(item.startswith("answer_correctness=") for item in failures)
    assert any(item.startswith("faithfulness=") for item in failures)
    bad_multimodal = dict(multimodal, recall=0.5, p95_latency_ms=2000.0)
    failures = gate_failures(args, retrieval, answer, bad_multimodal)
    assert any(item.startswith("multimodal_recall=") for item in failures)
    assert any(item.startswith("p95_multimodal_ms=") for item in failures)
    bad_multimodal_answer = dict(
        multimodal_answer,
        citation_accuracy=0.5,
        faithfulness=0.5,
    )
    failures = gate_failures(args, retrieval, answer, multimodal, bad_multimodal_answer)
    assert any(item.startswith("multimodal_citation_accuracy=") for item in failures)
    assert any(item.startswith("multimodal_faithfulness=") for item in failures)


def test_release_gate_main_pass() -> None:
    retrieval = {
        "recall": 1.0,
        "mrr": 1.0,
        "ndcg": 1.0,
        "permission_leakage_failures": 0,
        "p95_latency_ms": 10.0,
        "stage_p95_latency_ms": {"rerank": 20.0},
    }
    answer = {
        "citation_accuracy": 1.0,
        "evidence_hit_rate": 1.0,
        "refusal_quality": 1.0,
        "answer_correctness": 1.0,
        "faithfulness": 1.0,
    }
    multimodal_retrieval = {
        "recall": 1.0,
        "mrr": 1.0,
        "ndcg": 1.0,
        "p95_latency_ms": 12.0,
    }
    multimodal_answer = {
        "citation_accuracy": 1.0,
        "evidence_hit_rate": 1.0,
        "answer_correctness": 1.0,
        "faithfulness": 1.0,
    }

    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "eval.jsonl"
        multimodal_input = Path(tmp) / "multimodal_eval.jsonl"
        multimodal_answer_input = Path(tmp) / "multimodal_answer_eval.jsonl"
        json_output = Path(tmp) / "release_gate_report.json"
        for path in [input_path, multimodal_input, multimodal_answer_input]:
            path.write_text("{}\n", encoding="utf-8")

        with (
            patch(
                "release_gate.evaluate_retrieval",
                side_effect=[retrieval, multimodal_retrieval],
            ) as retrieval_mock,
            patch(
                "release_gate.evaluate_answers",
                side_effect=[answer, multimodal_answer],
            ) as answer_mock,
        ):
            output = run_release_gate_main(
                [
                    "release_gate.py",
                    "--input",
                    str(input_path),
                    "--retrieval-mode",
                    "rerank",
                    "--multimodal-input",
                    str(multimodal_input),
                    "--multimodal-answer-input",
                    str(multimodal_answer_input),
                    "--retrieval-limit",
                    "7",
                    "--candidate-limit",
                    "11",
                    "--context-limit",
                    "4",
                    "--json-output",
                    str(json_output),
                ]
            )

        assert retrieval_mock.call_count == 2
        assert retrieval_mock.call_args_list[0].kwargs == {
            "input_path": input_path,
            "limit": 7,
            "mode": "rerank",
        }
        assert retrieval_mock.call_args_list[1].kwargs == {
            "input_path": multimodal_input,
            "limit": 7,
            "mode": "multimodal",
        }
        assert answer_mock.call_count == 2
        assert answer_mock.call_args_list[0].kwargs == {
            "input_path": input_path,
            "candidate_limit": 11,
            "context_limit": 4,
        }
        assert answer_mock.call_args_list[1].kwargs == {
            "input_path": multimodal_answer_input,
            "candidate_limit": 11,
            "context_limit": 4,
            "mode": "multimodal",
        }
        assert '"status": "pass"' in output
        assert "release_gate=ok" in output
        report = json.loads(json_output.read_text(encoding="utf-8"))
        assert report["status"] == "pass"
        assert report["retrieval"] == retrieval
        assert report["answer"] == answer
        assert report["multimodal_retrieval"] == multimodal_retrieval
        assert report["multimodal_answer"] == multimodal_answer


def test_release_gate_main_fail() -> None:
    retrieval = {
        "recall": 0.5,
        "mrr": 1.0,
        "ndcg": 1.0,
        "permission_leakage_failures": 0,
        "p95_latency_ms": 10.0,
        "stage_p95_latency_ms": {"rerank": 20.0},
    }
    answer = {
        "citation_accuracy": 1.0,
        "evidence_hit_rate": 1.0,
        "refusal_quality": 1.0,
        "answer_correctness": 1.0,
        "faithfulness": 1.0,
    }

    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "eval.jsonl"
        json_output = Path(tmp) / "release_gate_report_fail.json"
        input_path.write_text("{}\n", encoding="utf-8")

        with (
            patch("release_gate.evaluate_retrieval", return_value=retrieval),
            patch("release_gate.evaluate_answers", return_value=answer),
        ):
            try:
                run_release_gate_main(
                    [
                        "release_gate.py",
                        "--input",
                        str(input_path),
                        "--retrieval-mode",
                        "rerank",
                        "--json-output",
                        str(json_output),
                    ]
                )
            except SystemExit as exc:
                assert str(exc) == "release_gate=failed"
            else:
                raise AssertionError("release_gate.main() should fail on unmet gates")

        report = json.loads(json_output.read_text(encoding="utf-8"))
        assert report["status"] == "fail"
        assert report["failures"]
        assert any(item.startswith("recall=") for item in report["failures"])


def run_release_gate_main(argv: list[str]) -> str:
    old_argv = sys.argv
    stdout = io.StringIO()
    sys.argv = argv
    try:
        with redirect_stdout(stdout):
            release_gate.main()
    finally:
        sys.argv = old_argv
    return stdout.getvalue()


class Args:
    input = Path("unused")
    min_recall = 1.0
    min_mrr = 1.0
    min_ndcg = 1.0
    max_leakage_failures = 0
    max_p95_retrieval_ms = 800.0
    max_p95_rerank_ms = 1500.0
    min_multimodal_recall = 1.0
    min_multimodal_mrr = 1.0
    min_multimodal_ndcg = 1.0
    max_p95_multimodal_ms = 1000.0
    min_citation_accuracy = 1.0
    min_evidence_hit_rate = 1.0
    min_refusal_quality = 1.0
    min_answer_correctness = 1.0
    min_faithfulness = 1.0
    min_multimodal_citation_accuracy = 1.0
    min_multimodal_evidence_hit_rate = 1.0
    min_multimodal_answer_correctness = 1.0
    min_multimodal_faithfulness = 1.0


if __name__ == "__main__":
    main()
