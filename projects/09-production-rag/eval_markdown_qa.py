from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from rag_core.config import load_config
from rag_core.model_api_retry import call_model_api_with_retries


HEADING_PATTERN = re.compile(r"^###\s+(\d+)\.\s*(.+?)\s*$")


@dataclass(frozen=True)
class MarkdownQaCase:
    number: int
    question: str
    reference_answer: str


@dataclass(frozen=True)
class QaEvaluation:
    number: int
    question: str
    answer: str
    citation_count: int
    latency_ms: float
    answer_correctness: float
    groundedness: float
    retrieval_sufficiency: float
    issue: str
    evaluation_error: str = ""


def main() -> None:
    args = parse_args()
    cases = parse_markdown_qa(args.input)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit(f"No Markdown QA cases found in {args.input}")

    source = resolve_source(args)
    doc_ids = list(source.get("child_doc_ids") or []) or [str(source["doc_id"])]
    config = load_config()
    judge_client = build_judge_client(config)
    evaluations = load_previous_evaluations(args.output) if args.resume else []
    completed_numbers = {
        item.number for item in evaluations if not item.evaluation_error
    }
    evaluations = [item for item in evaluations if not item.evaluation_error]
    pending_cases = [case for case in cases if case.number not in completed_numbers]
    with ThreadPoolExecutor(
        max_workers=min(args.concurrency, max(1, len(pending_cases))),
        thread_name_prefix="markdown-qa-eval",
    ) as executor:
        futures = {
            executor.submit(
                evaluate_case,
                args,
                case,
                doc_ids,
                judge_client,
                config.llm_model,
            ): case
            for case in pending_cases
        }
        for future in as_completed(futures):
            case = futures[future]
            try:
                evaluation = future.result()
            except Exception as exc:  # noqa: BLE001 - preserve partial baseline progress.
                evaluation = QaEvaluation(
                    number=case.number,
                    question=case.question,
                    answer="",
                    citation_count=0,
                    latency_ms=0.0,
                    answer_correctness=0.0,
                    groundedness=0.0,
                    retrieval_sufficiency=0.0,
                    issue="",
                    evaluation_error=str(exc)[:1000],
                )
            evaluations.append(evaluation)
            if evaluation.evaluation_error:
                print(f"case={evaluation.number} error={evaluation.evaluation_error}")
            else:
                print(
                    f"case={evaluation.number} citations={evaluation.citation_count} "
                    f"correctness={evaluation.answer_correctness:.2f} "
                    f"groundedness={evaluation.groundedness:.2f} "
                    f"retrieval={evaluation.retrieval_sufficiency:.2f}"
                )
            evaluations.sort(key=lambda item: item.number)
            write_payload(args, source=source, evaluations=evaluations)
    evaluations.sort(key=lambda item: item.number)

    payload = evaluation_payload(args, source=source, evaluations=evaluations)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    write_payload(args, source=source, evaluations=evaluations)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a real RAG API against numbered Markdown question/reference-answer pairs."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--token", required=True)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--acl-group", action="append", default=["engineering"])
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--candidate-limit", type=positive_int, default=20)
    parser.add_argument("--context-limit", type=positive_int, default=5)
    parser.add_argument("--concurrency", type=positive_int, default=1)
    parser.add_argument("--timeout", type=positive_float, default=180.0)
    parser.add_argument("--limit", type=non_negative_int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse successful cases from an existing --output and retry missing/failed cases.",
    )
    return parser.parse_args()


def evaluate_case(
    args: argparse.Namespace,
    case: MarkdownQaCase,
    doc_ids: list[str],
    judge_client,
    model: str,
) -> QaEvaluation:
    started = time.perf_counter()
    response = call_model_api_with_retries(
        "markdown_qa_query",
        lambda: ask_rag(args, case.question, doc_ids=doc_ids),
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    citations = list(response.get("citations") or [])
    answer = str(response.get("answer") or "")
    judgment = judge_answer(
        judge_client,
        model=model,
        case=case,
        answer=answer,
        citations=citations,
    )
    return QaEvaluation(
        number=case.number,
        question=case.question,
        answer=answer,
        citation_count=len(citations),
        latency_ms=latency_ms,
        answer_correctness=judgment["answer_correctness"],
        groundedness=judgment["groundedness"],
        retrieval_sufficiency=judgment["retrieval_sufficiency"],
        issue=judgment["issue"],
    )


def parse_markdown_qa(path: Path) -> list[MarkdownQaCase]:
    cases: list[MarkdownQaCase] = []
    current_number: int | None = None
    question_lines: list[str] = []
    answer_lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = HEADING_PATTERN.match(raw_line)
        if match:
            append_markdown_case(cases, current_number, question_lines, answer_lines)
            current_number = int(match.group(1))
            question_lines = [match.group(2)]
            answer_lines = []
            continue
        if current_number is not None:
            answer_lines.append(raw_line)
    append_markdown_case(cases, current_number, question_lines, answer_lines)
    return cases


def append_markdown_case(
    cases: list[MarkdownQaCase],
    number: int | None,
    question_lines: list[str],
    answer_lines: list[str],
) -> None:
    if number is None:
        return
    question = clean_markdown(" ".join(question_lines))
    reference = clean_markdown("\n".join(answer_lines))
    if question and reference:
        cases.append(
            MarkdownQaCase(
                number=number,
                question=question,
                reference_answer=reference,
            )
        )


def clean_markdown(value: str) -> str:
    value = re.sub(r"\*\*(.+?)\*\*", r"\1", value)
    value = re.sub(r"^\s*[*+-]\s+", "", value, flags=re.MULTILINE)
    value = re.sub(r"^\s*\d+\.\s+", "", value, flags=re.MULTILINE)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def resolve_source(args: argparse.Namespace) -> dict[str, Any]:
    query = urlencode({"tenant_id": args.tenant_id})
    response = request_json(
        args,
        method="GET",
        path=f"sources?{query}",
    )
    matches = [
        source
        for source in response.get("sources") or []
        if str(source.get("doc_id")) == args.doc_id and source.get("status") == "ready"
    ]
    if not matches:
        raise SystemExit(f"Ready source not found: {args.doc_id}")
    return max(matches, key=lambda source: int(source.get("doc_version") or 0))


def ask_rag(args: argparse.Namespace, question: str, *, doc_ids: list[str]) -> dict[str, Any]:
    return request_json(
        args,
        method="POST",
        path="query",
        payload={
            "query": question,
            "tenant_id": args.tenant_id,
            "acl_groups": args.acl_group,
            "doc_ids": doc_ids,
            "candidate_limit": args.candidate_limit,
            "context_limit": args.context_limit,
        },
    )


def request_json(
    args: argparse.Namespace,
    *,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {args.token}",
        "X-RAG-Tenant-ID": args.tenant_id,
        "X-RAG-ACL-Groups": ",".join(args.acl_group),
    }
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        urljoin(args.base_url.rstrip("/") + "/", path),
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=args.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code} for {path}: {detail}") from exc
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Request failed for {path}: {exc}") from exc


def build_judge_client(config):
    if not config.llm_base_url or not config.llm_api_key:
        raise SystemExit("The configured NEW_API_URL/NEW_API_KEY is required for QA judging.")
    from openai import OpenAI

    return OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)


def judge_answer(
    client,
    *,
    model: str,
    case: MarkdownQaCase,
    answer: str,
    citations: list[dict[str, Any]],
) -> dict[str, float | str]:
    evidence = "\n\n".join(
        str(item.get("text") or item.get("text_preview") or "")
        for item in citations
    )[:16_000]
    prompt = f"""You are evaluating one RAG answer.
Return only a JSON object with numeric scores from 0 to 1:
- answer_correctness: factual agreement with the reference answer.
- groundedness: claims in the answer are supported by the retrieved evidence.
- retrieval_sufficiency: the retrieved evidence is sufficient to answer the question.
- issue: one short Chinese description of the main weakness, or an empty string.

Question:
{case.question}

Reference answer:
{case.reference_answer}

Retrieved evidence:
{evidence}

RAG answer:
{answer}
"""
    response = call_model_api_with_retries(
        "markdown_qa_judge",
        lambda: client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2048,
            response_format={"type": "json_object"},
        ),
    )
    message = response.choices[0].message
    content = message.content or getattr(message, "reasoning_content", None) or ""
    parsed = parse_json_object(content)
    return {
        "answer_correctness": bounded_score(parsed.get("answer_correctness")),
        "groundedness": bounded_score(parsed.get("groundedness")),
        "retrieval_sufficiency": bounded_score(parsed.get("retrieval_sufficiency")),
        "issue": str(parsed.get("issue") or "")[:500],
    }


def parse_json_object(value: str) -> dict[str, Any]:
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Judge did not return JSON: {value[:500]}")
    parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Judge JSON must be an object")
    return parsed


def bounded_score(value: Any) -> float:
    return round(max(0.0, min(1.0, float(value))), 4)


def summarize_evaluations(evaluations: list[QaEvaluation]) -> dict[str, Any]:
    successful = [item for item in evaluations if not item.evaluation_error]
    return {
        "evaluated_count": len(successful),
        "error_count": len(evaluations) - len(successful),
        "answer_correctness": mean(item.answer_correctness for item in successful),
        "groundedness": mean(item.groundedness for item in successful),
        "retrieval_sufficiency": mean(item.retrieval_sufficiency for item in successful),
        "citation_count": {
            "avg": mean(float(item.citation_count) for item in successful),
            "zero_count": sum(1 for item in successful if item.citation_count == 0),
        },
        "latency_ms": {
            "avg": mean(item.latency_ms for item in successful),
            "p95": percentile([item.latency_ms for item in successful], 0.95),
        },
        "weak_cases": [
            {
                "number": item.number,
                "answer_correctness": item.answer_correctness,
                "groundedness": item.groundedness,
                "retrieval_sufficiency": item.retrieval_sufficiency,
                "issue": item.issue,
            }
            for item in successful
            if min(item.answer_correctness, item.groundedness, item.retrieval_sufficiency) < 0.8
        ],
        "evaluation_errors": [
            {"number": item.number, "error": item.evaluation_error}
            for item in evaluations
            if item.evaluation_error
        ],
    }


def load_previous_evaluations(path: Path | None) -> list[QaEvaluation]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [QaEvaluation(**item) for item in payload.get("cases") or []]


def evaluation_payload(
    args: argparse.Namespace,
    *,
    source: dict[str, Any],
    evaluations: list[QaEvaluation],
) -> dict[str, Any]:
    return {
        "input": str(args.input),
        "source_doc_id": source["doc_id"],
        "query_count": len(evaluations),
        "metrics": summarize_evaluations(evaluations),
        "cases": [asdict(item) for item in evaluations],
    }


def write_payload(
    args: argparse.Namespace,
    *,
    source: dict[str, Any],
    evaluations: list[QaEvaluation],
) -> None:
    if not args.output:
        return
    payload = evaluation_payload(args, source=source, evaluations=evaluations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def mean(values) -> float:
    values = list(values)
    return round(statistics.fmean(values), 4) if values else 0.0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return round(ordered[index], 2)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


if __name__ == "__main__":
    main()
