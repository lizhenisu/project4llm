from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from rag_core.config import RagConfig
from rag_core.model_api_retry import call_model_api_with_retries
from rag_core.prompts import ANSWER_SYSTEM_PROMPT
from rag_core.prompts import build_answer_prompt
from rag_core.prompts import format_evidence_header as prompt_evidence_header
from rag_core.prompts import format_location as prompt_format_location
from rag_core.types import SearchHit


@dataclass(frozen=True)
class AnswerGeneration:
    answer: str
    llm_model: str
    llm_backend: str
    latency_ms: float
    token_usage: dict[str, int]


def build_prompt(query: str, hits: list[SearchHit]) -> str:
    return build_answer_prompt(query, hits)


def generate_answer(config: RagConfig, query: str, hits: list[SearchHit]) -> AnswerGeneration:
    prompt = build_prompt(query, hits)
    start = perf_counter()
    if config.answer_backend != "llm":
        raise ValueError(
            f"Unsupported RAG_ANSWER_BACKEND={config.answer_backend!r}; use llm"
        )
    if not config.llm_base_url or not config.llm_api_key:
        raise RuntimeError("NEW_API_URL/NEW_API_KEY must be configured for answer generation.")

    from openai import OpenAI

    client = OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)
    response = call_model_api_with_retries(
        "answer_generation",
        lambda: client.chat.completions.create(
            model=config.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": ANSWER_SYSTEM_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
        ),
    )
    answer = response.choices[0].message.content or ""
    if not answer.strip():
        raise RuntimeError("LLM answer generation returned empty content.")
    return AnswerGeneration(
        answer=answer,
        llm_model=config.llm_model,
        llm_backend="newapi",
        latency_ms=elapsed_ms(start),
        token_usage=extract_usage(response),
    )


def generate_chat(config: RagConfig, messages: list[dict[str, str]]) -> AnswerGeneration:
    """Pure chat mode — no system prompt, just pass conversation history as-is to LLM."""
    start = perf_counter()
    if config.answer_backend != "llm":
        raise ValueError(
            f"Unsupported RAG_ANSWER_BACKEND={config.answer_backend!r}; use llm"
        )
    if not config.llm_base_url or not config.llm_api_key:
        raise RuntimeError("NEW_API_URL/NEW_API_KEY must be configured for chat generation.")

    from openai import OpenAI

    client = OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)
    response = call_model_api_with_retries(
        "chat_generation",
        lambda: client.chat.completions.create(
            model=config.llm_model,
            messages=messages,
        ),
    )
    answer = response.choices[0].message.content or ""
    if not answer.strip():
        raise RuntimeError("LLM chat generation returned empty content.")
    return AnswerGeneration(
        answer=answer,
        llm_model=config.llm_model,
        llm_backend="newapi",
        latency_ms=elapsed_ms(start),
        token_usage=extract_usage(response),
    )


def elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 2)


def format_location(metadata: dict) -> str:
    return prompt_format_location(metadata)


def format_evidence_header(index: int, hit: SearchHit) -> str:
    return prompt_evidence_header(index, hit)


def extract_usage(response) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if not usage:
        return {}
    return {
        key: int(value)
        for key, value in {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }.items()
        if value is not None
    }
