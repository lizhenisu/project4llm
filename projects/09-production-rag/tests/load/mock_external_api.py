from __future__ import annotations

import hashlib
import os
import random
import time
from typing import Any

from fastapi import FastAPI, HTTPException


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def create_app() -> FastAPI:
    app = FastAPI(title="Production RAG mock external API")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    def chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
        maybe_fail("MOCK_LLM_ERROR_RATE")
        sleep_ms(env_int("MOCK_LLM_LATENCY_MS", 10000))
        content = mock_chat_content(payload)
        prompt_tokens = count_prompt_tokens(payload.get("messages", []))
        completion_tokens = max(8, min(env_int("MOCK_LLM_COMPLETION_TOKENS", 128), 2048))
        return {
            "id": "mock-chat-completion",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model") or os.environ.get("MOCK_LLM_MODEL", "mock-llm"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    @app.post("/v1/embeddings")
    @app.post("/embeddings")
    def embeddings(payload: dict[str, Any]) -> dict[str, Any]:
        maybe_fail("MOCK_EMBEDDING_ERROR_RATE")
        sleep_ms(env_int("MOCK_EMBEDDING_LATENCY_MS", 120))
        inputs = normalize_inputs(payload.get("input", []))
        dim = int(payload.get("dimensions") or env_int("MOCK_EMBEDDING_DIM", 1024))
        embedding_mode = os.environ.get("MOCK_EMBEDDING_MODE", "random").lower()
        return {
            "object": "list",
            "model": payload.get("model") or "mock-embedding",
            "data": [
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": mock_embedding(input_item, dim=dim, mode=embedding_mode),
                }
                for index, input_item in enumerate(inputs)
            ],
            "usage": {"prompt_tokens": sum(token_count(str(item)) for item in inputs)},
        }

    @app.post("/v1/rerank")
    @app.post("/rerank")
    def rerank(payload: dict[str, Any]) -> dict[str, Any]:
        maybe_fail("MOCK_RERANK_ERROR_RATE")
        sleep_ms(env_int("MOCK_RERANK_LATENCY_MS", 180))
        documents = list(payload.get("documents") or [])
        top_n = int(payload.get("top_n") or len(documents))
        query = str(payload.get("query") or "")
        rerank_mode = os.environ.get("MOCK_RERANK_MODE", "identity").lower()
        scored = mock_rerank_results(query, documents, mode=rerank_mode)
        return {
            "id": "mock-rerank",
            "results": scored[:top_n],
            "usage": {"total_tokens": sum(token_count(str(document)) for document in documents)},
        }

    return app


def sleep_ms(milliseconds: int) -> None:
    if milliseconds > 0:
        time.sleep(milliseconds / 1000)


def maybe_fail(env_name: str) -> None:
    error_rate = env_float(env_name, 0.0)
    if error_rate > 0 and random.random() < error_rate:
        raise HTTPException(status_code=429, detail=f"mocked failure from {env_name}")


def normalize_inputs(raw_input: Any) -> list[Any]:
    if isinstance(raw_input, list):
        return raw_input
    return [raw_input]


def mock_chat_content(payload: dict[str, Any]) -> str:
    mode = os.environ.get("MOCK_LLM_MODE", "echo").lower()
    messages = payload.get("messages") or []
    user_text = ""
    if messages:
        user_text = str(messages[-1].get("content") or "")
    if mode == "echo":
        return extract_original_query(user_text) or user_text
    max_tokens = int(payload.get("max_tokens") or env_int("MOCK_LLM_COMPLETION_TOKENS", 128))
    if max_tokens <= 64:
        return user_text[:160] or "mock rewritten query"
    return (
        "这是 mock 外部大模型返回的压测回答。"
        "它用于稳定测量 Production RAG 的并发、流式事件和检索链路。"
    )


def extract_original_query(prompt: str) -> str:
    for marker, end_marker in (
        ("当前问题:\n", None),
        ("问题:\n", "\n\n当前系统日期:"),
    ):
        start = prompt.rfind(marker)
        if start < 0:
            continue
        value = prompt[start + len(marker) :]
        if end_marker and end_marker in value:
            value = value.split(end_marker, 1)[0]
        return value.strip()
    return prompt.strip()


def mock_embedding(value: Any, *, dim: int, mode: str) -> list[float]:
    if mode == "hash":
        return deterministic_vector(value, dim=dim)
    if mode == "zero":
        return [0.0] * dim
    return random_vector(dim=dim)


def random_vector(*, dim: int) -> list[float]:
    return [round(random.uniform(-1.0, 1.0), 6) for _index in range(dim)]


def deterministic_vector(value: Any, *, dim: int) -> list[float]:
    text = str(value)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vector: list[float] = []
    for index in range(dim):
        byte = digest[index % len(digest)]
        vector.append(round((byte / 255.0) * 2 - 1, 6))
    return vector


def mock_rerank_results(query: str, documents: list[Any], *, mode: str) -> list[dict[str, float | int]]:
    if mode == "lexical":
        scored = [
            {
                "index": index,
                "relevance_score": relevance_score(query, str(document), index),
            }
            for index, document in enumerate(documents)
        ]
        scored.sort(key=lambda item: item["relevance_score"], reverse=True)
        return scored
    return [
        {
            "index": index,
            "relevance_score": round(1.0 - (index * 0.000001), 6),
        }
        for index, _document in enumerate(documents)
    ]


def relevance_score(query: str, document: str, index: int) -> float:
    query_terms = set(query.lower().split())
    doc_terms = set(document.lower().split())
    overlap = len(query_terms & doc_terms)
    tie_breaker = 1.0 / (index + 10)
    return round(overlap + tie_breaker, 6)


def token_count(text: str) -> int:
    return max(1, len(text.split()))


def count_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(token_count(str(message.get("content") or "")) for message in messages)


app = create_app()
