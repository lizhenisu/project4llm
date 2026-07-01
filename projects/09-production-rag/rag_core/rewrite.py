from __future__ import annotations

import json
import re
import threading
import time

from rag_core.config import RagConfig
from rag_core.model_api_retry import (
    call_model_api_with_retries,
    chat_completion_with_fallback,
    is_transient_model_api_error,
)
from rag_core.prompts import QUERY_REWRITE_SYSTEM_PROMPT, build_query_rewrite_prompt
from rag_core.text_utils import normalize_text
from rag_core.types import RewriteResult


_CROSS_LINGUAL_LOCK = threading.Lock()
_CROSS_LINGUAL_CACHE: dict[tuple[str, str, str], str] = {}


def rewrite_query(
    query: str,
    *,
    history: list[str] | None,
    source_summaries: list[str] | None = None,
    config: RagConfig,
) -> RewriteResult:
    backend = config.query_rewrite_backend
    original = normalize_text(query)
    if backend == "llm":
        try:
            rewritten = llm_rewrite(original, history or [], source_summaries or [], config)
        except RuntimeError:
            rewritten = original
        return RewriteResult(original, rewritten, backend)
    raise ValueError(
        f"Unsupported RAG_QUERY_REWRITE_BACKEND={backend!r}; use llm"
    )


def llm_rewrite(query: str, history: list[str], source_summaries: list[str], config: RagConfig) -> str:
    if not config.llm_base_url or not config.llm_api_key:
        raise RuntimeError("NEW_API_URL/NEW_API_KEY must be configured for llm query rewrite.")

    history_window = max(0, config.query_rewrite_history_turns)
    selected_history = history[-history_window:] if history_window else []
    history_text = "\n".join(selected_history)
    source_summary_text = "\n".join(source_summaries[:20])
    completion = chat_completion_with_fallback(
        config=config,
        operation="query_rewrite",
        messages=[
            {
                "role": "system",
                "content": QUERY_REWRITE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_query_rewrite_prompt(
                    source_summary_text=source_summary_text,
                    history_text=history_text,
                    query=query,
                ),
            },
        ],
        max_tokens=max(1024, config.query_rewrite_max_tokens),
        response_format={"type": "json_object"},
    )
    response = completion.response
    message = response.choices[0].message
    rewritten = message.content or getattr(message, "reasoning_content", None) or ""
    if not rewritten.strip():
        raise RuntimeError("LLM query rewrite returned empty content.")
    normalized = parse_rewrite_response(rewritten)
    if needs_cross_lingual_expansion(query, normalized):
        try:
            expansion = llm_cross_lingual_expansion(
                completion.client,
                query=query,
                config=config,
                model=completion.model,
                backend=completion.backend,
            )
        except Exception:
            expansion = ""
        if expansion and latin_terms(expansion) - latin_terms(normalized):
            normalized = normalize_text(f"{normalized} {expansion}")
    return normalized


def parse_rewrite_response(value: str) -> str:
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end >= start:
        try:
            payload = json.loads(value[start : end + 1])
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            query = str(payload.get("query") or "").strip()
            english_keywords = str(payload.get("english_keywords") or "").strip()
            combined = " ".join(item for item in (query, english_keywords) if item)
            if combined:
                return normalize_text(combined)
    return normalize_text(value)


def needs_cross_lingual_expansion(original: str, rewritten: str) -> bool:
    if not re.search(r"[\u3400-\u9fff]", original):
        return False
    return not (latin_terms(rewritten) - latin_terms(original))


def latin_terms(value: str) -> set[str]:
    return {
        term.lower()
        for term in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{2,}", value)
    }


def llm_cross_lingual_expansion(
    client,
    *,
    query: str,
    config: RagConfig,
    model: str | None = None,
    backend: str = "newapi",
) -> str:
    selected_model = model or config.llm_model
    cache_key = (str(config.llm_base_url or ""), selected_model, query)
    with _CROSS_LINGUAL_LOCK:
        cached = _CROSS_LINGUAL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        for attempt in range(1, 4):
            try:
                response = call_model_api_with_retries(
                    "query_rewrite_cross_lingual",
                    lambda: client.chat.completions.create(
                        model=selected_model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Translate the Chinese technical search query into concise English "
                                    "retrieval keywords. Return only English keywords and preserve formulas, "
                                    "numbers, acronyms, and proper nouns. Do not answer the query."
                                ),
                            },
                            {"role": "user", "content": query},
                        ],
                        max_tokens=1024,
                    ),
                    usage_provider=backend,
                    usage_model=selected_model,
                )
                message = response.choices[0].message
                content = message.content or getattr(message, "reasoning_content", None) or ""
                expansion = normalize_text(content)
                if expansion:
                    if len(_CROSS_LINGUAL_CACHE) >= 512:
                        _CROSS_LINGUAL_CACHE.pop(next(iter(_CROSS_LINGUAL_CACHE)))
                    _CROSS_LINGUAL_CACHE[cache_key] = expansion
                return expansion
            except Exception as exc:
                if attempt >= 3 or not is_transient_model_api_error(exc):
                    raise
                time.sleep(float(2 ** (attempt - 1)))
    return ""
