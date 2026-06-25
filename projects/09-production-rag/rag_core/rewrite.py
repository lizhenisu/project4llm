from __future__ import annotations

from rag_core.config import RagConfig
from rag_core.model_api_retry import call_model_api_with_retries
from rag_core.prompts import QUERY_REWRITE_SYSTEM_PROMPT, build_query_rewrite_prompt
from rag_core.text_utils import normalize_text
from rag_core.types import RewriteResult


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

    from openai import OpenAI

    history_window = max(0, config.query_rewrite_history_turns)
    selected_history = history[-history_window:] if history_window else []
    history_text = "\n".join(selected_history)
    source_summary_text = "\n".join(source_summaries[:20])
    client = OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)
    response = call_model_api_with_retries(
        "query_rewrite",
        lambda: client.chat.completions.create(
            model=config.llm_model,
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
            max_tokens=max(1, config.query_rewrite_max_tokens),
        ),
    )
    rewritten = response.choices[0].message.content or ""
    if not rewritten.strip():
        raise RuntimeError("LLM query rewrite returned empty content.")
    return normalize_text(rewritten)
