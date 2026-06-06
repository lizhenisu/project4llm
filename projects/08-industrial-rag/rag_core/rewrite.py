from __future__ import annotations

from rag_core.config import RagConfig
from rag_core.text_utils import normalize_text
from rag_core.types import RewriteResult


def rewrite_query(
    query: str,
    *,
    history: list[str] | None,
    config: RagConfig,
) -> RewriteResult:
    backend = config.query_rewrite_backend
    original = normalize_text(query)
    if backend == "none":
        return RewriteResult(original, original, backend)
    if backend == "llm":
        return RewriteResult(original, llm_rewrite(original, history or [], config), backend)
    raise ValueError(
        f"Unsupported RAG_QUERY_REWRITE_BACKEND={backend!r}; use none/llm"
    )


def llm_rewrite(query: str, history: list[str], config: RagConfig) -> str:
    if not config.llm_base_url or not config.llm_api_key:
        raise RuntimeError("NEW_API_URL/NEW_API_KEY must be configured for llm query rewrite.")

    from openai import OpenAI

    history_text = "\n".join(history[-6:])
    client = OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)
    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是 RAG 查询改写器。只输出一个适合检索的中文查询，"
                    "不要添加用户没有表达的权限、租户或事实。"
                ),
            },
            {
                "role": "user",
                "content": f"对话历史:\n{history_text}\n\n当前问题:\n{query}",
            },
        ],
        max_tokens=128,
    )
    rewritten = response.choices[0].message.content or query
    return normalize_text(rewritten)
