from __future__ import annotations

import os
import sys
from dataclasses import replace
from types import SimpleNamespace

from rag_core.config import load_config
from rag_core.rerankers import build_reranker
from rag_core.rewrite import llm_rewrite


class FakeCompletions:
    last_request: dict[str, object] | None = None

    def create(self, **kwargs: object) -> SimpleNamespace:
        FakeCompletions.last_request = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="改写后的独立检索 query"),
                )
            ]
        )


class FakeOpenAI:
    def __init__(self, *, base_url: str, api_key: str) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions())


def main() -> None:
    old_env = {
        name: os.environ.get(name)
        for name in [
            "NEW_API_URL",
            "NEW_API_KEY",
            "RAG_QUERY_REWRITE_HISTORY_TURNS",
            "RAG_QUERY_REWRITE_MAX_TOKENS",
        ]
    }
    old_openai = sys.modules.get("openai")
    try:
        os.environ["NEW_API_URL"] = "https://llm.example/v1"
        os.environ["NEW_API_KEY"] = "test-key"
        os.environ["RAG_QUERY_REWRITE_HISTORY_TURNS"] = "2"
        os.environ["RAG_QUERY_REWRITE_MAX_TOKENS"] = "384"
        sys.modules["openai"] = SimpleNamespace(OpenAI=FakeOpenAI)

        config = load_config()
        rewritten = llm_rewrite(
            "现在怎么处理？",
            ["第一轮", "第二轮", "第三轮"],
            config,
        )
        assert rewritten == "改写后的独立检索 query"
        request = FakeCompletions.last_request
        assert request is not None
        assert request["max_tokens"] == 384
        user_message = request["messages"][1]["content"]  # type: ignore[index]
        assert "第一轮" not in user_message
        assert "第二轮" in user_message
        assert "第三轮" in user_message

        try:
            build_reranker(replace(config, rerank_backend="lexical"))
        except ValueError as exc:
            assert "use none/siliconflow/bge" in str(exc)
        else:
            raise AssertionError("lexical rerank backend should be rejected")
    finally:
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if old_openai is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = old_openai

    print("smoke_rewrite_rerank_config=ok")


if __name__ == "__main__":
    main()
