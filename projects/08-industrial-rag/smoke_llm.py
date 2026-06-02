from __future__ import annotations

from rag_core.config import load_config


def main() -> None:
    config = load_config()
    if not config.llm_base_url or not config.llm_api_key:
        print("smoke_llm=skipped; OPENAI_BASE_URL/OPENAI_API_KEY not configured")
        return

    from openai import OpenAI

    client = OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)
    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {"role": "system", "content": "只回答 OK。"},
            {"role": "user", "content": "连通性测试"},
        ],
        max_tokens=16,
    )
    content = response.choices[0].message.content or ""
    print(f"smoke_llm=ok model={config.llm_model} content={content[:80]}")


if __name__ == "__main__":
    main()

