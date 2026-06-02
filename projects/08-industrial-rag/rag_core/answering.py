from __future__ import annotations

from rag_core.config import RagConfig
from rag_core.types import SearchHit


def build_prompt(query: str, hits: list[SearchHit]) -> str:
    evidence = []
    for index, hit in enumerate(hits, start=1):
        evidence.append(
            "\n".join(
                [
                    f"[{index}] doc_id={hit.doc_id}, title={hit.title}, "
                    f"source_uri={hit.source_uri}, chunk_index={hit.chunk_index}",
                    hit.text,
                ]
            )
        )
    evidence_text = "\n\n".join(evidence) if evidence else "无"
    return f"""问题:
{query}

证据:
{evidence_text}

要求:
- 只使用证据回答。
- 每个关键结论后标注引用编号。
- 如果证据不足，回答“当前知识库没有足够证据”。
"""


def generate_answer(config: RagConfig, query: str, hits: list[SearchHit]) -> str:
    prompt = build_prompt(query, hits)
    if not config.llm_base_url or not config.llm_api_key:
        citations = " ".join(f"[{index}]" for index, _ in enumerate(hits, start=1))
        if not hits:
            return "当前知识库没有足够证据。"
        return (
            "未配置 OPENAI_BASE_URL/OPENAI_API_KEY，返回检索证据摘要：\n"
            f"{hits[0].text[:500]}\n\n引用：{citations}"
        )

    from openai import OpenAI

    client = OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)
    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {
                "role": "system",
                "content": "你是企业知识库问答助手。只根据给定证据回答。",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""

