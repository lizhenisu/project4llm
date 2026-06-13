from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from rag_core.config import RagConfig
from rag_core.types import SearchHit


@dataclass(frozen=True)
class AnswerGeneration:
    answer: str
    llm_model: str
    llm_backend: str
    latency_ms: float
    token_usage: dict[str, int]


def build_prompt(query: str, hits: list[SearchHit]) -> str:
    evidence = []
    for index, hit in enumerate(hits, start=1):
        evidence.append(
            "\n".join(
                [
                    format_evidence_header(index, hit),
                    hit.text,
                ]
            )
        )
    evidence_text = "\n\n".join(evidence) if evidence else "无"
    image_rule = (
        "\n- 图片证据来自 OCR/caption 或图片向量召回，可能不完整；回答时必须把它当作图片派生证据。"
        if any(hit.source_type == "image" for hit in hits)
        else ""
    )
    return f"""问题:
{query}

证据:
{evidence_text}

要求:
- 只使用证据回答。
- 每个关键结论后标注引用编号。
- 如果证据不足，回答“当前知识库没有足够证据”。
{image_rule}
"""


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


def elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 2)


def format_location(metadata: dict) -> str:
    if not metadata:
        return ""
    if "page_start" in metadata and "page_end" in metadata:
        start = metadata["page_start"]
        end = metadata["page_end"]
        return f", page={start}" if start == end else f", pages={start}-{end}"
    if "page_no" in metadata:
        return f", page={metadata['page_no']}"
    if "row_start" in metadata and "row_end" in metadata:
        return f", rows={metadata['row_start']}-{metadata['row_end']}"
    if "bbox" in metadata and metadata["bbox"]:
        return f", bbox={metadata['bbox']}"
    return ""


def format_evidence_header(index: int, hit: SearchHit) -> str:
    parts = [
        f"[{index}] doc_id={hit.doc_id}",
        f"title={hit.title}",
        f"source_type={hit.source_type}",
        f"source_uri={hit.source_uri}",
        f"chunk_index={hit.chunk_index}",
    ]
    location = format_location(hit.metadata)
    if location:
        parts.append(location.lstrip(", "))
    if hit.source_type == "image":
        image_uri = hit.metadata.get("image_uri") or hit.source_uri
        parts.append(f"image_uri={image_uri}")
        if hit.metadata.get("linked_doc_id"):
            parts.append(f"linked_doc_id={hit.metadata['linked_doc_id']}")
    return ", ".join(parts)


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
