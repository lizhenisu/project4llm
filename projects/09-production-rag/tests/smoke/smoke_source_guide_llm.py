from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from rag_core.config import load_config
from rag_core.source_guides import load_source_guide
from rag_core.sources import SourceSummary, generate_ingested_source_guides
from rag_core.types import SourceDocument


class FakeOpenAI:
    calls: list[dict] = []

    def __init__(self, *, base_url: str, api_key: str) -> None:
        assert base_url == "https://llm.example/v1"
        assert api_key == "test-key"
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, *, model: str, messages: list[dict], temperature: float):
        assert model == "test-llm"
        assert temperature == 0.2
        user_prompt = messages[-1]["content"]
        assert "不要直接复制原文长句" in user_prompt
        assert "创维集团AI研究院实习介绍资料" in user_prompt
        self.calls.append({"messages": messages, "temperature": temperature})
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="这份资料概述创维 AI 研究院的定位、使命、工作方式和实习招聘信息，适合回答研究院职责、研究方向与岗位要求等问题。"
                    )
                )
            ]
        )


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        config = replace(
            load_config(),
            object_store_dir=Path(temp_dir) / "object_store",
            runtime_dir=Path(temp_dir) / "runtime",
            llm_base_url="https://llm.example/v1",
            llm_api_key="test-key",
            llm_model="test-llm",
        )
        docs = [
            SourceDocument(
                tenant_id="team_a",
                doc_id="internship-guide/page-1",
                doc_version=1,
                source_type="pdf",
                source_uri="/tmp/internship-guide.pdf",
                title="创维 AI 研究院实习介绍资料 p1",
                text="创维集团AI研究院实习介绍资料\n一、研究院概况\n定位：集团技术中枢与AI中台。\n",
                acl_groups=["engineering"],
                metadata={"relative_path": "创维 AI 研究院实习介绍资料.pdf", "page_no": 1},
            )
        ]

        old_openai = sys.modules.get("openai")
        FakeOpenAI.calls = []
        sys.modules["openai"] = SimpleNamespace(OpenAI=FakeOpenAI)
        try:
            generate_ingested_source_guides(
                config=config,
                sources=[
                    SourceSummary(
                        doc_id="internship-guide",
                        title="创维 AI 研究院实习介绍资料.pdf",
                        source_type="pdf",
                        source_uri="/tmp/internship-guide.pdf",
                        doc_version=1,
                        chunk_count=1,
                        acl_groups=["engineering"],
                        status="ready",
                        current=True,
                        child_doc_ids=["internship-guide/page-1"],
                    )
                ],
                docs=docs,
            )
            generate_ingested_source_guides(
                config=config,
                sources=[
                    SourceSummary(
                        doc_id="internship-guide",
                        title="创维 AI 研究院实习介绍资料.pdf",
                        source_type="pdf",
                        source_uri="/tmp/internship-guide.pdf",
                        doc_version=1,
                        chunk_count=1,
                        acl_groups=["engineering"],
                        status="ready",
                        current=True,
                        child_doc_ids=["internship-guide/page-1"],
                    )
                ],
                docs=docs,
            )
            guide = load_source_guide(
                config.object_store_dir,
                tenant_id="team_a",
                source_doc_id="internship-guide",
                doc_version=1,
            )
            cached = load_source_guide(
                config.object_store_dir,
                tenant_id="team_a",
                source_doc_id="internship-guide",
                doc_version=1,
            )
        finally:
            if old_openai is None:
                sys.modules.pop("openai", None)
            else:
                sys.modules["openai"] = old_openai

    assert len(FakeOpenAI.calls) == 1
    assert guide == cached
    assert "适合回答研究院职责" in guide
    assert not guide.startswith("创维集团AI研究院实习介绍资料\n一、研究院概况")
    print("source guide llm smoke passed")


if __name__ == "__main__":
    main()
