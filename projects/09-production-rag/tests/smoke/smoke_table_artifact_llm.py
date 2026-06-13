from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from rag_core.artifacts import create_table_artifact, load_artifact
from rag_core.config import load_config
from rag_core.object_store import archive_source_documents
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
        assert "JSON schema" in user_prompt
        assert "岗位职责" in user_prompt
        self.calls.append({"messages": messages, "temperature": temperature})
        content = {
            "title": "实习岗位数据表格",
            "columns": ["岗位", "职责", "要求"],
            "rows": [
                ["大模型应用开发实习生", "开发 RAG 与智能体应用", "熟悉 Python 和 LLM"],
                ["前端工程实习生", "构建知识库交互界面", "熟悉 TypeScript"],
            ],
            "summary": "该表格用于比较实习岗位的职责和要求。",
        }
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(content, ensure_ascii=False)))]
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
        archive_source_documents(
            config.object_store_dir,
            [
                SourceDocument(
                    tenant_id="team_a",
                    doc_id="internship-guide/page-1",
                    doc_version=1,
                    source_type="pdf",
                    source_uri="/tmp/internship-guide.pdf",
                    title="创维 AI 研究院实习介绍资料",
                    text="岗位职责：开发 RAG 与智能体应用。要求：熟悉 Python、TypeScript 和 LLM。",
                    acl_groups=["engineering"],
                    metadata={"relative_path": "创维 AI 研究院实习介绍资料.pdf", "page_no": 1},
                )
            ],
        )

        old_openai = sys.modules.get("openai")
        FakeOpenAI.calls = []
        sys.modules["openai"] = SimpleNamespace(OpenAI=FakeOpenAI)
        try:
            artifact = create_table_artifact(
                config,
                title="实习岗位数据表格",
                tenant_id="team_a",
                source_doc_ids=["internship-guide/page-1"],
                acl_groups=["engineering"],
            )
        finally:
            if old_openai is None:
                sys.modules.pop("openai", None)
            else:
                sys.modules["openai"] = old_openai

        loaded = load_artifact(config, tenant_id="team_a", artifact_id=artifact.id)

    assert len(FakeOpenAI.calls) == 1
    assert loaded is not None
    assert loaded.artifact_type == "table"
    assert loaded.table is not None
    assert loaded.table["columns"] == ["岗位", "职责", "要求"]
    assert loaded.table["rows"][0][0] == "大模型应用开发实习生"
    print("table artifact llm smoke passed")


if __name__ == "__main__":
    main()
