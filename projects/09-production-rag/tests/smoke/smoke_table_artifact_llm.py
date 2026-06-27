from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from rag_core.artifacts import create_table_artifact, load_artifact, merge_tables, normalize_table
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
        self.calls.append({"messages": messages, "temperature": temperature})
        if "SECOND_CHUNK_UNIQUE_MARKER" in user_prompt:
            rows = [
                ["未提及", "未提及", "未提及"],
                ["后半部分岗位", "覆盖长文档后半段", "不能丢弃第二块"],
            ]
        elif "岗位职责" in user_prompt:
            rows = [
                ["大模型应用开发实习生", "开发 RAG 与智能体应用", "熟悉 Python 和 LLM"],
                ["文档中未提及。", "暂无", "N/A"],
                ["前端工程实习生", "构建知识库交互界面", "熟悉 TypeScript"],
            ]
        else:
            rows = [
                ["未说明", "未知", "无相关信息"],
                ["补充说明", "覆盖长文档中间部分", "不能跳过中间块"],
            ]
        content = {
            "title": "实习岗位数据表格",
            "columns": ["岗位", "职责", "要求"],
            "rows": rows,
            "summary": "该表格用于比较实习岗位的职责和要求。",
        }
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(content, ensure_ascii=False)))]
        )


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        old_table_chunk_chars = os.environ.get("RAG_TABLE_CHUNK_CHARS")
        old_artifact_workers = os.environ.get("RAG_ARTIFACT_LLM_WORKERS")
        os.environ["RAG_TABLE_CHUNK_CHARS"] = "1000"
        os.environ["RAG_ARTIFACT_LLM_WORKERS"] = "1"
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
                    text=(
                        "岗位职责：开发 RAG 与智能体应用。要求：熟悉 Python、TypeScript 和 LLM。\n"
                        + "补充说明。" * 2200
                        + "\nSECOND_CHUNK_UNIQUE_MARKER：后半部分岗位不能被丢弃。"
                    ),
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
            restore_env("RAG_TABLE_CHUNK_CHARS", old_table_chunk_chars)
            restore_env("RAG_ARTIFACT_LLM_WORKERS", old_artifact_workers)

        loaded = load_artifact(config, tenant_id="team_a", artifact_id=artifact.id)

    assert len(FakeOpenAI.calls) >= 2
    assert any("SECOND_CHUNK_UNIQUE_MARKER" in call["messages"][-1]["content"] for call in FakeOpenAI.calls)
    assert loaded is not None
    assert loaded.artifact_type == "table"
    assert loaded.table is not None
    assert loaded.table["columns"] == ["岗位", "职责", "要求"]
    assert loaded.table["rows"][0][0] == "大模型应用开发实习生"
    assert loaded.table["rows"][-1][0] == "后半部分岗位"
    assert all(any(cell not in {"未提及", "未说明", "暂无"} for cell in row) for row in loaded.table["rows"])
    assert normalize_table(
        {
            "columns": ["甲", "乙"],
            "rows": [["未提及。", "N/A"], ["有效信息", "未提及"]],
        },
        default_title="测试",
    )["rows"] == [["有效信息", "未提及"]]
    assert merge_tables(
        [
            {
                "title": "测试",
                "columns": ["甲", "乙"],
                "rows": [["有效信息", "值"]],
                "summary": "",
            },
            {
                "title": "测试",
                "columns": ["丙", "丁"],
                "rows": [["无法对齐", "仍无法对齐"]],
                "summary": "",
            },
        ],
        default_title="测试",
    )["rows"] == [["有效信息", "值"]]
    print("table artifact llm smoke passed")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
