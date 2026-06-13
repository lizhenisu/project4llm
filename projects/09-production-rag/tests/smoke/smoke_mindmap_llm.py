from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from rag_core.artifacts import build_mindmap_root
from rag_core.config import RagConfig
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
        self.calls.append({"messages": messages, "temperature": temperature})
        if "原文块:" in user_prompt:
            content = {
                "label": "局部主题",
                "children": [
                    {
                        "label": "研究院概况",
                        "children": [
                            {"label": "定位：集团技术中枢与AI中台", "children": []},
                            {"label": "使命：打造通用AI能力基座", "children": []},
                        ],
                    }
                ],
            }
        else:
            content = {
                "label": "实习招聘思维导图",
                "children": [
                    {
                        "label": "研究院概况",
                        "children": [
                            {"label": "定位：集团技术中枢与AI中台", "children": []},
                            {"label": "使命：打造通用AI能力基座", "children": []},
                            {"label": "工作模式：自由探索、平台输出", "children": []},
                        ],
                    },
                    {
                        "label": "核心研究方向",
                        "children": [
                            {"label": "大语言模型工程化", "children": []},
                            {"label": "智能体操作系统", "children": []},
                        ],
                    },
                ],
            }
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(content, ensure_ascii=False)))]
        )


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        config = make_config(Path(temp_dir))
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
                        "创维集团AI研究院实习介绍资料\n"
                        "一、研究院概况\n"
                        "定位：集团技术中枢与AI中台。\n"
                        "使命：打造通用AI能力基座。\n"
                        "工作模式：自由探索、深度攻坚、平台输出。\n"
                    ),
                    metadata={"relative_path": "创维 AI 研究院实习介绍资料.pdf", "page_no": 1},
                )
            ],
        )

        old_openai = sys.modules.get("openai")
        FakeOpenAI.calls = []
        sys.modules["openai"] = SimpleNamespace(OpenAI=FakeOpenAI)
        try:
            root = build_mindmap_root(
                title="实习招聘思维导图",
                config=config,
                tenant_id="team_a",
                source_doc_ids=["internship-guide/page-1"],
            )
        finally:
            if old_openai is None:
                sys.modules.pop("openai", None)
            else:
                sys.modules["openai"] = old_openai

    assert len(FakeOpenAI.calls) == 2
    assert root["label"] == "实习招聘思维导图"
    assert [child["label"] for child in root["children"]] == ["研究院概况", "核心研究方向"]
    assert [child["label"] for child in root["children"][0]["children"]] == [
        "定位：集团技术中枢与AI中台",
        "使命：打造通用AI能力基座",
        "工作模式：自由探索、平台输出",
    ]
    print("mindmap llm smoke passed")


def make_config(object_store_dir: Path) -> RagConfig:
    return RagConfig(
        milvus_uri="memory://unused",
        milvus_token=None,
        collection_name="unused",
        embedding_model="BAAI/bge-m3",
        embedding_backend="siliconflow",
        embedding_dim=3,
        embedding_batch_size=2,
        embedding_max_length=8192,
        rerank_model="BAAI/bge-reranker-v2-m3",
        rerank_backend="siliconflow",
        rerank_batch_size=8,
        rerank_max_length=1024,
        image_embedding_backend="clip",
        image_embedding_model="openai/clip-vit-base-patch32",
        image_embedding_dim=512,
        image_embedding_batch_size=8,
        model_device="auto",
        model_dtype="auto",
        llm_base_url="https://llm.example/v1",
        llm_api_key="test-key",
        llm_model="test-llm",
        siliconflow_base_url="https://api.siliconflow.cn",
        siliconflow_api_key="test-key",
        answer_backend="llm",
        chunk_size=700,
        chunk_overlap=100,
        reset_collection=False,
        runtime_dir=object_store_dir / "runtime",
        object_store_dir=object_store_dir,
        pii_policy="warn",
        max_context_chars=6000,
        max_chunks_per_doc=2,
        min_rerank_score=None,
        query_rewrite_backend="llm",
        query_rewrite_history_turns=6,
        query_rewrite_max_tokens=256,
        require_auth_context=False,
        api_token=None,
        dense_hnsw_m=16,
        dense_hnsw_ef_construction=100,
        dense_search_ef=128,
        image_hnsw_m=16,
        image_hnsw_ef_construction=100,
        image_search_ef=128,
        sparse_drop_ratio_build=0.2,
        sparse_drop_ratio_search=0.0,
    )


if __name__ == "__main__":
    main()
