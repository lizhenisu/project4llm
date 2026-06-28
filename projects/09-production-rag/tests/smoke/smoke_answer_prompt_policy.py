from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from rag_core.prompts import (
    ANSWER_SYSTEM_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    build_answer_prompt,
    build_query_rewrite_prompt,
)
from rag_core.rewrite import (
    latin_terms,
    llm_cross_lingual_expansion,
    needs_cross_lingual_expansion,
    parse_rewrite_response,
)
from rag_core.types import SearchHit


def main() -> None:
    hit = SearchHit(
        id="job-hit",
        score=0.9,
        text="创维集团AI研究院是深圳大学人工智能学院的实习基地。实习生每周到岗不少于5天。",
        doc_id="skyworth-ai-internship",
        title="创维 AI 研究院实习介绍",
        source_uri="memory://skyworth-ai-internship.pdf",
        source_type="pdf",
        chunk_index=3,
        tenant_id="team_a",
        acl_groups=["students"],
        metadata={"page_start": 2, "page_end": 2},
    )
    prompt = build_answer_prompt(
        "ai院和腾讯的合作关系以及放假措施",
        [hit],
        current_date=date(2026, 6, 14),
    )

    assert "不要机械拒答" in prompt
    assert "来源中未提及/没有具体说明" in prompt
    assert "整理证据中相关的可用信息" in prompt
    assert "通用常识" in prompt
    assert "不是来自知识库证据" in prompt
    assert "当前系统日期:\n2026-06-14" in prompt
    assert "[1] doc_id=skyworth-ai-internship" in prompt
    assert "每周到岗不少于5天" in prompt
    assert "严格依据检索证据" in ANSWER_SYSTEM_PROMPT
    assert "不要用外部知识补齐事实或因果解释" in ANSWER_SYSTEM_PROMPT
    assert "不要为了让解释更完整而补充证据未出现的理论动机" in prompt
    assert "原文优先于批次摘要或综合摘要" in prompt
    assert "不要为了凑足数量而跨章节补入相关设置" in prompt
    assert "不要猜测一个原文未列出的标题" in prompt
    assert "不要把 A/B 合并后留下空缺" in prompt
    assert "中文概念加对应英文关键词" in QUERY_REWRITE_SYSTEM_PROMPT
    assert "不要回答问题" in QUERY_REWRITE_SYSTEM_PROMPT
    assert "english_keywords" in QUERY_REWRITE_SYSTEM_PROMPT
    rewrite_prompt = build_query_rewrite_prompt(
        source_summary_text="English Transformer paper",
        history_text="",
        query="优化器超参数是什么？",
    )
    assert "补充原文中可能出现的英文技术关键词" in rewrite_prompt
    assert "English terms" in rewrite_prompt
    assert needs_cross_lingual_expansion(
        "Transformer 使用什么优化器？",
        "Transformer 使用什么优化器？",
    )
    assert not needs_cross_lingual_expansion(
        "Transformer 使用什么优化器？",
        "Transformer optimizer Adam hyperparameters",
    )
    assert latin_terms("Adam β1 optimizer epsilon") == {"adam", "optimizer", "epsilon"}
    assert parse_rewrite_response(
        '{"query":"优化器超参数","english_keywords":"Adam optimizer beta epsilon"}'
    ) == "优化器超参数 Adam optimizer beta epsilon"
    fake_client = FakeClient()
    fake_config = SimpleNamespace(llm_base_url="http://synthetic.test/v1", llm_model="synthetic")
    first = llm_cross_lingual_expansion(
        fake_client,
        query="合成优化器问题",
        config=fake_config,
    )
    second = llm_cross_lingual_expansion(
        fake_client,
        query="合成优化器问题",
        config=fake_config,
    )
    assert first == second == "synthetic optimizer keywords"
    assert fake_client.calls == 1
    print("smoke_answer_prompt_policy=ok")


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create),
        )

    def create(self, **_kwargs):
        self.calls += 1
        message = SimpleNamespace(content="synthetic optimizer keywords")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


if __name__ == "__main__":
    main()
