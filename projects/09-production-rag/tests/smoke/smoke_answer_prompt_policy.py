from __future__ import annotations

from datetime import date

from rag_core.prompts import ANSWER_SYSTEM_PROMPT, build_answer_prompt
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
    assert "可靠助手" in ANSWER_SYSTEM_PROMPT
    print("smoke_answer_prompt_policy=ok")


if __name__ == "__main__":
    main()
