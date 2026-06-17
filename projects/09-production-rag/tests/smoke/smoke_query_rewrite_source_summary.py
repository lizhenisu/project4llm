from __future__ import annotations

import tempfile
from pathlib import Path

from rag_core.prompts import build_query_rewrite_prompt
from rag_core.source_guides import load_source_guides_for_rewrite, save_source_guide


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        object_store_dir = Path(temp_dir)
        save_source_guide(
            object_store_dir,
            tenant_id="tenant-a",
            source_doc_id="source-parent",
            doc_version=2,
            title="自然辩证法",
            guide="这份资料总结自然辩证法的核心概念，适合回答科学技术哲学相关问题。",
            model="smoke",
        )

        summaries = load_source_guides_for_rewrite(
            object_store_dir,
            tenant_id="tenant-a",
            doc_ids=["source-parent"],
            current_doc_versions={"source-parent": 2},
        )
        assert summaries and "自然辩证法" in summaries[0]

        child_fallback = load_source_guides_for_rewrite(
            object_store_dir,
            tenant_id="tenant-a",
            doc_ids=["source-parent/page-1/image-1"],
            current_doc_versions={"source-parent": 2},
        )
        assert child_fallback == summaries

        child_only_current_versions = load_source_guides_for_rewrite(
            object_store_dir,
            tenant_id="tenant-a",
            doc_ids=["source-parent/page-1"],
            current_doc_versions={"source-parent/page-1": 2, "source-parent/page-2": 2},
        )
        assert child_only_current_versions == summaries

        prompt = build_query_rewrite_prompt(
            source_summary_text="\n".join(summaries),
            history_text="用户询问过科学技术哲学。",
            query="它怎么看技术发展？",
        )
        assert prompt.index("资料摘要:") < prompt.index("对话历史:") < prompt.index("当前问题:")
        assert "自然辩证法" in prompt
        assert "科学技术哲学" in prompt

    print("query rewrite source summary smoke passed")


if __name__ == "__main__":
    main()
