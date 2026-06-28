from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.config import load_config  # noqa: E402
from rag_core.document_scope import SELECTED_DOC_EXTRACT, build_scope_plan, source_guide_hits  # noqa: E402
from rag_core.section_summaries import (  # noqa: E402
    SOURCE_SECTION_SUMMARIES_PATH,
    delete_source_section_summaries,
    load_source_section_summaries,
)
from rag_core.sources import SourceSummary, generate_ingested_source_guides  # noqa: E402
from rag_core.types import SourceDocument  # noqa: E402


def main() -> None:
    old_fast_path = os.environ.get("RAG_SOURCE_GUIDE_FAST_PATH_CHARS")
    old_summary_chars = os.environ.get("RAG_SECTION_SUMMARY_CHARS")
    os.environ["RAG_SOURCE_GUIDE_FAST_PATH_CHARS"] = "10000"
    os.environ["RAG_SECTION_SUMMARY_CHARS"] = "120"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(
                load_config(),
                object_store_dir=Path(tmp) / "object_store",
                runtime_dir=Path(tmp) / "runtime",
            )
            docs = source_documents()
            generate_ingested_source_guides(
                config=config,
                sources=[source_summary()],
                docs=docs,
            )

            assert (config.object_store_dir / SOURCE_SECTION_SUMMARIES_PATH).exists()
            sections = load_source_section_summaries(
                config.object_store_dir,
                tenant_id="section-tenant",
                source_keys={("section-source", 1)},
            )
            assert len(sections) >= 2
            assert sections[0].title == "第 1 页"
            assert sections[0].summary.startswith("SECTION_HEAD_MARKER")
            assert any("SECTION_TAIL_MARKER" in section.summary for section in sections)
            assert load_source_section_summaries(
                config.object_store_dir,
                tenant_id="other-tenant",
                source_keys={("section-source", 1)},
            ) == []

            extraction_plan = build_scope_plan(
                config=config,
                tenant_id="section-tenant",
                query="从这份资料中提取所有指标",
                doc_ids=["section-source"],
                doc_version=1,
                include_all_sources=False,
            )
            assert extraction_plan.route.intent == SELECTED_DOC_EXTRACT
            assert len(extraction_plan.section_summaries) == len(sections)
            hits = source_guide_hits(
                extraction_plan.guides,
                section_summaries=extraction_plan.section_summaries,
                tenant_id="section-tenant",
                acl_groups=["engineering"],
            )
            assert hits[0].source_type == "source_summary"
            assert all(hit.source_type == "section_summary" for hit in hits[1:])
            assert any("章节提取摘要" in hit.text for hit in hits)

            summary_plan = build_scope_plan(
                config=config,
                tenant_id="section-tenant",
                query="总结这份资料",
                doc_ids=["section-source"],
                doc_version=1,
                include_all_sources=False,
            )
            assert summary_plan.section_summaries == []

            removed = delete_source_section_summaries(
                config.object_store_dir,
                tenant_id="section-tenant",
                source_doc_ids={"section-source"},
                doc_version=1,
            )
            assert removed == len(sections)
            assert load_source_section_summaries(
                config.object_store_dir,
                tenant_id="section-tenant",
                source_keys={("section-source", 1)},
            ) == []
    finally:
        restore_env("RAG_SOURCE_GUIDE_FAST_PATH_CHARS", old_fast_path)
        restore_env("RAG_SECTION_SUMMARY_CHARS", old_summary_chars)
    print("smoke_section_summaries=ok")


def source_documents() -> list[SourceDocument]:
    long_page = "SECTION_HEAD_MARKER " + ("deployment metric details " * 20) + " SECTION_TAIL_MARKER"
    return [
        SourceDocument(
            tenant_id="section-tenant",
            doc_id="section-source/page-1",
            doc_version=1,
            source_type="txt",
            source_uri="memory://section-source/page-1",
            title="Section Source",
            text=long_page,
            acl_groups=["engineering"],
            metadata={"page_no": 1},
        ),
        SourceDocument(
            tenant_id="section-tenant",
            doc_id="section-source/page-2",
            doc_version=1,
            source_type="txt",
            source_uri="memory://section-source/page-2",
            title="Section Source",
            text="Second page contains latency, throughput, and error-rate targets.",
            acl_groups=["engineering"],
            metadata={"page_no": 2},
        ),
    ]


def source_summary() -> SourceSummary:
    return SourceSummary(
        doc_id="section-source",
        title="Section Source",
        source_type="txt",
        source_uri="memory://section-source",
        doc_version=1,
        chunk_count=2,
        acl_groups=["engineering"],
        status="ready",
        current=True,
        child_doc_ids=["section-source/page-1", "section-source/page-2"],
    )


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
