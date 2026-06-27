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
from rag_core.jsonl_store import read_object_jsonl  # noqa: E402
from rag_core.source_guides import (  # noqa: E402
    SOURCE_GUIDES_PATH,
    get_or_create_source_guide,
)
from rag_core.types import SourceDocument  # noqa: E402


def main() -> None:
    old_limit = os.environ.get("RAG_SOURCE_GUIDE_FAST_PATH_CHARS")
    os.environ["RAG_SOURCE_GUIDE_FAST_PATH_CHARS"] = "300"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(
                load_config(),
                object_store_dir=Path(tmp) / "object_store",
                runtime_dir=Path(tmp) / "runtime",
                llm_base_url="",
                llm_api_key="",
            )
            result = get_or_create_source_guide(
                config=config,
                tenant_id="fast-path-tenant",
                source_doc_id="tiny-note",
                doc_version=1,
                doc_title="Tiny note.txt",
                docs=[source_doc(text="Deployment owner: platform team. Retry limit: 3.")],
            )
            rows = read_object_jsonl(config.object_store_dir, SOURCE_GUIDES_PATH)

            assert result.title == "Tiny note.txt"
            assert "Deployment owner: platform team" in result.guide
            assert rows[0]["model"] == "deterministic-fast-path"

            try:
                get_or_create_source_guide(
                    config=config,
                    tenant_id="fast-path-tenant",
                    source_doc_id="large-note",
                    doc_version=1,
                    doc_title="Large note.txt",
                    docs=[source_doc(text="large text " * 100)],
                )
            except RuntimeError as exc:
                assert "must be configured for source guide generation" in str(exc)
            else:
                raise AssertionError("large text unexpectedly bypassed LLM guide generation")
    finally:
        restore_env("RAG_SOURCE_GUIDE_FAST_PATH_CHARS", old_limit)
    print("smoke_source_guide_fast_path=ok")


def source_doc(*, text: str) -> SourceDocument:
    return SourceDocument(
        tenant_id="fast-path-tenant",
        doc_id="tiny-note",
        doc_version=1,
        source_type="txt",
        source_uri="memory://tiny-note",
        title="Tiny note.txt",
        text=text,
        acl_groups=["engineering"],
        metadata={},
    )


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
