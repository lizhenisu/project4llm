from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.config import load_config  # noqa: E402
from rag_core.sources import ingest_source_documents  # noqa: E402
from rag_core.types import SourceDocument  # noqa: E402


class FakeEmbeddingModel:
    model_name = "synthetic-progress-embedding"

    def __init__(self, dim: int) -> None:
        self.dim = dim

    @staticmethod
    def count_tokens(text: str) -> int:
        return max(1, len(text.split()))

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.01] * self.dim for _text in texts]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="rag-ingestion-progress-") as tmp:
        root = Path(tmp)
        config = replace(
            load_config(),
            metadata_database_url=None,
            object_store_backend="local",
            object_store_dir=root / "object_store",
            runtime_dir=root / "runtime",
        )
        doc = SourceDocument(
            tenant_id="synthetic-progress-tenant",
            doc_id="synthetic-progress-doc",
            doc_version=1,
            source_type="txt",
            source_uri="/synthetic/progress.txt",
            title="Synthetic progress",
            text="synthetic ingestion progress must survive worker process boundaries",
            acl_groups=["engineering"],
        )
        observations: list[tuple[str, int, str]] = []
        indexed_entities: list[dict] = []

        with (
            patch("rag_core.sources.connect", return_value=object()),
            patch("rag_core.sources.ensure_collection"),
            patch(
                "rag_core.sources.build_embedding_model",
                return_value=FakeEmbeddingModel(config.embedding_dim),
            ),
            patch("rag_core.sources.generate_ingested_source_guides"),
            patch(
                "rag_core.sources.upsert_entities",
                side_effect=lambda _client, *, collection_name, entities: indexed_entities.extend(entities),
            ),
        ):
            summary = ingest_source_documents(
                config=config,
                docs=[doc],
                progress_callback=lambda stage, percent, detail: observations.append(
                    (stage, percent, detail)
                ),
            )

    assert summary.document_count == 1
    assert summary.chunk_count == 1
    assert len(indexed_entities) == 1
    assert observations == [
        ("preparing", 32, ""),
        ("chunking", 40, "0/1 个文档"),
        ("text_embedding", 50, "0/1 个文本片段"),
        ("text_embedding", 62, "1/1 个文本片段"),
        ("summarizing", 72, "1 个文档"),
        ("indexing", 82, "0/1 个向量"),
        ("persisting", 92, "1/1 个向量"),
        ("finalizing", 99, ""),
    ]
    assert all(left[1] <= right[1] for left, right in zip(observations, observations[1:], strict=False))
    print("smoke_ingestion_progress=ok")


if __name__ == "__main__":
    main()
