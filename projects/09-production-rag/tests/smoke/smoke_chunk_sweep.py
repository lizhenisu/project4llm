from __future__ import annotations

import os
import tempfile
from pathlib import Path

from rag_core.config import FIXTURE_DATA_DIR
from rag_core.config import load_config
from rag_core.milvus_store import connect
from sweep_chunking import ChunkSpec, sweep_chunking


def main() -> None:
    old_milvus_uri = os.environ.get("MILVUS_URI")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MILVUS_URI"] = str(Path(tmp) / "chunk_sweep.db")
        try:
            rows = sweep_chunking(
                input_path=FIXTURE_DATA_DIR / "sample_docs.jsonl",
                eval_input_path=FIXTURE_DATA_DIR / "eval_queries.jsonl",
                specs=[
                    ChunkSpec(chunk_size=80, overlap=10),
                    ChunkSpec(chunk_size=160, overlap=20),
                ],
                mode="hybrid",
                limit=5,
            )
            assert len(rows) == 2
            client = connect(load_config())
            for row in rows:
                assert row["doc_count"] == 4
                assert row["chunk_count"] >= 4
                assert row["upserted"] == row["chunk_count"]
                assert row["recall"] == 1.0
                assert row["mrr"] == 1.0
                assert row["stage_p95_latency_ms"]["milvus_search"] >= 0.0
                assert row["cleanup"] == "dropped"
                assert not client.has_collection(str(row["temporary_collection"]))
        finally:
            restore_env("MILVUS_URI", old_milvus_uri)
    print("smoke_chunk_sweep=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
