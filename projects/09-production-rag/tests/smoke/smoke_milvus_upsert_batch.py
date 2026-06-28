from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.milvus_store import upsert_entities


class FakeMilvusClient:
    def __init__(self) -> None:
        self.batches: list[list[dict]] = []

    def upsert(self, *, collection_name: str, data: list[dict]) -> dict[str, int]:
        assert collection_name == "smoke_collection"
        self.batches.append(data)
        return {"upsert_count": len(data)}


def main() -> None:
    client = FakeMilvusClient()
    entities = [{"id": str(index)} for index in range(5)]
    total = upsert_entities(
        client,
        collection_name="smoke_collection",
        entities=entities,
        batch_size=2,
    )
    assert total == 5
    assert [len(batch) for batch in client.batches] == [2, 2, 1]
    assert [item["id"] for batch in client.batches for item in batch] == ["0", "1", "2", "3", "4"]
    print("smoke_milvus_upsert_batch=ok")


if __name__ == "__main__":
    main()
