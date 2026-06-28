from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tests.load.milvus_seed_load import (  # noqa: E402
    deterministic_vector,
    expected_chunk_count,
    iter_chunks,
)


def main() -> None:
    args = argparse.Namespace(
        tenant_id="synthetic-capacity",
        tenant_count=2,
        documents_per_tenant=3,
        chunks_per_document=4,
        acl_group="engineering",
    )
    chunks = list(iter_chunks(args))
    assert len(chunks) == expected_chunk_count(args) == 24
    assert {chunk.tenant_id for chunk in chunks} == {
        "synthetic-capacity-0000",
        "synthetic-capacity-0001",
    }
    assert len({(chunk.tenant_id, chunk.doc_id, chunk.chunk_index) for chunk in chunks}) == 24
    assert all(chunk.metadata["synthetic"] is True for chunk in chunks)

    first = deterministic_vector(0, 32)
    second = deterministic_vector(1, 32)
    assert first == deterministic_vector(0, 32)
    assert first != second
    assert math.isclose(math.sqrt(sum(value * value for value in first)), 1.0)
    print("smoke_milvus_seed_load=ok")


if __name__ == "__main__":
    main()
