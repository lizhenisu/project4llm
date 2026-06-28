from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.events import append_event


@contextmanager
def patched_env(**values: str):
    old_values = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def main() -> None:
    test_event_log_limits_keep_json_parseable_and_key_fields()
    print("smoke_event_log_limits=ok")


def test_event_log_limits_keep_json_parseable_and_key_fields() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        runtime_dir = Path(tmp) / "runtime"
        with patched_env(
            RAG_EVENT_MAX_JSON_BYTES="2048",
            RAG_EVENT_MAX_STRING_CHARS="128",
            RAG_EVENT_MAX_LIST_ITEMS="5",
            RAG_EVENT_MAX_DICT_ITEMS="8",
        ):
            append_event(
                runtime_dir,
                "answer_events",
                {
                    "request_id": "event-limit-smoke",
                    "query": "怎么总结这些资料？" * 200,
                    "trace": {
                        "retrieval_mode": "hybrid_dense_sparse_rerank",
                        "stage_latency_ms": {"embedding": 1.0, "milvus_search": 2.0, "rerank": 3.0},
                        "current_versions": {f"doc-{index}": index for index in range(200)},
                        "filter_expr": "doc_id in [" + ",".join(f'"doc-{index}"' for index in range(2000)) + "]",
                    },
                    "final_context": [
                        {"doc_id": f"doc-{index}", "text_preview": "evidence " * 200}
                        for index in range(40)
                    ],
                    "llm": {
                        "llm_model": "test-model",
                        "latency_ms": 10.0,
                        "answer": "answer " * 2000,
                    },
                },
            )
        line = (runtime_dir / "answer_events.jsonl").read_text(encoding="utf-8").splitlines()[0]
        event = json.loads(line)

    assert len(line.encode("utf-8")) < 4096
    assert event["request_id"] == "event-limit-smoke"
    assert event["_event_truncated"] is True
    assert event["_original_json_bytes"] > len(line.encode("utf-8"))
    assert event["trace"]["stage_latency_ms"]["rerank"] == 3.0
    assert "_truncated_items" in event["final_context"][-1]


if __name__ == "__main__":
    main()
