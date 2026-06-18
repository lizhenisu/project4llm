from __future__ import annotations

import tempfile
from pathlib import Path

from rag_core.events import append_event
from monitor_events import summarize_runtime_events


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        runtime_dir = Path(tmp) / "runtime"
        append_event(
            runtime_dir,
            "retrieval_events",
            {
                "request_id": "monitor-retrieval-1",
                "trace": {
                    "retrieval_mode": "hybrid_dense_sparse_rerank",
                    "context_count": 2,
                    "source_types": ["md", "image"],
                    "stage_latency_ms": {
                        "embedding": 10.0,
                        "milvus_search": 20.0,
                        "rerank": 30.0,
                    },
                },
                "final_context": [
                    {"doc_id": "rag-runbook", "source_type": "md"},
                    {
                        "doc_id": "dashboard-screenshot",
                        "source_type": "image",
                        "metadata": {
                            "fusion": {
                                "channels": {
                                    "text_hybrid": 1,
                                    "image_vector": 1,
                                }
                            }
                        },
                    },
                ],
            },
        )
        append_event(
            runtime_dir,
            "answer_events",
            {
                "request_id": "monitor-answer-1",
                "trace": {
                    "retrieval_mode": "multimodal_text_image_fusion_rerank",
                    "context_count": 1,
                    "source_types": ["image"],
                    "stage_latency_ms": {
                        "embedding": 12.0,
                        "milvus_search": 24.0,
                        "rerank": 36.0,
                    },
                },
                "final_context": [
                    {
                        "doc_id": "dashboard-screenshot",
                        "source_type": "image",
                        "metadata": {
                            "fusion": {
                                "channels": {
                                    "text_hybrid": 1,
                                    "image_vector": 1,
                                }
                            }
                        },
                    }
                ],
                "llm": {
                    "llm_model": "test-model",
                    "llm_backend": "newapi",
                    "latency_ms": 5.0,
                    "token_usage": {},
                },
            },
        )
        append_event(
            runtime_dir,
            "feedback_events",
            {
                "request_id": "monitor-answer-1",
                "rating": 1,
                "comment": "ok",
            },
        )

        summary = summarize_runtime_events(runtime_dir)

    assert summary["retrieval_events"] == 1
    assert summary["answer_events"] == 1
    assert summary["feedback_events"] == 1
    assert summary["retrieval_modes"] == {
        "hybrid_dense_sparse_rerank": 1,
        "multimodal_text_image_fusion_rerank": 1,
    }
    assert summary["requested_source_types"] == {"image": 2, "md": 1}
    assert summary["context"]["avg"] == 1.5
    assert summary["context_source_types"] == {"image": 2, "md": 1}
    assert summary["fusion_channels"] == {"image_vector": 2, "text_hybrid": 2}
    assert summary["stage_latency_ms"]["embedding"]["count"] == 2
    assert summary["stage_latency_ms"]["milvus_search"]["p95"] == 24.0
    assert summary["llm_latency_ms"]["p50"] == 5.0
    assert summary["top_context_docs"]["dashboard-screenshot"] == 2
    assert summary["feedback_ratings"] == {"1": 1}
    print("smoke_monitoring=ok")


if __name__ == "__main__":
    main()
