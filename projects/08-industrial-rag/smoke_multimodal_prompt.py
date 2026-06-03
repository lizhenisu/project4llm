from __future__ import annotations

from rag_core.answering import build_prompt, format_evidence_header
from rag_core.types import ImageDocument, SearchHit
from ingest_image import normalize_image_metadata


def main() -> None:
    hit = SearchHit(
        id="image-hit",
        score=1.0,
        text=(
            "OCR:\nRAG Dashboard p95 latency recall@50 rerank latency error rate\n"
            "图片描述:\n一张 RAG 线上监控面板截图。"
        ),
        doc_id="dashboard-screenshot",
        title="RAG 监控面板截图",
        source_uri="images/rag-dashboard.png",
        source_type="image",
        chunk_index=0,
        tenant_id="team_a",
        acl_groups=["ops"],
        metadata={
            "image_uri": "images/rag-dashboard.png",
            "linked_doc_id": "rag-runbook",
            "bbox": [10, 20, 200, 120],
            "caption": "一张 RAG 线上监控面板截图。",
            "ocr_text": "RAG Dashboard p95 latency recall@50 rerank latency error rate",
        },
    )
    header = format_evidence_header(1, hit)
    assert "source_type=image" in header
    assert "image_uri=images/rag-dashboard.png" in header
    assert "linked_doc_id=rag-runbook" in header
    assert "bbox=[10, 20, 200, 120]" in header

    prompt = build_prompt("RAG 监控面板显示了什么？", [hit])
    assert "图片证据来自 OCR/caption 或图片向量召回" in prompt
    assert "可能不完整" in prompt
    assert "source_type=image" in prompt

    metadata = normalize_image_metadata(
        ImageDocument(
            tenant_id="team_a",
            doc_id="dashboard-screenshot",
            doc_version=1,
            source_uri="images/rag-dashboard.png",
            title="RAG 监控面板截图",
            ocr_text="RAG Dashboard",
            caption="监控面板截图",
            acl_groups=["ops"],
            metadata={"linked_doc_id": "rag-runbook"},
        )
    )
    assert metadata["image_uri"] == "images/rag-dashboard.png"
    assert metadata["bbox"] == []
    assert metadata["linked_doc_id"] == "rag-runbook"
    assert metadata["ocr_text"] == "RAG Dashboard"
    assert metadata["caption"] == "监控面板截图"
    print("smoke_multimodal_prompt=ok")


if __name__ == "__main__":
    main()
