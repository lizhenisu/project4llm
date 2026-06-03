#!/usr/bin/env python3
"""下载 BGE-M3 和 bge-reranker-v2-m3 模型。

沙箱环境网络受限，请在本地终端执行此脚本。
需要 3-5 GB 磁盘空间和稳定的网络连接。

用法:
    python projects/08-industrial-rag/download_models.py

环境变量（可选）:
    HF_ENDPOINT     Hugging Face 镜像，国内建议 https://hf-mirror.com
    HF_TOKEN        Hugging Face token（提升下载速度）
"""

from __future__ import annotations

import os
import sys
import time


def download_embedding(model_id: str) -> None:
    """下载 embedding 模型并做快速推理测试。"""
    print(f"\n{'='*60}")
    print(f"下载 embedding 模型: {model_id}")
    print(f"{'='*60}")
    start = time.monotonic()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit(
            "sentence-transformers 未安装。请先执行: "
            "pip install sentence-transformers"
        )

    model = SentenceTransformer(model_id, device="cpu")
    elapsed = time.monotonic() - start
    dim = model.get_sentence_embedding_dimension()
    print(f"完成: dim={dim}, 耗时 {elapsed:.0f}s")

    test_vec = model.encode(["hello world", "Milvus RAG"])
    print(f"推理测试通过: shape={test_vec.shape}")


def download_reranker(model_id: str) -> None:
    """下载 reranker 模型并做快速推理测试。"""
    print(f"\n{'='*60}")
    print(f"下载 reranker 模型: {model_id}")
    print(f"{'='*60}")
    start = time.monotonic()

    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(model_id)
    elapsed = time.monotonic() - start
    print(f"完成: 耗时 {elapsed:.0f}s")

    scores = reranker.predict([
        ("RAG 检索变慢排查", "检查 topK 和 HNSW ef 参数"),
        ("RAG 检索变慢排查", "今天天气很好"),
    ])
    print(f"推理测试通过: scores={[f'{s:.3f}' for s in scores]}")


def main() -> None:
    if "HF_ENDPOINT" not in os.environ:
        print("提示: 国内用户建议 export HF_ENDPOINT='https://hf-mirror.com'\n")

    download_embedding("BAAI/bge-m3")

    try:
        download_reranker("BAAI/bge-reranker-v2-m3")
    except Exception as exc:
        print(f"Reranker 下载失败 (非致命): {exc}")
        print("embedding 已就绪，可稍后重试 reranker。")

    print(f"\n{'='*60}")
    print("模型下载完成！运行教学 walkthrough:")
    print("  python projects/08-industrial-rag/walkthrough_core_rag.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
