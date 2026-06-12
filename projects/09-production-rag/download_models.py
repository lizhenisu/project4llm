#!/usr/bin/env python3
"""下载 BGE-M3、bge-reranker-v2-m3 和 CLIP 模型。

首次使用需运行此脚本，需要 ~5 GB 磁盘空间。
BGE/reranker 通过 ModelScope 下载；CLIP 使用 Hugging Face Hub。

用法:
    python projects/09-production-rag/download_models.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


MODELSCOPE_CACHE = Path.home() / ".cache" / "modelscope" / "hub" / "models" / "BAAI"


def _download(repo_id: str, subdir: str) -> Path:
    """Download a model from ModelScope and return its local path."""
    from modelscope import snapshot_download

    target = MODELSCOPE_CACHE / subdir
    if target.is_dir() and (
        (target / "pytorch_model.bin").exists()
        or (target / "model.safetensors").exists()
    ):
        print(f"  已存在: {target}")
        return target

    print(f"  正在从 ModelScope 下载 {repo_id}...")
    try:
        snapshot_download(repo_id, ignore_file_pattern=["onnx/*", "*.jpg", "*.webp", "colbert_linear.pt", "sparse_linear.pt"])
    except ImportError:
        sys.exit("modelscope 未安装。先执行: uv add modelscope")
    return target


def _verify_embedding(path: Path) -> None:
    from sentence_transformers import SentenceTransformer

    print(f"  加载模型验证: {path}")
    start = time.monotonic()
    model = SentenceTransformer(str(path), device="cpu")
    dim = model.get_sentence_embedding_dimension()
    elapsed = time.monotonic() - start
    print(f"  dim={dim}, load_time={elapsed:.1f}s — 通过")
    vec = model.encode(["hello world", "Milvus RAG 向量检索"])
    print(f"  推理 shape={vec.shape} — 通过")


def _verify_reranker(path: Path) -> None:
    from sentence_transformers import CrossEncoder

    print(f"  加载模型验证: {path}")
    start = time.monotonic()
    reranker = CrossEncoder(str(path))
    elapsed = time.monotonic() - start
    print(f"  load_time={elapsed:.1f}s — 通过")
    scores = reranker.predict([
        ("RAG 检索变慢排查", "检查 topK 和 HNSW ef 参数"),
        ("RAG 检索变慢排查", "今天天气很好"),
    ])
    print(f"  scores={[f'{s:.3f}' for s in scores]} — 通过")


def _verify_clip(model_name: str) -> None:
    from transformers import CLIPModel, CLIPProcessor

    print(f"  加载 CLIP 验证: {model_name}")
    start = time.monotonic()
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    elapsed = time.monotonic() - start
    batch = processor(text=["RAG dashboard screenshot"], padding=True, return_tensors="pt")
    features = model.get_text_features(**batch)
    print(f"  dim={features.shape[-1]}, load_time={elapsed:.1f}s — 通过")


def main() -> None:
    print("=" * 60)
    print("下载 RAG 模型")
    print("=" * 60)

    # BGE-M3 embedding
    emb_path = _download("BAAI/bge-m3", "bge-m3")
    _verify_embedding(emb_path)

    # bge-reranker-v2-m3
    rerank_path = _download("BAAI/bge-reranker-v2-m3", "bge-reranker-v2-m3")
    _verify_reranker(rerank_path)

    # CLIP image/text embedding
    _verify_clip("openai/clip-vit-base-patch32")

    print()
    print("=" * 60)
    print("模型下载并验证完成！")
    print("运行教学 walkthrough:")
    print("  python projects/09-production-rag/walkthrough_core_rag.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
