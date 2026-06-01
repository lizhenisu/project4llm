# AGENTS.md

## Project Overview

This repository is a hands-on Chinese learning curriculum for preparing LLM / AI agent / RAG engineering interviews. It contains:

- `notes/`: concept notes and interview-oriented explanations.
- `projects/`: runnable mini projects that turn each concept into code.
- `target.md`: the overall 12-week LLM training roadmap.
- `README.md`: the recommended learning order and core commands.

The project principle is: every concept should map to runnable code, and every script should be explainable in terms of inputs, outputs, shapes, metrics, or retrieval behavior.

## Environment Rule

Before running any Python code, always activate the local virtual environment:

```bash
source .venv/bin/activate   # For Linux/macOS/WSL
# Or for Windows: .venv\Scripts\activate
```

Use the activated environment for all Python, `uv`, and package commands.

## Dependency Management

This project uses `uv` and keeps dependencies in `pyproject.toml` plus `uv.lock`.

Install or sync existing dependencies with:

```bash
source .venv/bin/activate
uv sync
```

Add a new package with:

```bash
source .venv/bin/activate
uv add package-name
```

For optional extras, quote the package spec:

```bash
source .venv/bin/activate
uv add "pymilvus[milvus_lite]"
```

Do not edit `uv.lock` manually. Let `uv add` or `uv sync` update it.

If `pip` is unavailable inside `.venv`, prefer `uv add` / `uv sync` instead of trying to bootstrap pip.

## Running Code

Run commands from the repository root unless a script explicitly says otherwise.

Common smoke commands:

```bash
source .venv/bin/activate
python projects/01-ml-basics/train_mlp.py --epochs 3
python projects/03-tokenizer-and-data/train_bpe_tokenizer.py
python projects/03-tokenizer-and-data/data_pipeline.py
python projects/02-transformer-from-scratch/smoke_test.py
python projects/02-transformer-from-scratch/train_tiny_gpt.py --steps 20
python projects/04-sft-qwen-lora/build_sft_dataset.py
python projects/05-dpo-preference/build_preference_dataset.py
python projects/07-milvus-rag/milvus_lite_rag_demo.py
```

For a quick syntax check:

```bash
source .venv/bin/activate
python -m py_compile path/to/script.py
```

For dependency consistency:

```bash
source .venv/bin/activate
uv pip check
```

## Milvus Lite Note

`projects/07-milvus-rag/milvus_lite_rag_demo.py` starts a local Milvus Lite server and binds a local gRPC port. In sandboxed environments this may require elevated permission to bind `127.0.0.1`.

The generated local database directory is ignored by git:

```text
projects/07-milvus-rag/milvus_lite_demo.db/
```

Do not commit local Milvus database files, caches, `__pycache__/`, or `.venv/`.
