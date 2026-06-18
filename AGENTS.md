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

Run commands from the repository root unless a script explicitly says otherwise. Prefer focused smoke checks for the files or project being changed instead of running unrelated demos.

For a quick Python syntax check:

```bash
source .venv/bin/activate
python -m py_compile path/to/script.py
```

For dependency consistency:

```bash
source .venv/bin/activate
uv pip check
```

## Production RAG Development Workflow

For `projects/09-production-rag`, avoid rebuilding Docker containers for routine development.

Use Docker for Milvus dependencies, run the FastAPI backend locally with reload, and use the Vite dev server for frontend hot reload:

```bash
cd projects/09-production-rag
docker compose up -d milvus

source ../../.venv/bin/activate
uvicorn serve:app --reload --host 0.0.0.0 --port 8008

cd frontend
npm run dev -- --host 0.0.0.0
```

Open the development UI at:

```text
http://localhost:5173/
```

When a frontend test needs an authenticated session, use the fixed-token development URL:

```text
http://localhost:5173/#token=production-rag-fixed-test-login-token
```

The Vite dev server proxies `/api/*` to the containerized backend at `http://127.0.0.1:8008`, so frontend changes under `projects/09-production-rag/frontend/src/` should be visible immediately in the browser without rebuilding `rag-web`.

Only rebuild containers when validating production packaging, Dockerfile changes, nginx config changes, backend dependency/image changes, or before a final production-like check.

```bash
cd projects/09-production-rag
docker compose up -d --build rag-api rag-web
```

The production-style UI remains:

```text
http://localhost:8080/
```

## Milvus Lite Note

Do not commit local Milvus database files, caches, `__pycache__/`, `.venv/`, generated runtime directories, or object-store data.

## Privacy And Sensitive Data

The repository must not contain user-private information or secrets. This includes, but is not limited to:

- Real names or identifiable personal profiles.
- Email addresses, phone numbers, addresses, ID numbers, account names, or other contact details.
- API keys, tokens, passwords, cookies, session IDs, private keys, `.env` values, or service credentials.
- Resumes, CVs, screenshots, documents, PDFs, uploads, exports, logs, database files, or object-store data that may contain private user information.

Use synthetic placeholders in committed examples and fixtures. Do not copy real user-provided content into committed files.

## Branch Management Rule

Development happens on the `dev` branch. The `main` branch is reserved for stable version releases only.

Do not implement ordinary code changes directly on `main`. Merge `dev` into `main` only when the user explicitly asks to publish a stable release.

**Do NOT automatically merge `dev` into `main` unless explicitly asked.** Always ask the user before merging or force-pushing to `main`. The `dev` branch is for development and experimentation; only the user decides when to promote changes to `main`.

## Commit Message Rule

For small changes, a one-sentence English commit message is enough.

For larger changes, use one summary sentence followed by bullet points:

```text
Short summary sentence

- First concrete change
- Second concrete change
- Verification or migration note
```
