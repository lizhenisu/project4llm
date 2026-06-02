#!/usr/bin/env bash
set -euo pipefail

python wait_milvus.py
python schema.py
uvicorn serve:app --host 0.0.0.0 --port "${RAG_API_PORT:-8008}"
