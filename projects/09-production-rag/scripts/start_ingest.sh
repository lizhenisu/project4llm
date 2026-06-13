#!/usr/bin/env bash
set -euo pipefail

python wait_milvus.py
python schema.py

if [[ -z "${RAG_TEXT_INPUT:-}" && -z "${RAG_IMAGE_INPUT:-}" ]]; then
  echo "Set at least one of RAG_TEXT_INPUT or RAG_IMAGE_INPUT before running ingest." >&2
  exit 2
fi

if [[ -n "${RAG_TEXT_INPUT:-}" ]]; then
  python ingest_text.py --input "$RAG_TEXT_INPUT"
fi

if [[ -n "${RAG_IMAGE_INPUT:-}" ]]; then
  python ingest_image.py --input "$RAG_IMAGE_INPUT"
fi
