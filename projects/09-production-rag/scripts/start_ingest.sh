#!/usr/bin/env bash
set -euo pipefail

python wait_milvus.py
python schema.py

if [[ -n "${RAG_TEXT_INPUT:-}" ]]; then
  python ingest_text.py --input "$RAG_TEXT_INPUT"
else
  echo "RAG_TEXT_INPUT is not set; skipping text ingest."
fi

if [[ -n "${RAG_IMAGE_INPUT:-}" ]]; then
  python ingest_image.py --input "$RAG_IMAGE_INPUT"
else
  echo "RAG_IMAGE_INPUT is not set; skipping image ingest."
fi
