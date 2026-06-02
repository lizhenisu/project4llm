#!/usr/bin/env bash
set -euo pipefail

python wait_milvus.py
python schema.py
python ingest_text.py
python ingest_image.py
