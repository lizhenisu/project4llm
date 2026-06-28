#!/usr/bin/env bash
set -euo pipefail

python wait_milvus.py
python schema.py
python check_config.py
exec python ingestion_worker.py
