#!/usr/bin/env bash
set -euo pipefail

API_WORKERS="${RAG_API_WORKERS:-1}"
API_LIMIT_CONCURRENCY="${RAG_API_LIMIT_CONCURRENCY:-256}"
API_KEEP_ALIVE_SECONDS="${RAG_API_KEEP_ALIVE_SECONDS:-15}"
API_GRACEFUL_SHUTDOWN_SECONDS="${RAG_API_GRACEFUL_SHUTDOWN_SECONDS:-30}"

require_positive_integer() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer, got: ${value}" >&2
    exit 2
  fi
}

require_non_negative_integer() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "${name} must be a non-negative integer, got: ${value}" >&2
    exit 2
  fi
}

require_positive_integer "RAG_API_WORKERS" "$API_WORKERS"
require_positive_integer "RAG_API_LIMIT_CONCURRENCY" "$API_LIMIT_CONCURRENCY"
require_non_negative_integer "RAG_API_KEEP_ALIVE_SECONDS" "$API_KEEP_ALIVE_SECONDS"
require_non_negative_integer "RAG_API_GRACEFUL_SHUTDOWN_SECONDS" "$API_GRACEFUL_SHUTDOWN_SECONDS"

python wait_milvus.py
python schema.py
python check_config.py
exec uvicorn serve:app \
  --host 0.0.0.0 \
  --port "${RAG_API_PORT:-8008}" \
  --workers "$API_WORKERS" \
  --limit-concurrency "$API_LIMIT_CONCURRENCY" \
  --timeout-keep-alive "$API_KEEP_ALIVE_SECONDS" \
  --timeout-graceful-shutdown "$API_GRACEFUL_SHUTDOWN_SECONDS"
