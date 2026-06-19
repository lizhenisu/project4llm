#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"

API_BASE_URL="${RAG_LOAD_BASE_URL:-http://127.0.0.1:8008}"
MOCK_INTERNAL_BASE_URL="${RAG_LOAD_MOCK_INTERNAL_BASE_URL:-http://rag-mock-external-api:18080}"
MOCK_LLM_LATENCY_MS="${MOCK_LLM_LATENCY_MS:-10000}"
MOCK_LLM_MODE="${MOCK_LLM_MODE:-echo}"
MOCK_EMBEDDING_MODE="${MOCK_EMBEDDING_MODE:-random}"
MOCK_RERANK_MODE="${MOCK_RERANK_MODE:-identity}"
KEEP_CONTAINERS="${RAG_LOAD_KEEP_CONTAINERS:-0}"
MIN_CONCURRENCY=1
MAX_CONCURRENCY=64
INCLUDE_SOURCE_IDENTIFIERS=false
CLEANED_UP=false
LOAD_CONTAINERS_STARTED=false

cleanup() {
  local status=$?
  if [[ "$CLEANED_UP" == "true" ]]; then
    exit "$status"
  fi
  CLEANED_UP=true
  trap - EXIT INT TERM
  if [[ "$KEEP_CONTAINERS" == "1" || "$KEEP_CONTAINERS" == "true" ]]; then
    echo "Keeping load-test containers because RAG_LOAD_KEEP_CONTAINERS=${KEEP_CONTAINERS}."
    exit "$status"
  fi
  if [[ "$LOAD_CONTAINERS_STARTED" != "true" ]]; then
    exit "$status"
  fi
  echo "Stopping load-test containers..."
  (
    cd "$PROJECT_DIR"
    docker compose --profile load-test --profile ingest down --remove-orphans
  )
  exit "$status"
}

trap cleanup EXIT INT TERM

usage() {
  cat >&2 <<'EOF'
Usage:
  deploy_and_load_test_2c2g.sh [--min-concurrency N] [--max-concurrency N] [--include-source-identifiers]

Allowed options:
  --min-concurrency N          Lowest concurrency to start from. Default: 1
  --max-concurrency N          Highest concurrency to search up to. Default: 64
  --include-source-identifiers Include limited doc_id/doc_version details in the report.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --min-concurrency)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      MIN_CONCURRENCY="$2"
      shift 2
      ;;
    --max-concurrency)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      MAX_CONCURRENCY="$2"
      shift 2
      ;;
    --include-source-identifiers)
      INCLUDE_SOURCE_IDENTIFIERS=true
      shift
      ;;
    *)
      echo "Unsupported option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if ! [[ "$MIN_CONCURRENCY" =~ ^[0-9]+$ && "$MAX_CONCURRENCY" =~ ^[0-9]+$ ]]; then
  echo "--min-concurrency and --max-concurrency must be positive integers." >&2
  exit 2
fi
if (( MIN_CONCURRENCY < 1 || MAX_CONCURRENCY < MIN_CONCURRENCY )); then
  echo "Invalid concurrency range: min=${MIN_CONCURRENCY}, max=${MAX_CONCURRENCY}." >&2
  exit 2
fi

LOAD_TEST_ARGS=(
  --search-limit
  --search-min-concurrency "$MIN_CONCURRENCY"
  --search-max-concurrency "$MAX_CONCURRENCY"
)
if [[ "$INCLUDE_SOURCE_IDENTIFIERS" == "true" ]]; then
  LOAD_TEST_ARGS+=(--include-source-identifiers)
fi

cd "$PROJECT_DIR"

if [[ ! -f ".env" ]]; then
  echo "Missing $PROJECT_DIR/.env. Create it before deploying the production containers." >&2
  exit 1
fi

echo "Stopping existing Production RAG compose containers..."
docker compose --profile load-test --profile ingest down --remove-orphans

echo "Building and starting production containers with the mock external API..."
LOAD_CONTAINERS_STARTED=true
NEW_API_URL="${MOCK_INTERNAL_BASE_URL}/v1" \
NEW_API_KEY="${NEW_API_KEY:-mock-key}" \
SILICONFLOW_URL="${MOCK_INTERNAL_BASE_URL}" \
SILICONFLOW_API_KEY="${SILICONFLOW_API_KEY:-mock-key}" \
MOCK_LLM_LATENCY_MS="$MOCK_LLM_LATENCY_MS" \
MOCK_LLM_MODE="$MOCK_LLM_MODE" \
MOCK_EMBEDDING_MODE="$MOCK_EMBEDDING_MODE" \
MOCK_RERANK_MODE="$MOCK_RERANK_MODE" \
docker compose --profile load-test up -d --build rag-mock-external-api rag-api rag-web

echo "Waiting for API health at ${API_BASE_URL}/health..."
for attempt in $(seq 1 90); do
  if curl -fsS "${API_BASE_URL}/health" >/dev/null 2>&1; then
    echo "API is healthy."
    break
  fi
  if [[ "$attempt" == "90" ]]; then
    echo "API did not become healthy in time. Recent rag-api logs:" >&2
    docker compose logs --tail=120 rag-api >&2 || true
    exit 1
  fi
  sleep 2
done

cd "$REPO_DIR"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

echo "Deployment is healthy. Searching approximate RAG concurrency limit from ${MIN_CONCURRENCY} to ${MAX_CONCURRENCY}..."
"$PROJECT_DIR/scripts/run_2c2g_load_test.sh" \
  --base-url "$API_BASE_URL" \
  "${LOAD_TEST_ARGS[@]}"
