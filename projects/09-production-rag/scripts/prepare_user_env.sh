#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_DIR="${ROOT_DIR}/projects/09-production-rag"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/practice4llm-uv-cache}"
WARNINGS=()

cd "${ROOT_DIR}"

section() {
  echo
  echo "== $1 =="
}

warn() {
  WARNINGS+=("$1")
  echo "WARN: $1"
}

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "ERROR: Missing .venv. Create or sync the project environment first."
  exit 1
fi

source .venv/bin/activate

section "Python"
python --version

section "uv"
uv --version

section "Python dependency check"
UV_CACHE_DIR="${UV_CACHE_DIR}" uv pip check

section "Node.js"
if command -v node >/dev/null 2>&1; then
  node --version
else
  warn "node not found. Install Node.js 20+ or 22+ before starting the TypeScript frontend."
fi

section "npm"
if command -v npm >/dev/null 2>&1; then
  if ! npm --version; then
    warn "npm exists but is not usable. In WSL this often means npm points to Windows Node; install Linux-side Node/npm or fix PATH."
  fi
else
  warn "npm not found. Install npm together with Node.js."
fi

section "Docker"
if command -v docker >/dev/null 2>&1; then
  docker --version || true
  docker compose version || true
  if docker ps >/dev/null 2>&1; then
    echo "docker daemon=reachable"
  else
    warn "docker CLI exists, but daemon is not reachable or permission is denied. Enable Docker Desktop WSL integration, start Docker, or add this Linux user to the docker group."
  fi
else
  warn "docker not found. Install Docker in WSL or enable Docker Desktop WSL integration."
fi

section "09 project checks"
python -m py_compile \
  "${PROJECT_DIR}/serve.py" \
  "${PROJECT_DIR}/schema.py" \
  "${PROJECT_DIR}/ingest_text.py" \
  "${PROJECT_DIR}/ingest_image.py"
echo "py_compile=ok"

echo
echo "User-level preparation check finished."
if [[ "${#WARNINGS[@]}" -gt 0 ]]; then
  echo
  echo "Action needed:"
  for item in "${WARNINGS[@]}"; do
    echo "- ${item}"
  done
  echo
  echo "This script only checks the environment. It never installs packages or changes system configuration."
fi
