#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

BLOGRAG_BIN="${BLOGRAG_BIN:-${REPO_ROOT}/.venv/bin/blograg}"
INDEX_DIR="${BLOGRAG_INDEX_DIR:-${REPO_ROOT}/index}"
ENV_FILE="${BLOGRAG_ENV_FILE:-${REPO_ROOT}/.env.local}"
MCP_HOST="${BLOGRAG_MCP_HOST:-127.0.0.1}"
MCP_PORT="${BLOGRAG_MCP_PORT:-8765}"
HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
TRANSFORMERS_NO_ADVISORY_WARNINGS="${TRANSFORMERS_NO_ADVISORY_WARNINGS:-1}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [[ ! -x "${BLOGRAG_BIN}" ]]; then
  echo "blograg executable not found at ${BLOGRAG_BIN}" >&2
  echo "Install project dependencies first, for example: .venv/bin/python -m pip install -e '.[dev]'" >&2
  exit 1
fi

if [[ ! -f "${INDEX_DIR}/blograg/manifest.json" ]]; then
  echo "blograg index not found at ${INDEX_DIR}" >&2
  echo "Build it first with: ${BLOGRAG_BIN} build --blog-dir /path/to/blog --index-dir ${INDEX_DIR}" >&2
  exit 1
fi

exec "${BLOGRAG_BIN}" serve \
  --index-dir "${INDEX_DIR}" \
  --transport streamable-http \
  --host "${MCP_HOST}" \
  --port "${MCP_PORT}"
