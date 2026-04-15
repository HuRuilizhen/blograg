#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

BLOGRAG_BIN="${BLOGRAG_BIN:-${REPO_ROOT}/.venv/bin/blograg}"
SERVE_SCRIPT="${REPO_ROOT}/scripts/serve_mcp.sh"
RESTART_SCRIPT="${REPO_ROOT}/scripts/restart_mcp_http.sh"

client=""
blog_dir=""
index_dir="${REPO_ROOT}/index"
server_name="blograg"
rebuild="false"
host="${BLOGRAG_MCP_HOST:-127.0.0.1}"
port="${BLOGRAG_MCP_PORT:-8765}"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --client codex|openclaw|both [options]

Options:
  --client CLIENT         codex, openclaw, or both
  --blog-dir PATH         source blog directory; required when building a new index
  --index-dir PATH        persisted index directory (default: ${index_dir})
  --server-name NAME      MCP server name to register (default: ${server_name})
  --host HOST             HTTP MCP bind host (default: ${host})
  --port PORT             HTTP MCP bind port (default: ${port})
  --rebuild               rebuild the index before registering
  -h, --help              show this help

Examples:
  $(basename "$0") --client codex --blog-dir /path/to/blog
  $(basename "$0") --client both --blog-dir /path/to/blog --index-dir /tmp/blograg-index
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --client)
      client="${2:-}"
      shift 2
      ;;
    --blog-dir)
      blog_dir="${2:-}"
      shift 2
      ;;
    --index-dir)
      index_dir="${2:-}"
      shift 2
      ;;
    --server-name)
      server_name="${2:-}"
      shift 2
      ;;
    --host)
      host="${2:-}"
      shift 2
      ;;
    --port)
      port="${2:-}"
      shift 2
      ;;
    --rebuild)
      rebuild="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${client}" ]]; then
  echo "--client is required" >&2
  usage >&2
  exit 1
fi

if [[ ! -x "${BLOGRAG_BIN}" ]]; then
  echo "blograg executable not found at ${BLOGRAG_BIN}" >&2
  exit 1
fi

needs_build="false"
if [[ "${rebuild}" == "true" || ! -f "${index_dir}/blograg/manifest.json" ]]; then
  needs_build="true"
fi

if [[ "${needs_build}" == "true" ]]; then
  if [[ -z "${blog_dir}" ]]; then
    echo "--blog-dir is required because the index is missing or --rebuild was requested" >&2
    exit 1
  fi
  "${BLOGRAG_BIN}" build --blog-dir "${blog_dir}" --index-dir "${index_dir}"
fi

BLOGRAG_INDEX_DIR="${index_dir}" BLOGRAG_MCP_HOST="${host}" BLOGRAG_MCP_PORT="${port}" \
  bash "${RESTART_SCRIPT}"

server_url="http://${host}:${port}/mcp"

register_codex() {
  codex mcp add "${server_name}" --url "${server_url}"
}

register_openclaw() {
  local config_json
  config_json=$(
    printf '{"url":"%s"}' "${server_url}"
  )
  openclaw mcp set "${server_name}" "${config_json}"
}

case "${client}" in
  codex)
    register_codex
    ;;
  openclaw)
    register_openclaw
    ;;
  both)
    register_codex
    register_openclaw
    ;;
  *)
    echo "Unsupported client: ${client}" >&2
    exit 1
    ;;
esac

echo "Registered ${server_name} for ${client} using URL ${server_url}"
