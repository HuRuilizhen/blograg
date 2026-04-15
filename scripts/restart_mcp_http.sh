#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

HTTP_SCRIPT="${REPO_ROOT}/scripts/serve_mcp.sh"
PID_FILE="${BLOGRAG_MCP_PID_FILE:-/tmp/blograg-mcp-http.pid}"
LOG_FILE="${BLOGRAG_MCP_LOG_FILE:-/tmp/blograg-mcp-http.log}"
MCP_HOST="${BLOGRAG_MCP_HOST:-127.0.0.1}"
MCP_PORT="${BLOGRAG_MCP_PORT:-8765}"
READY_URL="${BLOGRAG_MCP_READY_URL:-http://${MCP_HOST}:${MCP_PORT}/mcp}"

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    kill "${existing_pid}" 2>/dev/null || true
    sleep 1
    if kill -0 "${existing_pid}" 2>/dev/null; then
      kill -9 "${existing_pid}" 2>/dev/null || true
    fi
  fi
fi

nohup bash "${HTTP_SCRIPT}" </dev/null >"${LOG_FILE}" 2>&1 &
new_pid=$!
echo "${new_pid}" > "${PID_FILE}"

for _ in $(seq 1 50); do
  if ! kill -0 "${new_pid}" 2>/dev/null; then
    echo "blograg HTTP MCP server exited before becoming ready" >&2
    echo "Last log output:" >&2
    tail -n 50 "${LOG_FILE}" >&2 || true
    exit 1
  fi
  status_code="$(curl --silent --output /dev/null --write-out '%{http_code}' --max-time 1 "${READY_URL}" || true)"
  if [[ -n "${status_code}" && "${status_code}" != "000" ]]; then
    echo "Started blograg HTTP MCP server on PID ${new_pid}"
    echo "PID file: ${PID_FILE}"
    echo "Log file: ${LOG_FILE}"
    echo "Ready URL: ${READY_URL}"
    exit 0
  fi
  sleep 0.2
done

echo "blograg HTTP MCP server did not become ready in time" >&2
echo "Last log output:" >&2
tail -n 50 "${LOG_FILE}" >&2 || true
exit 1
