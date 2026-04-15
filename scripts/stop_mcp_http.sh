#!/usr/bin/env bash

set -euo pipefail

PID_FILE="${BLOGRAG_MCP_PID_FILE:-/tmp/blograg-mcp-http.pid}"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "No PID file found at ${PID_FILE}"
  exit 0
fi

pid="$(cat "${PID_FILE}")"

if [[ -z "${pid}" ]]; then
  rm -f "${PID_FILE}"
  echo "Removed empty PID file ${PID_FILE}"
  exit 0
fi

if ! kill -0 "${pid}" 2>/dev/null; then
  rm -f "${PID_FILE}"
  echo "No running blograg HTTP MCP server found for PID ${pid}"
  exit 0
fi

kill "${pid}" 2>/dev/null || true
sleep 1

if kill -0 "${pid}" 2>/dev/null; then
  kill -9 "${pid}" 2>/dev/null || true
fi

rm -f "${PID_FILE}"
echo "Stopped blograg HTTP MCP server (PID ${pid})"
