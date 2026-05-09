"""Managed background process helpers for the blograg MCP server."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TransportMode = Literal["streamable-http", "stdio"]


@dataclass(slots=True, frozen=True)
class ServerStatus:
    """Observed process and HTTP readiness state for one managed server."""

    pid_file: Path
    log_file: Path
    url: str
    pid: int | None
    process_running: bool
    http_ready: bool
    http_status_code: int | None
    detail: str


def build_server_url(*, host: str, port: int) -> str:
    """Return the canonical MCP HTTP URL for one server binding."""

    return f"http://{host}:{port}/mcp"


def read_pid(pid_file: Path) -> int | None:
    """Read a PID file when it contains a valid process ID."""

    if not pid_file.is_file():
        return None
    raw_text = pid_file.read_text(encoding="utf-8").strip()
    if not raw_text:
        return None
    try:
        pid = int(raw_text)
    except ValueError:
        return None
    if pid <= 0:
        return None
    return pid


def is_process_running(pid: int) -> bool:
    """Return whether a process appears to be alive."""

    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def start_server(
    *,
    index_dir: Path,
    host: str,
    port: int,
    transport: TransportMode,
    pid_file: Path,
    log_file: Path,
    config_dir: Path | None,
    force_restart: bool,
    ready_timeout_seconds: float = 10.0,
) -> ServerStatus:
    """Start the blograg MCP server in the background and wait for readiness."""

    existing_pid = read_pid(pid_file)
    if existing_pid is not None and is_process_running(existing_pid):
        if not force_restart:
            raise RuntimeError(
                f"blograg server is already running with PID {existing_pid}. "
                "Use `blograg status` to inspect it or `blograg stop` first."
            )
        stop_server(pid_file=pid_file)
    elif existing_pid is not None:
        pid_file.unlink(missing_ok=True)

    pid_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "blograg",
        "serve",
        "--index-dir",
        str(index_dir),
        "--transport",
        transport,
        "--host",
        host,
        "--port",
        str(port),
    ]
    environment = os.environ.copy()
    if config_dir is not None:
        environment["BLOGRAG_CONFIG_DIR"] = str(config_dir)
    with log_file.open("ab") as log_handle:
        if os.name == "nt":
            creationflags = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
            )
            process = subprocess.Popen(  # noqa: S603
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
                creationflags=creationflags,
            )
        else:
            process = subprocess.Popen(  # noqa: S603
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
    pid_file.write_text(f"{process.pid}\n", encoding="utf-8")

    url = build_server_url(host=host, port=port)
    deadline = time.monotonic() + ready_timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            detail = _tail_log(log_file)
            raise RuntimeError(
                "blograg server exited before becoming ready."
                + (f" Last log output:\n{detail}" if detail else "")
            )
        status = get_server_status(pid_file=pid_file, log_file=log_file, url=url)
        if status.http_ready:
            return status
        time.sleep(0.2)

    status = get_server_status(pid_file=pid_file, log_file=log_file, url=url)
    raise RuntimeError(
        f"blograg server did not become ready in time for {url}."
        + (f" Last log output:\n{_tail_log(log_file)}" if log_file.is_file() else "")
    )


def stop_server(*, pid_file: Path) -> str:
    """Stop one managed background server."""

    pid = read_pid(pid_file)
    if pid is None:
        pid_file.unlink(missing_ok=True)
        return f"No running blograg server found at {pid_file}."
    if not is_process_running(pid):
        pid_file.unlink(missing_ok=True)
        return f"Removed stale PID file {pid_file}."

    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not is_process_running(pid):
            pid_file.unlink(missing_ok=True)
            return f"Stopped blograg server (PID {pid})."
        time.sleep(0.1)

    if hasattr(signal, "SIGKILL"):
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.2)
    pid_file.unlink(missing_ok=True)
    return f"Stopped blograg server (PID {pid}) after force kill."


def get_server_status(*, pid_file: Path, log_file: Path, url: str) -> ServerStatus:
    """Inspect one managed server from its PID file and expected URL."""

    pid = read_pid(pid_file)
    process_running = pid is not None and is_process_running(pid)
    http_ready, http_status_code, detail = probe_server(url)
    return ServerStatus(
        pid_file=pid_file,
        log_file=log_file,
        url=url,
        pid=pid,
        process_running=process_running,
        http_ready=http_ready,
        http_status_code=http_status_code,
        detail=detail,
    )


def probe_server(url: str) -> tuple[bool, int | None, str]:
    """Probe the MCP HTTP endpoint and report whether it looks ready."""

    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=1.0) as response:  # noqa: S310
            status_code = response.getcode()
            return status_code == 200, status_code, f"HTTP {status_code}"
    except HTTPError as error:
        if error.code in {400, 405}:
            return True, error.code, f"HTTP {error.code}"
        return False, error.code, f"HTTP {error.code}"
    except URLError as error:
        return False, None, str(error.reason)


def _tail_log(log_file: Path, max_lines: int = 20) -> str:
    """Return the last few log lines when available."""

    if not log_file.is_file():
        return ""
    lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])
