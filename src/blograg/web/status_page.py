"""Render browser-facing status pages for the local MCP server."""

from __future__ import annotations

import socket
from dataclasses import dataclass
from html import escape
from importlib.resources import files
from string import Template

from blograg.indexing import BlogRAGIndex


@dataclass(slots=True, frozen=True)
class StatusPageContext:
    """View model for the browser-facing status page."""

    bind_address: str
    health_url: str
    index_dir: str
    lan_endpoint: str | None
    local_endpoint: str
    mcp_url: str
    paragraph_count: int


def render_status_page(*, index: BlogRAGIndex, host: str, port: int, mcp_url: str) -> str:
    """Render the local browser-facing status page."""

    template = Template(_load_status_page_template())
    health_url = f"http://{host}:{port}/healthz"
    context = StatusPageContext(
        bind_address=escape(f"{host}:{port}"),
        health_url=escape(health_url),
        index_dir=escape(str(index.index_dir)),
        lan_endpoint=_resolve_lan_endpoint(host=host, port=port),
        local_endpoint=escape(f"http://127.0.0.1:{port}/mcp"),
        mcp_url=escape(mcp_url),
        paragraph_count=len(index.paragraph_records),
    )
    return template.substitute(
        bind_address=context.bind_address,
        health_url=context.health_url,
        index_dir=context.index_dir,
        lan_endpoint=context.lan_endpoint or "Not exposed on LAN",
        lan_visibility="available" if context.lan_endpoint is not None else "not-exposed",
        local_endpoint=context.local_endpoint,
        mcp_url=context.mcp_url,
        paragraph_count=context.paragraph_count,
    )


def _load_status_page_template() -> str:
    """Load the packaged status page template."""

    return files("blograg.web").joinpath("templates", "status.html").read_text(encoding="utf-8")


def _resolve_lan_endpoint(*, host: str, port: int) -> str | None:
    """Return a best-effort LAN endpoint when the server is not loopback-only."""

    normalized_host = host.strip().lower()
    if normalized_host in {"127.0.0.1", "localhost", "::1"}:
        return None
    if normalized_host == "0.0.0.0":
        lan_ip = _detect_primary_lan_ip()
        if lan_ip is None:
            return None
        return escape(f"http://{lan_ip}:{port}/mcp")
    return escape(f"http://{host}:{port}/mcp")


def _detect_primary_lan_ip() -> str | None:
    """Detect one likely primary LAN IPv4 address for browser display."""

    probe_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe_socket.connect(("10.255.255.255", 1))
        detected_ip = probe_socket.getsockname()[0]
    except OSError:
        return None
    finally:
        probe_socket.close()
    if detected_ip.startswith("127."):
        return None
    return detected_ip
