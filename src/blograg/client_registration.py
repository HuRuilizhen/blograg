"""External MCP client registration helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Literal, cast

ClientName = Literal["codex", "openclaw"]


@dataclass(slots=True, frozen=True)
class ClientRegistrationStatus:
    """Observed registration state for one MCP client."""

    client: ClientName
    configured: bool
    detail: str
    url: str | None = None


def register_client(*, client: ClientName, server_name: str, url: str) -> str:
    """Register one MCP server URL with one supported client."""

    executable = shutil.which(client)
    if executable is None:
        raise RuntimeError(f"Required client executable `{client}` was not found on PATH.")

    if client == "codex":
        subprocess.run(  # noqa: S603
            [executable, "mcp", "add", server_name, "--url", url],
            check=True,
        )
        return f"Registered `{server_name}` for codex using {url}."

    payload = json.dumps({"url": url}, separators=(",", ":"))
    subprocess.run(  # noqa: S603
        [executable, "mcp", "set", server_name, payload],
        check=True,
    )
    return f"Registered `{server_name}` for openclaw using {url}."


def get_client_registration_status(
    *,
    client: ClientName,
    server_name: str,
) -> ClientRegistrationStatus:
    """Return whether one server name is configured for one supported client."""

    executable = shutil.which(client)
    if executable is None:
        return ClientRegistrationStatus(
            client=client,
            configured=False,
            detail=f"`{client}` executable not found on PATH.",
        )

    if client == "codex":
        command = [executable, "mcp", "get", server_name, "--json"]
    else:
        command = [executable, "mcp", "show", server_name, "--json"]

    result = subprocess.run(  # noqa: S603
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (
            result.stderr.strip() or result.stdout.strip() or f"`{server_name}` is not configured."
        )
        return ClientRegistrationStatus(
            client=client,
            configured=False,
            detail=detail,
        )

    try:
        payload = cast(dict[str, Any], json.loads(result.stdout))
    except json.JSONDecodeError:
        detail = result.stdout.strip() or "Configured, but client output was not valid JSON."
        return ClientRegistrationStatus(
            client=client,
            configured=True,
            detail=detail,
        )

    if client == "codex":
        url = payload.get("url")
    else:
        mcp_servers = payload.get("mcpServers")
        if isinstance(mcp_servers, dict):
            typed_servers = cast(dict[str, Any], mcp_servers)
            server_payload = typed_servers.get(server_name)
            if isinstance(server_payload, dict):
                typed_payload = cast(dict[str, Any], server_payload)
                url = typed_payload.get("url")
            else:
                url = None
        else:
            url = None

    detail = f"Configured for {url}." if isinstance(url, str) and url else "Configured."
    return ClientRegistrationStatus(
        client=client,
        configured=True,
        detail=detail,
        url=url if isinstance(url, str) else None,
    )
