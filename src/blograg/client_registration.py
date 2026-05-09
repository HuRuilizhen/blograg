"""External MCP client registration helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Literal

ClientName = Literal["codex", "openclaw"]


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
