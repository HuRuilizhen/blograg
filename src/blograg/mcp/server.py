"""FastMCP server construction for blograg."""

from __future__ import annotations

from dataclasses import asdict

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

from blograg.indexing import BlogRAGIndex
from blograg.web import render_status_page


def create_mcp_server(
    index: BlogRAGIndex,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> FastMCP:
    """Create a minimal MCP server for one loaded blograg index."""

    mcp_url = f"http://{host}:{port}/mcp"
    server = FastMCP(
        name="blograg",
        instructions=(
            "Retrieve heading-delimited paragraphs from one local Jekyll-style blog index."
        ),
        host=host,
        port=port,
    )

    @server.tool(
        name="retrieve_paragraphs",
        description=(
            "Retrieve structured paragraph results from the loaded blog index for one query."
        ),
        structured_output=True,
    )
    def retrieve_paragraphs(  # pyright: ignore[reportUnusedFunction]
        query: str, top_k: int = 5
    ) -> list[dict[str, object]]:
        """Retrieve paragraph-first MCP results from the loaded index."""

        return [asdict(result) for result in index.retrieve_paragraphs(query=query, top_k=top_k)]

    @server.custom_route("/", methods=["GET"])
    async def root_page(request: Request) -> Response:  # pyright: ignore[reportUnusedFunction]
        del request
        return HTMLResponse(render_status_page(index=index, host=host, port=port, mcp_url=mcp_url))

    @server.custom_route("/healthz", methods=["GET"])
    async def healthz(request: Request) -> Response:  # pyright: ignore[reportUnusedFunction]
        del request
        return JSONResponse(
            {
                "status": "ok",
                "service": "blograg",
                "mcp_url": mcp_url,
                "index_dir": str(index.index_dir),
                "paragraph_count": len(index.paragraph_records),
                "tools": ["retrieve_paragraphs"],
            }
        )

    return server
