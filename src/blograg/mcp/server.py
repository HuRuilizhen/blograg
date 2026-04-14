"""FastMCP server construction for blograg."""

from __future__ import annotations

from dataclasses import asdict

from mcp.server.fastmcp import FastMCP

from blograg.indexing import BlogRAGIndex


def create_mcp_server(index: BlogRAGIndex) -> FastMCP:
    """Create a minimal MCP server for one loaded blograg index."""

    server = FastMCP(
        name="blograg",
        instructions=(
            "Retrieve heading-delimited paragraphs from one local Jekyll-style blog index."
        ),
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

    return server
