"""CLI entrypoints for blograg."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer

from blograg.config import build_config, resolve_labelgen_cache_dir
from blograg.indexing import build_index, load_index
from blograg.mcp import create_mcp_server

app = typer.Typer(help="Build and serve a local Jekyll-blog paragraph retriever.")
_BLOG_DIR_OPTION = typer.Option(..., exists=True, file_okay=False, dir_okay=True)
_INDEX_DIR_BUILD_OPTION = typer.Option(..., file_okay=False, dir_okay=True)
_INDEX_DIR_SERVE_OPTION = typer.Option(..., file_okay=False, dir_okay=True)
_CONCEPT_EXTRACTOR_OPTION = typer.Option(
    "heuristic",
    help="Concept extraction mode to use during build: spacy, heuristic, or llm.",
)
_LLM_PROVIDER_OPTION = typer.Option(
    "mistral",
    help="LLM provider for concept extraction when --concept-extractor=llm.",
)
_LLM_MODEL_OPTION = typer.Option(
    None,
    help="Provider model name for LLM concept extraction.",
)
_LLM_BASE_URL_OPTION = typer.Option(
    None,
    help="Optional provider base URL or chat completions URL override.",
)
_LLM_API_KEY_ENV_VAR_OPTION = typer.Option(
    None,
    help="Optional API key environment variable override for LLM concept extraction.",
)
_LABELGEN_CACHE_DIR_OPTION = typer.Option(
    None,
    help=(
        "Optional cache directory override for provider-backed LLM concept extraction. "
        "Precedence: $LABELGEN_CACHE_DIR > --labelgen-cache-dir > upstream default."
    ),
)
_TRANSPORT_OPTION = typer.Option(
    "streamable-http",
    help="MCP transport to use: streamable-http or stdio.",
)
_HOST_OPTION = typer.Option(
    "127.0.0.1",
    help="Host to bind for HTTP MCP transport.",
)
_PORT_OPTION = typer.Option(
    8765,
    min=1,
    max=65535,
    help="Port to bind for HTTP MCP transport.",
)
_LLM_BATCH_SIZE_OPTION = typer.Option(
    8,
    min=1,
    help="Paragraph batch size per LLM extraction request.",
)
_LLM_MAX_CONCEPTS_OPTION = typer.Option(
    12,
    min=1,
    help="Maximum extracted concepts per paragraph in LLM mode.",
)
_LLM_OUTPUT_CONTRACT_OPTION = typer.Option(
    "auto",
    help="Preferred structured-output contract mode for LLM extraction.",
)


@app.command()
def build(
    blog_dir: Path = _BLOG_DIR_OPTION,
    index_dir: Path = _INDEX_DIR_BUILD_OPTION,
    concept_extractor: Literal["spacy", "heuristic", "llm"] = _CONCEPT_EXTRACTOR_OPTION,
    llm_provider: Literal["openai", "mistral", "qwen", "ollama"] = _LLM_PROVIDER_OPTION,
    llm_model: str | None = _LLM_MODEL_OPTION,
    llm_base_url: str | None = _LLM_BASE_URL_OPTION,
    llm_api_key_env_var: str | None = _LLM_API_KEY_ENV_VAR_OPTION,
    labelgen_cache_dir: str | None = _LABELGEN_CACHE_DIR_OPTION,
    llm_batch_size: int = _LLM_BATCH_SIZE_OPTION,
    llm_max_concepts_per_paragraph: int = _LLM_MAX_CONCEPTS_OPTION,
    llm_output_contract_mode: Literal[
        "auto", "json_schema", "json_object", "prompt_only"
    ] = _LLM_OUTPUT_CONTRACT_OPTION,
) -> None:
    """Build a fresh local index from one blog directory."""

    config = build_config(
        concept_extractor=concept_extractor,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key_env_var=llm_api_key_env_var,
        labelgen_cache_dir=labelgen_cache_dir,
        llm_batch_size=llm_batch_size,
        llm_max_concepts_per_paragraph=llm_max_concepts_per_paragraph,
        llm_output_contract_mode=llm_output_contract_mode,
    )
    index = build_index(blog_dir=blog_dir, index_dir=index_dir, config=config)
    typer.echo(
        f"Built blograg index with {len(index.paragraph_records)} paragraphs at "
        f"{(index_dir / 'blograg').resolve()}"
    )


@app.command()
def serve(
    index_dir: Path = _INDEX_DIR_SERVE_OPTION,
    labelgen_cache_dir: str | None = _LABELGEN_CACHE_DIR_OPTION,
    transport: Literal["streamable-http", "stdio"] = _TRANSPORT_OPTION,
    host: str = _HOST_OPTION,
    port: int = _PORT_OPTION,
) -> None:
    """Load an existing local index and start the MCP server."""

    try:
        index = load_index(index_dir=index_dir)
    except (FileNotFoundError, RuntimeError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    cache_dir = resolve_labelgen_cache_dir(labelgen_cache_dir)
    if cache_dir is not None:
        index.pipeline.config.labelgen.extraction.llm.cache_dir = cache_dir
    server = create_mcp_server(index, host=host, port=port)
    server.run(transport=transport)
