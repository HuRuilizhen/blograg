"""CLI entrypoints for blograg."""

from __future__ import annotations

import subprocess
from contextlib import nullcontext
from pathlib import Path
from typing import Literal, cast

import typer

from blograg.client_registration import register_client
from blograg.config import (
    ConceptExtractorMode,
    LLMProvider,
    build_config,
    resolve_labelgen_cache_dir,
)
from blograg.indexing import build_index, load_index
from blograg.mcp import create_mcp_server
from blograg.service_manager import build_server_url, get_server_status, start_server, stop_server
from blograg.user_config import (
    ProviderSecrets,
    apply_provider_secret,
    config_value_map,
    get_config_paths,
    known_secret_providers,
    load_cli_config,
    load_provider_secrets,
    save_cli_config,
    save_provider_secrets,
    secret_status_map,
    set_config_value,
    set_secret_value,
    unset_config_value,
    unset_secret_value,
)

app = typer.Typer(help="Build and serve a local Jekyll-blog paragraph retriever.")
config_app = typer.Typer(help="Manage persistent blograg defaults and local secrets.")
_BLOG_DIR_OPTION = typer.Option(None, file_okay=False, dir_okay=True)
_INDEX_DIR_BUILD_OPTION = typer.Option(None, file_okay=False, dir_okay=True)
_INDEX_DIR_SERVE_OPTION = typer.Option(None, file_okay=False, dir_okay=True)
_CONCEPT_EXTRACTOR_OPTION = typer.Option(
    None,
    help="Concept extraction mode to use during build: spacy, heuristic, or llm.",
)
_LLM_PROVIDER_OPTION = typer.Option(
    None,
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
    None,
    help="MCP transport to use: streamable-http or stdio.",
)
_HOST_OPTION = typer.Option(
    None,
    help="Host to bind for HTTP MCP transport.",
)
_PORT_OPTION = typer.Option(
    None,
    min=1,
    max=65535,
    help="Port to bind for HTTP MCP transport.",
)
_PID_FILE_OPTION = typer.Option(
    None,
    file_okay=True,
    dir_okay=False,
    help="Optional PID file override for managed background service commands.",
)
_LOG_FILE_OPTION = typer.Option(
    None,
    file_okay=True,
    dir_okay=False,
    help="Optional log file override for managed background service commands.",
)
_LLM_BATCH_SIZE_OPTION = typer.Option(
    None,
    min=1,
    help="Paragraph batch size per LLM extraction request.",
)
_LLM_MAX_CONCEPTS_OPTION = typer.Option(
    None,
    min=1,
    help="Maximum extracted concepts per paragraph in LLM mode.",
)
_LLM_OUTPUT_CONTRACT_OPTION = typer.Option(
    None,
    help="Preferred structured-output contract mode for LLM extraction.",
)
_API_KEY_VALUE_OPTION = typer.Option(
    None,
    "--api-key",
    help="Provider API key. Omit to enter it interactively.",
    hide_input=True,
)
_FORCE_RESTART_OPTION = typer.Option(
    False,
    "--force-restart",
    help="Stop an existing managed server before starting a new one.",
)
_REGISTER_CLIENT_OPTION = typer.Option(
    ...,
    help="MCP client to register: codex, openclaw, or both.",
)
app.add_typer(config_app, name="config")


@app.command()
def build(
    blog_dir: Path | None = _BLOG_DIR_OPTION,
    index_dir: Path | None = _INDEX_DIR_BUILD_OPTION,
    concept_extractor: ConceptExtractorMode | None = _CONCEPT_EXTRACTOR_OPTION,
    llm_provider: LLMProvider | None = _LLM_PROVIDER_OPTION,
    llm_model: str | None = _LLM_MODEL_OPTION,
    llm_base_url: str | None = _LLM_BASE_URL_OPTION,
    llm_api_key_env_var: str | None = _LLM_API_KEY_ENV_VAR_OPTION,
    labelgen_cache_dir: str | None = _LABELGEN_CACHE_DIR_OPTION,
    llm_batch_size: int | None = _LLM_BATCH_SIZE_OPTION,
    llm_max_concepts_per_paragraph: int | None = _LLM_MAX_CONCEPTS_OPTION,
    llm_output_contract_mode: Literal["auto", "json_schema", "json_object", "prompt_only"]
    | None = _LLM_OUTPUT_CONTRACT_OPTION,
) -> None:
    """Build a fresh local index from one blog directory."""

    persisted_config = load_cli_config()
    provider_secrets = load_provider_secrets()
    resolved_blog_dir = _require_existing_directory(
        blog_dir or _coerce_path(persisted_config.default_blog_dir),
        option_name="blog-dir",
        guidance="Provide `--blog-dir` or configure `default_blog_dir` first.",
    )
    resolved_index_dir = _require_directory_path(
        index_dir or _coerce_path(persisted_config.default_index_dir),
        option_name="index-dir",
        guidance="Provide `--index-dir` or configure `default_index_dir` first.",
        must_exist=False,
    )
    resolved_concept_extractor = (
        concept_extractor or persisted_config.build.concept_extractor or "heuristic"
    )
    resolved_llm_provider = llm_provider or persisted_config.build.llm_provider or "mistral"
    resolved_llm_model = llm_model if llm_model is not None else persisted_config.build.llm_model
    resolved_llm_base_url = (
        llm_base_url if llm_base_url is not None else persisted_config.build.llm_base_url
    )
    resolved_llm_api_key_env_var = (
        llm_api_key_env_var
        if llm_api_key_env_var is not None
        else persisted_config.build.llm_api_key_env_var
    )
    resolved_labelgen_cache_dir = (
        labelgen_cache_dir
        if labelgen_cache_dir is not None
        else persisted_config.build.labelgen_cache_dir
    )
    resolved_llm_batch_size = (
        llm_batch_size if llm_batch_size is not None else persisted_config.build.llm_batch_size or 8
    )
    resolved_llm_max_concepts = (
        llm_max_concepts_per_paragraph
        if llm_max_concepts_per_paragraph is not None
        else persisted_config.build.llm_max_concepts_per_paragraph or 12
    )
    resolved_llm_output_contract_mode = (
        llm_output_contract_mode
        if llm_output_contract_mode is not None
        else persisted_config.build.llm_output_contract_mode or "auto"
    )
    if resolved_concept_extractor == "llm" and not resolved_llm_model:
        typer.echo(
            "LLM extraction requires a configured model name. "
            "Provide `--llm-model` or configure `build.llm_model` first.",
            err=True,
        )
        raise typer.Exit(code=1)

    config = build_config(
        concept_extractor=resolved_concept_extractor,
        llm_provider=resolved_llm_provider,
        llm_model=resolved_llm_model,
        llm_base_url=resolved_llm_base_url,
        llm_api_key_env_var=resolved_llm_api_key_env_var,
        labelgen_cache_dir=resolved_labelgen_cache_dir,
        llm_batch_size=resolved_llm_batch_size,
        llm_max_concepts_per_paragraph=resolved_llm_max_concepts,
        llm_output_contract_mode=resolved_llm_output_contract_mode,
    )
    with _provider_secret_context(
        concept_extractor=resolved_concept_extractor,
        provider=resolved_llm_provider,
        api_key_env_var=resolved_llm_api_key_env_var,
        secrets=provider_secrets,
    ):
        index = build_index(blog_dir=resolved_blog_dir, index_dir=resolved_index_dir, config=config)
        typer.echo(
            f"Built blograg index with {len(index.paragraph_records)} paragraphs at "
            f"{(resolved_index_dir / 'blograg').resolve()}"
        )


@app.command()
def serve(
    index_dir: Path | None = _INDEX_DIR_SERVE_OPTION,
    labelgen_cache_dir: str | None = _LABELGEN_CACHE_DIR_OPTION,
    transport: Literal["streamable-http", "stdio"] | None = _TRANSPORT_OPTION,
    host: str | None = _HOST_OPTION,
    port: int | None = _PORT_OPTION,
) -> None:
    """Load an existing local index and start the MCP server."""

    persisted_config = load_cli_config()
    provider_secrets = load_provider_secrets()
    resolved_index_dir = _require_directory_path(
        index_dir or _coerce_path(persisted_config.default_index_dir),
        option_name="index-dir",
        guidance="Provide `--index-dir` or configure `default_index_dir` first.",
        must_exist=False,
    )
    resolved_transport = transport or persisted_config.serve.transport or "streamable-http"
    resolved_host = host or persisted_config.serve.host or "127.0.0.1"
    resolved_port = port or persisted_config.serve.port or 8765
    try:
        index = load_index(index_dir=resolved_index_dir)
    except (FileNotFoundError, RuntimeError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    cache_dir = resolve_labelgen_cache_dir(labelgen_cache_dir)
    if cache_dir is not None:
        index.pipeline.config.labelgen.extraction.llm.cache_dir = cache_dir
    server = create_mcp_server(index, host=resolved_host, port=resolved_port)
    labelgen_config = index.pipeline.config.labelgen
    llm_config = labelgen_config.extraction.llm
    resolved_extractor_mode_method = getattr(labelgen_config, "resolved_extractor_mode", None)
    resolved_extractor_mode = (
        resolved_extractor_mode_method()
        if callable(resolved_extractor_mode_method)
        else "heuristic"
    )
    with _provider_secret_context(
        concept_extractor=cast(ConceptExtractorMode, resolved_extractor_mode),
        provider=cast(LLMProvider, getattr(llm_config, "provider", "mistral")),
        api_key_env_var=getattr(llm_config, "api_key_env_var", None),
        secrets=provider_secrets,
    ):
        server.run(transport=resolved_transport)


@app.command()
def start(
    index_dir: Path | None = _INDEX_DIR_SERVE_OPTION,
    transport: Literal["streamable-http", "stdio"] | None = _TRANSPORT_OPTION,
    host: str | None = _HOST_OPTION,
    port: int | None = _PORT_OPTION,
    pid_file: Path | None = _PID_FILE_OPTION,
    log_file: Path | None = _LOG_FILE_OPTION,
    force_restart: bool = _FORCE_RESTART_OPTION,
) -> None:
    """Start the MCP server in the background and wait for readiness."""

    persisted_config = load_cli_config()
    config_paths = get_config_paths()
    resolved_index_dir = _require_directory_path(
        index_dir or _coerce_path(persisted_config.default_index_dir),
        option_name="index-dir",
        guidance="Provide `--index-dir` or configure `default_index_dir` first.",
        must_exist=False,
    )
    resolved_transport = transport or persisted_config.serve.transport or "streamable-http"
    resolved_host = host or persisted_config.serve.host or "127.0.0.1"
    resolved_port = port or persisted_config.serve.port or 8765
    resolved_pid_file = pid_file or config_paths.pid_path
    resolved_log_file = log_file or config_paths.log_path

    try:
        status = start_server(
            index_dir=resolved_index_dir,
            host=resolved_host,
            port=resolved_port,
            transport=resolved_transport,
            pid_file=resolved_pid_file,
            log_file=resolved_log_file,
            config_dir=config_paths.config_dir,
            force_restart=force_restart,
        )
    except RuntimeError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Started blograg server (PID {status.pid}).")
    typer.echo(f"PID file: {status.pid_file}")
    typer.echo(f"Log file: {status.log_file}")
    typer.echo(f"URL: {status.url}")


@app.command()
def stop(
    pid_file: Path | None = _PID_FILE_OPTION,
) -> None:
    """Stop the managed background MCP server."""

    config_paths = get_config_paths()
    resolved_pid_file = pid_file or config_paths.pid_path
    message = stop_server(pid_file=resolved_pid_file)
    typer.echo(message)


@app.command()
def status(
    pid_file: Path | None = _PID_FILE_OPTION,
    log_file: Path | None = _LOG_FILE_OPTION,
    host: str | None = _HOST_OPTION,
    port: int | None = _PORT_OPTION,
    url: str | None = typer.Option(None, help="Optional MCP URL override."),
) -> None:
    """Report managed process state and MCP HTTP readiness."""

    persisted_config = load_cli_config()
    config_paths = get_config_paths()
    resolved_pid_file = pid_file or config_paths.pid_path
    resolved_log_file = log_file or config_paths.log_path
    resolved_host = host or persisted_config.serve.host or "127.0.0.1"
    resolved_port = port or persisted_config.serve.port or 8765
    resolved_url = url or build_server_url(host=resolved_host, port=resolved_port)
    observed_status = get_server_status(
        pid_file=resolved_pid_file,
        log_file=resolved_log_file,
        url=resolved_url,
    )

    typer.echo(f"pid_file={observed_status.pid_file}")
    typer.echo(f"log_file={observed_status.log_file}")
    typer.echo(f"url={observed_status.url}")
    typer.echo(f"pid={observed_status.pid if observed_status.pid is not None else 'missing'}")
    typer.echo(f"process_running={'yes' if observed_status.process_running else 'no'}")
    typer.echo(f"http_ready={'yes' if observed_status.http_ready else 'no'}")
    if observed_status.http_status_code is not None:
        typer.echo(f"http_status={observed_status.http_status_code}")
    typer.echo(f"detail={observed_status.detail}")


@app.command()
def register(
    client: Literal["codex", "openclaw", "both"] = _REGISTER_CLIENT_OPTION,
    server_name: str = typer.Option("blograg", help="MCP server name to register."),
    host: str | None = _HOST_OPTION,
    port: int | None = _PORT_OPTION,
    url: str | None = typer.Option(None, help="Optional MCP URL override."),
) -> None:
    """Register the MCP endpoint with Codex and/or OpenClaw."""

    persisted_config = load_cli_config()
    resolved_host = host or persisted_config.serve.host or "127.0.0.1"
    resolved_port = port or persisted_config.serve.port or 8765
    resolved_url = url or build_server_url(host=resolved_host, port=resolved_port)
    clients = ["codex", "openclaw"] if client == "both" else [client]

    try:
        for current_client in clients:
            message = register_client(
                client=cast(Literal["codex", "openclaw"], current_client),
                server_name=server_name,
                url=resolved_url,
            )
            typer.echo(message)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error


@config_app.command("path")
def config_path() -> None:
    """Show user config and secret file locations."""

    paths = get_config_paths()
    typer.echo(f"config_dir={paths.config_dir}")
    typer.echo(f"config_file={paths.config_path}")
    typer.echo(f"secrets_file={paths.secrets_path}")


@config_app.command("show")
def config_show() -> None:
    """Show persisted config values and masked secret state."""

    paths = get_config_paths()
    config = load_cli_config()
    secrets = load_provider_secrets()
    typer.echo(f"config_dir={paths.config_dir}")
    typer.echo(f"config_file={paths.config_path}")
    typer.echo(f"secrets_file={paths.secrets_path}")

    values = config_value_map(config)
    if values:
        typer.echo("[config]")
        for key, value in values.items():
            typer.echo(f"{key} = {value}")

    typer.echo("[secrets]")
    for provider, configured in secret_status_map(secrets).items():
        typer.echo(f"{provider} = {'configured' if configured else 'missing'}")


@config_app.command("set")
def config_set(key: str, value: str) -> None:
    """Set one persisted config value."""

    config = load_cli_config()
    try:
        set_config_value(config, key, value)
    except (KeyError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    save_cli_config(config)
    typer.echo(f"Set `{key}`.")


@config_app.command("unset")
def config_unset(key: str) -> None:
    """Unset one persisted config value."""

    config = load_cli_config()
    try:
        unset_config_value(config, key)
    except KeyError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    save_cli_config(config)
    typer.echo(f"Unset `{key}`.")


@config_app.command("set-secret")
def config_set_secret(
    provider: str,
    api_key: str | None = _API_KEY_VALUE_OPTION,
) -> None:
    """Persist one provider API key locally."""

    normalized_provider = _parse_secret_provider(provider)
    resolved_api_key = api_key
    if resolved_api_key is None:
        resolved_api_key = typer.prompt(f"{normalized_provider} API key", hide_input=True)
    secrets = load_provider_secrets()
    set_secret_value(secrets, normalized_provider, resolved_api_key)
    save_provider_secrets(secrets)
    typer.echo(f"Stored secret for `{normalized_provider}`.")


@config_app.command("unset-secret")
def config_unset_secret(provider: str) -> None:
    """Remove one persisted provider API key."""

    normalized_provider = _parse_secret_provider(provider)
    secrets = load_provider_secrets()
    unset_secret_value(secrets, normalized_provider)
    save_provider_secrets(secrets)
    typer.echo(f"Removed secret for `{normalized_provider}`.")


@config_app.command("wizard")
def config_wizard() -> None:
    """Interactively initialize blograg defaults and local secrets."""

    config = load_cli_config()
    secrets = load_provider_secrets()
    config.default_blog_dir = _blank_to_none(
        typer.prompt(
            "Default blog directory",
            default=config.default_blog_dir or "",
            show_default=bool(config.default_blog_dir),
        )
    )
    config.default_index_dir = _blank_to_none(
        typer.prompt(
            "Default index directory",
            default=config.default_index_dir or "",
            show_default=bool(config.default_index_dir),
        )
    )
    config.serve.host = _blank_to_none(
        typer.prompt(
            "Default MCP host",
            default=config.serve.host or "127.0.0.1",
        )
    )
    config.serve.port = int(
        typer.prompt(
            "Default MCP port",
            default=str(config.serve.port or 8765),
        )
    )
    config.serve.transport = cast(
        Literal["streamable-http", "stdio"],
        typer.prompt(
            "Default MCP transport",
            default=config.serve.transport or "streamable-http",
        ),
    )
    config.build.concept_extractor = cast(
        ConceptExtractorMode,
        typer.prompt(
            "Default concept extractor",
            default=config.build.concept_extractor or "heuristic",
        ),
    )
    config.build.labelgen_cache_dir = _blank_to_none(
        typer.prompt(
            "Default labelgen cache directory",
            default=config.build.labelgen_cache_dir or "",
            show_default=bool(config.build.labelgen_cache_dir),
        )
    )
    if config.build.concept_extractor == "llm":
        config.build.llm_provider = cast(
            LLMProvider,
            typer.prompt(
                "Default LLM provider",
                default=config.build.llm_provider or "mistral",
            ),
        )
        config.build.llm_model = _blank_to_none(
            typer.prompt(
                "Default LLM model",
                default=config.build.llm_model or "",
                show_default=bool(config.build.llm_model),
            )
        )
        config.build.llm_base_url = _blank_to_none(
            typer.prompt(
                "Default LLM base URL",
                default=config.build.llm_base_url or "",
                show_default=bool(config.build.llm_base_url),
            )
        )
        config.build.llm_api_key_env_var = _blank_to_none(
            typer.prompt(
                "LLM API key env var name",
                default=config.build.llm_api_key_env_var or "",
                show_default=bool(config.build.llm_api_key_env_var),
            )
        )
        config.build.llm_batch_size = int(
            typer.prompt(
                "Default LLM batch size",
                default=str(config.build.llm_batch_size or 8),
            )
        )
        config.build.llm_max_concepts_per_paragraph = int(
            typer.prompt(
                "Default max concepts per paragraph",
                default=str(config.build.llm_max_concepts_per_paragraph or 12),
            )
        )
        config.build.llm_output_contract_mode = cast(
            Literal["auto", "json_schema", "json_object", "prompt_only"],
            typer.prompt(
                "Default LLM output contract mode",
                default=config.build.llm_output_contract_mode or "auto",
            ),
        )
        provider = cast(str, config.build.llm_provider)
        provider_api_key = _blank_to_none(
            typer.prompt(
                f"{provider} API key",
                default="",
                show_default=False,
                hide_input=True,
            )
        )
        if provider_api_key is not None:
            set_secret_value(secrets, _parse_secret_provider(provider), provider_api_key)

    save_cli_config(config)
    save_provider_secrets(secrets)
    typer.echo("Saved blograg config and secrets.")


def _coerce_path(value: str | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser()


def _require_existing_directory(
    path: Path | None,
    *,
    option_name: str,
    guidance: str,
) -> Path:
    if path is None:
        typer.echo(guidance, err=True)
        raise typer.Exit(code=1)
    if not path.exists():
        typer.echo(f"`--{option_name}` directory does not exist: {path}", err=True)
        raise typer.Exit(code=1)
    if not path.is_dir():
        typer.echo(f"`--{option_name}` must be a directory: {path}", err=True)
        raise typer.Exit(code=1)
    return path


def _require_directory_path(
    path: Path | None,
    *,
    option_name: str,
    guidance: str,
    must_exist: bool,
) -> Path:
    if path is None:
        typer.echo(guidance, err=True)
        raise typer.Exit(code=1)
    if path.exists() and not path.is_dir():
        typer.echo(f"`--{option_name}` must be a directory: {path}", err=True)
        raise typer.Exit(code=1)
    if must_exist and not path.exists():
        typer.echo(f"`--{option_name}` directory does not exist: {path}", err=True)
        raise typer.Exit(code=1)
    return path


def _provider_secret_context(
    *,
    concept_extractor: ConceptExtractorMode,
    provider: LLMProvider,
    api_key_env_var: str | None,
    secrets: ProviderSecrets,
):
    if concept_extractor != "llm":
        return nullcontext()
    return apply_provider_secret(
        provider=provider,
        configured_env_var=api_key_env_var,
        secrets=secrets,
    )


def _parse_secret_provider(provider: str) -> Literal["openai", "mistral", "qwen", "ollama"]:
    if provider not in known_secret_providers():
        typer.echo(
            "Unsupported provider. Supported providers: " + ", ".join(known_secret_providers()),
            err=True,
        )
        raise typer.Exit(code=1)
    return cast(Literal["openai", "mistral", "qwen", "ollama"], provider)


def _blank_to_none(value: str) -> str | None:
    stripped = value.strip()
    if not stripped:
        return None
    return stripped
