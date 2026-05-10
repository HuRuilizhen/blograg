"""CLI entrypoints for blograg."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Literal, cast

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from blograg.client_registration import register_client
from blograg.config import (
    ConceptExtractorMode,
    LabelFreeFallbackStrategy,
    LLMProvider,
    RetrievalStrategy,
    build_config,
    resolve_labelgen_cache_dir,
)
from blograg.indexing import BuildProgressUpdate, build_index, load_index
from blograg.mcp import create_mcp_server
from blograg.service_manager import (
    build_browser_url,
    build_health_url,
    build_server_url,
    derive_health_url,
    get_server_status,
    start_server,
    stop_server,
)
from blograg.user_config import (
    CLIConfig,
    ConfigPaths,
    ProviderSecrets,
    apply_provider_secret,
    config_value_map,
    config_value_map_all,
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
_console = Console()
_error_console = Console(stderr=True)
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
_RETRIEVAL_STRATEGY_OPTION = typer.Option(
    None,
    help=(
        "Advanced retrieval strategy override: "
        "greedy_label_coverage_semantic_rerank or label_gate_semantic_rank."
    ),
)
_LABEL_FREE_FALLBACK_STRATEGY_OPTION = typer.Option(
    None,
    help=(
        "Advanced label-free fallback strategy override: concept_overlap_only, "
        "concept_overlap_semantic_rerank, concept_gate_semantic_rank, or semantic_only."
    ),
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
    retrieval_strategy: RetrievalStrategy | None = _RETRIEVAL_STRATEGY_OPTION,
    label_free_fallback_strategy: LabelFreeFallbackStrategy | None = (
        _LABEL_FREE_FALLBACK_STRATEGY_OPTION
    ),
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
    resolved_retrieval_strategy = (
        retrieval_strategy
        if retrieval_strategy is not None
        else (
            persisted_config.retrieval.retrieval_strategy or "greedy_label_coverage_semantic_rerank"
        )
    )
    resolved_label_free_fallback_strategy = (
        label_free_fallback_strategy
        if label_free_fallback_strategy is not None
        else persisted_config.retrieval.label_free_fallback_strategy or "semantic_only"
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
        retrieval_strategy=resolved_retrieval_strategy,
        label_free_fallback_strategy=resolved_label_free_fallback_strategy,
    )
    started_at = time.monotonic()
    with _provider_secret_context(
        concept_extractor=resolved_concept_extractor,
        provider=resolved_llm_provider,
        api_key_env_var=resolved_llm_api_key_env_var,
        secrets=provider_secrets,
    ):
        progress_display = _BuildProgressDisplay()
        try:
            index = build_index(
                blog_dir=resolved_blog_dir,
                index_dir=resolved_index_dir,
                config=config,
                progress_callback=progress_display.update,
            )
        finally:
            progress_display.finish()
    build_seconds = time.monotonic() - started_at
    typer.echo("Build complete.")
    typer.echo(f"Paragraphs: {len(index.paragraph_records)}")
    typer.echo(f"Index directory: {(resolved_index_dir / 'blograg').resolve()}")
    typer.echo(f"Extractor: {resolved_concept_extractor}")
    if resolved_concept_extractor == "llm":
        typer.echo(f"LLM provider: {resolved_llm_provider}")
        if resolved_llm_model:
            typer.echo(f"LLM model: {resolved_llm_model}")
    typer.echo(f"Elapsed: {build_seconds:.1f}s")


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
    if resolved_transport == "streamable-http":
        _quiet_streamable_http_manager_logs()
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
    typer.echo(f"MCP endpoint: {status.mcp_url}")
    typer.echo(f"Health endpoint: {status.health_url}")
    typer.echo(f"Open {build_browser_url(host=resolved_host, port=resolved_port)} in your browser.")


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
    resolved_mcp_url = url or build_server_url(host=resolved_host, port=resolved_port)
    resolved_health_url = (
        derive_health_url(resolved_mcp_url)
        if url
        else build_health_url(
            host=resolved_host,
            port=resolved_port,
        )
    )
    observed_status = get_server_status(
        pid_file=resolved_pid_file,
        log_file=resolved_log_file,
        mcp_url=resolved_mcp_url,
        health_url=resolved_health_url,
    )
    rows = [
        ("Server", "running" if observed_status.process_running else "stopped"),
        ("PID", str(observed_status.pid) if observed_status.pid is not None else "missing"),
        ("MCP endpoint", observed_status.mcp_url),
        ("Health endpoint", observed_status.health_url),
        ("PID file", str(observed_status.pid_file)),
        ("Log file", str(observed_status.log_file)),
        ("HTTP status", "ready" if observed_status.http_ready else "not ready"),
    ]
    if observed_status.http_status_code is not None:
        rows.append(("HTTP code", str(observed_status.http_status_code)))
    rows.append(("Detail", observed_status.detail))
    _print_key_value_table("Status", rows)


@app.command()
def doctor() -> None:
    """Check local configuration, index readiness, service state, and client tooling."""

    persisted_config = load_cli_config()
    provider_secrets = load_provider_secrets()
    config_paths = get_config_paths()
    issues: list[str] = []

    doctor_rows: list[tuple[str, str, str, str]] = [
        (
            "Configuration",
            "Config file",
            _status_label(config_paths.config_path.is_file()),
            str(config_paths.config_path),
        ),
        (
            "Configuration",
            "Secrets file",
            _status_label(config_paths.secrets_path.is_file()),
            str(config_paths.secrets_path),
        ),
    ]
    if not config_paths.config_path.is_file():
        issues.append(f"Config file: {config_paths.config_path}")
    if not config_paths.secrets_path.is_file():
        issues.append(f"Secrets file: {config_paths.secrets_path}")

    default_index_dir = _coerce_path(persisted_config.default_index_dir)
    if default_index_dir is None:
        doctor_rows.append(("Index", "Default index", "WARN", "No default_index_dir configured."))
        issues.append("Default index: No default_index_dir configured.")
    else:
        doctor_rows.append(("Index", "Default index", "OK", str(default_index_dir)))
        index_issues = _validate_index_directory(default_index_dir)
        if index_issues:
            for issue in index_issues:
                doctor_rows.append(("Index", "Index artifact", "WARN", issue))
                issues.append(f"Index artifact: {issue}")
        else:
            doctor_rows.append(("Index", "Index artifacts", "OK", "Index looks complete."))

    resolved_host = persisted_config.serve.host or "127.0.0.1"
    resolved_port = persisted_config.serve.port or 8765
    status = get_server_status(
        pid_file=config_paths.pid_path,
        log_file=config_paths.log_path,
        mcp_url=build_server_url(host=resolved_host, port=resolved_port),
        health_url=build_health_url(host=resolved_host, port=resolved_port),
    )
    process_detail = f"PID {status.pid}" if status.pid is not None else "No PID file."
    doctor_rows.append(
        ("Service", "Managed process", _status_label(status.process_running), process_detail)
    )
    if not status.process_running:
        issues.append(f"Managed process: {process_detail}")
    doctor_rows.append(("Service", "HTTP health", _status_label(status.http_ready), status.detail))
    if not status.http_ready:
        issues.append(f"HTTP health: {status.detail}")

    for client in ("codex", "openclaw"):
        executable = shutil.which(client)
        detail = executable or "Not found on PATH."
        doctor_rows.append(
            (
                "Clients",
                f"{client} executable",
                _status_label(executable is not None),
                detail,
            )
        )
        if executable is None:
            issues.append(f"{client} executable: {detail}")

    extractor = persisted_config.build.concept_extractor or "heuristic"
    doctor_rows.append(("LLM", "Extractor mode", "OK", extractor))
    if extractor == "llm":
        model = persisted_config.build.llm_model
        provider = persisted_config.build.llm_provider or "mistral"
        env_var = persisted_config.build.llm_api_key_env_var or _default_api_key_env_var(provider)
        secret_present = _provider_secret_present(provider_secrets, provider)
        env_present = bool(env_var and os.environ.get(env_var))
        doctor_rows.append(("LLM", "LLM provider", "OK", provider))
        model_detail = model or "Missing build.llm_model."
        model_ok = model is not None
        doctor_rows.append(("LLM", "LLM model", _status_label(model_ok), model_detail))
        if not model_ok:
            issues.append(f"LLM model: {model_detail}")
        credential_ok = secret_present or env_present
        credential_detail = (
            f"Secret configured for {provider}."
            if secret_present
            else f"Environment variable {env_var} is set."
            if env_present
            else f"Missing secret or environment variable {env_var}."
        )
        doctor_rows.append(
            ("LLM", "LLM credential", _status_label(credential_ok), credential_detail)
        )
        if not credential_ok:
            issues.append(f"LLM credential: {credential_detail}")

    _print_doctor_table(doctor_rows)
    if issues:
        _error_console.print(f"Doctor found {len(issues)} issue(s).")
        raise typer.Exit(code=1)
    _console.print("Doctor found no issues.")


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
    _print_key_value_table(
        "Config paths",
        [
            ("Config dir", str(paths.config_dir)),
            ("Config file", str(paths.config_path)),
            ("Secrets file", str(paths.secrets_path)),
        ],
    )


@config_app.command("show")
def config_show(
    show_all: bool = typer.Option(
        False,
        "--all",
        help="Show all known config keys, including unset values and runtime defaults.",
    ),
) -> None:
    """Show persisted config values and masked secret state."""

    paths = get_config_paths()
    config = load_cli_config()
    secrets = load_provider_secrets()
    _print_key_value_table(
        "Config paths",
        [
            ("Config dir", str(paths.config_dir)),
            ("Config file", str(paths.config_path)),
            ("Secrets file", str(paths.secrets_path)),
        ],
    )

    values = config_value_map_all(config) if show_all else config_value_map(config)
    if values:
        _print_key_value_table("Config", list(values.items()))

    secret_rows = [
        (provider, "configured" if configured else "missing")
        for provider, configured in secret_status_map(secrets).items()
    ]
    _print_key_value_table("Secrets", secret_rows)


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
    paths = get_config_paths()
    _print_wizard_intro(paths)

    _print_wizard_step(
        "Step 1",
        "Paths",
        "Set the default blog source and index output directories.",
    )
    config.default_blog_dir = _prompt_optional_text(
        "Default blog directory",
        config.default_blog_dir,
    )
    config.default_index_dir = _prompt_optional_text(
        "Default index directory",
        config.default_index_dir,
    )

    _print_wizard_step(
        "Step 2",
        "Server",
        "Choose how the local MCP server should listen by default.",
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
        _prompt_choice(
            "Default MCP transport",
            ["streamable-http", "stdio"],
            config.serve.transport or "streamable-http",
        ),
    )

    _print_wizard_step(
        "Step 3",
        "Build defaults",
        "Choose how blograg should build new indexes by default.",
    )
    config.build.concept_extractor = cast(
        ConceptExtractorMode,
        _prompt_choice(
            "Default concept extractor",
            ["heuristic", "spacy", "llm"],
            config.build.concept_extractor or "heuristic",
        ),
    )
    config.build.labelgen_cache_dir = _prompt_optional_text(
        "Default labelgen cache directory",
        config.build.labelgen_cache_dir,
    )

    if config.build.concept_extractor == "llm":
        _print_wizard_step(
            "Step 4",
            "LLM",
            "Configure provider defaults and an optional local API key.",
        )
        config.build.llm_provider = cast(
            LLMProvider,
            _prompt_choice(
                "Default LLM provider",
                ["mistral", "openai", "qwen", "deepseek", "ollama"],
                config.build.llm_provider or "mistral",
            ),
        )
        config.build.llm_model = _prompt_optional_text(
            "Default LLM model",
            config.build.llm_model,
        )
        config.build.llm_base_url = _prompt_optional_text(
            "Default LLM base URL",
            config.build.llm_base_url,
        )
        config.build.llm_api_key_env_var = _prompt_optional_text(
            "LLM API key env var name",
            config.build.llm_api_key_env_var,
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
            _prompt_choice(
                "Default LLM output contract mode",
                ["auto", "json_schema", "json_object", "prompt_only"],
                config.build.llm_output_contract_mode or "auto",
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

    _print_wizard_summary(config, secrets)
    should_save = typer.confirm("Write these settings to disk?", default=True)
    if not should_save:
        typer.echo("Aborted without writing config files.")
        raise typer.Exit(code=1)

    save_cli_config(config)
    save_provider_secrets(secrets)
    typer.echo("Saved blograg config and secrets.")
    if config.default_blog_dir and config.default_index_dir:
        typer.echo("Next: run `blograg build`, then `blograg start`.")
    elif config.default_index_dir:
        typer.echo("Next: run `blograg start` when your index is ready.")
    else:
        typer.echo("Next: configure an index path or pass `--index-dir` when serving.")


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


def _parse_secret_provider(
    provider: str,
) -> Literal["openai", "mistral", "qwen", "ollama", "deepseek"]:
    if provider not in known_secret_providers():
        typer.echo(
            "Unsupported provider. Supported providers: " + ", ".join(known_secret_providers()),
            err=True,
        )
        raise typer.Exit(code=1)
    return cast(Literal["openai", "mistral", "qwen", "ollama", "deepseek"], provider)


def _blank_to_none(value: str) -> str | None:
    stripped = value.strip()
    if not stripped:
        return None
    return stripped


def _prompt_optional_text(label: str, current_value: str | None) -> str | None:
    return _blank_to_none(
        typer.prompt(
            label,
            default=current_value or "",
            show_default=bool(current_value),
        )
    )


def _prompt_choice(label: str, options: list[str], default_value: str) -> str:
    choice_table = Table(box=None, show_header=False, pad_edge=False, show_edge=False)
    choice_table.add_column(no_wrap=True, style="bold cyan", min_width=3)
    choice_table.add_column()
    for index, option in enumerate(options, start=1):
        if option == default_value:
            rendered_option = f"[bold cyan]{option}[/bold cyan] [dim](default)[/dim]"
        else:
            rendered_option = option
        choice_table.add_row(f"{index}.", rendered_option)
    _console.print(
        Panel(
            choice_table,
            title=Text(label, style="dim"),
            title_align="left",
            border_style="dim",
            box=box.ROUNDED,
        )
    )
    _console.print("[dim]Press Enter to accept the default.[/dim]")
    default_index = options.index(default_value) + 1
    raw_choice = typer.prompt("Select an option", default=str(default_index))
    try:
        choice_index = int(raw_choice)
    except ValueError as error:
        raise typer.BadParameter("Enter the number of one listed option.") from error
    if choice_index < 1 or choice_index > len(options):
        raise typer.BadParameter("Enter the number of one listed option.")
    return options[choice_index - 1]


def _print_wizard_intro(paths: ConfigPaths) -> None:
    lines = [
        "This wizard saves default blograg paths, server settings, and optional LLM credentials.",
        f"Config file: {paths.config_path}",
        f"Secrets file: {paths.secrets_path}",
    ]
    _console.print(
        Panel(
            "\n".join(lines),
            title=Text("blograg config wizard", style="dim"),
            title_align="left",
            border_style="dim",
            box=box.ROUNDED,
        )
    )


def _print_wizard_step(step: str, title: str, description: str) -> None:
    _console.print("")
    _console.print(
        Panel(
            description,
            title=Text(f"{step} · {title}", style="dim"),
            title_align="left",
            border_style="dim",
            box=box.ROUNDED,
        )
    )


def _print_wizard_summary(config: CLIConfig, secrets: ProviderSecrets) -> None:
    rows: list[tuple[str, str]] = [
        ("default_blog_dir", config.default_blog_dir or "unset"),
        ("default_index_dir", config.default_index_dir or "unset"),
        ("serve.host", config.serve.host or "unset"),
        ("serve.port", str(config.serve.port or "unset")),
        ("serve.transport", config.serve.transport or "unset"),
        ("build.concept_extractor", config.build.concept_extractor or "unset"),
        ("build.labelgen_cache_dir", config.build.labelgen_cache_dir or "unset"),
    ]
    if config.build.concept_extractor == "llm":
        rows.extend(
            [
                ("build.llm_provider", config.build.llm_provider or "unset"),
                ("build.llm_model", config.build.llm_model or "unset"),
                ("build.llm_base_url", config.build.llm_base_url or "unset"),
                ("build.llm_api_key_env_var", config.build.llm_api_key_env_var or "unset"),
                (
                    "build.llm_batch_size",
                    str(config.build.llm_batch_size or "unset"),
                ),
                (
                    "build.llm_max_concepts_per_paragraph",
                    str(config.build.llm_max_concepts_per_paragraph or "unset"),
                ),
                (
                    "build.llm_output_contract_mode",
                    config.build.llm_output_contract_mode or "unset",
                ),
            ]
        )
    _console.print("")
    _print_key_value_table("Wizard summary", rows)
    secret_rows = [
        (provider, "configured" if configured else "missing")
        for provider, configured in secret_status_map(secrets).items()
    ]
    _print_key_value_table("Secrets", secret_rows)


def _validate_index_directory(index_dir: Path) -> list[str]:
    required_paths = [
        index_dir / "blograg" / "manifest.json",
        index_dir / "blograg" / "paragraphs.json",
        index_dir / "blograg" / "labelrag",
    ]
    issues: list[str] = []
    for path in required_paths:
        if path.name == "labelrag":
            if not path.is_dir():
                issues.append(f"Missing artifact directory: {path}")
        elif not path.is_file():
            issues.append(f"Missing artifact file: {path}")
    return issues


def _status_label(ok: bool) -> str:
    return "OK" if ok else "WARN"


def _print_key_value_table(title: str, rows: list[tuple[str, str]]) -> None:
    table = Table(box=None, show_header=False, pad_edge=False, show_edge=False)
    table.add_column(style="bold cyan", no_wrap=True, min_width=12)
    table.add_column(overflow="fold")
    for field, value in rows:
        table.add_row(field, value)
    _console.print(
        Panel(
            table,
            title=Text(title, style="dim"),
            title_align="left",
            border_style="dim",
            box=box.ROUNDED,
        )
    )


def _print_doctor_table(rows: list[tuple[str, str, str, str]]) -> None:
    table = Table(box=None, header_style="bold", show_edge=False, pad_edge=False)
    table.add_column("Section", style="bold cyan", no_wrap=True)
    table.add_column("Check", style="bold cyan", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Detail", overflow="fold")
    for section, check, status, detail in rows:
        table.add_row(section, check, status, detail)
    _console.print(
        Panel(
            table,
            title=Text("Doctor", style="dim"),
            title_align="left",
            border_style="dim",
            box=box.ROUNDED,
        )
    )


def _provider_secret_present(secrets: ProviderSecrets, provider: LLMProvider) -> bool:
    return getattr(secrets, provider) is not None


def _default_api_key_env_var(provider: LLMProvider) -> str:
    return {
        "openai": "OPENAI_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "qwen": "DASHSCOPE_API_KEY",
        "ollama": "OLLAMA_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }[provider]


class _BuildProgressDisplay:
    """Minimal terminal progress display for local index builds."""

    def __init__(self) -> None:
        self._active = False

    def update(self, event: BuildProgressUpdate) -> None:
        if not sys.stderr.isatty():
            return
        stage_label = {
            "extract": "Extracting paragraphs",
            "embed": "Embedding paragraphs",
            "save": "Saving index",
        }[event.stage]
        current = ""
        if event.current_paragraph_id is not None:
            current = f" | current={event.current_paragraph_id}"
            if event.current_paragraph_text_preview:
                current += f" | {event.current_paragraph_text_preview}"
        line = f"\r{stage_label}: {event.processed}/{event.total}{current}"
        print(line, end="", file=sys.stderr, flush=True)
        self._active = True

    def finish(self) -> None:
        if self._active and sys.stderr.isatty():
            print(file=sys.stderr, flush=True)
        self._active = False


def _quiet_streamable_http_manager_logs() -> None:
    """Suppress noisy MCP transport lifecycle info logs while keeping uvicorn output intact."""

    logging.getLogger("mcp.server.streamable_http_manager").setLevel(
        max(logging.WARNING, logging.getLogger().level)
    )
