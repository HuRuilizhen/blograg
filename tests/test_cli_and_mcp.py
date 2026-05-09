"""Tests for the CLI and MCP server wiring."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import anyio
import pytest
from typer.testing import CliRunner

import blograg.cli
from blograg.cli import app
from blograg.config import BlogRAGConfig
from blograg.indexing import build_index
from blograg.mcp import create_mcp_server
from blograg.service_manager import ServerStatus
from blograg.user_config import (
    BuildDefaults,
    CLIConfig,
    ProviderSecrets,
    ServeDefaults,
    save_provider_secrets,
)
from blograg.user_config import save_cli_config as save_user_cli_config
from tests.testsupport import FakeEmbeddingProvider

runner = CliRunner()


def test_create_mcp_server_exposes_retrieve_paragraphs_tool(tmp_path: Path) -> None:
    blog_dir = _write_blog(
        tmp_path,
        {
            "_posts/2026-04-14-jekyll.md": (
                "---\n"
                "title: Jekyll Notes\n"
                "---\n"
                "\n"
                "## Front Matter\n"
                "Jekyll front matter is stored at the top of the post.\n"
            )
        },
    )
    index = build_index(
        blog_dir=blog_dir,
        index_dir=tmp_path / "index",
        config=BlogRAGConfig(),
        embedding_provider=FakeEmbeddingProvider(),
    )

    server = create_mcp_server(index)

    tools = anyio.run(server.list_tools)
    result = anyio.run(
        lambda: server.call_tool(
            "retrieve_paragraphs",
            {"query": "front matter", "top_k": 1},
        )
    )
    _, structured_payload = cast(tuple[object, dict[str, Any]], result)
    payload = cast(list[dict[str, Any]], structured_payload["result"])

    assert [tool.name for tool in tools] == ["retrieve_paragraphs"]
    assert payload[0]["paragraph_id"] == "jekyll::p001"
    assert payload[0]["trace"]["retrieval_strategy"]


def test_build_command_reports_written_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    @dataclass(slots=True)
    class _FakeIndex:
        paragraph_records: dict[str, object]

    built_with: dict[str, object] = {}

    def fake_build_index(*, blog_dir: Path, index_dir: Path, config: object) -> _FakeIndex:
        built_with["blog_dir"] = blog_dir
        built_with["index_dir"] = index_dir
        built_with["config"] = config
        return _FakeIndex(paragraph_records={"p1": object(), "p2": object()})

    monkeypatch.setattr(blograg.cli, "build_index", fake_build_index)
    (tmp_path / "blog").mkdir()

    result = runner.invoke(
        app,
        [
            "build",
            "--blog-dir",
            str(tmp_path / "blog"),
            "--index-dir",
            str(tmp_path / "index"),
        ],
    )

    assert result.exit_code == 0
    assert built_with["blog_dir"] == tmp_path / "blog"
    assert built_with["index_dir"] == tmp_path / "index"
    assert "Built blograg index with 2 paragraphs" in result.stdout


def test_build_command_can_select_llm_extraction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = {}

    @dataclass(slots=True)
    class _FakeIndex:
        paragraph_records: dict[str, object]

    def fake_build_index(*, blog_dir: Path, index_dir: Path, config: object) -> _FakeIndex:
        captured["blog_dir"] = blog_dir
        captured["index_dir"] = index_dir
        captured["config"] = config
        return _FakeIndex(paragraph_records={"p1": object()})

    monkeypatch.setattr(blograg.cli, "build_index", fake_build_index)
    (tmp_path / "blog").mkdir()

    result = runner.invoke(
        app,
        [
            "build",
            "--blog-dir",
            str(tmp_path / "blog"),
            "--index-dir",
            str(tmp_path / "index"),
            "--concept-extractor",
            "llm",
            "--llm-provider",
            "mistral",
            "--llm-model",
            "mistral-small",
            "--llm-base-url",
            "https://api.mistral.ai/v1/chat/completions",
            "--llm-api-key-env-var",
            "MISTRAL_API_KEY",
            "--labelgen-cache-dir",
            "/tmp/blograg-cache-cli",
        ],
    )

    config = cast(BlogRAGConfig, captured["config"])
    llm_config = config.labelrag_pipeline.labelgen.extraction.llm

    assert result.exit_code == 0
    assert config.labelrag_pipeline.labelgen.resolved_extractor_mode() == "llm"
    assert llm_config.provider == "mistral"
    assert llm_config.model == "mistral-small"
    assert llm_config.base_url == "https://api.mistral.ai/v1/chat/completions"
    assert llm_config.api_key_env_var == "MISTRAL_API_KEY"
    assert llm_config.cache_dir == "/tmp/blograg-cache-cli"


def test_build_command_can_select_spacy_extraction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    @dataclass(slots=True)
    class _FakeIndex:
        paragraph_records: dict[str, object]

    def fake_build_index(*, blog_dir: Path, index_dir: Path, config: object) -> _FakeIndex:
        captured["config"] = config
        return _FakeIndex(paragraph_records={"p1": object()})

    monkeypatch.setattr(blograg.cli, "build_index", fake_build_index)
    (tmp_path / "blog").mkdir()

    result = runner.invoke(
        app,
        [
            "build",
            "--blog-dir",
            str(tmp_path / "blog"),
            "--index-dir",
            str(tmp_path / "index"),
            "--concept-extractor",
            "spacy",
        ],
    )

    config = cast(BlogRAGConfig, captured["config"])
    assert result.exit_code == 0
    assert config.labelrag_pipeline.labelgen.resolved_extractor_mode() == "spacy"
    assert config.labelrag_pipeline.labelgen.use_nlp_extractor is True


def test_build_command_uses_persisted_defaults_and_secret(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    @dataclass(slots=True)
    class _FakeIndex:
        paragraph_records: dict[str, object]

    def fake_build_index(*, blog_dir: Path, index_dir: Path, config: object) -> _FakeIndex:
        captured["blog_dir"] = blog_dir
        captured["index_dir"] = index_dir
        captured["config"] = config
        captured["api_key"] = os.environ.get("MISTRAL_API_KEY")
        return _FakeIndex(paragraph_records={"p1": object()})

    monkeypatch.setenv("BLOGRAG_CONFIG_DIR", str(tmp_path / "config-root"))
    monkeypatch.setattr(blograg.cli, "build_index", fake_build_index)
    blog_dir = tmp_path / "blog"
    blog_dir.mkdir()
    save_user_cli_config(
        CLIConfig(
            default_blog_dir=str(blog_dir),
            default_index_dir=str(tmp_path / "index"),
            build=BuildDefaults(
                concept_extractor="llm",
                llm_provider="mistral",
                llm_model="mistral-small",
            ),
        )
    )
    save_provider_secrets(ProviderSecrets(mistral="secret-value"))

    result = runner.invoke(app, ["build"])

    config = cast(BlogRAGConfig, captured["config"])
    llm_config = config.labelrag_pipeline.labelgen.extraction.llm
    assert result.exit_code == 0
    assert captured["blog_dir"] == blog_dir
    assert captured["index_dir"] == tmp_path / "index"
    assert captured["api_key"] == "secret-value"
    assert llm_config.provider == "mistral"
    assert llm_config.model == "mistral-small"


def test_serve_command_loads_index_and_runs_stdio_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded_with: dict[str, Path] = {}
    run_arguments: dict[str, object] = {}
    fake_index = SimpleNamespace(
        pipeline=SimpleNamespace(
            config=SimpleNamespace(
                labelgen=SimpleNamespace(
                    extraction=SimpleNamespace(llm=SimpleNamespace(cache_dir=".labelgen-cache"))
                )
            )
        )
    )

    class _FakeServer:
        def run(self, *, transport: str) -> None:
            run_arguments["transport"] = transport

    def fake_load_index(*, index_dir: Path) -> object:
        loaded_with["index_dir"] = index_dir
        return fake_index

    def fake_create_mcp_server(index: object, *, host: str, port: int) -> _FakeServer:
        assert index is fake_index
        run_arguments["host"] = host
        run_arguments["port"] = port
        return _FakeServer()

    monkeypatch.setattr(blograg.cli, "load_index", fake_load_index)
    monkeypatch.setattr(blograg.cli, "create_mcp_server", fake_create_mcp_server)
    (tmp_path / "index").mkdir()

    result = runner.invoke(
        app,
        [
            "serve",
            "--index-dir",
            str(tmp_path / "index"),
        ],
    )

    assert result.exit_code == 0
    assert loaded_with["index_dir"] == tmp_path / "index"
    assert run_arguments["transport"] == "streamable-http"
    assert run_arguments["host"] == "127.0.0.1"
    assert run_arguments["port"] == 8765


def test_serve_command_uses_persisted_defaults_and_secret(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded_with: dict[str, Path] = {}
    run_arguments: dict[str, object] = {}
    fake_index = SimpleNamespace(
        pipeline=SimpleNamespace(
            config=SimpleNamespace(
                labelgen=SimpleNamespace(
                    resolved_extractor_mode=lambda: "llm",
                    extraction=SimpleNamespace(
                        llm=SimpleNamespace(
                            cache_dir=".labelgen-cache",
                            provider="mistral",
                            api_key_env_var=None,
                        )
                    ),
                )
            )
        )
    )

    class _FakeServer:
        def run(self, *, transport: str) -> None:
            run_arguments["transport"] = transport
            run_arguments["api_key"] = os.environ.get("MISTRAL_API_KEY")

    def fake_load_index(*, index_dir: Path) -> object:
        loaded_with["index_dir"] = index_dir
        return fake_index

    def fake_create_mcp_server(index: object, *, host: str, port: int) -> _FakeServer:
        assert index is fake_index
        run_arguments["host"] = host
        run_arguments["port"] = port
        return _FakeServer()

    monkeypatch.setenv("BLOGRAG_CONFIG_DIR", str(tmp_path / "config-root"))
    monkeypatch.setattr(blograg.cli, "load_index", fake_load_index)
    monkeypatch.setattr(blograg.cli, "create_mcp_server", fake_create_mcp_server)
    save_user_cli_config(
        CLIConfig(
            default_index_dir=str(tmp_path / "index"),
            serve=ServeDefaults(
                host="0.0.0.0",
                port=8877,
                transport="stdio",
            ),
        )
    )
    save_provider_secrets(ProviderSecrets(mistral="secret-value"))

    result = runner.invoke(app, ["serve"])

    assert result.exit_code == 0
    assert loaded_with["index_dir"] == tmp_path / "index"
    assert run_arguments["transport"] == "stdio"
    assert run_arguments["host"] == "0.0.0.0"
    assert run_arguments["port"] == 8877
    assert run_arguments["api_key"] == "secret-value"


def test_serve_command_applies_labelgen_cache_dir_from_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_arguments: dict[str, object] = {}
    fake_index = SimpleNamespace(
        pipeline=SimpleNamespace(
            config=SimpleNamespace(
                labelgen=SimpleNamespace(
                    extraction=SimpleNamespace(llm=SimpleNamespace(cache_dir=".labelgen-cache"))
                )
            )
        )
    )

    class _FakeServer:
        def run(self, *, transport: str) -> None:
            run_arguments["transport"] = transport

    def fake_load_index(*, index_dir: Path) -> object:
        return fake_index

    def fake_create_mcp_server(index: object, *, host: str, port: int) -> _FakeServer:
        assert index is fake_index
        run_arguments["host"] = host
        run_arguments["port"] = port
        return _FakeServer()

    monkeypatch.setenv("LABELGEN_CACHE_DIR", "/tmp/blograg-cache")
    monkeypatch.setattr(blograg.cli, "load_index", fake_load_index)
    monkeypatch.setattr(blograg.cli, "create_mcp_server", fake_create_mcp_server)
    (tmp_path / "index").mkdir()

    result = runner.invoke(
        app,
        [
            "serve",
            "--index-dir",
            str(tmp_path / "index"),
        ],
    )

    assert result.exit_code == 0
    assert fake_index.pipeline.config.labelgen.extraction.llm.cache_dir == "/tmp/blograg-cache"
    assert run_arguments["transport"] == "streamable-http"


def test_serve_command_prefers_environment_over_cli_for_labelgen_cache_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_arguments: dict[str, object] = {}
    fake_index = SimpleNamespace(
        pipeline=SimpleNamespace(
            config=SimpleNamespace(
                labelgen=SimpleNamespace(
                    extraction=SimpleNamespace(llm=SimpleNamespace(cache_dir=".labelgen-cache"))
                )
            )
        )
    )

    class _FakeServer:
        def run(self, *, transport: str) -> None:
            run_arguments["transport"] = transport

    def fake_load_index(*, index_dir: Path) -> object:
        return fake_index

    def fake_create_mcp_server(index: object, *, host: str, port: int) -> _FakeServer:
        assert index is fake_index
        run_arguments["host"] = host
        run_arguments["port"] = port
        return _FakeServer()

    monkeypatch.setenv("LABELGEN_CACHE_DIR", "/tmp/blograg-cache-env")
    monkeypatch.setattr(blograg.cli, "load_index", fake_load_index)
    monkeypatch.setattr(blograg.cli, "create_mcp_server", fake_create_mcp_server)
    (tmp_path / "index").mkdir()

    result = runner.invoke(
        app,
        [
            "serve",
            "--index-dir",
            str(tmp_path / "index"),
            "--labelgen-cache-dir",
            "/tmp/blograg-cache-cli",
        ],
    )

    assert result.exit_code == 0
    assert fake_index.pipeline.config.labelgen.extraction.llm.cache_dir == "/tmp/blograg-cache-env"
    assert run_arguments["transport"] == "streamable-http"


def test_serve_command_can_select_http_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_arguments: dict[str, object] = {}
    fake_index = SimpleNamespace(
        pipeline=SimpleNamespace(
            config=SimpleNamespace(
                labelgen=SimpleNamespace(
                    extraction=SimpleNamespace(llm=SimpleNamespace(cache_dir=".labelgen-cache"))
                )
            )
        )
    )

    class _FakeServer:
        def run(self, *, transport: str) -> None:
            run_arguments["transport"] = transport

    def fake_load_index(*, index_dir: Path) -> object:
        return fake_index

    def fake_create_mcp_server(index: object, *, host: str, port: int) -> _FakeServer:
        assert index is fake_index
        run_arguments["host"] = host
        run_arguments["port"] = port
        return _FakeServer()

    monkeypatch.setattr(blograg.cli, "load_index", fake_load_index)
    monkeypatch.setattr(blograg.cli, "create_mcp_server", fake_create_mcp_server)
    (tmp_path / "index").mkdir()

    result = runner.invoke(
        app,
        [
            "serve",
            "--index-dir",
            str(tmp_path / "index"),
            "--transport",
            "streamable-http",
            "--host",
            "127.0.0.1",
            "--port",
            "8877",
        ],
    )

    assert result.exit_code == 0
    assert run_arguments["transport"] == "streamable-http"
    assert run_arguments["host"] == "127.0.0.1"
    assert run_arguments["port"] == 8877


def test_config_commands_persist_values_and_mask_secrets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BLOGRAG_CONFIG_DIR", str(tmp_path / "config-root"))

    set_result = runner.invoke(app, ["config", "set", "default_index_dir", str(tmp_path / "index")])
    secret_result = runner.invoke(
        app,
        ["config", "set-secret", "mistral", "--api-key", "secret-value"],
    )
    show_result = runner.invoke(app, ["config", "show"])

    assert set_result.exit_code == 0
    assert secret_result.exit_code == 0
    assert show_result.exit_code == 0
    assert f"default_index_dir = {tmp_path / 'index'}" in show_result.stdout
    assert "mistral = configured" in show_result.stdout
    assert "secret-value" not in show_result.stdout


def test_start_command_uses_managed_runtime_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("BLOGRAG_CONFIG_DIR", str(tmp_path / "config-root"))
    save_user_cli_config(CLIConfig(default_index_dir=str(tmp_path / "index")))

    def fake_start_server(**kwargs: object) -> ServerStatus:
        captured.update(kwargs)
        return ServerStatus(
            pid_file=cast(Path, kwargs["pid_file"]),
            log_file=cast(Path, kwargs["log_file"]),
            url="http://127.0.0.1:8765/mcp",
            pid=12345,
            process_running=True,
            http_ready=True,
            http_status_code=405,
            detail="HTTP 405",
        )

    monkeypatch.setattr(blograg.cli, "start_server", fake_start_server)

    result = runner.invoke(app, ["start"])

    assert result.exit_code == 0
    assert captured["index_dir"] == tmp_path / "index"
    assert captured["pid_file"] == tmp_path / "config-root" / "server.pid"
    assert captured["log_file"] == tmp_path / "config-root" / "server.log"
    assert "Started blograg server (PID 12345)." in result.stdout


def test_stop_command_uses_managed_runtime_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("BLOGRAG_CONFIG_DIR", str(tmp_path / "config-root"))

    def fake_stop_server(*, pid_file: Path) -> str:
        captured["pid_file"] = pid_file
        return "Stopped blograg server (PID 12345)."

    monkeypatch.setattr(blograg.cli, "stop_server", fake_stop_server)

    result = runner.invoke(app, ["stop"])

    assert result.exit_code == 0
    assert captured["pid_file"] == tmp_path / "config-root" / "server.pid"
    assert "Stopped blograg server" in result.stdout


def test_status_command_reports_managed_runtime_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BLOGRAG_CONFIG_DIR", str(tmp_path / "config-root"))
    save_user_cli_config(CLIConfig(serve=ServeDefaults(host="0.0.0.0", port=8877)))

    def fake_get_server_status(*, pid_file: Path, log_file: Path, url: str) -> ServerStatus:
        assert pid_file == tmp_path / "config-root" / "server.pid"
        assert log_file == tmp_path / "config-root" / "server.log"
        assert url == "http://0.0.0.0:8877/mcp"
        return ServerStatus(
            pid_file=pid_file,
            log_file=log_file,
            url=url,
            pid=12345,
            process_running=True,
            http_ready=True,
            http_status_code=405,
            detail="HTTP 405",
        )

    monkeypatch.setattr(blograg.cli, "get_server_status", fake_get_server_status)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "process_running=yes" in result.stdout
    assert "http_ready=yes" in result.stdout
    assert "http_status=405" in result.stdout


def test_register_command_registers_both_clients(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BLOGRAG_CONFIG_DIR", str(tmp_path / "config-root"))
    save_user_cli_config(CLIConfig(serve=ServeDefaults(host="0.0.0.0", port=8877)))
    calls: list[tuple[str, str, str]] = []

    def fake_register_client(*, client: str, server_name: str, url: str) -> str:
        calls.append((client, server_name, url))
        return f"registered {client}"

    monkeypatch.setattr(blograg.cli, "register_client", fake_register_client)

    result = runner.invoke(app, ["register", "--client", "both", "--server-name", "blograg-local"])

    assert result.exit_code == 0
    assert calls == [
        ("codex", "blograg-local", "http://0.0.0.0:8877/mcp"),
        ("openclaw", "blograg-local", "http://0.0.0.0:8877/mcp"),
    ]
    assert "registered codex" in result.stdout
    assert "registered openclaw" in result.stdout


def test_register_command_reports_registration_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_register_client(*, client: str, server_name: str, url: str) -> str:
        del client, server_name, url
        raise RuntimeError("codex executable not found")

    monkeypatch.setattr(blograg.cli, "register_client", fake_register_client)

    result = runner.invoke(app, ["register", "--client", "codex"])

    assert result.exit_code == 1
    assert "codex executable not found" in result.stderr


def _write_blog(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create a temporary blog directory with the provided file contents."""

    blog_dir = tmp_path / "blog"
    for relative_path, content in files.items():
        destination = blog_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    return blog_dir
