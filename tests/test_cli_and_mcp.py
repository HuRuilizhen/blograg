"""Tests for the CLI and MCP server wiring."""

from __future__ import annotations

import os
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import anyio
import pytest
from starlette.testclient import TestClient
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
    RetrievalDefaults,
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
    assert payload[0]["trace"]["score_kind"]


def test_create_mcp_server_exposes_browser_status_routes(tmp_path: Path) -> None:
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
    server = create_mcp_server(index, host="127.0.0.1", port=8765)

    with TestClient(server.streamable_http_app()) as client:
        root_response = client.get("/")
        health_response = client.get("/healthz")

    assert root_response.status_code == 200
    assert "<title>blograg</title>" in root_response.text
    assert "blograg MCP server" in root_response.text
    assert "http://127.0.0.1:8765/mcp" in root_response.text
    assert str(index.index_dir) in root_response.text
    assert "Health endpoint" in root_response.text
    assert "retrieve_paragraphs" in root_response.text
    assert "query: string" in root_response.text
    assert "top_k: integer" in root_response.text
    assert "Quick actions" not in root_response.text

    assert health_response.status_code == 200
    assert health_response.json() == {
        "status": "ok",
        "service": "blograg",
        "mcp_url": "http://127.0.0.1:8765/mcp",
        "index_dir": str(index.index_dir),
        "paragraph_count": 1,
        "tools": ["retrieve_paragraphs"],
    }


def test_build_command_reports_written_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    @dataclass(slots=True)
    class _FakeIndex:
        paragraph_records: dict[str, object]

    built_with: dict[str, object] = {}

    def fake_build_index(
        *,
        blog_dir: Path,
        index_dir: Path,
        config: object,
        progress_callback: object | None = None,
    ) -> _FakeIndex:
        del progress_callback
        built_with["blog_dir"] = blog_dir
        built_with["index_dir"] = index_dir
        built_with["config"] = config
        return _FakeIndex(paragraph_records={"p1": object(), "p2": object()})

    monkeypatch.setenv("BLOGRAG_CONFIG_DIR", str(tmp_path / "config-root"))
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
    assert "Build complete." in result.stdout
    assert "Paragraphs: 2" in result.stdout
    assert "Extractor: heuristic" in result.stdout


def test_build_command_can_select_llm_extraction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = {}

    @dataclass(slots=True)
    class _FakeIndex:
        paragraph_records: dict[str, object]

    def fake_build_index(
        *,
        blog_dir: Path,
        index_dir: Path,
        config: object,
        progress_callback: object | None = None,
    ) -> _FakeIndex:
        del progress_callback
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

    def fake_build_index(
        *,
        blog_dir: Path,
        index_dir: Path,
        config: object,
        progress_callback: object | None = None,
    ) -> _FakeIndex:
        del blog_dir, index_dir, progress_callback
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

    def fake_build_index(
        *,
        blog_dir: Path,
        index_dir: Path,
        config: object,
        progress_callback: object | None = None,
    ) -> _FakeIndex:
        del progress_callback
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
            retrieval=RetrievalDefaults(
                retrieval_strategy="label_gate_semantic_rank",
                label_free_fallback_strategy="concept_overlap_semantic_rerank",
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
    assert config.labelrag_pipeline.retrieval.retrieval_strategy == "label_gate_semantic_rank"
    assert (
        config.labelrag_pipeline.retrieval.label_free_fallback_strategy
        == "concept_overlap_semantic_rerank"
    )


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
    monkeypatch.setenv("BLOGRAG_CONFIG_DIR", str(tmp_path / "config-root"))
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
    assert "Config paths" in show_result.stdout
    assert "Config" in show_result.stdout
    assert "Secrets" in show_result.stdout
    assert "default_index_dir" in show_result.stdout
    assert "index" in show_result.stdout
    assert "mistral" in show_result.stdout
    assert "configured" in show_result.stdout
    assert "secret-value" not in show_result.stdout


def test_config_show_all_includes_unset_and_default_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BLOGRAG_CONFIG_DIR", str(tmp_path / "config-root"))

    result = runner.invoke(app, ["config", "show", "--all"])

    assert result.exit_code == 0
    assert "default_blog_dir" in result.stdout
    assert "unset" in result.stdout
    assert "serve.host" in result.stdout
    assert "default: 127.0.0.1" in result.stdout
    assert "build.labelgen_cache_dir" in result.stdout
    assert "default: upstream default" in result.stdout
    assert ".labelgen-cache" in result.stdout
    assert "retrieval.retrieval_strategy" in result.stdout
    assert "default:" in result.stdout
    assert "greedy_label_coverage_semantic_reran" in result.stdout


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
            mcp_url="http://127.0.0.1:8765/mcp",
            health_url="http://127.0.0.1:8765/healthz",
            pid=12345,
            process_running=True,
            http_ready=True,
            http_status_code=200,
            detail="HTTP 200",
        )

    monkeypatch.setattr(blograg.cli, "start_server", fake_start_server)

    result = runner.invoke(app, ["start"])

    assert result.exit_code == 0
    assert captured["index_dir"] == tmp_path / "index"
    assert captured["pid_file"] == tmp_path / "config-root" / "server.pid"
    assert captured["log_file"] == tmp_path / "config-root" / "server.log"
    assert "Started blograg server (PID 12345)." in result.stdout
    assert "MCP endpoint: http://127.0.0.1:8765/mcp" in result.stdout
    assert "Health endpoint: http://127.0.0.1:8765/healthz" in result.stdout


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

    def fake_get_server_status(
        *, pid_file: Path, log_file: Path, mcp_url: str, health_url: str
    ) -> ServerStatus:
        assert pid_file == tmp_path / "config-root" / "server.pid"
        assert log_file == tmp_path / "config-root" / "server.log"
        assert mcp_url == "http://0.0.0.0:8877/mcp"
        assert health_url == "http://0.0.0.0:8877/healthz"
        return ServerStatus(
            pid_file=pid_file,
            log_file=log_file,
            mcp_url=mcp_url,
            health_url=health_url,
            pid=12345,
            process_running=True,
            http_ready=True,
            http_status_code=200,
            detail="HTTP 200",
        )

    monkeypatch.setattr(blograg.cli, "get_server_status", fake_get_server_status)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "Status" in result.stdout
    assert "Server" in result.stdout
    assert "running" in result.stdout
    assert "MCP endpoint" in result.stdout
    assert "http://0.0.0.0:8877/mcp" in result.stdout
    assert "Health endpoint" in result.stdout
    assert "http://0.0.0.0:8877/healthz" in result.stdout
    assert "HTTP status" in result.stdout
    assert "ready" in result.stdout
    assert "HTTP code" in result.stdout
    assert "200" in result.stdout


def test_logs_reads_recent_lines_from_managed_log_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_root = tmp_path / "config-root"
    log_path = config_root / "server.log"
    monkeypatch.setenv("BLOGRAG_CONFIG_DIR", str(config_root))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("one\ntwo\nthree\n", encoding="utf-8")

    result = runner.invoke(app, ["logs", "--tail", "2"])

    assert result.exit_code == 0
    assert "two" in result.stdout
    assert "three" in result.stdout
    assert "one" not in result.stdout


def test_logs_follow_streams_appended_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_root = tmp_path / "config-root"
    log_path = config_root / "server.log"
    monkeypatch.setenv("BLOGRAG_CONFIG_DIR", str(config_root))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("one\ntwo\nthree\n", encoding="utf-8")

    def fake_follow_log(path: Path) -> Generator[str, None, None]:
        assert path == log_path
        yield "four"
        raise KeyboardInterrupt

    monkeypatch.setattr(blograg.cli, "_follow_log", fake_follow_log)

    result = runner.invoke(app, ["logs", "--tail", "1", "--follow"])

    assert result.exit_code == 0
    assert "three" in result.stdout
    assert "four" in result.stdout


def test_logs_fails_clearly_when_log_file_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_root = tmp_path / "config-root"
    monkeypatch.setenv("BLOGRAG_CONFIG_DIR", str(config_root))

    result = runner.invoke(app, ["logs"])

    assert result.exit_code == 1
    assert "No managed server log file found" in result.stderr
    assert "blograg start" in result.stderr


def test_doctor_command_reports_actionable_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BLOGRAG_CONFIG_DIR", str(tmp_path / "config-root"))
    index_dir = tmp_path / "index"
    save_user_cli_config(
        CLIConfig(
            default_index_dir=str(index_dir),
            build=BuildDefaults(
                concept_extractor="llm",
                llm_provider="mistral",
                llm_model="mistral-small",
            ),
            serve=ServeDefaults(host="127.0.0.1", port=8765),
        )
    )
    save_provider_secrets(ProviderSecrets(mistral="secret-value"))
    (index_dir / "blograg" / "labelrag").mkdir(parents=True)
    (index_dir / "blograg" / "manifest.json").write_text("{}", encoding="utf-8")
    (index_dir / "blograg" / "paragraphs.json").write_text("[]", encoding="utf-8")

    def fake_get_server_status(
        *, pid_file: Path, log_file: Path, mcp_url: str, health_url: str
    ) -> ServerStatus:
        del pid_file, log_file, mcp_url, health_url
        return ServerStatus(
            pid_file=tmp_path / "config-root" / "server.pid",
            log_file=tmp_path / "config-root" / "server.log",
            mcp_url="http://127.0.0.1:8765/mcp",
            health_url="http://127.0.0.1:8765/healthz",
            pid=12345,
            process_running=True,
            http_ready=True,
            http_status_code=200,
            detail="HTTP 200",
        )

    monkeypatch.setattr(blograg.cli, "get_server_status", fake_get_server_status)

    def fake_which(name: str) -> str:
        return f"/usr/bin/{name}"

    monkeypatch.setattr(blograg.cli.shutil, "which", fake_which)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "Doctor" in result.stdout
    assert "Configuration" in result.stdout
    assert "Clients" in result.stdout
    assert "Doctor found no issues." in result.stdout
    assert "Index artifacts" in result.stdout
    assert "Index looks complete." in result.stdout
    assert "codex executable" in result.stdout
    assert "/usr/bin/codex" in result.stdout


def test_doctor_command_fails_when_key_setup_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BLOGRAG_CONFIG_DIR", str(tmp_path / "config-root"))
    save_user_cli_config(
        CLIConfig(
            build=BuildDefaults(
                concept_extractor="llm",
                llm_provider="mistral",
            )
        )
    )

    def fake_get_server_status(
        *, pid_file: Path, log_file: Path, mcp_url: str, health_url: str
    ) -> ServerStatus:
        del pid_file, log_file, mcp_url, health_url
        return ServerStatus(
            pid_file=tmp_path / "config-root" / "server.pid",
            log_file=tmp_path / "config-root" / "server.log",
            mcp_url="http://127.0.0.1:8765/mcp",
            health_url="http://127.0.0.1:8765/healthz",
            pid=None,
            process_running=False,
            http_ready=False,
            http_status_code=None,
            detail="Connection refused",
        )

    monkeypatch.setattr(blograg.cli, "get_server_status", fake_get_server_status)

    def fake_missing_which(_name: str) -> None:
        return None

    monkeypatch.setattr(blograg.cli.shutil, "which", fake_missing_which)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "LLM model" in result.stdout
    assert "Missing build.llm_model." in result.stdout
    assert "Doctor found" in result.stderr


def test_register_show_reports_client_registration_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    del tmp_path

    def fake_get_client_registration_status(*, client: str, server_name: str) -> object:
        assert server_name == "blograg-local"
        if client == "codex":
            return SimpleNamespace(
                configured=True,
                detail="Configured for http://127.0.0.1:8765/mcp.",
                url="http://127.0.0.1:8765/mcp",
            )
        return SimpleNamespace(
            configured=False,
            detail="`blograg-local` is not configured.",
            url=None,
        )

    monkeypatch.setattr(
        blograg.cli,
        "get_client_registration_status",
        fake_get_client_registration_status,
    )

    result = runner.invoke(app, ["register", "--show", "--server-name", "blograg-local"])

    assert result.exit_code == 0
    assert "Bindings" in result.stdout
    assert "codex" in result.stdout
    assert "openclaw" in result.stdout
    assert "http://127.0.0.1:8765/mcp" in result.stdout
    assert "not configured" in result.stdout


def test_register_command_registers_single_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BLOGRAG_CONFIG_DIR", str(tmp_path / "config-root"))
    save_user_cli_config(CLIConfig(serve=ServeDefaults(host="0.0.0.0", port=8877)))
    calls: list[tuple[str, str, str]] = []

    def fake_register_client(*, client: str, server_name: str, url: str) -> str:
        calls.append((client, server_name, url))
        return f"registered {client}"

    monkeypatch.setattr(blograg.cli, "register_client", fake_register_client)

    result = runner.invoke(app, ["register", "--client", "codex", "--server-name", "blograg-local"])

    assert result.exit_code == 0
    assert calls == [("codex", "blograg-local", "http://0.0.0.0:8877/mcp")]
    assert "registered codex" in result.stdout


def test_register_requires_client_without_show() -> None:
    result = runner.invoke(app, ["register"])

    assert result.exit_code == 1
    assert "Provide `--client` or use `--show`" in result.stderr


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
