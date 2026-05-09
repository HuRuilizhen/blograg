"""Tests for the CLI and MCP server wiring."""

from __future__ import annotations

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


def _write_blog(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create a temporary blog directory with the provided file contents."""

    blog_dir = tmp_path / "blog"
    for relative_path, content in files.items():
        destination = blog_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    return blog_dir
