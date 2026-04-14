"""Tests for blograg index build, load, and retrieval behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blograg.config import BlogRAGConfig
from blograg.indexing import build_index, load_index
from blograg.version import __version__
from tests.testsupport import FakeEmbeddingProvider


def test_build_index_writes_outer_layout_and_metadata(tmp_path: Path) -> None:
    blog_dir = _write_blog(
        tmp_path,
        {
            "_posts/2026-04-14-my-post.md": (
                "---\ntitle: My Post\n---\n\nIntro.\n\n## Section\nBody.\n"
            )
        },
    )
    index_dir = tmp_path / "index"

    build_index(
        blog_dir=blog_dir,
        index_dir=index_dir,
        config=BlogRAGConfig(),
        embedding_provider=FakeEmbeddingProvider(),
    )

    manifest_path = index_dir / "blograg" / "manifest.json"
    paragraphs_path = index_dir / "blograg" / "paragraphs.json"
    labelrag_dir = index_dir / "blograg" / "labelrag"

    assert manifest_path.is_file()
    assert paragraphs_path.is_file()
    assert labelrag_dir.is_dir()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["blograg_version"] == __version__
    assert manifest["schema_version"] == 1
    assert manifest["paragraph_count"] == 2
    assert manifest["labelrag_persistence_format"] == "json.gz"

    paragraphs_payload = json.loads(paragraphs_path.read_text(encoding="utf-8"))
    assert [item["paragraph_id"] for item in paragraphs_payload] == [
        "my-post::p001",
        "my-post::p002",
    ]


def test_load_index_retrieves_structured_paragraph_results(tmp_path: Path) -> None:
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
                "\n"
                "## Paragraph Retrieval\n"
                "Paragraph retrieval should keep section metadata.\n"
            )
        },
    )
    index_dir = tmp_path / "index"
    config = BlogRAGConfig()
    provider = FakeEmbeddingProvider()

    build_index(
        blog_dir=blog_dir,
        index_dir=index_dir,
        config=config,
        embedding_provider=provider,
    )
    loaded_index = load_index(
        index_dir=index_dir,
        config=config,
        embedding_provider=provider,
    )

    results = loaded_index.retrieve_paragraphs("front matter metadata", top_k=1)

    assert len(results) == 1
    assert results[0].paragraph_id == "jekyll::p001"
    assert results[0].post_title == "Jekyll Notes"
    assert results[0].slug == "jekyll"
    assert results[0].section_heading == "Front Matter"
    assert results[0].trace.retrieval_strategy
    assert isinstance(results[0].trace.score, float)


def test_load_index_fails_clearly_when_manifest_is_missing(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Run `blograg build"):
        load_index(index_dir=index_dir, embedding_provider=FakeEmbeddingProvider())


def test_build_index_rejects_empty_blog(tmp_path: Path) -> None:
    blog_dir = tmp_path / "blog"
    blog_dir.mkdir()

    with pytest.raises(ValueError, match="No supported markdown posts"):
        build_index(
            blog_dir=blog_dir,
            index_dir=tmp_path / "index",
            embedding_provider=FakeEmbeddingProvider(),
        )


def _write_blog(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create a temporary blog directory with the provided file contents."""

    blog_dir = tmp_path / "blog"
    for relative_path, content in files.items():
        destination = blog_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    return blog_dir
