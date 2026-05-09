"""Tests for blograg index build, load, and retrieval behavior."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Barrier, Lock, Thread
from time import sleep
from types import SimpleNamespace

import pytest

from blograg.config import BlogRAGConfig, build_config
from blograg.indexing import BlogRAGIndex, BuildProgressUpdate, build_index, load_index
from blograg.models import ParagraphRecord
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


def test_build_index_rebuilds_cleanly_over_existing_blograg_directory(tmp_path: Path) -> None:
    blog_dir = _write_blog(
        tmp_path,
        {
            "_posts/2026-04-14-my-post.md": (
                "---\ntitle: My Post\n---\n\nIntro.\n\n## Section\nBody.\n"
            )
        },
    )
    index_dir = tmp_path / "index"
    provider = FakeEmbeddingProvider()

    build_index(
        blog_dir=blog_dir,
        index_dir=index_dir,
        config=BlogRAGConfig(),
        embedding_provider=provider,
    )
    stale_artifact = index_dir / "blograg" / "stale.txt"
    stale_artifact.write_text("remove me", encoding="utf-8")

    rebuilt_index = build_index(
        blog_dir=blog_dir,
        index_dir=index_dir,
        config=BlogRAGConfig(),
        embedding_provider=provider,
    )

    assert stale_artifact.exists() is False
    assert sorted(rebuilt_index.paragraph_records) == ["my-post::p001", "my-post::p002"]


def test_load_index_fails_clearly_when_paragraph_metadata_is_missing(tmp_path: Path) -> None:
    blog_dir = _write_blog(
        tmp_path,
        {"_posts/2026-04-14-my-post.md": "---\n---\n\n## Section\nBody.\n"},
    )
    index_dir = tmp_path / "index"

    build_index(
        blog_dir=blog_dir,
        index_dir=index_dir,
        embedding_provider=FakeEmbeddingProvider(),
    )
    (index_dir / "blograg" / "paragraphs.json").unlink()

    with pytest.raises(FileNotFoundError, match="Run `blograg build"):
        load_index(index_dir=index_dir, embedding_provider=FakeEmbeddingProvider())


def test_load_index_fails_clearly_when_labelrag_snapshot_is_missing(tmp_path: Path) -> None:
    blog_dir = _write_blog(
        tmp_path,
        {"_posts/2026-04-14-my-post.md": "---\n---\n\n## Section\nBody.\n"},
    )
    index_dir = tmp_path / "index"

    build_index(
        blog_dir=blog_dir,
        index_dir=index_dir,
        embedding_provider=FakeEmbeddingProvider(),
    )
    labelrag_dir = index_dir / "blograg" / "labelrag"
    for child in labelrag_dir.iterdir():
        if child.is_file():
            child.unlink()
    labelrag_dir.rmdir()

    with pytest.raises(FileNotFoundError, match="Run `blograg build"):
        load_index(index_dir=index_dir, embedding_provider=FakeEmbeddingProvider())


def test_load_index_fails_when_manifest_count_does_not_match_paragraph_metadata(
    tmp_path: Path,
) -> None:
    blog_dir = _write_blog(
        tmp_path,
        {"_posts/2026-04-14-my-post.md": "---\n---\n\n## Section\nBody.\n"},
    )
    index_dir = tmp_path / "index"

    build_index(
        blog_dir=blog_dir,
        index_dir=index_dir,
        embedding_provider=FakeEmbeddingProvider(),
    )
    manifest_path = index_dir / "blograg" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["paragraph_count"] = 999
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match the manifest paragraph count"):
        load_index(index_dir=index_dir, embedding_provider=FakeEmbeddingProvider())


def test_retrieve_paragraphs_rejects_non_positive_top_k(tmp_path: Path) -> None:
    blog_dir = _write_blog(
        tmp_path,
        {"_posts/2026-04-14-my-post.md": "---\n---\n\n## Section\nBody.\n"},
    )
    index_dir = tmp_path / "index"

    index = build_index(
        blog_dir=blog_dir,
        index_dir=index_dir,
        config=BlogRAGConfig(),
        embedding_provider=FakeEmbeddingProvider(),
    )

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        index.retrieve_paragraphs("section", top_k=0)


def test_retrieve_paragraphs_serializes_pipeline_limit_override() -> None:
    active_calls = 0
    max_active_calls = 0
    observed_limits: list[int] = []
    state_lock = Lock()
    start_barrier = Barrier(2)

    class _FakePipeline:
        def __init__(self) -> None:
            self.config = SimpleNamespace(retrieval=SimpleNamespace(max_paragraphs=8))

        def build_context(self, query: str) -> SimpleNamespace:
            del query
            nonlocal active_calls, max_active_calls
            with state_lock:
                active_calls += 1
                max_active_calls = max(max_active_calls, active_calls)
                observed_limits.append(self.config.retrieval.max_paragraphs)
            sleep(0.05)
            with state_lock:
                active_calls -= 1
            return SimpleNamespace(
                metadata={"retrieval_strategy": "test"},
                retrieved_paragraphs=[
                    SimpleNamespace(
                        paragraph_id="post::p001",
                        text="Paragraph body",
                        retrieval_score=1.0,
                    )
                ],
            )

    index = BlogRAGIndex(
        pipeline=_FakePipeline(),  # type: ignore[arg-type]
        paragraph_records={
            "post::p001": ParagraphRecord(
                paragraph_id="post::p001",
                text="Paragraph body",
                post_title="Post",
                slug="post",
                section_heading="Section",
                source_path="_posts/post.md",
                order_in_post=1,
            )
        },
        config=build_config(),
    )

    def run_retrieval(top_k: int) -> None:
        start_barrier.wait()
        results = index.retrieve_paragraphs("query", top_k=top_k)
        assert len(results) == 1

    first_thread = Thread(target=run_retrieval, args=(1,))
    second_thread = Thread(target=run_retrieval, args=(2,))

    first_thread.start()
    second_thread.start()
    first_thread.join()
    second_thread.join()

    assert max_active_calls == 1
    assert sorted(observed_limits) == [1, 2]
    assert index.pipeline.config.retrieval.max_paragraphs == 8


def test_build_index_rejects_empty_blog(tmp_path: Path) -> None:
    blog_dir = tmp_path / "blog"
    blog_dir.mkdir()

    with pytest.raises(ValueError, match="No supported markdown posts"):
        build_index(
            blog_dir=blog_dir,
            index_dir=tmp_path / "index",
            embedding_provider=FakeEmbeddingProvider(),
        )


def test_build_index_emits_progress_updates(tmp_path: Path) -> None:
    blog_dir = _write_blog(
        tmp_path,
        {
            "_posts/2026-04-14-my-post.md": (
                "---\ntitle: My Post\n---\n\nIntro paragraph.\n\n## Section\nBody paragraph.\n"
            )
        },
    )
    progress_events: list[BuildProgressUpdate] = []

    build_index(
        blog_dir=blog_dir,
        index_dir=tmp_path / "index",
        config=BlogRAGConfig(),
        embedding_provider=FakeEmbeddingProvider(),
        progress_callback=progress_events.append,
    )

    assert progress_events[0].stage == "extract"
    assert progress_events[0].processed == 0
    assert progress_events[0].total == 2
    assert progress_events[0].current_paragraph_id == "my-post::p001"
    assert progress_events[1].stage == "extract"
    assert progress_events[1].processed == 1
    assert progress_events[1].current_paragraph_id == "my-post::p002"
    assert progress_events[2].stage == "extract"
    assert progress_events[2].processed == 2
    assert progress_events[2].current_paragraph_id is None
    assert progress_events[3].stage == "embed"
    assert progress_events[3].processed == 2
    assert progress_events[4].stage == "save"
    assert progress_events[4].processed == 2


def _write_blog(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create a temporary blog directory with the provided file contents."""

    blog_dir = tmp_path / "blog"
    for relative_path, content in files.items():
        destination = blog_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    return blog_dir
