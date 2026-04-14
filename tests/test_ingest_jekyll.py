"""Tests for Jekyll-style blog ingestion."""

from pathlib import Path

from blograg.ingest import (
    discover_post_paths,
    load_blog_paragraphs,
    load_post,
    split_post_into_paragraphs,
)


def test_discover_post_paths_filters_supported_markdown_files(tmp_path: Path) -> None:
    blog_dir = tmp_path / "blog"
    blog_dir.mkdir()
    (blog_dir / "_posts").mkdir()
    (blog_dir / "_posts" / "2026-04-14-first-post.md").write_text("# Title\n", encoding="utf-8")
    (blog_dir / "_posts" / "draft.markdown").write_text("# Draft\n", encoding="utf-8")
    (blog_dir / "_posts" / "notes.txt").write_text("ignore", encoding="utf-8")

    discovered = discover_post_paths(blog_dir)

    assert [path.relative_to(blog_dir).as_posix() for path in discovered] == [
        "_posts/2026-04-14-first-post.md",
        "_posts/draft.markdown",
    ]


def test_load_post_prefers_front_matter_slug_and_title(tmp_path: Path) -> None:
    blog_dir = tmp_path / "blog"
    posts_dir = blog_dir / "_posts"
    posts_dir.mkdir(parents=True)
    post_path = posts_dir / "2026-04-14-ignored-slug.md"
    post_path.write_text(
        "---\ntitle: Actual Title\nslug: canonical-slug\n---\n\n# Heading\nBody.\n",
        encoding="utf-8",
    )

    post = load_post(post_path, blog_dir=blog_dir)

    assert post.post_title == "Actual Title"
    assert post.slug == "canonical-slug"
    assert post.source_path == "_posts/2026-04-14-ignored-slug.md"


def test_load_post_falls_back_to_jekyll_filename_slug(tmp_path: Path) -> None:
    blog_dir = tmp_path / "blog"
    posts_dir = blog_dir / "_posts"
    posts_dir.mkdir(parents=True)
    post_path = posts_dir / "2026-04-14-my-post.md"
    post_path.write_text("---\n---\n\ncontent", encoding="utf-8")

    post = load_post(post_path, blog_dir=blog_dir)

    assert post.slug == "my-post"
    assert post.post_title == "my-post"


def test_split_post_into_paragraphs_preserves_intro_and_headings(tmp_path: Path) -> None:
    blog_dir = tmp_path / "blog"
    posts_dir = blog_dir / "_posts"
    posts_dir.mkdir(parents=True)
    post_path = posts_dir / "2026-04-14-my-post.md"
    post_path.write_text(
        "---\n"
        "title: My Post\n"
        "---\n"
        "\n"
        "Intro paragraph.\n"
        "\n"
        "Still intro.\n"
        "\n"
        "## First Section\n"
        "Paragraph one.\n"
        "\n"
        "### Second Section\n"
        "Paragraph two.\n",
        encoding="utf-8",
    )

    post = load_post(post_path, blog_dir=blog_dir)
    paragraphs = split_post_into_paragraphs(post)

    assert [paragraph.paragraph_id for paragraph in paragraphs] == [
        "my-post::p001",
        "my-post::p002",
        "my-post::p003",
    ]
    assert [paragraph.section_heading for paragraph in paragraphs] == [
        None,
        "First Section",
        "Second Section",
    ]
    assert paragraphs[0].text == "Intro paragraph.\n\nStill intro."
    assert paragraphs[1].text == "## First Section\nParagraph one."
    assert paragraphs[2].text == "### Second Section\nParagraph two."
    assert [paragraph.order_in_post for paragraph in paragraphs] == [1, 2, 3]


def test_split_post_without_headings_returns_one_intro_paragraph(tmp_path: Path) -> None:
    blog_dir = tmp_path / "blog"
    posts_dir = blog_dir / "_posts"
    posts_dir.mkdir(parents=True)
    post_path = posts_dir / "standalone.md"
    post_path.write_text("---\n---\n\nPlain body only.\n", encoding="utf-8")

    post = load_post(post_path, blog_dir=blog_dir)
    paragraphs = split_post_into_paragraphs(post)

    assert len(paragraphs) == 1
    assert paragraphs[0].paragraph_id == "standalone::p001"
    assert paragraphs[0].section_heading is None
    assert paragraphs[0].text == "Plain body only."


def test_load_blog_paragraphs_returns_deterministic_source_order(tmp_path: Path) -> None:
    blog_dir = tmp_path / "blog"
    posts_dir = blog_dir / "_posts"
    posts_dir.mkdir(parents=True)
    (posts_dir / "2026-04-15-b-post.md").write_text("---\n---\n\n# B\nBody\n", encoding="utf-8")
    (posts_dir / "2026-04-14-a-post.md").write_text("---\n---\n\n# A\nBody\n", encoding="utf-8")

    paragraphs = load_blog_paragraphs(blog_dir)

    assert [paragraph.slug for paragraph in paragraphs] == ["a-post", "b-post"]
