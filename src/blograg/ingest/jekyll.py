"""Jekyll-style blog discovery and paragraph segmentation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import frontmatter
from markdown_it import MarkdownIt

from blograg.models import ParagraphRecord, PostRecord

_MARKDOWN_EXTENSIONS = {".md", ".markdown"}
_JEKYLL_POST_FILENAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-(?P<slug>.+)$")
_MARKDOWN_PARSER = MarkdownIt("commonmark")


@dataclass(slots=True, frozen=True)
class _SectionBoundary:
    """A heading boundary extracted from a markdown body."""

    line_index: int
    heading_text: str


def discover_post_paths(blog_dir: Path) -> list[Path]:
    """Return supported markdown post files under one blog directory."""

    if not blog_dir.exists():
        msg = f"Blog directory does not exist: {blog_dir}"
        raise FileNotFoundError(msg)
    if not blog_dir.is_dir():
        msg = f"Blog directory is not a directory: {blog_dir}"
        raise NotADirectoryError(msg)

    return sorted(
        path
        for path in blog_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _MARKDOWN_EXTENSIONS
    )


def load_blog_paragraphs(blog_dir: Path) -> list[ParagraphRecord]:
    """Load and segment all supported markdown posts from a blog directory."""

    paragraph_records: list[ParagraphRecord] = []
    for post_path in discover_post_paths(blog_dir):
        post_record = load_post(post_path, blog_dir=blog_dir)
        paragraph_records.extend(split_post_into_paragraphs(post_record))
    return paragraph_records


def load_post(post_path: Path, *, blog_dir: Path) -> PostRecord:
    """Load one markdown post and derive stable blograg metadata."""

    resolved_blog_dir = blog_dir.resolve()
    resolved_post_path = post_path.resolve()
    relative_source_path = resolved_post_path.relative_to(resolved_blog_dir).as_posix()

    parsed_post = frontmatter.load(str(resolved_post_path))
    slug = _derive_slug(parsed_post.metadata.get("slug"), resolved_post_path)
    title = _derive_title(parsed_post.metadata.get("title"), slug)

    return PostRecord(
        post_title=title,
        slug=slug,
        source_path=relative_source_path,
        body=parsed_post.content,
        front_matter=_normalize_front_matter(parsed_post.metadata),
    )


def split_post_into_paragraphs(post: PostRecord) -> list[ParagraphRecord]:
    """Split a parsed post into heading-delimited paragraph blocks."""

    boundaries = _extract_heading_boundaries(post.body)
    lines = post.body.splitlines(keepends=True)
    segments: list[tuple[str, str | None]] = []

    if boundaries:
        first_boundary = boundaries[0]
        intro_text = _strip_preserving_structure("".join(lines[: first_boundary.line_index]))
        if intro_text:
            segments.append((intro_text, None))

        for index, boundary in enumerate(boundaries):
            start_line = boundary.line_index
            end_line = (
                boundaries[index + 1].line_index if index + 1 < len(boundaries) else len(lines)
            )
            segment_text = _strip_preserving_structure("".join(lines[start_line:end_line]))
            if segment_text:
                segments.append((segment_text, boundary.heading_text))
    else:
        body_text = _strip_preserving_structure(post.body)
        if body_text:
            segments.append((body_text, None))

    return [
        ParagraphRecord(
            paragraph_id=f"{post.slug}::p{order_in_post:03d}",
            text=text,
            post_title=post.post_title,
            slug=post.slug,
            section_heading=section_heading,
            source_path=post.source_path,
            order_in_post=order_in_post,
        )
        for order_in_post, (text, section_heading) in enumerate(segments, start=1)
    ]


def _extract_heading_boundaries(markdown_text: str) -> list[_SectionBoundary]:
    """Return line-indexed heading boundaries in source order."""

    tokens = _MARKDOWN_PARSER.parse(markdown_text)
    boundaries: list[_SectionBoundary] = []

    for index, token in enumerate(tokens):
        if token.type != "heading_open" or token.map is None:
            continue
        if index + 1 >= len(tokens):
            continue
        heading_inline = tokens[index + 1]
        if heading_inline.type != "inline":
            continue
        boundaries.append(
            _SectionBoundary(
                line_index=token.map[0],
                heading_text=heading_inline.content.strip(),
            )
        )

    return boundaries


def _derive_slug(front_matter_slug: object, post_path: Path) -> str:
    """Derive the effective slug from front matter or a Jekyll filename."""

    if isinstance(front_matter_slug, str):
        normalized_slug = front_matter_slug.strip()
        if normalized_slug:
            return normalized_slug

    filename_stem = post_path.stem
    filename_match = _JEKYLL_POST_FILENAME_RE.match(filename_stem)
    if filename_match is not None:
        return filename_match.group("slug")
    return filename_stem


def _derive_title(front_matter_title: object, slug: str) -> str:
    """Derive the effective post title."""

    if isinstance(front_matter_title, str):
        normalized_title = front_matter_title.strip()
        if normalized_title:
            return normalized_title
    return slug


def _strip_preserving_structure(markdown_text: str) -> str:
    """Trim surrounding blank space while preserving internal markdown layout."""

    return markdown_text.strip()


def _normalize_front_matter(metadata: dict[Any, Any]) -> dict[str, Any]:
    """Convert parsed front matter into a JSON-like dictionary."""

    return {str(key): value for key, value in metadata.items()}
