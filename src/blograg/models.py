"""Data models for blograg ingestion, indexing, and retrieval."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class PostRecord:
    """A parsed Jekyll-style blog post before paragraph segmentation."""

    post_title: str
    slug: str
    source_path: str
    body: str
    front_matter: dict[str, Any] = field(default_factory=lambda: {})


@dataclass(slots=True, frozen=True)
class ParagraphRecord:
    """One heading-delimited paragraph block from a blog post."""

    paragraph_id: str
    text: str
    post_title: str
    slug: str
    section_heading: str | None
    source_path: str
    order_in_post: int


@dataclass(slots=True, frozen=True)
class RetrievalTrace:
    """Minimal retrieval trace returned to MCP callers."""

    retrieval_strategy: str
    score: float
    score_kind: str


@dataclass(slots=True, frozen=True)
class ParagraphResult:
    """Structured paragraph retrieval result exposed by blograg."""

    paragraph_id: str
    text: str
    post_title: str
    slug: str
    section_heading: str | None
    trace: RetrievalTrace
