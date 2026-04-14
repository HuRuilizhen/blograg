"""Build and load persisted blograg indexes."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from labelrag import EmbeddingProvider, Paragraph, RAGPipeline

from blograg.config import BlogRAGConfig, PersistenceFormat
from blograg.ingest import load_blog_paragraphs
from blograg.models import ParagraphRecord, ParagraphResult, RetrievalTrace
from blograg.version import __version__

_BLOGRAG_DIRNAME = "blograg"
_LABELRAG_DIRNAME = "labelrag"
_MANIFEST_FILENAME = "manifest.json"
_PARAGRAPHS_FILENAME = "paragraphs.json"
_SCHEMA_VERSION = 1


@dataclass(slots=True, frozen=True)
class _IndexLayout:
    """Resolved filesystem layout for one persisted blograg index."""

    root_dir: Path
    blograg_dir: Path
    labelrag_dir: Path
    manifest_path: Path
    paragraphs_path: Path


@dataclass(slots=True, frozen=True)
class _BlogRAGManifest:
    """Persisted outer-layer manifest for one blograg snapshot."""

    blograg_version: str
    schema_version: int
    paragraph_count: int
    labelrag_persistence_format: PersistenceFormat
    paragraphs_artifact: str
    labelrag_snapshot_dir: str


@dataclass(slots=True)
class BlogRAGIndex:
    """Loaded blograg index that can answer paragraph retrieval queries."""

    pipeline: RAGPipeline
    paragraph_records: dict[str, ParagraphRecord]
    config: BlogRAGConfig

    def retrieve_paragraphs(self, query: str, top_k: int | None = None) -> list[ParagraphResult]:
        """Retrieve structured paragraph-first results for one query."""

        requested_top_k = top_k or self.config.retrieval_default_top_k
        if requested_top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        original_limit = self.pipeline.config.retrieval.max_paragraphs
        self.pipeline.config.retrieval.max_paragraphs = requested_top_k
        try:
            retrieval_result = self.pipeline.build_context(query)
        finally:
            self.pipeline.config.retrieval.max_paragraphs = original_limit

        retrieval_strategy = str(retrieval_result.metadata.get("retrieval_strategy", "unknown"))
        return [
            _retrieval_result_item_to_paragraph_result(
                paragraph_id=item.paragraph_id,
                paragraph_records=self.paragraph_records,
                text=item.text,
                retrieval_strategy=retrieval_strategy,
                score=item.retrieval_score,
            )
            for item in retrieval_result.retrieved_paragraphs
            if item.paragraph_id in self.paragraph_records
        ]


def build_index(
    *,
    blog_dir: Path,
    index_dir: Path,
    config: BlogRAGConfig | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> BlogRAGIndex:
    """Build a fresh blograg index from a local blog directory."""

    effective_config = config or BlogRAGConfig()
    paragraph_records = load_blog_paragraphs(blog_dir)
    if not paragraph_records:
        msg = f"No supported markdown posts were found in {blog_dir}."
        raise ValueError(msg)

    layout = _build_layout(index_dir)
    _reset_blograg_directory(layout.blograg_dir)

    pipeline = RAGPipeline(
        config=effective_config.labelrag_pipeline,
        embedding_provider=embedding_provider,
    )
    pipeline.fit([_paragraph_record_to_labelrag_paragraph(record) for record in paragraph_records])
    pipeline.save(
        layout.labelrag_dir,
        format=effective_config.labelrag_persistence_format,
    )

    manifest = _BlogRAGManifest(
        blograg_version=__version__,
        schema_version=_SCHEMA_VERSION,
        paragraph_count=len(paragraph_records),
        labelrag_persistence_format=effective_config.labelrag_persistence_format,
        paragraphs_artifact=_PARAGRAPHS_FILENAME,
        labelrag_snapshot_dir=_LABELRAG_DIRNAME,
    )
    _write_json(layout.manifest_path, asdict(manifest))
    _write_json(
        layout.paragraphs_path,
        [_paragraph_record_to_dict(record) for record in paragraph_records],
    )

    return BlogRAGIndex(
        pipeline=pipeline,
        paragraph_records={record.paragraph_id: record for record in paragraph_records},
        config=effective_config,
    )


def load_index(
    *,
    index_dir: Path,
    config: BlogRAGConfig | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> BlogRAGIndex:
    """Load a previously built blograg index from disk."""

    effective_config = config or BlogRAGConfig()
    layout = _build_layout(index_dir)
    _require_artifact(layout.manifest_path, index_dir)
    _require_artifact(layout.paragraphs_path, index_dir)
    _require_directory(layout.labelrag_dir, index_dir)

    manifest = _load_manifest(layout.manifest_path)
    paragraph_records = _load_paragraph_records(layout.paragraphs_path)
    if manifest.paragraph_count != len(paragraph_records):
        raise RuntimeError(
            "Stored paragraph metadata does not match the manifest paragraph count. "
            "Run `blograg build` again."
        )

    pipeline = RAGPipeline.load(
        layout.labelrag_dir,
        format=manifest.labelrag_persistence_format,
        embedding_provider=embedding_provider,
    )
    return BlogRAGIndex(
        pipeline=pipeline,
        paragraph_records={record.paragraph_id: record for record in paragraph_records},
        config=effective_config,
    )


def _build_layout(index_dir: Path) -> _IndexLayout:
    """Resolve the filesystem layout for one blograg index root."""

    root_dir = index_dir.resolve()
    blograg_dir = root_dir / _BLOGRAG_DIRNAME
    labelrag_dir = blograg_dir / _LABELRAG_DIRNAME
    return _IndexLayout(
        root_dir=root_dir,
        blograg_dir=blograg_dir,
        labelrag_dir=labelrag_dir,
        manifest_path=blograg_dir / _MANIFEST_FILENAME,
        paragraphs_path=blograg_dir / _PARAGRAPHS_FILENAME,
    )


def _reset_blograg_directory(blograg_dir: Path) -> None:
    """Replace the outer blograg directory for a full rebuild."""

    if blograg_dir.exists():
        shutil.rmtree(blograg_dir)
    blograg_dir.mkdir(parents=True, exist_ok=True)


def _paragraph_record_to_labelrag_paragraph(record: ParagraphRecord) -> Paragraph:
    """Convert one blograg paragraph record into a labelrag paragraph."""

    return Paragraph(
        id=record.paragraph_id,
        text=record.text,
        metadata=_paragraph_record_to_dict(record),
    )


def _paragraph_record_to_dict(record: ParagraphRecord) -> dict[str, Any]:
    """Serialize paragraph metadata into a JSON-compatible dictionary."""

    return {
        "paragraph_id": record.paragraph_id,
        "text": record.text,
        "post_title": record.post_title,
        "slug": record.slug,
        "section_heading": record.section_heading,
        "source_path": record.source_path,
        "order_in_post": record.order_in_post,
    }


def _retrieval_result_item_to_paragraph_result(
    *,
    paragraph_id: str,
    paragraph_records: dict[str, ParagraphRecord],
    text: str,
    retrieval_strategy: str,
    score: float,
) -> ParagraphResult:
    """Map one labelrag retrieval item into the blograg MCP-facing contract."""

    record = paragraph_records[paragraph_id]
    return ParagraphResult(
        paragraph_id=record.paragraph_id,
        text=text,
        post_title=record.post_title,
        slug=record.slug,
        section_heading=record.section_heading,
        trace=RetrievalTrace(
            retrieval_strategy=retrieval_strategy,
            score=score,
        ),
    )


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    """Write one JSON payload to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_manifest(path: Path) -> _BlogRAGManifest:
    """Load and validate one blograg manifest."""

    raw_data = _load_json_object(path)
    schema_version = raw_data.get("schema_version")
    blograg_version = raw_data.get("blograg_version")
    if schema_version != _SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported blograg schema version {schema_version!r}. Run `blograg build` again."
        )
    if not isinstance(blograg_version, str) or not blograg_version:
        raise RuntimeError("Blograg manifest must include a non-empty blograg_version.")
    paragraph_count = raw_data.get("paragraph_count")
    persistence_format = _require_persistence_format(raw_data, "labelrag_persistence_format")
    paragraphs_artifact = raw_data.get("paragraphs_artifact")
    labelrag_snapshot_dir = raw_data.get("labelrag_snapshot_dir")
    if not isinstance(paragraph_count, int):
        raise RuntimeError("Blograg manifest must include an integer paragraph_count.")
    if not isinstance(paragraphs_artifact, str):
        raise RuntimeError("Blograg manifest must include paragraphs_artifact.")
    if paragraphs_artifact != _PARAGRAPHS_FILENAME:
        raise RuntimeError(
            "Blograg manifest paragraphs_artifact does not match the expected layout."
        )
    if not isinstance(labelrag_snapshot_dir, str):
        raise RuntimeError("Blograg manifest must include labelrag_snapshot_dir.")
    if labelrag_snapshot_dir != _LABELRAG_DIRNAME:
        raise RuntimeError(
            "Blograg manifest labelrag_snapshot_dir does not match the expected layout."
        )
    return _BlogRAGManifest(
        blograg_version=blograg_version,
        schema_version=_SCHEMA_VERSION,
        paragraph_count=paragraph_count,
        labelrag_persistence_format=persistence_format,
        paragraphs_artifact=paragraphs_artifact,
        labelrag_snapshot_dir=labelrag_snapshot_dir,
    )


def _load_paragraph_records(path: Path) -> list[ParagraphRecord]:
    """Load serialized paragraph records from disk."""

    raw_payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, list):
        raise RuntimeError("Paragraph metadata artifact must contain a JSON list.")

    paragraph_records: list[ParagraphRecord] = []
    for item in cast(list[object], raw_payload):
        if not isinstance(item, dict):
            raise RuntimeError("Each stored paragraph metadata item must be a JSON object.")
        payload = {str(key): value for key, value in cast(dict[object, object], item).items()}
        paragraph_records.append(
            ParagraphRecord(
                paragraph_id=_require_str(payload, "paragraph_id"),
                text=_require_str(payload, "text"),
                post_title=_require_str(payload, "post_title"),
                slug=_require_str(payload, "slug"),
                section_heading=_optional_str(payload.get("section_heading"), "section_heading"),
                source_path=_require_str(payload, "source_path"),
                order_in_post=_require_int(payload, "order_in_post"),
            )
        )
    return paragraph_records


def _require_artifact(path: Path, index_dir: Path) -> None:
    """Require one persisted file artifact."""

    if not path.is_file():
        raise FileNotFoundError(
            "Missing blograg artifact at "
            f"{path}. Run `blograg build --blog-dir ... --index-dir {index_dir}` first."
        )


def _require_directory(path: Path, index_dir: Path) -> None:
    """Require one persisted directory artifact."""

    if not path.is_dir():
        raise FileNotFoundError(
            "Missing labelrag snapshot directory at "
            f"{path}. Run `blograg build --blog-dir ... --index-dir {index_dir}` first."
        )


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""

    raw_payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise RuntimeError("Expected a JSON object.")
    payload = cast(dict[object, object], raw_payload)
    return {str(key): value for key, value in payload.items()}


def _require_str(payload: dict[str, Any], key: str) -> str:
    """Require a string field from a JSON object."""

    value = payload.get(key)
    if not isinstance(value, str):
        raise RuntimeError(f"Expected `{key}` to be a string.")
    return value


def _optional_str(value: object, key: str) -> str | None:
    """Require an optional string field from a JSON object."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeError(f"Expected `{key}` to be a string or null.")
    return value


def _require_int(payload: dict[str, Any], key: str) -> int:
    """Require an integer field from a JSON object."""

    value = payload.get(key)
    if not isinstance(value, int):
        raise RuntimeError(f"Expected `{key}` to be an integer.")
    return value


def _require_persistence_format(payload: dict[str, Any], key: str) -> PersistenceFormat:
    """Require a supported persistence format literal from a JSON object."""

    value = payload.get(key)
    if value not in {"json", "json.gz"}:
        raise RuntimeError(f"Expected `{key}` to be either `json` or `json.gz`.")
    return cast(PersistenceFormat, value)
