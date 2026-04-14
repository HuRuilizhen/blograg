"""Configuration models for blograg."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from labelgen import LabelGeneratorConfig
from labelrag import RAGPipelineConfig, RetrievalConfig

ConceptExtractorMode = Literal["spacy", "heuristic", "llm"]
LLMProvider = Literal["openai", "mistral", "qwen", "ollama"]
PersistenceFormat = Literal["json", "json.gz"]


@dataclass(slots=True)
class BlogRAGConfig:
    """Minimal configuration surface for the blograg MVP."""

    retrieval_default_top_k: int = 5
    labelrag_persistence_format: PersistenceFormat = "json.gz"
    labelrag_pipeline: RAGPipelineConfig = field(default_factory=lambda: _default_pipeline_config())


def _default_pipeline_config() -> RAGPipelineConfig:
    """Build a stable default pipeline configuration for the MVP."""

    return RAGPipelineConfig(
        labelgen=LabelGeneratorConfig(
            use_nlp_extractor=False,
            use_graph_community_detection=False,
        ),
        retrieval=RetrievalConfig(
            max_paragraphs=8,
            label_free_fallback_strategy="semantic_only",
        ),
    )


def build_config(
    *,
    concept_extractor: ConceptExtractorMode = "heuristic",
    llm_provider: LLMProvider = "mistral",
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key_env_var: str | None = None,
    llm_batch_size: int = 8,
    llm_max_concepts_per_paragraph: int = 12,
    llm_output_contract_mode: Literal["auto", "json_schema", "json_object", "prompt_only"] = "auto",
) -> BlogRAGConfig:
    """Build a blograg config from explicit CLI-facing options."""

    config = BlogRAGConfig()
    labelgen_config = config.labelrag_pipeline.labelgen
    labelgen_config.extractor_mode = concept_extractor
    labelgen_config.use_nlp_extractor = concept_extractor == "spacy"

    if concept_extractor == "llm":
        llm_config = labelgen_config.extraction.llm
        llm_config.provider = llm_provider
        llm_config.model = llm_model or ""
        llm_config.api_key_env_var = llm_api_key_env_var
        llm_config.base_url = llm_base_url
        llm_config.batch_size = llm_batch_size
        llm_config.max_concepts_per_paragraph = llm_max_concepts_per_paragraph
        llm_config.output_contract_mode = llm_output_contract_mode
    return config
