"""Tests for blograg configuration helpers."""

from blograg.config import build_config


def test_build_config_defaults_to_heuristic_extraction() -> None:
    config = build_config()

    assert config.labelrag_pipeline.labelgen.resolved_extractor_mode() == "heuristic"
    assert config.labelrag_pipeline.labelgen.extraction.llm.model == ""


def test_build_config_can_enable_spacy_extraction() -> None:
    config = build_config(concept_extractor="spacy")

    assert config.labelrag_pipeline.labelgen.resolved_extractor_mode() == "spacy"
    assert config.labelrag_pipeline.labelgen.use_nlp_extractor is True


def test_build_config_can_enable_llm_extraction() -> None:
    config = build_config(
        concept_extractor="llm",
        llm_provider="mistral",
        llm_model="mistral-small",
        llm_base_url="https://api.mistral.ai/v1/chat/completions",
        llm_api_key_env_var="MISTRAL_API_KEY",
        llm_batch_size=4,
        llm_max_concepts_per_paragraph=9,
        llm_output_contract_mode="json_object",
    )

    llm_config = config.labelrag_pipeline.labelgen.extraction.llm
    assert config.labelrag_pipeline.labelgen.resolved_extractor_mode() == "llm"
    assert config.labelrag_pipeline.labelgen.use_nlp_extractor is False
    assert llm_config.provider == "mistral"
    assert llm_config.model == "mistral-small"
    assert llm_config.base_url == "https://api.mistral.ai/v1/chat/completions"
    assert llm_config.api_key_env_var == "MISTRAL_API_KEY"
    assert llm_config.batch_size == 4
    assert llm_config.max_concepts_per_paragraph == 9
    assert llm_config.output_contract_mode == "json_object"
