"""Tests for user-level CLI config and secret storage."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from blograg.user_config import (
    BuildDefaults,
    CLIConfig,
    ProviderSecrets,
    RetrievalDefaults,
    ServeDefaults,
    apply_provider_secret,
    get_config_paths,
    load_cli_config,
    load_provider_secrets,
    save_cli_config,
    save_provider_secrets,
    set_config_value,
    unset_config_value,
)


def test_user_config_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLOGRAG_CONFIG_DIR", str(tmp_path / "config-root"))

    save_cli_config(
        CLIConfig(
            default_blog_dir="/tmp/blog",
            default_index_dir="/tmp/index",
            build=BuildDefaults(
                concept_extractor="llm",
                llm_provider="mistral",
                llm_model="mistral-small",
                llm_base_url="https://api.mistral.ai/v1/chat/completions",
                llm_api_key_env_var="MISTRAL_API_KEY",
                llm_batch_size=2,
                llm_max_concepts_per_paragraph=9,
                llm_output_contract_mode="json_object",
                labelgen_cache_dir="/tmp/cache",
            ),
            serve=ServeDefaults(
                host="127.0.0.1",
                port=8877,
            ),
            retrieval=RetrievalDefaults(
                retrieval_strategy="label_gate_semantic_rank",
                label_free_fallback_strategy="concept_overlap_semantic_rerank",
            ),
        )
    )
    save_provider_secrets(
        ProviderSecrets(
            mistral="secret-value",
        )
    )

    loaded_config = load_cli_config()
    loaded_secrets = load_provider_secrets()
    paths = get_config_paths()

    assert loaded_config.default_blog_dir == "/tmp/blog"
    assert loaded_config.default_index_dir == "/tmp/index"
    assert loaded_config.build.llm_provider == "mistral"
    assert loaded_config.build.llm_model == "mistral-small"
    assert loaded_config.serve.port == 8877
    assert loaded_config.retrieval.retrieval_strategy == "label_gate_semantic_rank"
    assert loaded_config.retrieval.label_free_fallback_strategy == "concept_overlap_semantic_rerank"
    assert loaded_secrets.mistral == "secret-value"
    assert paths.config_path.is_file()
    assert paths.secrets_path.is_file()


def test_apply_provider_secret_temporarily_sets_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    with apply_provider_secret(
        provider="mistral",
        configured_env_var=None,
        secrets=ProviderSecrets(mistral="secret-value"),
    ):
        assert os.environ["MISTRAL_API_KEY"] == "secret-value"

    assert "MISTRAL_API_KEY" not in os.environ


def test_retrieval_config_values_can_be_set_and_unset() -> None:
    config = CLIConfig()

    set_config_value(config, "retrieval.retrieval_strategy", "label_gate_semantic_rank")
    set_config_value(
        config,
        "retrieval.label_free_fallback_strategy",
        "concept_overlap_semantic_rerank",
    )

    assert config.retrieval.retrieval_strategy == "label_gate_semantic_rank"
    assert config.retrieval.label_free_fallback_strategy == "concept_overlap_semantic_rerank"

    unset_config_value(config, "retrieval.retrieval_strategy")
    unset_config_value(config, "retrieval.label_free_fallback_strategy")

    assert config.retrieval.retrieval_strategy is None
    assert config.retrieval.label_free_fallback_strategy is None
