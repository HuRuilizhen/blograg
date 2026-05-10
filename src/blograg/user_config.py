"""User-level CLI configuration and secret storage helpers."""

from __future__ import annotations

import os
import stat
import tomllib
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, cast

from blograg.config import (
    ConceptExtractorMode,
    LabelFreeFallbackStrategy,
    LLMProvider,
    RetrievalStrategy,
)

TransportMode = Literal["streamable-http", "stdio"]
LLMOutputContractMode = Literal["auto", "json_schema", "json_object", "prompt_only"]
SecretProvider = Literal["openai", "mistral", "qwen", "ollama", "deepseek"]
_DEFAULT_API_KEY_ENV_VARS: dict[SecretProvider, str] = {
    "openai": "OPENAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "ollama": "OLLAMA_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}
_CONFIG_DIR_ENV_VAR = "BLOGRAG_CONFIG_DIR"
_CONCEPT_EXTRACTOR_VALUES = {"spacy", "heuristic", "llm"}
_LLM_PROVIDER_VALUES = {"openai", "mistral", "qwen", "ollama", "deepseek"}
_LLM_OUTPUT_CONTRACT_VALUES = {"auto", "json_schema", "json_object", "prompt_only"}
_TRANSPORT_VALUES = {"streamable-http", "stdio"}
_RETRIEVAL_STRATEGY_VALUES = {
    "greedy_label_coverage_semantic_rerank",
    "label_gate_semantic_rank",
}
_LABEL_FREE_FALLBACK_STRATEGY_VALUES = {
    "concept_overlap_only",
    "concept_overlap_semantic_rerank",
    "concept_gate_semantic_rank",
    "semantic_only",
}
TomlScalar = str | int | bool
TomlValue = TomlScalar | dict[str, "TomlValue"]
TomlTable = dict[str, TomlValue]


@dataclass(slots=True)
class BuildDefaults:
    """Persisted CLI defaults for `blograg build`."""

    concept_extractor: ConceptExtractorMode | None = field(
        default=None,
        metadata={"display_default": "default: heuristic"},
    )
    llm_provider: LLMProvider | None = field(
        default=None,
        metadata={"display_default": "default: mistral"},
    )
    llm_model: str | None = field(default=None, metadata={"display_default": "unset"})
    llm_base_url: str | None = field(default=None, metadata={"display_default": "unset"})
    llm_api_key_env_var: str | None = field(default=None, metadata={"display_default": "unset"})
    llm_batch_size: int | None = field(default=None, metadata={"display_default": "default: 8"})
    llm_max_concepts_per_paragraph: int | None = field(
        default=None,
        metadata={"display_default": "default: 12"},
    )
    llm_output_contract_mode: LLMOutputContractMode | None = field(
        default=None,
        metadata={"display_default": "default: auto"},
    )
    labelgen_cache_dir: str | None = field(
        default=None,
        metadata={"display_default": "default: upstream default (.labelgen-cache)"},
    )


@dataclass(slots=True)
class ServeDefaults:
    """Persisted CLI defaults for `blograg serve`."""

    host: str | None = field(default=None, metadata={"display_default": "default: 127.0.0.1"})
    port: int | None = field(default=None, metadata={"display_default": "default: 8765"})
    transport: TransportMode | None = field(
        default=None,
        metadata={"display_default": "default: streamable-http"},
    )


@dataclass(slots=True)
class RetrievalDefaults:
    """Persisted CLI defaults for retrieval behavior."""

    retrieval_strategy: RetrievalStrategy | None = field(
        default=None,
        metadata={"display_default": "default: greedy_label_coverage_semantic_rerank"},
    )
    label_free_fallback_strategy: LabelFreeFallbackStrategy | None = field(
        default=None,
        metadata={"display_default": "default: semantic_only"},
    )


@dataclass(slots=True)
class CLIConfig:
    """Top-level persisted user configuration."""

    default_blog_dir: str | None = field(default=None, metadata={"display_default": "unset"})
    default_index_dir: str | None = field(default=None, metadata={"display_default": "unset"})
    build: BuildDefaults = field(default_factory=BuildDefaults)
    serve: ServeDefaults = field(default_factory=ServeDefaults)
    retrieval: RetrievalDefaults = field(default_factory=RetrievalDefaults)


@dataclass(slots=True)
class ProviderSecrets:
    """Provider API keys persisted locally for CLI usage."""

    openai: str | None = None
    mistral: str | None = None
    qwen: str | None = None
    ollama: str | None = None
    deepseek: str | None = None


@dataclass(slots=True)
class ConfigPaths:
    """Resolved filesystem locations for user config and secrets."""

    config_dir: Path
    config_path: Path
    secrets_path: Path
    pid_path: Path
    log_path: Path


def get_config_paths() -> ConfigPaths:
    """Return resolved user config paths for this machine."""

    override_dir = os.environ.get(_CONFIG_DIR_ENV_VAR)
    if override_dir:
        config_dir = Path(override_dir).expanduser().resolve()
    elif os.name == "nt":
        appdata_dir = os.environ.get("APPDATA")
        root_dir = Path(appdata_dir) if appdata_dir else Path.home() / "AppData" / "Roaming"
        config_dir = root_dir / "blograg"
    else:
        config_dir = Path.home() / ".config" / "blograg"

    return ConfigPaths(
        config_dir=config_dir,
        config_path=config_dir / "config.toml",
        secrets_path=config_dir / "secrets.toml",
        pid_path=config_dir / "server.pid",
        log_path=config_dir / "server.log",
    )


def load_cli_config() -> CLIConfig:
    """Load persisted user config when it exists."""

    paths = get_config_paths()
    if not paths.config_path.is_file():
        return CLIConfig()
    raw_payload = _load_toml_object(paths.config_path)
    build_section = _load_toml_section(raw_payload.get("build"))
    serve_section = _load_toml_section(raw_payload.get("serve"))
    retrieval_section = _load_toml_section(raw_payload.get("retrieval"))
    return CLIConfig(
        default_blog_dir=_optional_str(raw_payload.get("default_blog_dir")),
        default_index_dir=_optional_str(raw_payload.get("default_index_dir")),
        build=BuildDefaults(
            concept_extractor=cast(
                ConceptExtractorMode | None,
                _optional_literal(
                    build_section.get("concept_extractor"), _CONCEPT_EXTRACTOR_VALUES
                ),
            ),
            llm_provider=cast(
                LLMProvider | None,
                _optional_literal(build_section.get("llm_provider"), _LLM_PROVIDER_VALUES),
            ),
            llm_model=_optional_str(build_section.get("llm_model")),
            llm_base_url=_optional_str(build_section.get("llm_base_url")),
            llm_api_key_env_var=_optional_str(build_section.get("llm_api_key_env_var")),
            llm_batch_size=_optional_int(build_section.get("llm_batch_size")),
            llm_max_concepts_per_paragraph=_optional_int(
                build_section.get("llm_max_concepts_per_paragraph")
            ),
            llm_output_contract_mode=cast(
                LLMOutputContractMode | None,
                _optional_literal(
                    build_section.get("llm_output_contract_mode"),
                    _LLM_OUTPUT_CONTRACT_VALUES,
                ),
            ),
            labelgen_cache_dir=_optional_str(build_section.get("labelgen_cache_dir")),
        ),
        serve=ServeDefaults(
            host=_optional_str(serve_section.get("host")),
            port=_optional_int(serve_section.get("port")),
            transport=cast(
                TransportMode | None,
                _optional_literal(serve_section.get("transport"), _TRANSPORT_VALUES),
            ),
        ),
        retrieval=RetrievalDefaults(
            retrieval_strategy=cast(
                RetrievalStrategy | None,
                _optional_literal(
                    retrieval_section.get("retrieval_strategy"),
                    _RETRIEVAL_STRATEGY_VALUES,
                ),
            ),
            label_free_fallback_strategy=cast(
                LabelFreeFallbackStrategy | None,
                _optional_literal(
                    retrieval_section.get("label_free_fallback_strategy"),
                    _LABEL_FREE_FALLBACK_STRATEGY_VALUES,
                ),
            ),
        ),
    )


def save_cli_config(config: CLIConfig) -> None:
    """Persist user config to disk."""

    paths = get_config_paths()
    payload = _drop_none_values(asdict(config))
    _write_toml(paths.config_path, payload)


def load_provider_secrets() -> ProviderSecrets:
    """Load persisted provider secrets when available."""

    paths = get_config_paths()
    if not paths.secrets_path.is_file():
        return ProviderSecrets()
    raw_payload = _load_toml_object(paths.secrets_path)
    providers_section = _load_toml_section(raw_payload.get("providers"))
    return ProviderSecrets(
        openai=_load_provider_api_key(providers_section, "openai"),
        mistral=_load_provider_api_key(providers_section, "mistral"),
        qwen=_load_provider_api_key(providers_section, "qwen"),
        ollama=_load_provider_api_key(providers_section, "ollama"),
        deepseek=_load_provider_api_key(providers_section, "deepseek"),
    )


def save_provider_secrets(secrets: ProviderSecrets) -> None:
    """Persist provider secrets to disk with restricted permissions."""

    paths = get_config_paths()
    payload = cast(
        TomlTable,
        {
            "providers": _drop_none_values(
                {
                    "openai": {"api_key": secrets.openai},
                    "mistral": {"api_key": secrets.mistral},
                    "qwen": {"api_key": secrets.qwen},
                    "ollama": {"api_key": secrets.ollama},
                    "deepseek": {"api_key": secrets.deepseek},
                }
            )
        },
    )
    _write_toml(paths.secrets_path, payload)
    if os.name != "nt":
        paths.secrets_path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def config_value_map(config: CLIConfig) -> dict[str, str]:
    """Return flattened config values for human-readable display."""

    return dict(_iter_config_display_entries(config, include_unset=False))


def config_value_map_all(config: CLIConfig) -> dict[str, str]:
    """Return all known config values, including unset entries and runtime defaults."""

    return dict(_iter_config_display_entries(config, include_unset=True))


def secret_status_map(secrets: ProviderSecrets) -> dict[str, bool]:
    """Return masked secret presence state for display."""

    return {
        "openai": secrets.openai is not None,
        "mistral": secrets.mistral is not None,
        "qwen": secrets.qwen is not None,
        "ollama": secrets.ollama is not None,
        "deepseek": secrets.deepseek is not None,
    }


def set_config_value(config: CLIConfig, key: str, raw_value: str) -> None:
    """Set one persisted config key from CLI input."""

    if key == "default_blog_dir":
        config.default_blog_dir = raw_value
        return
    if key == "default_index_dir":
        config.default_index_dir = raw_value
        return
    if key == "serve.host":
        config.serve.host = raw_value
        return
    if key == "serve.port":
        config.serve.port = _parse_positive_int(key, raw_value)
        return
    if key == "serve.transport":
        config.serve.transport = cast(
            TransportMode,
            _parse_literal(key, raw_value, _TRANSPORT_VALUES),
        )
        return
    if key == "build.concept_extractor":
        config.build.concept_extractor = cast(
            ConceptExtractorMode,
            _parse_literal(key, raw_value, _CONCEPT_EXTRACTOR_VALUES),
        )
        return
    if key == "build.llm_provider":
        config.build.llm_provider = cast(
            LLMProvider,
            _parse_literal(key, raw_value, _LLM_PROVIDER_VALUES),
        )
        return
    if key == "build.llm_model":
        config.build.llm_model = raw_value
        return
    if key == "build.llm_base_url":
        config.build.llm_base_url = raw_value
        return
    if key == "build.llm_api_key_env_var":
        config.build.llm_api_key_env_var = raw_value
        return
    if key == "build.llm_batch_size":
        config.build.llm_batch_size = _parse_positive_int(key, raw_value)
        return
    if key == "build.llm_max_concepts_per_paragraph":
        config.build.llm_max_concepts_per_paragraph = _parse_positive_int(key, raw_value)
        return
    if key == "build.llm_output_contract_mode":
        config.build.llm_output_contract_mode = cast(
            LLMOutputContractMode,
            _parse_literal(key, raw_value, _LLM_OUTPUT_CONTRACT_VALUES),
        )
        return
    if key == "build.labelgen_cache_dir":
        config.build.labelgen_cache_dir = raw_value
        return
    if key == "retrieval.retrieval_strategy":
        config.retrieval.retrieval_strategy = cast(
            RetrievalStrategy,
            _parse_literal(key, raw_value, _RETRIEVAL_STRATEGY_VALUES),
        )
        return
    if key == "retrieval.label_free_fallback_strategy":
        config.retrieval.label_free_fallback_strategy = cast(
            LabelFreeFallbackStrategy,
            _parse_literal(key, raw_value, _LABEL_FREE_FALLBACK_STRATEGY_VALUES),
        )
        return
    raise KeyError(_unknown_config_key_message(key))


def unset_config_value(config: CLIConfig, key: str) -> None:
    """Unset one persisted config key."""

    if key == "default_blog_dir":
        config.default_blog_dir = None
        return
    if key == "default_index_dir":
        config.default_index_dir = None
        return
    if key == "serve.host":
        config.serve.host = None
        return
    if key == "serve.port":
        config.serve.port = None
        return
    if key == "serve.transport":
        config.serve.transport = None
        return
    if key == "build.concept_extractor":
        config.build.concept_extractor = None
        return
    if key == "build.llm_provider":
        config.build.llm_provider = None
        return
    if key == "build.llm_model":
        config.build.llm_model = None
        return
    if key == "build.llm_base_url":
        config.build.llm_base_url = None
        return
    if key == "build.llm_api_key_env_var":
        config.build.llm_api_key_env_var = None
        return
    if key == "build.llm_batch_size":
        config.build.llm_batch_size = None
        return
    if key == "build.llm_max_concepts_per_paragraph":
        config.build.llm_max_concepts_per_paragraph = None
        return
    if key == "build.llm_output_contract_mode":
        config.build.llm_output_contract_mode = None
        return
    if key == "build.labelgen_cache_dir":
        config.build.labelgen_cache_dir = None
        return
    if key == "retrieval.retrieval_strategy":
        config.retrieval.retrieval_strategy = None
        return
    if key == "retrieval.label_free_fallback_strategy":
        config.retrieval.label_free_fallback_strategy = None
        return
    raise KeyError(_unknown_config_key_message(key))


def set_secret_value(secrets: ProviderSecrets, provider: SecretProvider, api_key: str) -> None:
    """Set one provider API key."""

    setattr(secrets, provider, api_key)


def unset_secret_value(secrets: ProviderSecrets, provider: SecretProvider) -> None:
    """Unset one provider API key."""

    setattr(secrets, provider, None)


def resolve_provider_api_key_env_var(
    provider: SecretProvider,
    configured_env_var: str | None = None,
) -> str:
    """Resolve the effective provider API key environment variable name."""

    return configured_env_var or _DEFAULT_API_KEY_ENV_VARS[provider]


@contextmanager
def apply_provider_secret(
    *,
    provider: SecretProvider,
    configured_env_var: str | None,
    secrets: ProviderSecrets,
) -> Generator[None, None, None]:
    """Temporarily expose one configured provider secret through the expected env var."""

    api_key = getattr(secrets, provider)
    if api_key is None:
        yield
        return

    env_var = resolve_provider_api_key_env_var(provider, configured_env_var)
    original_value = os.environ.get(env_var)
    os.environ[env_var] = api_key
    try:
        yield
    finally:
        if original_value is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = original_value


def known_config_keys() -> list[str]:
    """Return supported CLI config keys."""

    return [key for key, _ in _iter_config_display_entries(CLIConfig(), include_unset=True)]


def known_secret_providers() -> list[str]:
    """Return supported secret provider keys."""

    return ["openai", "mistral", "qwen", "ollama", "deepseek"]


def _load_provider_api_key(
    providers_section: dict[str, Any],
    provider: SecretProvider,
) -> str | None:
    provider_section = _load_toml_section(providers_section.get(provider))
    return _optional_str(provider_section.get("api_key"))


def _write_toml(path: Path, payload: TomlTable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_serialize_toml(payload), encoding="utf-8")


def _serialize_toml(payload: TomlTable) -> str:
    scalar_lines: list[str] = []
    section_lines: list[str] = []

    for key, value in payload.items():
        if isinstance(value, dict):
            section_lines.extend(_serialize_toml_section([key], value))
        else:
            scalar_lines.append(f"{key} = {_toml_literal(value)}")

    output_lines = scalar_lines
    if scalar_lines and section_lines:
        output_lines = [*scalar_lines, "", *section_lines]
    elif section_lines:
        output_lines = section_lines
    return "\n".join(output_lines) + "\n"


def _serialize_toml_section(prefix: list[str], payload: TomlTable) -> list[str]:
    scalar_lines: list[str] = []
    nested_sections: list[list[str]] = []

    for key, value in payload.items():
        if isinstance(value, dict):
            nested_sections.append(_serialize_toml_section([*prefix, key], value))
        else:
            scalar_lines.append(f"{key} = {_toml_literal(value)}")

    lines: list[str] = []
    if scalar_lines:
        lines.append(f"[{'.'.join(prefix)}]")
        lines.extend(scalar_lines)

    for section in nested_sections:
        if lines:
            lines.append("")
        lines.extend(section)

    return lines


def _toml_literal(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    raise TypeError(f"Unsupported TOML value type: {type(value)!r}")


def _drop_none_values(payload: dict[str, Any]) -> TomlTable:
    cleaned: TomlTable = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, dict):
            nested = _drop_none_values(cast(dict[str, Any], value))
            if nested:
                cleaned[key] = nested
            continue
        cleaned[key] = cast(TomlScalar, value)
    return cleaned


def _load_toml_object(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw_payload = cast(dict[object, object], tomllib.load(handle))
    return {str(key): value for key, value in raw_payload.items()}


def _load_toml_section(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RuntimeError("Expected TOML table.")
    typed_value = cast(dict[object, object], value)
    return {str(key): item for key, item in typed_value.items()}


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeError("Expected string or null configuration value.")
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise RuntimeError("Expected integer or null configuration value.")
    return value


def _optional_literal(value: object, allowed_values: set[str]) -> str | None:
    parsed = _optional_str(value)
    if parsed is None:
        return None
    if parsed not in allowed_values:
        raise RuntimeError(f"Unsupported configuration value {parsed!r}.")
    return parsed


def _parse_positive_int(key: str, raw_value: str) -> int:
    try:
        parsed = int(raw_value)
    except ValueError as error:
        raise ValueError(f"`{key}` expects an integer value.") from error
    if parsed <= 0:
        raise ValueError(f"`{key}` expects an integer greater than zero.")
    return parsed


def _parse_literal(key: str, raw_value: str, allowed_values: set[str]) -> str:
    if raw_value not in allowed_values:
        supported_values = ", ".join(sorted(allowed_values))
        raise ValueError(f"`{key}` must be one of: {supported_values}.")
    return raw_value


def _iter_config_display_entries(
    value: object,
    *,
    prefix: str = "",
    include_unset: bool,
) -> Generator[tuple[str, str], None, None]:
    if not is_dataclass(value):
        raise TypeError("Expected dataclass config object.")

    for dataclass_field in fields(value):
        key = f"{prefix}.{dataclass_field.name}" if prefix else dataclass_field.name
        field_value = getattr(value, dataclass_field.name)
        if is_dataclass(field_value):
            yield from _iter_config_display_entries(
                field_value,
                prefix=key,
                include_unset=include_unset,
            )
            continue
        if field_value is None:
            if include_unset:
                yield key, str(dataclass_field.metadata.get("display_default", "unset"))
            continue
        yield key, str(field_value)


def _unknown_config_key_message(key: str) -> str:
    return f"Unsupported config key `{key}`. Supported keys: " + ", ".join(known_config_keys())
