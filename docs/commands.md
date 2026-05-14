# Command Reference

This document is the detailed CLI reference for `blograg 0.0.2`.

For the shorter product overview and first-run path, see the repository
`README.md`.

## Top-Level Commands

```text
blograg --version
blograg build
blograg serve
blograg start
blograg stop
blograg status
blograg logs
blograg doctor
blograg register
blograg config ...
```

`blograg --version` prints the installed CLI version and exits.

## `blograg build`

Build a fresh local index from one Jekyll-style blog directory.

Typical usage:

```bash
blograg build --blog-dir /path/to/blog --index-dir /path/to/index
```

Common options:

- `--concept-extractor {heuristic|spacy|llm}`
- `--llm-provider {openai|mistral|qwen|ollama|deepseek}`
- `--llm-model TEXT`
- `--llm-base-url TEXT`
- `--llm-api-key-env-var TEXT`
- `--labelgen-cache-dir TEXT`
- `--llm-batch-size INTEGER`
- `--llm-max-concepts-per-paragraph INTEGER`
- `--llm-output-contract-mode {auto|json_schema|json_object|prompt_only}`

Notes:

- `build` is a full rebuild command.
- It does not reuse incremental index state.
- Retrieval strategy settings are not part of build-time behavior.
- If `--labelgen-cache-dir` is not set, `blograg` defaults to
  `labelgen-cache` under the resolved `blograg` config directory.
- In `llm` mode, the build summary reports the effective cache path along with
  the provider and model.

## `blograg serve`

Load an existing index and start the MCP server in the foreground.

Typical usage:

```bash
blograg serve --index-dir /path/to/index
```

Useful options:

- `--host TEXT`
- `--port INTEGER`
- `--labelgen-cache-dir TEXT`
- `--retrieval-strategy {greedy_label_coverage_semantic_rerank|label_gate_semantic_rank}`
- `--label-free-fallback-strategy {concept_overlap_only|concept_overlap_semantic_rerank|concept_gate_semantic_rank|semantic_only}`

Notes:

- `serve` does not rebuild automatically.
- Runtime retrieval strategy overrides are applied after the index is loaded.

## `blograg start`

Start the HTTP MCP server in the background and wait for readiness.

Typical usage:

```bash
blograg start --index-dir /path/to/index
```

Useful options:

- `--host TEXT`
- `--port INTEGER`
- `--retrieval-strategy ...`
- `--label-free-fallback-strategy ...`
- `--pid-file FILE`
- `--log-file FILE`
- `--force-restart`

Notes:

- `start` launches `blograg serve` in a managed subprocess.
- Runtime retrieval overrides are forwarded to that subprocess.

## `blograg stop`

Stop the managed background server.

Typical usage:

```bash
blograg stop
```

Optional override:

- `--pid-file FILE`

## `blograg status`

Inspect managed process state and HTTP readiness.

Typical usage:

```bash
blograg status
```

Useful options:

- `--pid-file FILE`
- `--log-file FILE`
- `--host TEXT`
- `--port INTEGER`
- `--url TEXT`

Notes:

- `status` reports managed process state and HTTP readiness.
- It also reports the effective LabelGen cache path used by local defaults.
- Retrieval strategy overrides are not shown here because one-off `start`
  overrides do not persist back to config.

## `blograg logs`

Inspect the managed server log.

Typical usage:

```bash
blograg logs
blograg logs --tail 100
blograg logs --follow
blograg logs --tail 100 --follow
```

Notes:

- Default behavior prints the last 50 lines.
- `--follow` keeps streaming appended output until interrupted.

## `blograg doctor`

Run a local diagnostics pass.

Current checks include:

- config and secrets file presence
- index path and artifact completeness
- managed server process state
- HTTP readiness
- configured retrieval defaults
- effective LabelGen cache path
- client executable presence
- client registration state
- LLM provider/model/credential requirements when using `llm`

Typical usage:

```bash
blograg doctor
```

## `blograg register`

Register the MCP endpoint with a local client, or inspect binding state.

Register one client:

```bash
blograg register --client codex
blograg register --client openclaw
```

Inspect binding state:

```bash
blograg register --show
blograg register --show --server-name blograg-local
```

Useful options:

- `--client {codex|openclaw}`
- `--server-name TEXT`
- `--host TEXT`
- `--port INTEGER`
- `--url TEXT`
- `--show`

Rules:

- `--show` and `--client` are mutually exclusive.
- A successful registration prints a suggested next step to verify bindings.

## `blograg config`

Persistent user-level defaults and local provider secrets.

Subcommands:

```text
blograg config path
blograg config show
blograg config set
blograg config unset
blograg config set-secret
blograg config unset-secret
blograg config wizard
```

### `blograg config path`

Show resolved config file locations.

### `blograg config show`

Show persisted config and masked secret state.

```bash
blograg config show
blograg config show --all
```

`--all` includes unset values and documented runtime defaults.

### `blograg config set`

Set one persisted config key.

Examples:

```bash
blograg config set default_index_dir /path/to/index
blograg config set build.llm_model mistral-small
blograg config set retrieval.retrieval_strategy label_gate_semantic_rank
```

### `blograg config unset`

Unset one persisted config key.

Example:

```bash
blograg config unset retrieval.retrieval_strategy
```

### `blograg config set-secret`

Persist one provider API key locally.

Examples:

```bash
blograg config set-secret mistral --api-key your-key-here
blograg config set-secret deepseek
```

If `--api-key` is omitted, the command prompts interactively.

### `blograg config unset-secret`

Remove one stored provider API key.

### `blograg config wizard`

Interactive first-run setup for:

- default blog and index paths
- default serve host/port
- build defaults
- optional LLM provider settings
- optional provider API key storage

Recommended first-run path:

```bash
blograg config wizard
blograg build
blograg start
blograg register --client codex
```
