# blograg

`blograg` is a local MCP-oriented retrieval tool for one Jekyll-style blog.
It uses [`labelrag`](https://github.com/HuRuilizhen/labelrag) as the retrieval
core and treats heading-delimited markdown sections as the paragraph unit.

Concept extraction and retrieval configuration are intentionally thin wrappers
around upstream `labelrag` and
[`labelgen`](https://github.com/HuRuilizhen/labelgen) capabilities. For deeper
configuration semantics, prefer the upstream repositories and their public API
documentation.

## MVP Scope

The current MVP supports:

- one local blog directory
- Jekyll-style front matter parsing
- heading-delimited paragraph segmentation
- full rebuild only
- a minimal MCP tool surface centered on `retrieve_paragraphs`

The current MVP does not support:

- token chunking or overlap windows
- incremental indexing
- multiple blog roots
- runtime rebuilds from the MCP server
- alternate storage backends

## Installation

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## CLI

Initialize or inspect persistent defaults and local provider secrets:

```bash
blograg config wizard
blograg config show
```

Build a fresh local index:

```bash
blograg build --blog-dir /path/to/blog --index-dir /path/to/index
```

Serve an existing index over MCP Streamable HTTP:

```bash
blograg serve --index-dir /path/to/index
```

`serve` does not rebuild automatically. If the index is missing or incomplete,
run `build` first.

Start the HTTP MCP server in the background:

```bash
blograg start --index-dir /path/to/index
```

Inspect managed server status:

```bash
blograg status
```

Inspect managed client bindings:

```bash
blograg register --show
```

Inspect managed server logs:

```bash
blograg logs
blograg logs --follow
```

Stop the managed background server:

```bash
blograg stop
```

## MCP Client Setup

`blograg` exposes an MCP server over Streamable HTTP through:

```bash
blograg serve --index-dir /path/to/index
```

Register the MCP endpoint for Codex and/or OpenClaw:

```bash
blograg register --client codex
blograg register --client openclaw
```

If you want a managed local HTTP service first:

```bash
blograg start --index-dir /path/to/index
blograg register --client codex
blograg register --client openclaw
```

You can also register a specific URL directly:

```bash
blograg register \
  --client codex \
  --server-name blograg-local \
  --url http://127.0.0.1:8765/mcp
```

The generated registrations point clients at a local Streamable HTTP endpoint
instead of a stdio subprocess. That avoids stdio transport breakage from noisy
third-party model-loading logs and keeps Codex/OpenClaw on the same stable URL.

## Persistent Config

`blograg` now supports persistent user-level config and secrets:

- `config.toml`
- `secrets.toml`

Default locations:

- macOS/Linux: `~/.config/blograg/`
- Windows: `%AppData%/blograg/`

Useful commands:

```bash
blograg config path
blograg config show
blograg config set default_index_dir /path/to/index
blograg config set build.llm_model mistral-small
blograg config set-secret mistral --api-key your-key-here
```

`config show` masks secret values and reports only whether each provider key is
configured.

## LLM Credentials

If your index was built with `--concept-extractor llm`, upstream query analysis
still needs the corresponding provider API key at serve time.

You can either:

- store the key through `blograg config set-secret ...`
- or continue providing it through environment variables

Example persistent secret setup:

```bash
blograg config set-secret mistral --api-key your-key-here
```

Example environment variable setup:

```bash
MISTRAL_API_KEY=your-key-here
```

If you configure both an environment variable and a stored secret, explicit CLI
arguments still take precedence over persisted defaults.

## Concept Extraction Modes

`blograg build` exposes the current upstream concept extraction modes:

- `spacy`
- `heuristic`
- `llm`

The CLI option names are thin wrappers around upstream `labelgen` configuration.
The following build options currently map directly to LLM extraction settings:

- `--concept-extractor`
- `--llm-provider`
- `--llm-model`
- `--llm-base-url`
- `--llm-api-key-env-var`
- `--llm-batch-size`
- `--llm-max-concepts-per-paragraph`
- `--llm-output-contract-mode`

## LLM Build Example

Example Mistral build:

```bash
MISTRAL_API_KEY=your-key-here \
blograg build \
  --blog-dir /path/to/blog \
  --index-dir /path/to/index \
  --concept-extractor llm \
  --llm-provider mistral \
  --llm-model mistral-small \
  --llm-base-url https://api.mistral.ai/v1/chat/completions \
  --llm-batch-size 1
```

Practical notes:

- `--llm-batch-size 1` is the safest current setting for Mistral concept
  extraction in real builds.
- `--llm-base-url` accepts either a provider base URL or a full chat-completions
  endpoint URL, following upstream behavior.

## LLM Serve Requirements

If an index was built with `--concept-extractor llm`, the `serve` process also
needs access to the corresponding provider API key because query analysis still
uses the fitted upstream concept-extraction path.

Example:

```bash
blograg serve --index-dir /path/to/index
```

When the matching provider secret is stored through `blograg config set-secret`,
you do not need to export the API key again before `serve`.

## MCP Tool

The server exposes one tool:

```text
retrieve_paragraphs(query: str, top_k: int = 5)
```

Each result includes:

- `paragraph_id`
- `text`
- `post_title`
- `slug`
- `section_heading`
- `trace.retrieval_strategy`
- `trace.score`
- `trace.score_kind`

## Paragraph Segmentation Rules

- A paragraph begins at a markdown heading.
- A paragraph ends immediately before the next heading.
- Content before the first heading is preserved as a standalone intro paragraph.
- Intro paragraphs have no `section_heading`.
- Paragraph IDs follow `slug::pNNN`.

## Index Layout

`blograg build` writes an outer `blograg` directory inside the chosen index root:

```text
/path/to/index/
  blograg/
    manifest.json
    paragraphs.json
    labelrag/
      config.json.gz
      corpus_index.json.gz
      fit_result.json.gz
      label_generator.json.gz
      manifest.json.gz
      paragraph_embeddings.npz
```

The outer layer stores `blograg`-specific metadata such as the outer schema
version and `blograg` package version. The inner `labelrag` directory is a
standard persisted `labelrag` snapshot and remains responsible for upstream
pipeline configuration details.

## Runtime Notes

- The default MVP configuration uses heuristic concept extraction instead of the
  spaCy extractor, so it does not require downloading a spaCy language model.
- The default embedding provider is still `sentence-transformers`, so the first
  real build or first real query may download the configured embedding model if
  it is not already cached locally.
- When using `spacy`, install a compatible spaCy pipeline such as
  `en_core_web_sm` as described in the upstream
  [`labelrag` README](https://github.com/HuRuilizhen/labelrag/blob/main/README.md).

## Upstream References

- `labelrag` repository:
  `https://github.com/HuRuilizhen/labelrag`
- `labelrag` README:
  `https://github.com/HuRuilizhen/labelrag/blob/main/README.md`
- `labelrag` public API notes:
  `https://github.com/HuRuilizhen/labelrag/blob/main/docs/public_api.md`
- `labelgen` repository:
  `https://github.com/HuRuilizhen/labelgen`
- `labelgen` public API notes:
  `https://github.com/HuRuilizhen/labelgen/blob/main/docs/public_api.md`

## Development Checks

Run the local quality gates before committing:

```bash
pytest
ruff check .
ruff format --check .
pyright
```
