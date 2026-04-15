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

## MCP Client Setup

`blograg` exposes an MCP server over Streamable HTTP through:

```bash
blograg serve --index-dir /path/to/index
```

For project-local client registration, this repository also includes:

- `scripts/serve_mcp.sh`
  - starts the HTTP MCP server in the foreground from this repository's `.venv`
  - reads the index path from `BLOGRAG_INDEX_DIR`
  - automatically loads `.env.local` from the repository root when present
  - defaults to `index/` in the repository root
- `scripts/restart_mcp_http.sh`
  - restarts the local HTTP MCP server in the background
  - writes a PID file at `/tmp/blograg-mcp-http.pid`
  - writes logs to `/tmp/blograg-mcp-http.log`
- `scripts/stop_mcp_http.sh`
  - stops the background HTTP MCP server started through the PID file
- `scripts/setup_mcp.sh`
  - optionally builds the index
  - restarts the local HTTP MCP server
  - registers the server URL for `codex`, `openclaw`, or both

## Service Lifecycle

Use the wrapper scripts when you want a project-local HTTP MCP service:

Start in the foreground for local debugging:

```bash
bash scripts/serve_mcp.sh
```

Restart in the background:

```bash
bash scripts/restart_mcp_http.sh
```

Stop the background server:

```bash
bash scripts/stop_mcp_http.sh
```

Register the MCP endpoint for Codex and OpenClaw:

```bash
bash scripts/setup_mcp.sh --client both
```

`setup_mcp.sh` is a convenience entry point for registration. It is not a
process manager. If you need to restart or stop the local HTTP service later,
use `restart_mcp_http.sh` and `stop_mcp_http.sh` directly.

One-command registration examples:

```bash
scripts/setup_mcp.sh --client codex --blog-dir /path/to/blog
scripts/setup_mcp.sh --client openclaw --blog-dir /path/to/blog
scripts/setup_mcp.sh --client both --blog-dir /path/to/blog
```

If you already have a built index, point registration at it directly:

```bash
scripts/setup_mcp.sh \
  --client both \
  --index-dir /path/to/index
```

If you want to rebuild before registration:

```bash
scripts/setup_mcp.sh \
  --client codex \
  --blog-dir /path/to/blog \
  --index-dir /path/to/index \
  --rebuild
```

The generated registrations point clients at a local Streamable HTTP endpoint
instead of a stdio subprocess. That avoids stdio transport breakage from noisy
third-party model-loading logs and keeps Codex/OpenClaw on the same stable URL.

## Local Env File

If your index was built with `--concept-extractor llm`, upstream query analysis
still needs the corresponding provider API key at serve time.

The recommended project-local setup is a root `.env.local` file that is loaded
by `scripts/serve_mcp.sh` before it starts `blograg serve`.

Example `.env.local`:

```bash
MISTRAL_API_KEY=your-key-here
```

Then the one-command setup remains the same:

```bash
bash scripts/setup_mcp.sh --client both --index-dir /path/to/index
```

Because the wrapper script loads `.env.local`, neither Codex nor OpenClaw needs
to store the API key directly in their MCP configuration.

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
MISTRAL_API_KEY=your-key-here \
blograg serve --index-dir /path/to/index
```

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
