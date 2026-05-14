# blograg

`blograg` is a local MCP-oriented retrieval tool for one Jekyll-style blog.
It uses [`labelrag`](https://github.com/HuRuilizhen/labelrag) as the retrieval
core and treats heading-delimited markdown sections as the paragraph unit.
It is designed to:

- build a paragraph index from one Jekyll-style repository
- serve that index over MCP Streamable HTTP
- inspect service state from the CLI and a lightweight browser page
- register the HTTP endpoint with local MCP clients such as Codex or OpenClaw

Detailed command reference lives in
[`docs/commands.md`](https://github.com/HuRuilizhen/blograg-mcp/blob/main/docs/commands.md).

## Installation

Recommended for most users:

```bash
pipx install blograg
```

If you prefer `pip`:

```bash
python -m pip install blograg
```

If you use Homebrew:

```bash
brew install HuRuilizhen/tap/blograg
```

## Quick Start

Initialize local defaults and optional provider secrets:

```bash
blograg config wizard
```

Build an index:

```bash
blograg build --blog-dir /path/to/blog --index-dir /path/to/index
```

Start the managed HTTP service:

```bash
blograg start --index-dir /path/to/index
```

Inspect service state:

```bash
blograg status
blograg logs --follow
blograg doctor
```

Open the browser status page:

```text
http://127.0.0.1:8765/
```

Register the MCP endpoint with a client:

```bash
blograg register --client codex
blograg register --show
```

## Core Commands

Most day-to-day usage is centered on:

- `blograg config wizard`
- `blograg build`
- `blograg serve`
- `blograg start`
- `blograg status`
- `blograg logs`
- `blograg doctor`
- `blograg register`

For command-by-command examples and option summaries, see
[`docs/commands.md`](https://github.com/HuRuilizhen/blograg-mcp/blob/main/docs/commands.md).

## Persistent Config

`blograg` stores user-level config and secrets in:

- `config.toml`
- `secrets.toml`

Default locations:

- macOS/Linux: `~/.config/blograg/`
- Windows: `%AppData%/blograg/`

Useful commands:

```bash
blograg config path
blograg config show
blograg config show --all
blograg config set default_index_dir /path/to/index
blograg config set retrieval.retrieval_strategy label_gate_semantic_rank
blograg config set-secret mistral --api-key your-key-here
```

`config show` masks secret values and only reports whether each provider key is
configured.

## MCP Service Model

`blograg serve` loads an existing index and starts the MCP server. It does not
rebuild automatically. If the index is missing or incomplete, run `build`
first.

The default transport is Streamable HTTP. The default HTTP binding is:

- host: `127.0.0.1`
- port: `8765`

If you need LAN access, bind explicitly:

```bash
blograg serve --host 0.0.0.0 --port 8765
```

Current HTTP endpoints:

- `/mcp`
- `/`
- `/healthz`

The browser page at `/` is a lightweight status page, not a separate web app.

## MCP Client Registration

Register the local endpoint with one client at a time:

```bash
blograg register --client codex
blograg register --client openclaw
```

Inspect current registration state:

```bash
blograg register --show
blograg register --show --server-name blograg-local
```

You can also register an explicit URL:

```bash
blograg register \
  --client codex \
  --server-name blograg-local \
  --url http://127.0.0.1:8765/mcp
```

## LLM Usage

`blograg build` supports the upstream extraction modes:

- `heuristic`
- `spacy`
- `llm`

Example LLM build:

```bash
MISTRAL_API_KEY=your-key-here \
blograg build \
  --blog-dir /path/to/blog \
  --index-dir /path/to/index \
  --concept-extractor llm \
  --llm-provider mistral \
  --llm-model mistral-small
```

If an index was built with `--concept-extractor llm`, query analysis at serve
time still needs access to the corresponding provider API key. You can provide
it through:

- `blograg config set-secret ...`
- environment variables such as `MISTRAL_API_KEY`

## Retrieval Output

The server currently exposes one tool:

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

## Index Layout

`blograg build` writes an outer `blograg` directory inside the chosen index
root:

```text
/path/to/index/
  blograg/
    manifest.json
    paragraphs.json
    labelrag/
      ...
```

The outer layer stores `blograg`-specific metadata and paragraph source
metadata. The inner `labelrag` directory is a normal persisted upstream
snapshot.

## Runtime Notes

- The default build mode is `heuristic`, so the default path does not require a
  spaCy model download.
- The default embedding provider still comes from upstream `labelrag`, so the
  first real build or query may download the configured embedding model.
- Advanced retrieval runtime settings live under persisted `retrieval.*`
  config keys and can also be overridden through `serve` and `start`.

## Development Checks

```bash
pytest
ruff check .
ruff format --check .
pyright
python -m build
twine check dist/*
```
