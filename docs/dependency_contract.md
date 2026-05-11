# Dependency Contract

This document records the minimum upstream surface that `blograg 0.0.0`
intentionally depends on.

The goal is not to restate all of `labelrag` or `labelgen`, but to make the
release-facing dependency boundary explicit.

## Dependency Targets

Current intended dependency targets:

- `labelrag >= 0.1.3`
- `paralabelgen == 0.2.3` indirectly through `labelrag`

Contract boundary:

- `blograg` depends directly on `labelrag`
- `blograg` does not orchestrate `paralabelgen` as a separate top-level
  pipeline layer
- extraction, concept assignment, and label assignment remain upstream
  responsibilities through `labelrag`

## `labelrag` Contract

`blograg` depends on the following public `labelrag` surface:

- `RAGPipeline`
- `RAGPipelineConfig`
- `RetrievalConfig`
- `build_context(...)`
- `save(...)`
- `load(...)`

Current assumptions:

- `RAGPipeline.fit(...)` accepts paragraph text input that `blograg` can
  provide
- `build_context(question)` returns paragraph retrieval results suitable for
  powering `retrieve_paragraphs`
- retrieved paragraph items expose:
  - `paragraph_id`
  - `text`
  - `retrieval_score`
  - `retrieval_score_kind`
- retrieval metadata exposes at least:
  - `retrieval_strategy`
- `save(...)` and `load(...)` persist and restore a static fitted state
- the pipeline can be constructed through `RAGPipeline(config)` without a
  mandatory explicit provider override

Current `labelrag` capabilities that matter to `blograg`:

- paragraph-level retrieval
- semantic reranking on the main path
- configurable label-free fallback strategies
- retrieval score kind reporting
- persistence via snapshot save/load

## `labelgen` And `paralabelgen` Contract

`blograg` does not call `labelgen` or `paralabelgen` as its own orchestration
layer, but it depends on `labelrag` behavior that is shaped by them.

Current assumptions:

- the default local path is still stable with heuristic extraction
- the LLM extraction path remains upstream behavior
- concept extraction, community detection, and label assignment semantics are
  not reimplemented inside `blograg`

`blograg` should not duplicate or override:

- concept extraction logic
- label assignment logic
- community inference logic

## Config Exposure Rules

`blograg 0.0.0` intentionally exposes only a narrow subset of upstream
configuration.

Current exposed areas:

- build-time extractor selection and LLM extraction settings
- runtime retrieval strategy selection
- runtime label-free fallback strategy selection

Current non-goals:

- full pass-through of all upstream config models
- full inspection or debugging APIs from upstream libraries

## Non-Goals For This Contract

This contract does not promise:

- compatibility with future incremental-ingestion APIs
- sqlite backend support
- multi-backend storage selection
- direct orchestration of `paralabelgen`
- a stable promise that every future upstream retrieval metadata field will be
  forwarded automatically

Those areas belong to later versions if and when `blograg` expands beyond the
initial public `0.0.0` release.
