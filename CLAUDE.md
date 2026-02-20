# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
cargo build                    # dev build
cargo build --release          # optimized build
cargo run -- ask "question"    # single question
cargo run -- -v ask "question" # verbose (per-hop timing)
cargo run -- eval eval/questions.jsonl  # batch evaluation
```

## Data Preparation (Python, one-time)

```bash
pip install mwparserfromhell sentence-transformers qdrant-client tqdm python-dotenv
python scripts/download_wiki.py           # download simplewiki dump
python scripts/chunk_wiki.py              # chunk into passages → data/passages.jsonl
python scripts/embed_and_upload.py --cloud-inference  # upload with server-side embedding
python scripts/embed_and_upload.py        # or: local embedding with sentence-transformers
```

All scripts are resumable. Use `--fresh` on embed_and_upload.py to recreate the collection.

## Architecture

Multi-hop retrieval agent: **Planner → [Search → Reader] × N → Synthesizer**

- **Planner**: Decomposes question into 1-4 search queries (JSON array)
- **Retriever** (Qdrant): Vector search via cloud inference (Document API)
- **Reader**: Evaluates passages, decides `Continue { follow_up_queries }` or `Synthesize`
- **Synthesizer**: Produces final answer from all accumulated passages

The agent loop is in `src/agent/mod.rs`. Hops are sequential (each depends on the reader's decision). All passages accumulate across hops. Every run writes structured JSON to `logs/runs.jsonl`.

## Key Design Decisions

**LLM via OpenRouter**: All LLM calls use the OpenAI chat completions format via OpenRouter (`src/llm/client.rs`). Any model available on OpenRouter can be used (Anthropic, OpenAI, Minimax, etc.) by setting the model IDs in `.env`. Cost is tracked per-request using `usage.cost` from the OpenRouter response.

**Qdrant cloud inference**: Embedding is done server-side by Qdrant Cloud (Document API). The Python upload script mirrors this with `--cloud-inference` flag.

**Qdrant indexing**: Bulk uploads disable indexing (`indexing_threshold=0`), re-enable after (`20000`). Upserts use `wait=False` for throughput, `wait=True` on final batch.

**Error fallbacks**: Planner defaults to raw question if JSON parse fails. Reader defaults to Synthesize if decision is unclear.

## Configuration

All config via `.env` (loaded by `dotenvy` in Rust, `python-dotenv` in Python). See `.env.example`. Key vars:

- `LLM_API_KEY` — OpenRouter API key (or any OpenAI-compatible provider)
- `LLM_BASE_URL` — LLM endpoint (defaults to `https://openrouter.ai/api/v1/chat/completions`)
- `QDRANT_URL`, `QDRANT_API_KEY` — Qdrant connection (cloud or local, gRPC port 6334)
- `EMBEDDING_MODEL` — model name for Qdrant cloud inference
- `PLANNER_MODEL`, `READER_MODEL`, `SYNTHESIZER_MODEL` — OpenRouter model IDs (e.g. `minimax/minimax-m2.5`, `anthropic/claude-haiku-4-5-20241022`)
- `MAX_HOPS`, `TOP_K` — agent behavior tuning

## Code Quality

Pre-commit git hook runs `cargo fmt --check` and `cargo clippy -- -D warnings`. Both must pass before committing.

## Crate Dependencies

No test framework configured. The project uses `anyhow` for error handling, `tokio` async runtime, `clap` derive for CLI, `qdrant-client` gRPC, and `reqwest` for the LLM API.
