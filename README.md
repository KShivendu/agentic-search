# Agentic Search

Multi-hop research agent over a large corpus. Decomposes complex questions into retrieval steps, accumulates context across hops, and synthesizes answers — with full instrumentation of every step.

## Architecture

```
Query → Planner → [Search Qdrant → Reader] × N hops → Synthesizer → Answer
```

All steps produce structured JSON logs: per-hop latency breakdowns, token usage, retrieval stats, and actual cost (via OpenRouter `usage.cost`).

## Stack

- **Core system**: Rust (async, `qdrant-client`, `reqwest`)
- **Vector DB**: Qdrant with cloud inference (server-side embedding)
- **Embeddings**: `mxbai-embed-large-v1` (1024d, embedded server-side by Qdrant)
- **LLM**: Any model via OpenRouter (OpenAI-compatible chat completions)
- **Data prep**: Python scripts for Wikipedia dump processing

## Quick Start

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env with your LLM_API_KEY (OpenRouter) and Qdrant credentials

# 2. Prepare corpus (Simple English Wikipedia)
pip install mwparserfromhell sentence-transformers qdrant-client tqdm python-dotenv
python scripts/download_wiki.py
python scripts/chunk_wiki.py
python scripts/embed_and_upload.py --cloud-inference

# 3. Run
cargo run -- ask "What connections exist between transistors and the space race?"
cargo run -- -v ask "How did the printing press influence the Protestant Reformation?"
cargo run -- eval eval/questions.jsonl
```

## Configuration

All config via `.env`. Key variables:

| Variable | Description | Default |
|---|---|---|
| `LLM_API_KEY` | OpenRouter API key (or any OpenAI-compatible provider) | *required* |
| `LLM_BASE_URL` | LLM endpoint | `https://openrouter.ai/api/v1/chat/completions` |
| `QDRANT_URL` | Qdrant gRPC endpoint | `http://localhost:6334` |
| `QDRANT_API_KEY` | Qdrant API key (for cloud) | *optional* |
| `PLANNER_MODEL` | Model for query decomposition | `anthropic/claude-haiku-4-5-20241022` |
| `READER_MODEL` | Model for passage evaluation | `anthropic/claude-haiku-4-5-20241022` |
| `SYNTHESIZER_MODEL` | Model for final answer | `anthropic/claude-sonnet-4-20250514` |
| `EMBEDDING_MODEL` | Qdrant cloud inference model | `mixedbread-ai/mxbai-embed-large-v1` |
| `MAX_HOPS` | Maximum retrieval hops | `7` |
| `TOP_K` | Results per search | `10` |

## Project Structure

```
src/
├── main.rs                 # CLI (ask, eval)
├── config.rs               # Env-based configuration
├── agent/                  # Multi-hop agent loop
│   ├── planner.rs          # Query decomposition
│   ├── reader.rs           # Passage evaluation + follow-up decisions
│   └── synthesizer.rs      # Final answer generation
├── retrieval/
│   └── qdrant.rs           # Vector search (cloud inference)
├── llm/
│   └── client.rs           # OpenAI-compatible LLM client
└── instrumentation/
    └── logger.rs           # Structured JSON logging

scripts/                    # Python data prep (one-time)
eval/questions.jsonl        # 35 multi-hop evaluation questions
```
