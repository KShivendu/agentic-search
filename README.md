# Agentic Search

Multi-hop research agent over a large corpus. Decomposes complex questions into retrieval steps, accumulates context across hops, and synthesizes answers — with full instrumentation of every step.

## Architecture

```
Query → Planner (Claude Haiku) → [Embed → Search Qdrant → Reader (Claude Haiku)] × N hops → Synthesizer (Claude Sonnet) → Answer
```

All steps produce structured JSON logs: per-hop latency breakdowns, token usage, retrieval stats, and cost estimates.

## Stack

- **Core system**: Rust (async, `qdrant-client`, `fastembed`, `reqwest`)
- **Vector DB**: Qdrant with Binary Quantization
- **Embeddings**: `mxbai-embed-large-v1` (1024d, local ONNX via fastembed)
- **LLM**: Anthropic Claude (Haiku for hops, Sonnet for synthesis)
- **Data prep**: Python scripts for Wikipedia dump processing

## Quick Start

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY

# 2. Start Qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 3. Prepare corpus (Simple English Wikipedia)
pip install mwparserfromhell sentence-transformers qdrant-client tqdm
python scripts/download_wiki.py
python scripts/chunk_wiki.py
python scripts/embed_and_upload.py

# 4. Run
cargo run -- ask "What connections exist between transistors and the space race?"
cargo run -- -v ask "How did the printing press influence the Protestant Reformation?"
cargo run -- eval eval/questions.jsonl
```

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
│   ├── qdrant.rs           # Vector search
│   └── embedder.rs         # Local embedding (fastembed)
├── llm/
│   └── anthropic.rs        # Claude API client
└── instrumentation/
    └── logger.rs           # Structured JSON logging

scripts/                    # Python data prep (one-time)
eval/questions.jsonl        # 35 multi-hop evaluation questions
```
