#!/usr/bin/env python3
"""Ingest BrowseComp-Plus corpus into Qdrant and convert queries to eval format.

Steps:
  1. Stream BrowseComp-Plus corpus from HuggingFace (~100K docs)
  2. Chunk each document into ~200-word passages (same strategy as chunk_wiki.py)
  3. Embed and upload to a separate Qdrant collection (browsecomp_passages)
  4. Convert decrypted queries + qrel_golds.txt → eval/browsecomp_questions.jsonl

Usage:
    # Full pipeline
    python scripts/ingest_browsecomp.py

    # Skip ingest, only convert queries (if already ingested)
    python scripts/ingest_browsecomp.py --queries-only

    # Ingest only, skip query conversion
    python scripts/ingest_browsecomp.py --ingest-only

    # Use pre-decrypted file from BrowseComp-Plus repo
    python scripts/ingest_browsecomp.py --decrypted-file /tmp/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl --qrels-gold /tmp/BrowseComp-Plus/topics-qrels/qrel_golds.txt
"""

import argparse
import json
import os
import re
import sys
import uuid

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

COLLECTION_NAME = "browsecomp_passages"
CHUNKS_FILE = "data/browsecomp_passages.jsonl"  # local cache of chunked passages
OUTPUT_FILE = "eval/browsecomp_questions.jsonl"
MIN_WORDS = 30
MAX_WORDS = 380  # ~505 tokens (1 word ≈ 1.33 tokens), fits within mxbai-embed-large-v1's 512 token limit
UPLOAD_BATCH_SIZE = 512  # larger batches = fewer round-trips to Qdrant Cloud

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")


# ── Chunking (same strategy as chunk_wiki.py) ─────────────────────────────────

def chunk_text(text: str, title: str, source_url: str) -> list[dict]:
    """Split text into ~200-word chunks respecting paragraph boundaries."""
    # Strip markdown frontmatter if present
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL).strip()

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk: list[str] = []
    current_words = 0

    for para in paragraphs:
        words = para.split()
        para_words = len(words)

        if para_words > MAX_WORDS:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                s_words = len(sentence.split())
                if current_words + s_words > MAX_WORDS and current_words >= MIN_WORDS:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_words = 0
                current_chunk.append(sentence)
                current_words += s_words
        elif current_words + para_words > MAX_WORDS and current_words >= MIN_WORDS:
            chunks.append(" ".join(current_chunk))
            current_chunk = [para]
            current_words = para_words
        else:
            current_chunk.append(para)
            current_words += para_words

    if current_words >= MIN_WORDS:
        chunks.append(" ".join(current_chunk))

    return [
        {
            "id": f"{source_url}_{i}",
            "text": chunk,
            "title": title,
            "chunk_index": i,
            "source_url": source_url,
        }
        for i, chunk in enumerate(chunks)
    ]


# ── Step 1: Download + chunk to local JSONL ───────────────────────────────────

def process_doc(args: tuple) -> list[dict]:
    """Chunk a single doc. Runs in worker process."""
    docid, url, text = args
    title_match = re.search(r"^title:\s*(.+)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else url
    chunks = chunk_text(text, title, url)
    for chunk in chunks:
        chunk["doc_id"] = docid
    return chunks


def download_and_chunk(chunks_file: str):
    """Stream corpus from HuggingFace, chunk with multiprocessing, save to local JSONL."""
    import multiprocessing as mp
    from datasets import load_dataset

    print("Streaming BrowseComp-Plus corpus from HuggingFace...")
    ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train", streaming=True)

    n_workers = max(1, mp.cpu_count() - 1)
    print(f"Chunking with {n_workers} workers → {chunks_file}")

    os.makedirs(os.path.dirname(chunks_file), exist_ok=True)

    total_chunks = 0
    batch = []
    BATCH_SIZE = 256

    pbar = tqdm(desc="Chunking docs", unit=" docs", dynamic_ncols=True)

    with open(chunks_file, "w") as out, mp.Pool(n_workers) as pool:
        for doc in ds:
            batch.append((doc["docid"], doc["url"], doc["text"]))
            pbar.update(1)

            if len(batch) >= BATCH_SIZE:
                for chunks in pool.imap_unordered(process_doc, batch):
                    for chunk in chunks:
                        out.write(json.dumps(chunk) + "\n")
                        total_chunks += 1
                pbar.set_postfix(chunks=f"{total_chunks:,}")
                batch = []

        if batch:
            for chunks in pool.imap_unordered(process_doc, batch):
                for chunk in chunks:
                    out.write(json.dumps(chunk) + "\n")
                    total_chunks += 1

    pbar.close()
    print(f"\nChunked {total_chunks:,} passages → {chunks_file}")
    return total_chunks


# ── Step 2: Upload from local JSONL ───────────────────────────────────────────

def ensure_collection(client, vector_dim: int) -> int:
    """Create collection if needed. Returns current point count."""
    from qdrant_client.models import Distance, OptimizersConfigDiff, VectorParams

    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        info = client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' exists ({info.points_count} points)")
        client.update_collection(
            collection_name=COLLECTION_NAME,
            optimizer_config=OptimizersConfigDiff(indexing_threshold=0),
        )
        return info.points_count

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
        on_disk_payload=True,
    )
    print(f"Created collection '{COLLECTION_NAME}' (dim={vector_dim})")
    return 0


def upload_chunks(client, chunks_file: str):
    """Upload pre-chunked local JSONL to Qdrant with cloud inference."""
    from qdrant_client.http.models import Document
    from qdrant_client.models import OptimizersConfigDiff, PointStruct

    MODEL_DIMS = {"mixedbread-ai/mxbai-embed-large-v1": 1024, "all-MiniLM-L6-v2": 384}
    vector_dim = MODEL_DIMS.get(MODEL_NAME, 1024)

    existing_points = ensure_collection(client, vector_dim)

    # Count total lines for progress
    print("Counting passages...")
    total_lines = sum(1 for _ in open(chunks_file))
    skip = (existing_points // UPLOAD_BATCH_SIZE) * UPLOAD_BATCH_SIZE
    print(f"Total passages: {total_lines:,} | Already uploaded: {existing_points:,} | Skipping: {skip:,}")

    batch: list[PointStruct] = []
    uploaded = 0

    pbar = tqdm(total=total_lines - skip, desc="Uploading", unit=" passages", dynamic_ncols=True)

    with open(chunks_file) as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            chunk = json.loads(line)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["id"]))
            batch.append(
                PointStruct(
                    id=point_id,
                    vector=Document(text=chunk["text"], model=MODEL_NAME),
                    payload={
                        "text": chunk["text"],
                        "title": chunk["title"],
                        "chunk_index": chunk["chunk_index"],
                        "source_url": chunk["source_url"],
                        "doc_id": chunk["doc_id"],
                    },
                )
            )

            if len(batch) >= UPLOAD_BATCH_SIZE:
                is_last = (i >= total_lines - 1)
                client.upsert(collection_name=COLLECTION_NAME, points=batch, wait=is_last)
                uploaded += len(batch)
                pbar.update(len(batch))
                batch = []

    if batch:
        client.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)
        pbar.update(len(batch))

    pbar.close()
    print("\nRe-enabling indexing...")
    client.update_collection(
        collection_name=COLLECTION_NAME,
        optimizer_config=OptimizersConfigDiff(indexing_threshold=20000),
    )
    info = client.get_collection(COLLECTION_NAME)
    print(f"Done. {info.points_count:,} points in '{COLLECTION_NAME}'.")


def ingest_corpus(client):
    """Full ingest: chunk locally first, then upload."""
    if not os.path.exists(CHUNKS_FILE):
        download_and_chunk(CHUNKS_FILE)
    else:
        lines = sum(1 for _ in open(CHUNKS_FILE))
        print(f"Using existing chunks file: {CHUNKS_FILE} ({lines:,} passages)")
    upload_chunks(client, CHUNKS_FILE)


# ── Query conversion ───────────────────────────────────────────────────────────

def convert_queries(decrypted_file: str, qrels_gold_file: str):
    """Convert decrypted queries + gold qrels → eval/browsecomp_questions.jsonl."""

    # Load gold doc mappings: query_id → [doc_ids]
    gold_docs: dict[str, list[str]] = {}
    with open(qrels_gold_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, docid, _ = parts[0], parts[1], parts[2], parts[3]
                gold_docs.setdefault(qid, []).append(docid)

    # Load already-written questions
    done = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["question"])

    written = 0
    with open(decrypted_file) as inp, open(OUTPUT_FILE, "a") as out:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            question = q["query"]
            answer = q["answer"]
            qid = str(q.get("query_id", ""))

            if question in done:
                continue

            entry = {
                "question": question,
                "expected_answer": answer,
                "type": "multi-hop",
                "source": "browsecomp-plus",
                "query_id": qid,
                # doc_ids from qrels — not Qdrant point IDs yet, but traceable
                "gold_doc_ids": gold_docs.get(qid, []),
                "key_claims": None,
                "gold_passages": None,
            }
            out.write(json.dumps(entry) + "\n")
            written += 1

    print(f"Written {written} questions to {OUTPUT_FILE}")
    print("Note: gold_doc_ids are BrowseComp doc IDs. gold_passages (Qdrant point IDs) need a separate mapping step.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest BrowseComp-Plus into Qdrant")
    parser.add_argument("--queries-only", action="store_true", help="Skip ingest, only convert queries")
    parser.add_argument("--ingest-only", action="store_true", help="Skip query conversion")
    parser.add_argument(
        "--decrypted-file",
        default="/tmp/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl",
        help="Path to decrypted queries JSONL",
    )
    parser.add_argument(
        "--qrels-gold",
        default="/tmp/BrowseComp-Plus/topics-qrels/qrel_golds.txt",
        help="Path to qrel_golds.txt",
    )
    args = parser.parse_args()

    if not args.queries_only:
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            print("Install qdrant-client: pip install qdrant-client")
            sys.exit(1)

        client_kwargs = {"url": QDRANT_URL.replace("6334", "6333"), "api_key": QDRANT_API_KEY, "timeout": 120}
        client_kwargs["cloud_inference"] = True
        client = QdrantClient(**client_kwargs)
        ingest_corpus(client)

    if not args.ingest_only:
        if not os.path.exists(args.decrypted_file):
            print(f"Decrypted file not found: {args.decrypted_file}")
            print("Run: cd /tmp/BrowseComp-Plus && python scripts_build_index/decrypt_dataset.py --output data/browsecomp_plus_decrypted.jsonl --generate-tsv topics-qrels/queries.tsv")
            sys.exit(1)
        convert_queries(args.decrypted_file, args.qrels_gold)


if __name__ == "__main__":
    main()
