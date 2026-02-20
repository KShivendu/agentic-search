#!/usr/bin/env python3
"""Embed passages locally and upload to Qdrant.

Resumable: checks how many points are already in Qdrant and skips them.
Streams passages from JSONL — does not load all into memory.

Strategy: embed in small batches (64), accumulate, then upsert in large
batches (1024) to minimize Qdrant round trips. Indexing is disabled during
upload and re-enabled after.
"""

import json
import os
import sys
import uuid

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Install sentence-transformers: pip install sentence-transformers")
    sys.exit(1)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        BinaryQuantization,
        BinaryQuantizationConfig,
        Distance,
        OptimizersConfigDiff,
        PointStruct,
        VectorParams,
    )
except ImportError:
    print("Install qdrant-client: pip install qdrant-client")
    sys.exit(1)

INPUT_FILE = "data/passages.jsonl"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "wiki_passages")
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBED_BATCH_SIZE = 256   # MiniLM is small enough for large CPU batches
UPLOAD_BATCH_SIZE = 2048  # large batches for Qdrant upsert (fewer round trips)

MODEL_DIMS = {
    "all-MiniLM-L6-v2": 384,
    "mixedbread-ai/mxbai-embed-large-v1": 1024,
}
VECTOR_DIM = MODEL_DIMS.get(MODEL_NAME, 384)


def ensure_collection(client: QdrantClient) -> int:
    """Create collection if needed with indexing disabled. Returns current point count."""
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
        vectors_config=VectorParams(
            size=VECTOR_DIM,
            distance=Distance.COSINE,
            quantization_config=BinaryQuantization(
                binary=BinaryQuantizationConfig(always_ram=True),
            ),
        ),
        optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
    )
    print(f"Created collection '{COLLECTION_NAME}' with BQ enabled, indexing disabled")
    return 0


def count_lines(filepath: str) -> int:
    """Count lines in a file without loading it all."""
    count = 0
    with open(filepath) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def iter_passages(filepath: str, skip: int = 0):
    """Stream passages from JSONL, optionally skipping the first `skip` lines."""
    skipped = 0
    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            if skipped < skip:
                skipped += 1
                continue
            yield json.loads(line)


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        print("Run chunk_wiki.py first.")
        sys.exit(1)

    print(f"Counting passages in {INPUT_FILE}...")
    total_passages = count_lines(INPUT_FILE)
    print(f"Total: {total_passages:,} passages")

    print(f"Connecting to Qdrant: {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    existing_points = ensure_collection(client)

    # Resume: skip passages already uploaded
    skip_passages = 0
    if existing_points > 0:
        skip_passages = (existing_points // UPLOAD_BATCH_SIZE) * UPLOAD_BATCH_SIZE
        print(f"Resuming: skipping first {skip_passages:,} passages (already uploaded)")

    remaining_count = total_passages - skip_passages

    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"\nEmbed (batch={EMBED_BATCH_SIZE}) → Upload (batch={UPLOAD_BATCH_SIZE})")
    print(f"Processing {remaining_count:,} passages...\n")

    pbar = tqdm(
        total=remaining_count,
        desc="Embed+Upload",
        unit=" passages",
        dynamic_ncols=True,
    )

    upload_buffer: list[PointStruct] = []
    embed_batch: list[dict] = []

    for passage in iter_passages(INPUT_FILE, skip=skip_passages):
        embed_batch.append(passage)

        if len(embed_batch) >= EMBED_BATCH_SIZE:
            # Embed small batch
            texts = [p["text"] for p in embed_batch]
            embeddings = model.encode(texts, show_progress_bar=False)

            for p, emb in zip(embed_batch, embeddings):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, p["id"]))
                upload_buffer.append(
                    PointStruct(
                        id=point_id,
                        vector=emb.tolist(),
                        payload={
                            "text": p["text"],
                            "title": p["title"],
                            "chunk_index": p["chunk_index"],
                            "passage_id": p["id"],
                        },
                    )
                )

            pbar.update(len(embed_batch))
            embed_batch = []

            # Flush to Qdrant when upload buffer is full
            if len(upload_buffer) >= UPLOAD_BATCH_SIZE:
                client.upsert(collection_name=COLLECTION_NAME, points=upload_buffer)
                upload_buffer = []

    # Handle remaining embed batch
    if embed_batch:
        texts = [p["text"] for p in embed_batch]
        embeddings = model.encode(texts, show_progress_bar=False)
        for p, emb in zip(embed_batch, embeddings):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, p["id"]))
            upload_buffer.append(
                PointStruct(
                    id=point_id,
                    vector=emb.tolist(),
                    payload={
                        "text": p["text"],
                        "title": p["title"],
                        "chunk_index": p["chunk_index"],
                        "passage_id": p["id"],
                    },
                )
            )
        pbar.update(len(embed_batch))

    # Flush remaining upload buffer
    if upload_buffer:
        client.upsert(collection_name=COLLECTION_NAME, points=upload_buffer)

    pbar.close()

    # Re-enable indexing
    print("\nUpload complete. Enabling indexing (threshold=20000)...")
    client.update_collection(
        collection_name=COLLECTION_NAME,
        optimizer_config=OptimizersConfigDiff(indexing_threshold=20000),
    )

    info = client.get_collection(COLLECTION_NAME)
    print(f"Done. {info.points_count:,} points in '{COLLECTION_NAME}'. Indexing will run in background.")


if __name__ == "__main__":
    main()
