#!/usr/bin/env python3
"""Embed passages locally and upload to Qdrant.

Uses sentence-transformers with mxbai-embed-large-v1 for embedding.
Creates a Qdrant collection with Binary Quantization enabled.
"""

import json
import os
import sys
import time
import uuid

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
        PointStruct,
        VectorParams,
    )
except ImportError:
    print("Install qdrant-client: pip install qdrant-client")
    sys.exit(1)

INPUT_FILE = "data/passages.jsonl"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "wiki_passages")
MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
BATCH_SIZE = 64
VECTOR_DIM = 1024  # mxbai-embed-large-v1 output dimension


def create_collection(client: QdrantClient):
    """Create Qdrant collection with Binary Quantization."""
    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in collections:
        info = client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists ({info.points_count} points)")
        resp = input("Recreate? [y/N] ").strip().lower()
        if resp != "y":
            return False
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_DIM,
            distance=Distance.COSINE,
            quantization_config=BinaryQuantization(
                binary=BinaryQuantizationConfig(always_ram=True),
            ),
        ),
    )
    print(f"Created collection '{COLLECTION_NAME}' with BQ enabled")
    return True


def load_passages(filepath: str) -> list[dict]:
    """Load passages from JSONL file."""
    passages = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                passages.append(json.loads(line))
    return passages


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        print("Run chunk_wiki.py first.")
        sys.exit(1)

    print(f"Loading passages from {INPUT_FILE}...")
    passages = load_passages(INPUT_FILE)
    print(f"Loaded {len(passages)} passages")

    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Connecting to Qdrant: {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)
    create_collection(client)

    print(f"Embedding and uploading in batches of {BATCH_SIZE}...")
    total_uploaded = 0
    start_time = time.time()

    for batch_start in range(0, len(passages), BATCH_SIZE):
        batch = passages[batch_start : batch_start + BATCH_SIZE]
        texts = [p["text"] for p in batch]

        # Embed batch
        embeddings = model.encode(texts, show_progress_bar=False)

        # Create Qdrant points
        points = []
        for passage, embedding in zip(batch, embeddings):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, passage["id"]))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "text": passage["text"],
                        "title": passage["title"],
                        "chunk_index": passage["chunk_index"],
                        "passage_id": passage["id"],
                    },
                )
            )

        # Upload to Qdrant
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total_uploaded += len(points)

        elapsed = time.time() - start_time
        rate = total_uploaded / elapsed if elapsed > 0 else 0
        print(
            f"\r  Uploaded {total_uploaded}/{len(passages)} ({rate:.0f} passages/s)",
            end="",
        )

    elapsed = time.time() - start_time
    print(f"\n\nDone: {total_uploaded} passages uploaded in {elapsed:.1f}s")
    print(f"Collection: {COLLECTION_NAME}")

    # Verify
    info = client.get_collection(COLLECTION_NAME)
    print(f"Qdrant reports {info.points_count} points in collection")


if __name__ == "__main__":
    main()
