#!/usr/bin/env python3
"""Embed passages and upload to Qdrant.

Supports two modes:
  --cloud-inference  : Send text to Qdrant Cloud, which embeds server-side (fast, no local GPU needed)
  (default)          : Embed locally with sentence-transformers, then upload vectors

Resumable: checks how many points are already in Qdrant and skips them.
Streams passages from JSONL — does not load all into memory.
Indexing is disabled during upload and re-enabled after.
"""

import json
import os
import sys
import uuid

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Document
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
UPLOAD_BATCH_SIZE = 128  # Qdrant cloud inference batch size
LOCAL_EMBED_BATCH_SIZE = 256
LOCAL_UPLOAD_BATCH_SIZE = 2048

MODEL_DIMS = {
    "all-MiniLM-L6-v2": 384,
    "mixedbread-ai/mxbai-embed-large-v1": 1024,
}
VECTOR_DIM = MODEL_DIMS.get(MODEL_NAME, 384)


def ensure_collection(client: QdrantClient, cloud_inference: bool) -> int:
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

    if cloud_inference:
        # For cloud inference, Qdrant manages vector config based on the model
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_DIM,
                distance=Distance.COSINE,
            ),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
        )
    else:
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
    print(f"Created collection '{COLLECTION_NAME}' (dim={VECTOR_DIM}), indexing disabled")
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


def upload_cloud_inference(client: QdrantClient, total_passages: int, skip_passages: int):
    """Upload using Qdrant cloud inference — server-side embedding."""
    remaining = total_passages - skip_passages
    print(f"\nCloud inference mode (model={MODEL_NAME})")
    print(f"Upload batch={UPLOAD_BATCH_SIZE}")
    print(f"Processing {remaining:,} passages...\n")

    pbar = tqdm(total=remaining, desc="Upload", unit=" passages", dynamic_ncols=True)
    batch: list[PointStruct] = []

    for passage in iter_passages(INPUT_FILE, skip=skip_passages):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, passage["id"]))
        batch.append(
            PointStruct(
                id=point_id,
                vector=Document(text=passage["text"], model=MODEL_NAME),
                payload={
                    "text": passage["text"],
                    "title": passage["title"],
                    "chunk_index": passage["chunk_index"],
                    "passage_id": passage["id"],
                },
            )
        )

        if len(batch) >= UPLOAD_BATCH_SIZE:
            client.upsert(collection_name=COLLECTION_NAME, points=batch, wait=False)
            pbar.update(len(batch))
            batch = []

    if batch:
        client.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)
        pbar.update(len(batch))

    pbar.close()


def upload_local_embedding(client: QdrantClient, total_passages: int, skip_passages: int):
    """Upload using local sentence-transformers embedding."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Install sentence-transformers: pip install sentence-transformers")
        sys.exit(1)

    remaining = total_passages - skip_passages

    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"\nLocal embedding mode")
    print(f"Embed batch={LOCAL_EMBED_BATCH_SIZE} → Upload batch={LOCAL_UPLOAD_BATCH_SIZE}")
    print(f"Processing {remaining:,} passages...\n")

    pbar = tqdm(total=remaining, desc="Embed+Upload", unit=" passages", dynamic_ncols=True)

    upload_buffer: list[PointStruct] = []
    embed_batch: list[dict] = []

    for passage in iter_passages(INPUT_FILE, skip=skip_passages):
        embed_batch.append(passage)

        if len(embed_batch) >= LOCAL_EMBED_BATCH_SIZE:
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

            if len(upload_buffer) >= LOCAL_UPLOAD_BATCH_SIZE:
                client.upsert(collection_name=COLLECTION_NAME, points=upload_buffer, wait=False)
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

    if upload_buffer:
        client.upsert(collection_name=COLLECTION_NAME, points=upload_buffer, wait=True)

    pbar.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Embed passages and upload to Qdrant")
    parser.add_argument(
        "--cloud-inference",
        action="store_true",
        help="Use Qdrant cloud inference (server-side embedding) instead of local",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing collection and start fresh",
    )
    args = parser.parse_args()

    cloud_inference = args.cloud_inference

    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        print("Run chunk_wiki.py first.")
        sys.exit(1)

    print(f"Counting passages in {INPUT_FILE}...")
    total_passages = count_lines(INPUT_FILE)
    print(f"Total: {total_passages:,} passages")

    print(f"Connecting to Qdrant: {QDRANT_URL}...")
    client_kwargs = {"url": QDRANT_URL, "api_key": QDRANT_API_KEY, "timeout": 120}
    if cloud_inference:
        client_kwargs["cloud_inference"] = True
    client = QdrantClient(**client_kwargs)

    # Delete collection if --fresh
    if args.fresh:
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME in collections:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")

    existing_points = ensure_collection(client, cloud_inference)

    # Resume: skip passages already uploaded
    skip_passages = 0
    batch_size = UPLOAD_BATCH_SIZE if cloud_inference else LOCAL_UPLOAD_BATCH_SIZE
    if existing_points > 0:
        skip_passages = (existing_points // batch_size) * batch_size
        print(f"Resuming: skipping first {skip_passages:,} passages (already uploaded)")

    if cloud_inference:
        upload_cloud_inference(client, total_passages, skip_passages)
    else:
        upload_local_embedding(client, total_passages, skip_passages)

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
