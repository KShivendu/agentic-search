#!/usr/bin/env python3
"""Download FRAMES benchmark and convert to eval/questions_annotated.jsonl format.

FRAMES (google/frames-benchmark) has 824 hand-curated multi-hop questions over
Wikipedia, each requiring 2-15 articles. Directly comparable to Perplexity/Exa
published numbers.

Usage:
    python scripts/load_frames.py                  # full dataset (824 questions)
    python scripts/load_frames.py --sample 50      # random subset
    python scripts/load_frames.py --no-annotate    # skip key_claims generation
    python scripts/load_frames.py --judge-model openai/gpt-4o
"""

import argparse
import json
import os
import random
import re
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

OUTPUT_FILE = "eval/questions_annotated.jsonl"

ANNOTATE_SYSTEM = """You are a factual claim extractor. Given a question and its expected answer, decompose the answer into 3-8 atomic factual claims.

Each claim should be:
- A single, independently verifiable fact
- Self-contained (understandable without the other claims)
- Specific (includes names, dates, numbers where present)

Respond with a JSON array of strings. Example:
["Gutenberg invented the printing press circa 1440", "Luther's 95 Theses were mass-printed"]"""


def load_frames():
    """Load FRAMES dataset via HuggingFace datasets library."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        sys.exit(1)

    print("Downloading FRAMES benchmark from HuggingFace...")
    ds = load_dataset("google/frames-benchmark", split="test")
    print(f"Loaded {len(ds)} questions")
    return ds


def annotate_claims(question: str, answer: str, model: str, api_key: str, base_url: str) -> list[str]:
    """Extract atomic key claims from an expected answer."""
    user_msg = f"Question: {question}\n\nExpected answer: {answer}\n\nExtract the atomic factual claims as a JSON array:"

    resp = requests.post(
        base_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": ANNOTATE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
        },
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    try:
        claims = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            claims = json.loads(match.group())
        else:
            print(f"  WARNING: Could not parse claims, using raw text", file=sys.stderr)
            claims = [content.strip()]

    return claims if isinstance(claims, list) else [str(claims)]


def main():
    parser = argparse.ArgumentParser(description="Load FRAMES benchmark into eval format")
    parser.add_argument("--sample", type=int, default=None, help="Random subset size (default: all 824)")
    parser.add_argument("--no-annotate", action="store_true", help="Skip key_claims generation")
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "openai/gpt-4.1-mini"))
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")

    if not args.no_annotate and not api_key:
        print("Error: LLM_API_KEY not set (required for claim annotation). Use --no-annotate to skip.", file=sys.stderr)
        sys.exit(1)

    ds = load_frames()
    examples = list(ds)

    if args.sample:
        random.seed(args.seed)
        examples = random.sample(examples, min(args.sample, len(examples)))
        print(f"Sampled {len(examples)} questions (seed={args.seed})")

    # Check for already-written questions (resumable)
    done = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    done.add(obj["question"])
        if done:
            print(f"Resuming — skipping {len(done)} already written questions")

    to_write = [ex for ex in examples if ex["Prompt"] not in done]
    if not to_write:
        print("All questions already in output file.")
        return

    print(f"Writing {len(to_write)} questions to {OUTPUT_FILE}")
    if not args.no_annotate:
        print(f"Annotating claims with {args.judge_model}...")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "a") as out:
        for i, ex in enumerate(to_write):
            question = ex["Prompt"]
            answer = ex["Answer"]
            # FRAMES stores wiki_links as a stringified list — parse it
            raw_links = ex.get("wiki_links", "[]")
            if isinstance(raw_links, str):
                try:
                    wiki_links = json.loads(raw_links.replace("'", '"'))
                except json.JSONDecodeError:
                    wiki_links = [s.strip().strip("'\"") for s in raw_links.strip("[]").split(",") if s.strip()]
            else:
                wiki_links = raw_links
            reasoning_types = ex.get("Reasoning_type", "")

            print(f"  [{i + 1}/{len(to_write)}] {question[:80]}...")

            key_claims = None
            if not args.no_annotate:
                key_claims = annotate_claims(question, answer, args.judge_model, api_key, base_url)
                print(f"    → {len(key_claims)} claims")

            entry = {
                "question": question,
                "expected_answer": answer,
                "type": "multi-hop",
                "reasoning_type": reasoning_types,  # e.g. "Tabular", "Multiple constraints"
                "wiki_links": wiki_links,           # source Wikipedia articles (not chunk IDs)
                "key_claims": key_claims,
                "gold_passages": None,              # chunk-level gold not available in FRAMES
            }

            out.write(json.dumps(entry) + "\n")
            out.flush()

    print(f"\nDone. {len(to_write)} questions written to {OUTPUT_FILE}")
    if args.no_annotate:
        print("Run annotate_claims.py or re-run without --no-annotate to generate key_claims.")


if __name__ == "__main__":
    main()
