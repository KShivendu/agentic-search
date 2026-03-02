#!/usr/bin/env python3
"""Decompose expected_answer into atomic key_claims using an LLM judge.

Reads eval/questions.jsonl, calls LLM to extract 3-8 key claims per question,
writes eval/questions_annotated.jsonl for human review.

Usage:
    python scripts/annotate_claims.py
    python scripts/annotate_claims.py --judge-model openai/gpt-4o
"""

import argparse
import json
import os
import re
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

INPUT_FILE = "eval/questions.jsonl"
OUTPUT_FILE = "eval/questions_annotated.jsonl"

SYSTEM_PROMPT = """You are a factual claim extractor. Given a question and its expected answer, decompose the answer into 3-8 atomic factual claims.

Each claim should be:
- A single, independently verifiable fact
- Self-contained (understandable without the other claims)
- Specific (includes names, dates, numbers where present)

Respond with a JSON array of strings. Example:
["Gutenberg invented the printing press circa 1440", "Luther's 95 Theses were mass-printed"]"""


def call_llm(question: str, expected_answer: str, model: str, api_key: str, base_url: str) -> list[str]:
    """Extract key claims from an expected answer."""
    user_msg = f"Question: {question}\n\nExpected answer: {expected_answer}\n\nExtract the atomic factual claims as a JSON array:"

    resp = requests.post(
        base_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    # Parse JSON array from response (with regex fallback)
    try:
        claims = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            claims = json.loads(match.group())
        else:
            print(f"  WARNING: Could not parse claims, using raw text", file=sys.stderr)
            claims = [content.strip()]

    if not isinstance(claims, list):
        claims = [str(claims)]

    return claims


def main():
    parser = argparse.ArgumentParser(description="Annotate eval questions with key claims")
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "openai/gpt-4.1-mini"))
    args = parser.parse_args()

    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
    if not api_key:
        print("Error: LLM_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Read input questions
    questions = []
    with open(INPUT_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    print(f"Annotating {len(questions)} questions with key claims using {args.judge_model}...")

    # Check for already-annotated questions (resumability)
    done = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    done.add(obj["question"])
        print(f"  Skipping {len(done)} already-annotated questions")

    with open(OUTPUT_FILE, "a") as out:
        for i, q in enumerate(questions):
            if q["question"] in done:
                continue

            print(f"  [{i+1}/{len(questions)}] {q['question'][:80]}...")
            claims = call_llm(q["question"], q["expected_answer"], args.judge_model, api_key, base_url)
            print(f"    → {len(claims)} claims")

            annotated = {
                **q,
                "key_claims": claims,
                "gold_passages": None,
            }
            out.write(json.dumps(annotated) + "\n")
            out.flush()

    print(f"\nDone! Output: {OUTPUT_FILE}")
    print("Review the claims, then replace eval/questions.jsonl with the annotated version.")


if __name__ == "__main__":
    main()
