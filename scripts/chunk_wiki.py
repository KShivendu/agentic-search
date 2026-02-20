#!/usr/bin/env python3
"""Parse Wikipedia XML dump and chunk into passages.

Resumable: if output file exists, skips already-processed articles.

Outputs JSONL with one passage per line:
  {"id": "...", "title": "...", "text": "...", "chunk_index": 0}
"""

import bz2
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

from tqdm import tqdm

try:
    import mwparserfromhell
except ImportError:
    print("Install mwparserfromhell: pip install mwparserfromhell")
    sys.exit(1)

INPUT_FILE = "data/simplewiki-latest-pages-articles.xml.bz2"
OUTPUT_FILE = "data/passages.jsonl"
MIN_WORDS = 30
MAX_WORDS = 300
TARGET_WORDS = 200

# Simple English Wikipedia: ~250K articles total, ~200K in main namespace
ESTIMATED_ARTICLES = 200_000

# MediaWiki XML namespace
MW_NS = "http://www.mediawiki.org/xml/export-0.11/"


def clean_wikitext(wikitext: str) -> str:
    """Convert wikitext to plain text using mwparserfromhell."""
    try:
        parsed = mwparserfromhell.parse(wikitext)
        text = parsed.strip_code(normalize=True, collapse=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()
    except Exception:
        return ""


def chunk_text(text: str, title: str) -> list[dict]:
    """Split text into chunks of ~TARGET_WORDS words, respecting paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = []
    current_words = 0

    for para in paragraphs:
        words = para.split()
        para_words = len(words)

        if para_words > MAX_WORDS:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                s_words = len(sentence.split())
                if current_words + s_words > MAX_WORDS and current_words >= MIN_WORDS:
                    chunks.append(
                        {
                            "title": title,
                            "text": " ".join(current_chunk),
                            "chunk_index": len(chunks),
                        }
                    )
                    current_chunk = []
                    current_words = 0
                current_chunk.append(sentence)
                current_words += s_words
        elif current_words + para_words > MAX_WORDS and current_words >= MIN_WORDS:
            chunks.append(
                {
                    "title": title,
                    "text": " ".join(current_chunk),
                    "chunk_index": len(chunks),
                }
            )
            current_chunk = [para]
            current_words = para_words
        else:
            current_chunk.append(para)
            current_words += para_words

    if current_words >= MIN_WORDS:
        chunks.append(
            {
                "title": title,
                "text": " ".join(current_chunk),
                "chunk_index": len(chunks),
            }
        )

    for i, chunk in enumerate(chunks):
        chunk["id"] = f"{title.replace(' ', '_')}_{i}"

    return chunks


def iter_articles(filepath: str):
    """Iterate over articles in a Wikipedia XML dump (bz2 compressed)."""
    with bz2.open(filepath, "rt", encoding="utf-8") as f:
        context = ET.iterparse(f, events=("end",))
        for event, elem in context:
            tag = elem.tag.replace(f"{{{MW_NS}}}", "")

            if tag == "page":
                ns_elem = elem.find(f"{{{MW_NS}}}ns")
                if ns_elem is not None and ns_elem.text == "0":
                    title_elem = elem.find(f"{{{MW_NS}}}title")
                    text_elem = elem.find(
                        f".//{{{MW_NS}}}revision/{{{MW_NS}}}text"
                    )

                    if title_elem is not None and text_elem is not None and text_elem.text:
                        title = title_elem.text
                        wikitext = text_elem.text

                        if wikitext.lower().startswith("#redirect"):
                            elem.clear()
                            continue

                        yield title, wikitext

                elem.clear()


def load_processed_titles(filepath: str) -> tuple[set[str], int]:
    """Load titles already in the output file for resumability. Returns (titles, chunk_count)."""
    titles = set()
    chunks = 0
    if not os.path.exists(filepath):
        return titles, chunks
    with open(filepath) as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    titles.add(data["title"])
                    chunks += 1
                except (json.JSONDecodeError, KeyError):
                    continue
    return titles, chunks


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        print("Run download_wiki.py first.")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Check for existing progress
    processed_titles, existing_chunks = load_processed_titles(OUTPUT_FILE)
    skip_count = len(processed_titles)

    if skip_count > 0:
        print(f"Resuming: {skip_count} articles ({existing_chunks} chunks) already processed")
        mode = "a"
    else:
        mode = "w"

    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}\n")

    total_chunks = existing_chunks

    pbar = tqdm(
        iter_articles(INPUT_FILE),
        total=ESTIMATED_ARTICLES,
        desc="Chunking",
        unit=" articles",
        dynamic_ncols=True,
    )

    with open(OUTPUT_FILE, mode) as out:
        for title, wikitext in pbar:
            if title in processed_titles:
                continue

            text = clean_wikitext(wikitext)

            if len(text.split()) < MIN_WORDS:
                continue

            chunks = chunk_text(text, title)
            for chunk in chunks:
                out.write(json.dumps(chunk) + "\n")
                total_chunks += 1

            pbar.set_postfix(chunks=f"{total_chunks:,}", refresh=False)

    print(f"\nTotal chunks: {total_chunks:,}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
