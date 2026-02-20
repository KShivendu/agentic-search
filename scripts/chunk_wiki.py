#!/usr/bin/env python3
"""Parse Wikipedia XML dump and chunk into passages.

Outputs JSONL with one passage per line:
  {"id": "...", "title": "...", "text": "...", "chunk_index": 0}
"""

import bz2
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

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

# MediaWiki XML namespace
MW_NS = "http://www.mediawiki.org/xml/export-0.11/"


def clean_wikitext(wikitext: str) -> str:
    """Convert wikitext to plain text using mwparserfromhell."""
    try:
        parsed = mwparserfromhell.parse(wikitext)
        # Remove templates, tags, etc.
        text = parsed.strip_code(normalize=True, collapse=True)
        # Clean up extra whitespace
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

        # If a single paragraph exceeds MAX_WORDS, split it by sentences
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

    # Don't forget the last chunk
    if current_words >= MIN_WORDS:
        chunks.append(
            {
                "title": title,
                "text": " ".join(current_chunk),
                "chunk_index": len(chunks),
            }
        )

    # Add IDs
    for i, chunk in enumerate(chunks):
        chunk["id"] = f"{title.replace(' ', '_')}_{i}"

    return chunks


def iter_articles(filepath: str):
    """Iterate over articles in a Wikipedia XML dump (bz2 compressed)."""
    with bz2.open(filepath, "rt", encoding="utf-8") as f:
        # Use iterparse to avoid loading the entire XML into memory
        context = ET.iterparse(f, events=("end",))
        for event, elem in context:
            tag = elem.tag.replace(f"{{{MW_NS}}}", "")

            if tag == "page":
                ns_elem = elem.find(f"{{{MW_NS}}}ns")
                # Only process main namespace (ns=0)
                if ns_elem is not None and ns_elem.text == "0":
                    title_elem = elem.find(f"{{{MW_NS}}}title")
                    text_elem = elem.find(
                        f".//{{{MW_NS}}}revision/{{{MW_NS}}}text"
                    )

                    if title_elem is not None and text_elem is not None and text_elem.text:
                        title = title_elem.text
                        wikitext = text_elem.text

                        # Skip redirects
                        if wikitext.lower().startswith("#redirect"):
                            elem.clear()
                            continue

                        yield title, wikitext

                elem.clear()


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        print("Run download_wiki.py first.")
        sys.exit(1)

    print(f"Processing: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    total_articles = 0
    total_chunks = 0

    with open(OUTPUT_FILE, "w") as out:
        for title, wikitext in iter_articles(INPUT_FILE):
            total_articles += 1
            text = clean_wikitext(wikitext)

            if len(text.split()) < MIN_WORDS:
                continue

            chunks = chunk_text(text, title)
            for chunk in chunks:
                out.write(json.dumps(chunk) + "\n")
                total_chunks += 1

            if total_articles % 1000 == 0:
                print(
                    f"  Processed {total_articles} articles, {total_chunks} chunks",
                    end="\r",
                )

    print(f"\nDone: {total_articles} articles → {total_chunks} chunks")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
