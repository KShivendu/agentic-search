#!/usr/bin/env python3
"""Download Simple English Wikipedia dump for the MVP corpus."""

import os
import urllib.request
import sys

# Simple English Wikipedia is ~200K articles — fast to download and process
DUMP_URL = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "simplewiki-latest-pages-articles.xml.bz2")


def download_with_progress(url: str, dest: str):
    """Download a file with progress reporting."""

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)")
            sys.stdout.flush()

    print(f"Downloading: {url}")
    print(f"Destination: {dest}")
    urllib.request.urlretrieve(url, dest, reporthook)
    print("\nDone.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
        print(f"File already exists: {OUTPUT_FILE} ({size_mb:.1f} MB)")
        resp = input("Re-download? [y/N] ").strip().lower()
        if resp != "y":
            return

    download_with_progress(DUMP_URL, OUTPUT_FILE)
    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"Downloaded {size_mb:.1f} MB to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
