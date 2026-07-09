"""
One-time setup script: downloads 8 sample line images from Teklia/IAM-line
via direct parquet URL and saves them to samples/ as static files.

Run once locally, then commit the samples/ directory to the repo.
Usage:  python download_samples.py
"""

import os
import io
import ssl
import urllib.request
import pandas as pd
from PIL import Image

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
NUM_SAMPLES = 8

# Direct parquet URL from HF dataset viewer API
PARQUET_URL = "https://huggingface.co/datasets/Teklia/IAM-line/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"


def main():
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    print("Downloading parquet file (first shard)...")
    # Create SSL context that skips verification for environments with cert issues
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(PARQUET_URL, headers={"User-Agent": "Python"})
    parquet_path = os.path.join(SAMPLES_DIR, "_temp.parquet")

    with urllib.request.urlopen(req, context=ctx) as resp:
        with open(parquet_path, "wb") as f:
            f.write(resp.read())

    print("Parsing parquet...")
    df = pd.read_parquet(parquet_path)

    for i in range(min(NUM_SAMPLES, len(df))):
        row = df.iloc[i]
        text = row["text"]

        # Image is stored as {"bytes": b"...", "path": ...} dict in parquet
        img_data = row["image"]
        if isinstance(img_data, dict):
            img_bytes = img_data.get("bytes", b"")
        else:
            img_bytes = img_data

        image = Image.open(io.BytesIO(img_bytes))

        filename = f"line_{i+1:02d}.png"
        filepath = os.path.join(SAMPLES_DIR, filename)
        image.save(filepath)

        meta_path = os.path.join(SAMPLES_DIR, f"line_{i+1:02d}.txt")
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"  [{i+1}/{NUM_SAMPLES}] Saved {filename}  GT: \"{text}\"")

    # Cleanup temp file
    os.remove(parquet_path)

    print(f"\nDone. {NUM_SAMPLES} samples saved to {SAMPLES_DIR}/")
    print("Commit the samples/ directory to your repo.")


if __name__ == "__main__":
    main()
