"""
One-time setup script: downloads the FULL Teklia/IAM-line dataset (all
splits, not just 8 samples like download_samples.py) via direct parquet
URLs, and writes line-crop PNGs plus data/train.csv and data/val.csv in the
image_path,text format finetune_trocr.py's LineOCRDataset expects.

Uses the same direct-parquet-URL approach as download_samples.py (rather
than datasets.load_dataset) because huggingface_hub's HTTP client hit a
"client has been closed" error in this environment -- the raw parquet URL
path is already proven to work here.

Run once locally:
    python prepare_iam_data.py
"""

import csv
import io
import os
import ssl
import urllib.request

import pandas as pd
from PIL import Image

# --- Keep all downloads/caches inside the project directory (D:), not
# scattered into C:\Users\...\.cache. ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LINES_DIR = os.path.join(DATA_DIR, "lines")

# --- This machine has known SSL/cert issues reaching HF Hub (see the same
# bypass in app.py and download_samples.py). ---
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

# Confirmed via https://datasets-server.huggingface.co/parquet?dataset=Teklia/IAM-line
# -- each split is a single parquet shard.
PARQUET_URLS = {
    "train": "https://huggingface.co/datasets/Teklia/IAM-line/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    "val": "https://huggingface.co/datasets/Teklia/IAM-line/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
}


def _download_parquet(url, dest_path):
    req = urllib.request.Request(url, headers={"User-Agent": "Python"})
    with urllib.request.urlopen(req, context=SSL_CTX, timeout=120) as resp:
        with open(dest_path, "wb") as f:
            f.write(resp.read())


def _write_split(df, split_name, csv_path):
    os.makedirs(LINES_DIR, exist_ok=True)
    rows_written = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "text"])
        for i, row in df.iterrows():
            text = row["text"]
            if not isinstance(text, str) or not text.strip():
                continue

            img_data = row["image"]
            img_bytes = img_data.get("bytes", b"") if isinstance(img_data, dict) else img_data
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            filename = f"{split_name}_{i:05d}.png"
            image.save(os.path.join(LINES_DIR, filename))
            writer.writerow([os.path.join("lines", filename), text])
            rows_written += 1
            if rows_written % 500 == 0:
                print(f"  [{split_name}] {rows_written} lines written...")
    print(f"{split_name}: {rows_written} lines -> {csv_path}")
    return rows_written


# Known-good sizes (from datasets-server.huggingface.co/parquet?dataset=Teklia/IAM-line)
# -- lets a re-run skip a redundant download if a prior interrupted run
# already fetched a split's parquet in full.
EXPECTED_SIZES = {"train": 167218165, "val": 24747019}


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for split_name, url in PARQUET_URLS.items():
        parquet_path = os.path.join(DATA_DIR, f"_temp_{split_name}.parquet")
        if os.path.exists(parquet_path) and os.path.getsize(parquet_path) == EXPECTED_SIZES.get(split_name):
            print(f"Reusing already-downloaded {split_name} parquet ({parquet_path}).")
        else:
            print(f"Downloading {split_name} parquet ({url})...")
            _download_parquet(url, parquet_path)

        print(f"Parsing {split_name} parquet...")
        df = pd.read_parquet(parquet_path)
        csv_name = "train.csv" if split_name == "train" else "val.csv"
        _write_split(df, split_name, os.path.join(DATA_DIR, csv_name))

        os.remove(parquet_path)

    print("\nDone. data/train.csv and data/val.csv are ready for finetune_trocr.py.")


if __name__ == "__main__":
    main()
