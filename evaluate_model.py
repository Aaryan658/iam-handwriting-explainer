"""
Evaluate any TrOCR checkpoint (a HF Hub model name or a local fine-tuned
directory, e.g. out/trocr-finetuned-v1) against a CSV of image_path,text
pairs -- used to get a real stock-vs-fine-tuned CER/WER comparison instead
of relying on published benchmark numbers.

Usage:
    python evaluate_model.py --model microsoft/trocr-base-handwritten --csv samples/ground_truth.csv --image-root samples
    python evaluate_model.py --model out/trocr-finetuned-v1 --csv data/val.csv --image-root data
"""

import argparse
import csv
import os

# --- Keep all HF downloads/caches inside the project directory (D:), not
# the default C:\Users\...\.cache -- must be set before importing
# transformers so it picks this up. ---
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_HF_CACHE_DIR = os.path.join(_PROJECT_ROOT, ".hf_cache")
os.environ.setdefault("HF_HOME", _HF_CACHE_DIR)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_HF_CACHE_DIR, "hub"))

# --- huggingface_hub's httpx client hits "Cannot send a request, as the
# client has been closed" in this environment (same issue app.py works
# around) -- forcing verify=False on every httpx.Client avoids it. ---
import httpx as _httpx
_old_httpx_init = _httpx.Client.__init__
def _patched_httpx_init(self, *a, **kw):
    kw["verify"] = False
    _old_httpx_init(self, *a, **kw)
_httpx.Client.__init__ = _patched_httpx_init

import jiwer
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def load_csv(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF Hub model name or local checkpoint dir")
    parser.add_argument("--csv", required=True, help="CSV with columns image_path,text")
    parser.add_argument("--image-root", default=None, help="Base dir for relative image_path (default: CSV's dir)")
    parser.add_argument("--num-beams", type=int, default=1, help="1 = greedy (matches current app.py behavior)")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows (quick sanity check)")
    args = parser.parse_args()

    image_root = args.image_root or os.path.dirname(os.path.abspath(args.csv))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}...")
    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device)
    model.eval()

    rows = load_csv(args.csv)
    if args.limit:
        rows = rows[: args.limit]
    print(f"Evaluating {len(rows)} rows from {args.csv} (num_beams={args.num_beams})...")

    references, hypotheses = [], []
    for i, row in enumerate(rows):
        image_path = row["image_path"] if os.path.isabs(row["image_path"]) else os.path.join(image_root, row["image_path"])
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_new_tokens=128, num_beams=args.num_beams)
        hypothesis = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        references.append(row["text"])
        hypotheses.append(hypothesis)

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(rows)} done...")

    cer = jiwer.cer(references, hypotheses)
    wer = jiwer.wer(references, hypotheses)
    print(f"\nModel: {args.model}")
    print(f"CSV:   {args.csv}  (n={len(rows)})")
    print(f"CER:   {cer * 100:.2f}%")
    print(f"WER:   {wer * 100:.2f}%")


if __name__ == "__main__":
    main()
