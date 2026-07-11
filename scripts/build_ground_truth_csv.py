"""One-time script: consolidate samples/line_*.txt sidecar files (written by
download_samples.py) into a single samples/ground_truth.csv."""
import csv
import glob
import os

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")


def main():
    rows = []
    for txt_path in sorted(glob.glob(os.path.join(SAMPLES_DIR, "line_*.txt"))):
        image_path = txt_path[:-4] + ".png"
        if not os.path.exists(image_path):
            continue
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        rows.append({"image_path": os.path.basename(image_path), "text": text})

    out_path = os.path.join(SAMPLES_DIR, "ground_truth.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "text"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
