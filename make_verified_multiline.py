"""
Composite line_01.png + line_02.png + line_03.png vertically with whitespace
gaps, save as samples/verified_multiline_test.png, then run
paragraph_pipeline.transcribe_paragraph() on it and print raw output.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
import paragraph_pipeline

SAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
OUT = os.path.join(SAMPLES, "verified_multiline_test.png")
GAP = 60  # px of white between lines

# ── load the three known-good single-line scans ────────────────────────────
imgs = [
    Image.open(os.path.join(SAMPLES, "line_01.png")).convert("RGB"),
    Image.open(os.path.join(SAMPLES, "line_02.png")).convert("RGB"),
    Image.open(os.path.join(SAMPLES, "line_03.png")).convert("RGB"),
]

for i, img in enumerate(imgs):
    print(f"line_0{i+1}.png  size={img.size}  mode={img.mode}")

# ── composite ──────────────────────────────────────────────────────────────
max_w = max(img.width for img in imgs)
total_h = sum(img.height for img in imgs) + GAP * (len(imgs) - 1)
canvas = Image.new("RGB", (max_w, total_h), (255, 255, 255))

y_cursor = 0
for img in imgs:
    canvas.paste(img, (0, y_cursor))
    y_cursor += img.height + GAP

canvas.save(OUT)
print(f"\nSaved: {OUT}  final_size={canvas.size}")

# ── run pipeline ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("paragraph_pipeline.transcribe_paragraph(samples/verified_multiline_test.png)")
print("=" * 70)

text, per_line = paragraph_pipeline.transcribe_paragraph(OUT)

print(f"--- segmented into {len(per_line)} line(s) ---")
for i, r in enumerate(per_line):
    print(f"line {i} (y={r['bbox'][0]}-{r['bbox'][1]}): {r['text']!r}")

print("--- reassembled paragraph ---")
print(text)

print("\n--- ground-truth comparison ---")
gt = {
    "line_01": "put down a resolution on the subject",
    "line_02": "and he is to be backed by Mr. Will",
    "line_03": "nominating any more Labour life Peers",
}
for k, v in gt.items():
    print(f"  {k}: {v!r}")
