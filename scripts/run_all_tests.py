"""
Raw output test — no summaries, no descriptions.
Step 3a: paragraph_pipeline.transcribe_paragraph on a clean multi-line image.
Step 3b: paragraph_pipeline.transcribe_paragraph on all 8 single-line samples.
"""
import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import paragraph_pipeline

SAMPLES_DIR = os.path.join(_REPO_ROOT, "samples")

# ── 3a ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("3a. CLEAN MULTI-LINE IMAGE: samples/clean_paragraph_test.png")
print("=" * 70)
paragraph_image = os.path.join(SAMPLES_DIR, "clean_paragraph_test.png")
text, per_line = paragraph_pipeline.transcribe_paragraph(paragraph_image)
print(f"--- segmented into {len(per_line)} line(s) ---")
for i, r in enumerate(per_line):
    print(f"line {i} (y={r['bbox'][0]}-{r['bbox'][1]}): {r['text']!r}")
print("--- reassembled paragraph ---")
print(text)
print()

# ── 3b ──────────────────────────────────────────────────────────────────────
single_line_samples = [
    ("samples/line_01.png", "samples/line_01.txt"),
    ("samples/line_02.png", "samples/line_02.txt"),
    ("samples/line_03.png", "samples/line_03.txt"),
    ("samples/line_04.png", "samples/line_04.txt"),
    ("samples/line_05.png", "samples/line_05.txt"),
    ("samples/line_06.png", "samples/line_06.txt"),
    ("samples/line_07.png", "samples/line_07.txt"),
    ("samples/line_08.png", "samples/line_08.txt"),
]

print("=" * 70)
print("3b. ALL SINGLE-LINE SAMPLES (regression check)")
print("=" * 70)
for img_rel, gt_rel in single_line_samples:
    img_path = os.path.join(_REPO_ROOT, img_rel)
    gt_path  = os.path.join(_REPO_ROOT, gt_rel)
    expected = open(gt_path, encoding="utf-8").read().strip()

    print(f"\n--- {img_rel} ---")
    text, per_line = paragraph_pipeline.transcribe_paragraph(img_path)
    print(f"segmented line count: {len(per_line)}")
    for i, r in enumerate(per_line):
        print(f"  line {i} (y={r['bbox'][0]}-{r['bbox'][1]}): {r['text']!r}")
    print(f"reassembled output : {text!r}")
    print(f"ground-truth txt   : {expected!r}")
