"""
Phase 2: run Phase 1's line segmentation over a full page/paragraph image,
transcribe each line with the existing single-line TrOCR pipeline from
app.py, and reassemble the lines into paragraph text.

Paragraph breaks are inferred from vertical gaps between consecutive line
bands: a gap noticeably larger than the median gap is treated as a
paragraph break (blank line), otherwise lines are joined with a single
newline.
"""

import sys
import statistics

from PIL import Image

from segmentation import segment_lines
from app import transcribe, transcribe_with_confidence_score, format_confidence_badge


def _join_lines(lines_text, gaps, paragraph_gap_ratio=1.6):
    """
    lines_text: list[str], one TrOCR transcription per line, in order.
    gaps: list[int or None], gap_before for each line (None for the first).
    """
    if not lines_text:
        return ""

    # Use the smallest gap as the "typical single-line spacing" baseline --
    # more robust than the median when there are only a couple of gaps to
    # sample from (a single paragraph break shouldn't skew the baseline).
    numeric_gaps = [g for g in gaps if g is not None]
    baseline_gap = min(numeric_gaps) if numeric_gaps else 0

    out = [lines_text[0]]
    for text, gap in zip(lines_text[1:], gaps[1:]):
        if baseline_gap > 0 and gap is not None and gap >= baseline_gap * paragraph_gap_ratio:
            out.append("")  # blank line -> paragraph break
        out.append(text)
    return "\n".join(out)


def transcribe_paragraph(image, segmentation_method="opencv"):
    """
    image: a PIL.Image, numpy array, or filepath string for a full
    page/paragraph handwriting scan.

    Returns (paragraph_text, per_line_results) where per_line_results is a
    list of {"text": str, "bbox": (y0, y1)} for inspection/debugging.
    """
    if isinstance(image, str):
        image = Image.open(image)

    lines = segment_lines(image, method=segmentation_method)
    if not lines:
        return "", []

    per_line_results = []
    for line in lines:
        text = transcribe(line["image"])
        per_line_results.append({"text": text, "bbox": line["bbox"]})

    paragraph_text = _join_lines(
        [r["text"] for r in per_line_results],
        [line["gap_before"] for line in lines],
    )
    return paragraph_text, per_line_results


def transcribe_paragraph_with_confidence(image=None, segmentation_method="opencv", lines=None):
    """
    Like transcribe_paragraph(), but also returns a model-grounded confidence
    badge aggregated across all segmented lines, using the same TrOCR
    token-probability signal as transcribe_with_confidence() in app.py.

    The aggregate uses the *weakest* line's confidence rather than an average
    -- a paragraph's trustworthiness is bounded by its worst-transcribed line,
    not smoothed out by its best one.

    Pass pre-segmented `lines` (from segmentation.segment_lines) to skip
    re-segmenting when the caller already segmented once to decide between
    the single-line and paragraph paths (e.g. app.py's upload routing).

    Returns (paragraph_text, confidence_md, per_line_results).
    """
    if lines is None:
        if isinstance(image, str):
            image = Image.open(image)
        lines = segment_lines(image, method=segmentation_method)

    if not lines:
        return "", "", []

    per_line_results = []
    line_probs = []
    for line in lines:
        text, prob = transcribe_with_confidence_score(line["image"])
        per_line_results.append({"text": text, "bbox": line["bbox"]})
        line_probs.append(prob)

    paragraph_text = _join_lines(
        [r["text"] for r in per_line_results],
        [line["gap_before"] for line in lines],
    )
    confidence_md = format_confidence_badge(min(line_probs))
    return paragraph_text, confidence_md, per_line_results


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "samples/education_paragraph.png"
    text, per_line = transcribe_paragraph(path)
    print(f"--- segmented into {len(per_line)} line(s) ---")
    for i, r in enumerate(per_line):
        print(f"line {i} (y={r['bbox'][0]}-{r['bbox'][1]}): {r['text']!r}")
    print("--- reassembled paragraph ---")
    print(text)
