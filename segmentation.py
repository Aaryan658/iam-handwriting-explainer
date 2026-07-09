"""
Phase 1 (+ Phase 3 extension point): segment a full-page/paragraph handwriting
image into individual line crops using classical OpenCV techniques, so each
crop can be fed to TrOCR (which only ever reads a single line at a time).

Pipeline: grayscale -> Sauvola adaptive binarization -> deskew -> horizontal
projection profile -> line bands -> padded crops of the *original* image.

Sauvola is implemented via cv2.boxFilter (local mean / local mean-of-squares)
rather than scikit-image, so this module has no extra dependencies beyond
opencv-python, numpy and Pillow, which the app already requires.
"""

import numpy as np
import cv2
from PIL import Image


def _to_gray(image):
    """Accept a PIL.Image or numpy array, return a float32 grayscale numpy array."""
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    return gray.astype(np.float32)


def sauvola_binarize(gray, window=25, k=0.2, r=128.0):
    """
    Local adaptive (Sauvola) binarization using box filters for the local
    mean/std, so the whole page is thresholded in O(1) per pixel instead of
    a per-pixel sliding window. Returns a uint8 mask where 255 = ink (foreground).
    """
    mean = cv2.boxFilter(gray, ddepth=-1, ksize=(window, window), normalize=True)
    mean_sq = cv2.boxFilter(gray * gray, ddepth=-1, ksize=(window, window), normalize=True)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))

    threshold = mean * (1.0 + k * ((std / r) - 1.0))
    ink = (gray < threshold).astype(np.uint8) * 255
    return ink


def deskew(gray, ink_mask):
    """
    Estimate and correct page rotation from the ink mask's minimum-area
    bounding rectangle. Returns (rotated_gray, rotated_ink_mask, angle_degrees).
    """
    ys, xs = np.where(ink_mask > 0)
    coords = np.column_stack((xs, ys)).astype(np.float32)  # minAreaRect wants (x, y)
    if len(coords) < 20:
        return gray, ink_mask, 0.0

    angle = cv2.minAreaRect(coords)[-1]
    # cv2.minAreaRect angles are in [-90, 0); normalize into [-45, 45] so a
    # near-horizontal page rotates by a few degrees, not ~90.
    if angle < -45:
        angle = 90 + angle

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_gray = cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    rotated_ink = cv2.warpAffine(ink_mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rotated_gray, rotated_ink, angle


def _line_bands(ink_mask, min_line_height_ratio=0.25, min_gap=2):
    """
    Horizontal projection profile: sum ink pixels per row, treat contiguous
    runs of non-zero rows (allowing gaps smaller than min_gap) as line bands.
    Filters out bands shorter than min_line_height_ratio * median band height
    (noise/specks rather than real text lines).
    """
    row_sums = ink_mask.sum(axis=1)
    is_text_row = row_sums > 0

    bands = []
    start = None
    gap = 0
    for y, has_text in enumerate(is_text_row):
        if has_text:
            if start is None:
                start = y
            gap = 0
        else:
            if start is not None:
                gap += 1
                if gap > min_gap:
                    bands.append((start, y - gap))
                    start = None
                    gap = 0
    if start is not None:
        bands.append((start, len(is_text_row) - 1))

    if not bands:
        return []

    heights = [b - a for a, b in bands]
    median_h = float(np.median(heights))
    min_h = median_h * min_line_height_ratio
    return [(a, b) for a, b in bands if (b - a) >= min_h]


def segment_lines_opencv(image, padding=4, window=25, k=0.2):
    """
    Default (Phase 1) line segmenter. Returns a list of dicts:
        {"image": PIL.Image, "bbox": (y0, y1), "gap_before": int or None}
    in top-to-bottom reading order. gap_before is the pixel gap to the
    previous line (None for the first line) -- used downstream to infer
    paragraph breaks.
    """
    pil_image = image if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")
    rgb = np.array(pil_image.convert("RGB"))
    gray = _to_gray(rgb)

    ink = sauvola_binarize(gray, window=window, k=k)
    _, ink_deskewed, angle = deskew(gray, ink)
    rotated_rgb = cv2.warpAffine(
        rgb,
        cv2.getRotationMatrix2D((rgb.shape[1] // 2, rgb.shape[0] // 2), angle, 1.0),
        (rgb.shape[1], rgb.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderValue=(255, 255, 255),
    )

    bands = _line_bands(ink_deskewed)
    h = rotated_rgb.shape[0]

    lines = []
    prev_bottom = None
    for (y0, y1) in bands:
        crop_top = max(0, y0 - padding)
        crop_bottom = min(h, y1 + padding)
        crop = rotated_rgb[crop_top:crop_bottom, :]
        gap_before = None if prev_bottom is None else max(0, y0 - prev_bottom)
        lines.append({
            "image": Image.fromarray(crop),
            "bbox": (crop_top, crop_bottom),
            "gap_before": gap_before,
        })
        prev_bottom = y1

    return lines


def segment_lines(image, method="opencv", **kwargs):
    """
    Phase-3 extension point: dispatches to a segmentation backend by name.
    "opencv" (default) is the classical Sauvola+projection-profile approach
    in this file -- fast, dependency-free, and good enough for clean/ruled
    pages, but it can misfire on heavily skewed, bleed-through, or
    marginalia-heavy historical scans (see project research notes).

    To swap in a learned line detector (e.g. Kraken's trainable baseline
    segmenter, or a fine-tuned YOLO/RF-DETR model) once real historical
    scans show OpenCV segmentation breaking down, add another branch here
    returning the same list-of-{"image", "bbox", "gap_before"} shape --
    paragraph_pipeline.py only depends on that shape, not on how it's produced.
    """
    if method == "opencv":
        return segment_lines_opencv(image, **kwargs)
    raise NotImplementedError(
        f"Segmentation method '{method}' is not implemented yet. "
        "This is a deliberate extension point -- see the module docstring."
    )
