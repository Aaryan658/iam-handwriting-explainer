# Corrections Feedback Loop & Multi-Engine Performance Metrics

## Problem

Two gaps in the current app:

1. `corrections_log.csv` is logged but never *used* — the "Learning From Your
   Corrections" dashboard panel implies the app learns from feedback, but
   nothing reads the log back into the pipeline. It's a display-only audit
   trail.
2. The Performance tab's WER/CER metrics (`_AGGREGATE_CER`, `_AGGREGATE_WER`
   in `app.py`) are computed once at startup against a fixed set of 8 bundled
   IAM samples (`samples/ground_truth.csv`). They never reflect real usage —
   uploaded images or corrected sample transcriptions don't move the numbers
   at all, and there's no comparison against other OCR engines.

## Goals

- Make Groq's correction step (`explain()`) actually use past user
  corrections as few-shot examples, so recurring TrOCR mistakes get caught
  more often — without retraining any model.
- Extend the Performance tab so it shows live accuracy computed from
  everything a user has corrected (Sample or Upload tab), not just the 8
  bundled samples.
- Add Tesseract and EasyOCR as comparison baselines alongside TrOCR (stock)
  and TrOCR+Groq (pipeline), for both the bundled benchmark and the live
  corrections data.

## Non-goals

- No model fine-tuning or weight updates of any kind (already tried and
  abandoned per prior session — underperformed stock TrOCR-large).
- No re-running inference on historical images at every Performance-tab
  render. All per-engine outputs for a correction are computed once, at
  logging time, and stored.
- No legacy-row compatibility code. `corrections_log.csv` currently has one
  row; it is migrated directly to the new schema as part of this change.

## Design

### 1. Groq few-shot correction loop

`get_recent_correction_examples(n=5)` (new function, `app.py`): reads
`corrections_log.csv`, returns the last 5 rows as
`(trocr_output, user_correction)` pairs, most recent first.

`explain()`'s Groq prompt gains a "Known correction patterns from past user
feedback" section built from these pairs, included only when the log has at
least one row (the section is omitted entirely otherwise — no placeholder
text).

This runs on every `explain()` call, both Sample and Upload tabs.

### 2. `corrections_log.csv` schema change

New schema (replaces the current one):

```
image_id, image_path, trocr_output, groq_output, tesseract_output, easyocr_output, user_correction, timestamp
```

- `image_id`: unchanged — `Sample N: line_NN.png` or the upload's basename.
- `image_path`: resolvable path to the actual image file.
  - Sample tab: derived from `SAMPLES_DIR` + the dropdown selection (files
    already committed to the repo — no copying needed).
  - Upload tab: the Gradio temp filepath is copied into a new
    `correction_images/` directory (created if missing) so the file survives
    past the Gradio temp-file lifecycle. Filename:
    `{timestamp}_{original_basename}`.
- `groq_output`: the corrected text extracted from `explain()`'s markdown
  output, using the same extraction regex `performance_metrics._extract_corrected`
  already implements (moved to a shared location — see Implementation notes).
  Empty string when `log_correction` fires from the textbox `Enter` submit
  before `Explain` has ever run for that transcription.
- `tesseract_output` / `easyocr_output`: computed once at log time by running
  both engines against `image_path`.
- `user_correction`, `timestamp`: unchanged.

`log_correction()` signature grows from
`(image_path_or_dropdown, trocr_text, user_correction)` to
`(image_path_or_dropdown, trocr_text, groq_text, user_correction)`. Both
Sample and Upload wiring pass their respective `*_explanation` component as
the new `groq_text` input.

**Migration**: the existing single data row in `corrections_log.csv` is
rewritten by hand to the new schema (`image_path` backfilled to
`samples/line_01.png`; `groq_output`/`tesseract_output`/`easyocr_output`
computed once during migration and filled in). This is a one-time data edit,
not runtime migration code.

### 3. OCR engine helpers

New module `ocr_engines.py`:

```python
def tesseract_transcribe(image_path: str) -> str: ...
def easyocr_transcribe(image_path: str) -> str: ...
```

- `tesseract_transcribe`: uses `pytesseract`, with
  `pytesseract.pytesseract.tesseract_cmd` hardcoded to
  `C:\Program Files\Tesseract-OCR\tesseract.exe` (installed via winget this
  session; not relying on PATH since it's not guaranteed to be set in every
  shell/process that runs the app).
- `easyocr_transcribe`: uses a module-level `easyocr.Reader(['en'])`
  (lazy-initialized once, reused across calls — matches how TrOCR's model is
  loaded once at import time elsewhere in `app.py`).

### 4. Performance tab changes

**Bundled 8-sample section** (`performance_metrics.py` +
`evaluate_stock_vs_pipeline`, renamed `evaluate_all_engines`): for each
ground-truth row, also run `tesseract_transcribe` and `easyocr_transcribe`,
compute CER/WER for both. The aggregate metrics table gains WER/CER rows for
Tesseract and EasyOCR; the per-sample comparison table gains
`Tesseract` / `Tesseract CER` / `EasyOCR` / `EasyOCR CER` columns.

**New "📈 Live Accuracy From Your Corrections" section**, added inside
`build_corrections_dashboard()` below the existing "most recent corrections"
table: aggregate WER/CER for all four engines (`trocr_output`, `groq_output`,
`tesseract_output`, `easyocr_output`, each vs `user_correction` as ground
truth), computed from every row in `corrections_log.csv`. Rows with an empty
`groq_output` are excluded only from the Groq aggregate, not the others.
Recomputed on every `build_corrections_dashboard()` call, so it updates
immediately after each new correction — no separate refresh mechanism.

This section is clearly separated from the bundled-sample table (different
heading, own "grows as you use the app" framing) rather than pooled into one
number, since the bundled set is a controlled benchmark and the corrections
set is a self-selected sample of images users found worth correcting (skews
toward errors) — merging them would misrepresent both.

### 5. Dependencies

- `pytesseract` (pip) + Tesseract-OCR 5.4.0 binary (installed via winget,
  `UB-Mannheim.TesseractOCR`)
- `easyocr` (pip) — downloads its own detection/recognition models on first
  use (~64MB)

Both added to `requirements.txt` (the runtime-deps file `app.py` already
lists `gradio`, `transformers`, `torch`, `groq`, etc. in).

## Testing

- Existing `evaluate_model.py` continues to work against `trocr_output`
  column semantics unchanged.
- Manual verification: log a correction from the Sample tab and from the
  Upload tab, confirm all four `*_output` columns populate in
  `corrections_log.csv`, confirm the Live Accuracy section updates in the
  browser without a page reload.
- Confirm `explain()`'s prompt includes the few-shot section only when
  `corrections_log.csv` has rows (empty-log case still needs to work for a
  fresh clone).
