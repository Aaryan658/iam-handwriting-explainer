# Corrections Feedback Loop & Multi-Engine Performance Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Groq's correction step use logged corrections as few-shot examples, and make the Performance tab show live, growing accuracy metrics (TrOCR stock, TrOCR+Groq, Tesseract, EasyOCR) computed from every correction a user logs — not just the 8 fixed bundled samples.

**Architecture:** `corrections_log.csv` grows two new denormalized columns (`groq_output`, plus the OCR-path metadata needed to re-derive `tesseract_output`/`easyocr_output`) computed once at logging time. `explain()` reads the tail of that same log to build a few-shot prompt block. `performance_metrics.py` and `build_corrections_dashboard()` both aggregate CER/WER across four engines using `jiwer`, one against the fixed bundled set, one against the growing corrections log.

**Tech Stack:** Python, Gradio, `jiwer` (already a dependency), `pytesseract` (new) + Tesseract-OCR binary (already installed via winget this session), `easyocr` (new).

**Reference spec:** [docs/superpowers/specs/2026-07-10-corrections-feedback-loop-design.md](../specs/2026-07-10-corrections-feedback-loop-design.md)

---

## Current state (for context)

- `app.py:342-374` — `log_correction(image_path_or_dropdown, trocr_text, user_correction)`: writes `image_id, trocr_output, user_correction, timestamp` to `corrections_log.csv`, only when `user_correction` is non-empty.
- `app.py:310-339` — `build_corrections_dashboard()`: reads the CSV, renders a "recent corrections" table.
- `app.py:488-640` — `explain(ocr_text, confidence_md)`: calls Groq with `SYSTEM_PROMPT`, no memory of past corrections.
- `app.py:999-1009` — startup block: `_perf.evaluate_stock_vs_pipeline(_GROUND_TRUTH)` computes `_AGGREGATE_CER`/`_AGGREGATE_WER` once against the 8 bundled samples.
- `app.py:1235-1287` — Performance tab: renders the aggregate metrics table and the per-sample stock-vs-pipeline comparison table.
- `performance_metrics.py` — `load_ground_truth`, `compute_cer_wer`, `_extract_corrected` (private, duplicated logic we'll share), `evaluate_stock_vs_pipeline`.
- `corrections_log.csv` — currently one row, old schema: `image_id, trocr_output, user_correction, timestamp`.
- Tesseract-OCR 5.4.0 is installed at `C:\Program Files\Tesseract-OCR\tesseract.exe` (not on PATH in every shell — hardcode the path).
- No test framework in this repo (no `tests/`, no `pytest` dependency). Verification here is manual: small `python -c` snippets to check function output, plus a browser click-through at the end, matching how the rest of `app.py` has been verified this session.

---

### Task 1: Share `extract_corrected_text` between `app.py` and `performance_metrics.py`

**Files:**
- Modify: `app.py:488` (insert new function after `explain()`, i.e. after line 640)
- Modify: `performance_metrics.py:22-33` (remove `_extract_corrected`, import shared version instead)

Both `log_correction()` (Task 5) and `performance_metrics.py`'s `evaluate_all_engines` (Task 8) need to pull the corrected transcription out of `explain()`'s markdown output. Today that logic (`_extract_corrected`) only lives in `performance_metrics.py`. Move it to `app.py` as a public function so `log_correction()` can use it too.

- [ ] **Step 1: Add `extract_corrected_text` to `app.py`**

Insert this immediately after the `explain()` function ends (after `app.py:640`, before the `# --- Paragraph pipeline UI wrapper ---` comment at line 643):

```python
def extract_corrected_text(explanation_markdown):
    """Pull the corrected transcription out of explain()'s markdown output.

    explain() renders the label as "**Corrected:**" and bolds individual
    "uncertain words" within the corrected text itself -- both are
    presentation markup, not part of the transcription, so they're stripped
    before any CER/WER comparison or logging.
    """
    match = re.search(r"Corrected:\**\s*(.+)", explanation_markdown or "")
    if not match:
        return ""
    return match.group(1).replace("**", "").strip()
```

- [ ] **Step 2: Verify it's importable and correct**

Run: `python -c "from app import extract_corrected_text; print(repr(extract_corrected_text('### LLM Correction + Confidence\n**Corrected:** put down a resolution\n**Confidence:** HIGH')))"`

Expected: `'put down a resolution'`

Run: `python -c "from app import extract_corrected_text; print(repr(extract_corrected_text('no corrected line here')))"`

Expected: `''`

- [ ] **Step 3: Update `performance_metrics.py` to use the shared function**

In `performance_metrics.py`, replace:

```python
from app import transcribe, explain
```

with:

```python
from app import transcribe, explain, extract_corrected_text
```

Delete the entire `_extract_corrected` function (lines 22-33):

```python
def _extract_corrected(explain_markdown):
    """Pull the corrected transcription out of explain()'s markdown output.

    explain() renders the label as "**Corrected:**" and bolds individual
    "uncertain words" within the corrected text itself (see app.py's
    display_corrected step) -- both are presentation markup, not part of the
    transcription, so they're stripped before CER/WER comparison.
    """
    match = re.search(r"Corrected:\**\s*(.+)", explain_markdown)
    if not match:
        return ""
    return match.group(1).replace("**", "").strip()
```

In `evaluate_stock_vs_pipeline` (line 49), replace:

```python
        pipeline_output = _extract_corrected(explain_output) or stock_output
```

with:

```python
        pipeline_output = extract_corrected_text(explain_output) or stock_output
```

The `import re` at the top of `performance_metrics.py` (line 4) is now unused — remove it.

- [ ] **Step 4: Verify performance_metrics.py still imports cleanly**

Run: `python -c "import performance_metrics"`

Expected: no output, no error.

- [ ] **Step 5: Commit**

```bash
git add app.py performance_metrics.py
git commit -m "refactor: share extract_corrected_text between app.py and performance_metrics.py"
```

---

### Task 2: Groq few-shot correction loop

**Files:**
- Modify: `app.py:488-536` (insert `get_recent_correction_examples`, wire into `explain()`)

- [ ] **Step 1: Add `get_recent_correction_examples` to `app.py`**

Insert immediately before `def explain(ocr_text, confidence_md=""):` (before line 488):

```python
def get_recent_correction_examples(n=5):
    """Read the last n logged corrections as (trocr_output, user_correction)
    pairs, most recent first, for use as Groq few-shot examples."""
    log_file = "corrections_log.csv"
    if not os.path.exists(log_file):
        return []

    import csv as _csv
    with open(log_file, "r", newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))

    recent = rows[-n:][::-1]
    return [
        (r["trocr_output"], r["user_correction"])
        for r in recent
        if r.get("trocr_output") and r.get("user_correction")
    ]
```

- [ ] **Step 2: Verify it reads the log correctly**

Run: `python -c "from app import get_recent_correction_examples; print(get_recent_correction_examples())"`

Expected: a list with one tuple (the current single row in `corrections_log.csv`), e.g.
`[('put down a resolution on the subject', 'this is a corrected sample transcription')]`

- [ ] **Step 3: Wire the examples into `explain()`'s Groq call**

In `explain()`, find this line (currently `app.py:534`):

```python
    # Pass raw OCR output directly to Groq
    corrected_ocr_text = ocr_text

    client = Groq(api_key=GROQ_API_KEY)
```

Replace with:

```python
    # Pass raw OCR output directly to Groq
    corrected_ocr_text = ocr_text

    correction_examples = get_recent_correction_examples()
    system_prompt = SYSTEM_PROMPT
    if correction_examples:
        examples_block = "\n".join(
            f'- OCR said "{wrong}" -> correct reading is "{right}"'
            for wrong, right in correction_examples
        )
        system_prompt = (
            SYSTEM_PROMPT
            + "\n\nKnown correction patterns from past user feedback (for "
            "reference only -- do not copy verbatim unless the current OCR "
            "output shows the same error):\n" + examples_block
        )

    client = Groq(api_key=GROQ_API_KEY)
```

Then find the `messages=[...]` block (currently `app.py:541-544`):

```python
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"OCR output:\n{corrected_ocr_text}"},
            ],
```

Replace `SYSTEM_PROMPT` with `system_prompt`:

```python
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"OCR output:\n{corrected_ocr_text}"},
            ],
```

- [ ] **Step 4: Verify the prompt changes without breaking explain()**

Run: `python -c "
from app import explain, GROQ_API_KEY
print('GROQ_API_KEY set:', bool(GROQ_API_KEY))
result = explain('put down a resolutoin on the subjact')
print(result)
"`

Expected: no exception; output starts with `### LLM Correction + Confidence` and includes a `**Corrected:**` line (assuming `GROQ_API_KEY` is set in the environment — if not, the printed check will show `False` and `explain()` will return the "GROQ_API_KEY not found" warning, which is also an acceptable pass for this step since it confirms no exception was raised by the new code path).

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: use logged corrections as Groq few-shot examples in explain()"
```

---

### Task 3: OCR comparison engines module

**Files:**
- Create: `ocr_engines.py`

- [ ] **Step 1: Create `ocr_engines.py`**

```python
"""Comparison OCR engines (Tesseract, EasyOCR) for benchmarking against
TrOCR, both on the bundled ground-truth set and on logged corrections."""
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

_easyocr_reader = None


def tesseract_transcribe(image_path):
    """Run Tesseract OCR on one image and return the raw transcription."""
    image = Image.open(image_path).convert("RGB")
    return pytesseract.image_to_string(image).strip()


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader


def easyocr_transcribe(image_path):
    """Run EasyOCR on one image and return the raw transcription."""
    reader = _get_easyocr_reader()
    results = reader.readtext(image_path, detail=0)
    return " ".join(results).strip()
```

- [ ] **Step 2: Add new dependencies to `requirements.txt`**

Append to `requirements.txt`:

```
pytesseract
easyocr
```

- [ ] **Step 3: Install the new dependencies**

Run: `pip install pytesseract easyocr`

Expected: both install successfully (easyocr will pull in additional dependencies like `opencv-python-headless`, `scikit-image` — this is expected).

- [ ] **Step 4: Verify both engines run against a real bundled sample**

Run: `python -c "
import ocr_engines as e
print('tesseract:', repr(e.tesseract_transcribe('samples/line_01.png')))
print('easyocr:', repr(e.easyocr_transcribe('samples/line_01.png')))
"`

Expected: two non-empty strings printed (exact OCR text will differ from TrOCR's output — that's fine, this just confirms both engines run end-to-end without exceptions). EasyOCR's first run downloads its detection/recognition models (~64MB) — expect a short delay and download progress output.

- [ ] **Step 5: Commit**

```bash
git add ocr_engines.py requirements.txt
git commit -m "feat: add Tesseract and EasyOCR comparison engines"
```

---

### Task 4: Gitignore the persisted-upload-images directory

**Files:**
- Modify: `.gitignore`

Upload-tab corrections will copy user-uploaded images into `correction_images/` (Task 5) so they survive past Gradio's temp-file lifecycle. This directory holds user-provided content and generated files — it shouldn't be committed.

- [ ] **Step 1: Add the entry**

Append to `.gitignore` (after line 17, `out/`):

```
correction_images/
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore correction_images/"
```

---

### Task 5: Extend `log_correction()` to capture per-engine outputs and a real image path

**Files:**
- Modify: `app.py:310-374` (`build_corrections_dashboard` stays as-is for now — Task 6 changes it; `log_correction` changes here)
- Modify: `app.py:1104-1132` (Sample tab wiring)
- Modify: `app.py:1195-1219` (Upload tab wiring)

- [ ] **Step 1: Replace `log_correction()`**

Replace the entire existing function (`app.py:342-374`):

```python
def log_correction(image_path_or_dropdown, trocr_text, user_correction):
    """Log the user's correction to corrections_log.csv if provided, and
    return the refreshed learning dashboard so the UI stays in sync."""
    if not user_correction or not user_correction.strip():
        return build_corrections_dashboard()

    import csv
    from datetime import datetime

    if image_path_or_dropdown:
        image_id = os.path.basename(image_path_or_dropdown)
    else:
        image_id = "unknown"

    log_file = "corrections_log.csv"
    file_exists = os.path.exists(log_file)

    try:
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["image_id", "trocr_output", "user_correction", "timestamp"])
            writer.writerow([
                image_id,
                trocr_text,
                user_correction.strip(),
                datetime.utcnow().isoformat()
            ])
        gr.Info("Correction logged successfully. Thank you!")
    except Exception as e:
        print(f"Error logging correction: {e}")

    return build_corrections_dashboard()
```

with:

```python
def _persist_correction_image(image_path):
    """Copy image_path into correction_images/ if it isn't already a
    permanent file under SAMPLES_DIR, so it survives past Gradio's
    temp-file lifecycle and can be re-run through OCR engines later."""
    if not image_path or not os.path.exists(image_path):
        return ""
    if os.path.abspath(image_path).startswith(os.path.abspath(SAMPLES_DIR)):
        return image_path

    import shutil
    from datetime import datetime

    os.makedirs("correction_images", exist_ok=True)
    dest_name = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}_{os.path.basename(image_path)}"
    dest_path = os.path.join("correction_images", dest_name)
    shutil.copyfile(image_path, dest_path)
    return dest_path


def log_correction(image_label, image_path, trocr_text, groq_explanation, user_correction):
    """Log the user's correction -- along with TrOCR's, Groq's, Tesseract's,
    and EasyOCR's output for the same image -- to corrections_log.csv, and
    return the refreshed learning dashboard so the UI stays in sync."""
    if not user_correction or not user_correction.strip():
        return build_corrections_dashboard()

    import csv
    from datetime import datetime

    image_id = os.path.basename(image_label) if image_label else "unknown"
    persisted_path = _persist_correction_image(image_path)
    groq_text = extract_corrected_text(groq_explanation) if groq_explanation else ""

    tesseract_text = ""
    easyocr_text = ""
    if persisted_path:
        try:
            tesseract_text = ocr_engines.tesseract_transcribe(persisted_path)
        except Exception as e:
            print(f"Tesseract comparison failed: {e}")
        try:
            easyocr_text = ocr_engines.easyocr_transcribe(persisted_path)
        except Exception as e:
            print(f"EasyOCR comparison failed: {e}")

    log_file = "corrections_log.csv"
    file_exists = os.path.exists(log_file)

    try:
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "image_id", "image_path", "trocr_output", "groq_output",
                    "tesseract_output", "easyocr_output", "user_correction", "timestamp"
                ])
            writer.writerow([
                image_id,
                persisted_path,
                trocr_text,
                groq_text,
                tesseract_text,
                easyocr_text,
                user_correction.strip(),
                datetime.utcnow().isoformat()
            ])
        gr.Info("Correction logged successfully. Thank you!")
    except Exception as e:
        print(f"Error logging correction: {e}")

    return build_corrections_dashboard()
```

- [ ] **Step 2: Add the `ocr_engines` import**

At the top of `app.py`, near the other local module import (`from segmentation import segment_lines`, line 24), add:

```python
import ocr_engines
```

- [ ] **Step 3: Update Sample tab wiring**

In `app.py`, find (around line 1109-1122):

```python
                sample_explain_btn.click(
                    fn=explain,
                    inputs=[sample_transcription, sample_confidence],
                    outputs=[sample_explanation],
                ).then(
                    fn=log_correction,
                    inputs=[sample_dropdown, sample_transcription, sample_verify],
                    outputs=[corrections_dashboard_display]
                )
                sample_verify.submit(
                    fn=log_correction,
                    inputs=[sample_dropdown, sample_transcription, sample_verify],
                    outputs=[corrections_dashboard_display]
                )
```

Replace with:

```python
                sample_explain_btn.click(
                    fn=explain,
                    inputs=[sample_transcription, sample_confidence],
                    outputs=[sample_explanation],
                ).then(
                    fn=log_correction,
                    inputs=[sample_dropdown, sample_image, sample_transcription, sample_explanation, sample_verify],
                    outputs=[corrections_dashboard_display]
                )
                sample_verify.submit(
                    fn=log_correction,
                    inputs=[sample_dropdown, sample_image, sample_transcription, sample_explanation, sample_verify],
                    outputs=[corrections_dashboard_display]
                )
```

(`sample_image` is already `type="filepath"` and already holds the resolved path to the bundled sample file — see `app.py:1049-1054` and `load_sample` at `app.py:1056-1059`.)

- [ ] **Step 4: Update Upload tab wiring**

In `app.py`, find (around line 1204-1217):

```python
                upload_explain_btn.click(
                    fn=explain,
                    inputs=[upload_transcription, upload_confidence],
                    outputs=[upload_explanation],
                ).then(
                    fn=log_correction,
                    inputs=[upload_image, upload_transcription, upload_verify],
                    outputs=[corrections_dashboard_display]
                )
                upload_verify.submit(
                    fn=log_correction,
                    inputs=[upload_image, upload_transcription, upload_verify],
                    outputs=[corrections_dashboard_display]
                )
```

Replace with:

```python
                upload_explain_btn.click(
                    fn=explain,
                    inputs=[upload_transcription, upload_confidence],
                    outputs=[upload_explanation],
                ).then(
                    fn=log_correction,
                    inputs=[upload_image, upload_image, upload_transcription, upload_explanation, upload_verify],
                    outputs=[corrections_dashboard_display]
                )
                upload_verify.submit(
                    fn=log_correction,
                    inputs=[upload_image, upload_image, upload_transcription, upload_explanation, upload_verify],
                    outputs=[corrections_dashboard_display]
                )
```

(`upload_image` is passed twice: once as the label source, once as the path source — both positions read the same `type="filepath"` component.)

- [ ] **Step 5: Verify `app.py` still imports and builds the UI without error**

Run: `python -c "from app import build_ui; demo = build_ui(); print('UI built OK')"`

Expected: `UI built OK` printed, no exception. (This does not start a server, just constructs the Gradio Blocks graph — it will fail fast if any wiring references an undefined variable or wrong arg count.)

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat: capture image path, Groq/Tesseract/EasyOCR output in corrections_log.csv"
```

---

### Task 6: Live Accuracy section in the corrections dashboard

**Files:**
- Modify: `app.py:310-339` (`build_corrections_dashboard`)

- [ ] **Step 1: Replace `build_corrections_dashboard()`**

Replace the entire existing function (`app.py:310-339`):

```python
def build_corrections_dashboard():
    """Summarize corrections_log.csv into a small 'the AI is learning from
    you' panel -- surfaces data the app already collects but never displays."""
    log_file = "corrections_log.csv"
    empty_msg = (
        "### 📈 Learning From Your Corrections\n"
        "_No corrections logged yet — use the correction box after Transcribe "
        "to help improve future accuracy._"
    )
    if not os.path.exists(log_file):
        return empty_msg

    import csv as _csv
    with open(log_file, "r", newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))

    if not rows:
        return empty_msg

    recent = rows[-5:][::-1]
    lines = [
        "### 📈 Learning From Your Corrections",
        f"**{len(rows)}** correction(s) logged so far. Most recent:",
        "",
        "| Sample | TrOCR Output | Your Correction |",
        "| :--- | :--- | :--- |",
    ]
    for r in recent:
        lines.append(f"| {r['image_id']} | {r['trocr_output'][:40]} | {r['user_correction'][:40]} |")
    return "\n".join(lines)
```

with:

```python
def _engine_cer_wer(rows, field):
    """Aggregate CER/WER for one engine's output column across every row
    that has both a value in that column and a user_correction to compare
    against. Returns (cer, wer) or None if no row qualifies."""
    import jiwer

    pairs = [(r[field], r["user_correction"]) for r in rows if r.get(field) and r.get("user_correction")]
    if not pairs:
        return None
    cers = [jiwer.cer(reference, hypothesis) for hypothesis, reference in pairs]
    wers = [jiwer.wer(reference, hypothesis) for hypothesis, reference in pairs]
    return sum(cers) / len(cers), sum(wers) / len(wers)


def build_corrections_dashboard():
    """Summarize corrections_log.csv into a 'recent corrections' table plus
    a live, growing per-engine accuracy table -- surfaces data the app
    already collects but never displayed before."""
    log_file = "corrections_log.csv"
    empty_msg = (
        "### 📈 Learning From Your Corrections\n"
        "_No corrections logged yet — use the correction box after Transcribe "
        "to help improve future accuracy._"
    )
    if not os.path.exists(log_file):
        return empty_msg

    import csv as _csv
    with open(log_file, "r", newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))

    if not rows:
        return empty_msg

    recent = rows[-5:][::-1]
    lines = [
        "### 📈 Learning From Your Corrections",
        f"**{len(rows)}** correction(s) logged so far. Most recent:",
        "",
        "| Sample | TrOCR Output | Your Correction |",
        "| :--- | :--- | :--- |",
    ]
    for r in recent:
        lines.append(f"| {r['image_id']} | {r['trocr_output'][:40]} | {r['user_correction'][:40]} |")

    lines += [
        "",
        "### 📈 Live Accuracy From Your Corrections",
        "_Computed from every correction you've logged — grows as you use the app._",
        "",
        "| Engine | WER | CER |",
        "| :--- | :---: | :---: |",
    ]
    engines = [
        ("Stock TrOCR", "trocr_output"),
        ("TrOCR + Groq (Pipeline)", "groq_output"),
        ("Tesseract", "tesseract_output"),
        ("EasyOCR", "easyocr_output"),
    ]
    for label, field in engines:
        result = _engine_cer_wer(rows, field)
        if result:
            cer, wer = result
            lines.append(f"| {label} | {wer*100:.2f}% | {cer*100:.2f}% |")
        else:
            lines.append(f"| {label} | _no data yet_ | _no data yet_ |")

    return "\n".join(lines)
```

- [ ] **Step 2: Verify against the current (pre-migration) log**

Run: `python -c "from app import build_corrections_dashboard; print(build_corrections_dashboard())"`

Expected: the "recent corrections" table renders as before, followed by a "Live Accuracy" table where `Stock TrOCR` shows real numbers (the CSV already has `trocr_output` and `user_correction`) and `TrOCR + Groq (Pipeline)`, `Tesseract`, `EasyOCR` all show `_no data yet_` (the old row has no `groq_output`/`tesseract_output`/`easyocr_output` columns yet — that's expected until Task 7's migration runs).

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add live per-engine accuracy table to corrections dashboard"
```

---

### Task 7: Migrate the existing `corrections_log.csv` row to the new schema

**Files:**
- Create (temporary, deleted at end of task): `migrate_corrections_log.py`
- Modify: `corrections_log.csv` (rewritten by the script)

The single existing row (`Sample 1: line_01.png, put down a resolution on the subject, this is a corrected sample transcription, 2026-07-07T09:22:10.207318`) predates the new schema. Rewrite it in place with real values for the new columns, computed once.

- [ ] **Step 1: Write the migration script**

```python
"""One-off migration: rewrite corrections_log.csv's old-schema rows into the
new schema (image_path, groq_output, tesseract_output, easyocr_output added).
Run once, then delete this file."""
import csv

from app import transcribe, explain, extract_corrected_text
import ocr_engines

with open("corrections_log.csv", "r", newline="", encoding="utf-8") as f:
    old_rows = list(csv.DictReader(f))

new_rows = []
for r in old_rows:
    image_path = f"samples/{r['image_id'].split(': ', 1)[1]}" if ": " in r["image_id"] else ""
    groq_output = ""
    tesseract_output = ""
    easyocr_output = ""
    if image_path:
        explanation = explain(r["trocr_output"])
        groq_output = extract_corrected_text(explanation)
        tesseract_output = ocr_engines.tesseract_transcribe(image_path)
        easyocr_output = ocr_engines.easyocr_transcribe(image_path)
    new_rows.append({
        "image_id": r["image_id"],
        "image_path": image_path,
        "trocr_output": r["trocr_output"],
        "groq_output": groq_output,
        "tesseract_output": tesseract_output,
        "easyocr_output": easyocr_output,
        "user_correction": r["user_correction"],
        "timestamp": r["timestamp"],
    })

fieldnames = ["image_id", "image_path", "trocr_output", "groq_output", "tesseract_output", "easyocr_output", "user_correction", "timestamp"]
with open("corrections_log.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(new_rows)

print(f"Migrated {len(new_rows)} row(s).")
for row in new_rows:
    print(row)
```

- [ ] **Step 2: Run it**

Run: `python migrate_corrections_log.py`

Expected: `Migrated 1 row(s).` followed by the row's contents printed, with non-empty `image_path`, `groq_output`, `tesseract_output`, and `easyocr_output` values.

- [ ] **Step 3: Verify the CSV**

Run: `python -c "
import csv
with open('corrections_log.csv') as f:
    rows = list(csv.DictReader(f))
print(rows[0].keys())
print(rows[0])
"`

Expected: `dict_keys(['image_id', 'image_path', 'trocr_output', 'groq_output', 'tesseract_output', 'easyocr_output', 'user_correction', 'timestamp'])` and one populated row.

- [ ] **Step 4: Verify the dashboard now shows real numbers for all four engines**

Run: `python -c "from app import build_corrections_dashboard; print(build_corrections_dashboard())"`

Expected: the "Live Accuracy" table now shows real WER/CER percentages for all four engines — no more `_no data yet_`.

- [ ] **Step 5: Delete the migration script**

```bash
rm migrate_corrections_log.py
```

- [ ] **Step 6: Commit**

```bash
git add corrections_log.csv
git commit -m "data: migrate corrections_log.csv to the multi-engine schema"
```

---

### Task 8: Four-engine comparison on the bundled ground-truth set

**Files:**
- Modify: `performance_metrics.py` (`evaluate_stock_vs_pipeline` → `evaluate_all_engines`)

- [ ] **Step 1: Add the `ocr_engines` import**

At the top of `performance_metrics.py`, add:

```python
import ocr_engines as _engines
```

- [ ] **Step 2: Replace `evaluate_stock_vs_pipeline`**

Replace the entire function (after Task 1's edits, this is the last function in the file):

```python
def evaluate_stock_vs_pipeline(ground_truth):
    """For each ground-truth row, run (1) stock transcribe() alone and (2) the
    full pipeline (transcribe -> explain()'s Groq correction), and compute
    CER/WER for both against the reference text."""
    results = []
    for row in ground_truth:
        image_path = f"samples/{row['image_path']}"
        reference = row["text"]

        stock_output = transcribe(image_path)
        stock_cer, stock_wer = compute_cer_wer(stock_output, reference)

        explain_output = explain(stock_output)
        pipeline_output = extract_corrected_text(explain_output) or stock_output
        pipeline_cer, pipeline_wer = compute_cer_wer(pipeline_output, reference)

        results.append({
            "image_path": row["image_path"],
            "reference": reference,
            "stock_output": stock_output,
            "stock_cer": stock_cer,
            "stock_wer": stock_wer,
            "pipeline_output": pipeline_output,
            "pipeline_cer": pipeline_cer,
            "pipeline_wer": pipeline_wer,
        })
    return results
```

with:

```python
def evaluate_all_engines(ground_truth):
    """For each ground-truth row, run stock TrOCR, the full pipeline
    (TrOCR -> Groq correction), Tesseract, and EasyOCR, and compute CER/WER
    for all four against the reference text."""
    results = []
    for row in ground_truth:
        image_path = f"samples/{row['image_path']}"
        reference = row["text"]

        stock_output = transcribe(image_path)
        stock_cer, stock_wer = compute_cer_wer(stock_output, reference)

        explain_output = explain(stock_output)
        pipeline_output = extract_corrected_text(explain_output) or stock_output
        pipeline_cer, pipeline_wer = compute_cer_wer(pipeline_output, reference)

        tesseract_output = _engines.tesseract_transcribe(image_path)
        tesseract_cer, tesseract_wer = compute_cer_wer(tesseract_output, reference)

        easyocr_output = _engines.easyocr_transcribe(image_path)
        easyocr_cer, easyocr_wer = compute_cer_wer(easyocr_output, reference)

        results.append({
            "image_path": row["image_path"],
            "reference": reference,
            "stock_output": stock_output,
            "stock_cer": stock_cer,
            "stock_wer": stock_wer,
            "pipeline_output": pipeline_output,
            "pipeline_cer": pipeline_cer,
            "pipeline_wer": pipeline_wer,
            "tesseract_output": tesseract_output,
            "tesseract_cer": tesseract_cer,
            "tesseract_wer": tesseract_wer,
            "easyocr_output": easyocr_output,
            "easyocr_cer": easyocr_cer,
            "easyocr_wer": easyocr_wer,
        })
    return results
```

- [ ] **Step 3: Verify it runs against the bundled set**

Run: `python -c "
import performance_metrics as p
gt = p.load_ground_truth('samples/ground_truth.csv')
results = p.evaluate_all_engines(gt[:2])
for r in results:
    print(r['image_path'], '| stock_cer', round(r['stock_cer'],3), '| pipeline_cer', round(r['pipeline_cer'],3), '| tesseract_cer', round(r['tesseract_cer'],3), '| easyocr_cer', round(r['easyocr_cer'],3))
"`

Expected: two lines printed, one per sample, each with four CER values (floats between 0 and probably-under-1). No exceptions.

- [ ] **Step 4: Commit**

```bash
git add performance_metrics.py
git commit -m "feat: add Tesseract and EasyOCR to bundled-sample evaluation"
```

---

### Task 9: Wire the four-engine comparison into the Performance tab

**Files:**
- Modify: `app.py:997-1009` (startup aggregate block)
- Modify: `app.py:1235-1244` (aggregate metrics table)
- Modify: `app.py:1272-1287` (per-sample comparison table)

- [ ] **Step 1: Update the startup block**

Replace (`app.py:997-1009`):

```python
import performance_metrics as _perf

try:
    _GROUND_TRUTH = _perf.load_ground_truth(os.path.join(SAMPLES_DIR, "ground_truth.csv"))
    _COMPARISON_ROWS = _perf.evaluate_stock_vs_pipeline(_GROUND_TRUTH)
    _AGGREGATE_CER = sum(r["pipeline_cer"] for r in _COMPARISON_ROWS) / len(_COMPARISON_ROWS)
    _AGGREGATE_WER = sum(r["pipeline_wer"] for r in _COMPARISON_ROWS) / len(_COMPARISON_ROWS)
except Exception as e:
    print(f"Performance metrics computation failed at startup: {e}")
    _COMPARISON_ROWS = []
    _AGGREGATE_CER = _AGGREGATE_WER = 0.0
```

with:

```python
import performance_metrics as _perf


def _aggregate(rows, field):
    return sum(r[field] for r in rows) / len(rows)


try:
    _GROUND_TRUTH = _perf.load_ground_truth(os.path.join(SAMPLES_DIR, "ground_truth.csv"))
    _COMPARISON_ROWS = _perf.evaluate_all_engines(_GROUND_TRUTH)
    _BUNDLED_METRICS = {
        "Stock TrOCR": (_aggregate(_COMPARISON_ROWS, "stock_wer"), _aggregate(_COMPARISON_ROWS, "stock_cer")),
        "TrOCR + Groq (Pipeline)": (_aggregate(_COMPARISON_ROWS, "pipeline_wer"), _aggregate(_COMPARISON_ROWS, "pipeline_cer")),
        "Tesseract": (_aggregate(_COMPARISON_ROWS, "tesseract_wer"), _aggregate(_COMPARISON_ROWS, "tesseract_cer")),
        "EasyOCR": (_aggregate(_COMPARISON_ROWS, "easyocr_wer"), _aggregate(_COMPARISON_ROWS, "easyocr_cer")),
    }
    _AGGREGATE_CER = _BUNDLED_METRICS["TrOCR + Groq (Pipeline)"][1]
    _AGGREGATE_WER = _BUNDLED_METRICS["TrOCR + Groq (Pipeline)"][0]
except Exception as e:
    print(f"Performance metrics computation failed at startup: {e}")
    _COMPARISON_ROWS = []
    _BUNDLED_METRICS = {}
    _AGGREGATE_CER = _AGGREGATE_WER = 0.0
```

(`_AGGREGATE_CER`/`_AGGREGATE_WER` are kept because nothing else in the file references them beyond the metrics table being replaced in Step 2 — searched and confirmed no other call sites.)

- [ ] **Step 2: Replace the aggregate metrics table**

Replace (`app.py:1235-1244`):

```python
                        gr.Markdown(
                            "### 📈 Batch Evaluation Metrics (8 Bundled Samples)\n"
                            "This table shows TrOCR + Groq pipeline performance metrics computed live at startup "
                            "against the 8 bundled ground-truth samples from the `Teklia/IAM-line` dataset."
                        )
                        gr.Markdown(
                            "| Metric | Value |\n"
                            "| :--- | :--- |\n"
                            f"| **Word Error Rate (WER)** | **{_AGGREGATE_WER * 100:.2f}%** |\n"
                            f"| **Character Error Rate (CER)** | **{_AGGREGATE_CER * 100:.2f}%** |\n"
                            f"| **Overall Word Accuracy** | **{(1 - _AGGREGATE_WER) * 100:.2f}%** |\n"
                            f"| **Overall Character Accuracy** | **{(1 - _AGGREGATE_CER) * 100:.2f}%** |"
                        )
```

with:

```python
                        gr.Markdown(
                            "### 📈 Batch Evaluation Metrics (8 Bundled Samples)\n"
                            "This table shows TrOCR, TrOCR+Groq, Tesseract, and EasyOCR performance metrics "
                            "computed live at startup against the 8 bundled ground-truth samples from the "
                            "`Teklia/IAM-line` dataset."
                        )
                        _bundled_metric_lines = [
                            "| Engine | WER | CER |",
                            "| :--- | :---: | :---: |",
                        ]
                        for _engine_label, (_wer, _cer) in _BUNDLED_METRICS.items():
                            _bundled_metric_lines.append(f"| **{_engine_label}** | **{_wer*100:.2f}%** | **{_cer*100:.2f}%** |")
                        gr.Markdown("\n".join(_bundled_metric_lines))
```

- [ ] **Step 3: Replace the per-sample comparison table**

Replace (`app.py:1272-1287`):

```python
                gr.Markdown("<hr>")
                gr.Markdown(
                    "### 👥 Stock TrOCR vs Full Pipeline (8-Sample Set)\n"
                    "A sample-by-sample comparison of raw TrOCR output against the full pipeline "
                    "(TrOCR followed by Groq correction), computed live at startup against the bundled ground truth."
                )
                _comparison_lines = [
                    "| Sample | Ground Truth | Stock TrOCR | Stock CER | Pipeline (TrOCR+Groq) | Pipeline CER |",
                    "| :--- | :--- | :--- | :---: | :--- | :---: |",
                ]
                for r in _COMPARISON_ROWS:
                    _comparison_lines.append(
                        f"| {r['image_path']} | {r['reference']} | {r['stock_output']} | "
                        f"{r['stock_cer']*100:.2f}% | {r['pipeline_output']} | {r['pipeline_cer']*100:.2f}% |"
                    )
                gr.Markdown("\n".join(_comparison_lines))
```

with:

```python
                gr.Markdown("<hr>")
                gr.Markdown(
                    "### 👥 Stock TrOCR vs Full Pipeline vs Tesseract vs EasyOCR (8-Sample Set)\n"
                    "A sample-by-sample comparison of raw TrOCR output against the full pipeline "
                    "(TrOCR followed by Groq correction), Tesseract, and EasyOCR, computed live at startup "
                    "against the bundled ground truth."
                )
                _comparison_lines = [
                    "| Sample | Ground Truth | Stock TrOCR | Stock CER | Pipeline (TrOCR+Groq) | Pipeline CER | Tesseract | Tesseract CER | EasyOCR | EasyOCR CER |",
                    "| :--- | :--- | :--- | :---: | :--- | :---: | :--- | :---: | :--- | :---: |",
                ]
                for r in _COMPARISON_ROWS:
                    _comparison_lines.append(
                        f"| {r['image_path']} | {r['reference']} | {r['stock_output']} | "
                        f"{r['stock_cer']*100:.2f}% | {r['pipeline_output']} | {r['pipeline_cer']*100:.2f}% | "
                        f"{r['tesseract_output']} | {r['tesseract_cer']*100:.2f}% | "
                        f"{r['easyocr_output']} | {r['easyocr_cer']*100:.2f}% |"
                    )
                gr.Markdown("\n".join(_comparison_lines))
```

- [ ] **Step 4: Verify the full app builds and the Performance tab computes without error**

Run: `python -c "
from app import build_ui, _BUNDLED_METRICS, _COMPARISON_ROWS
print('engines:', list(_BUNDLED_METRICS.keys()))
print('bundled rows:', len(_COMPARISON_ROWS))
demo = build_ui()
print('UI built OK')
"`

Expected:
```
engines: ['Stock TrOCR', 'TrOCR + Groq (Pipeline)', 'Tesseract', 'EasyOCR']
bundled rows: 8
UI built OK
```

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: show 4-engine comparison in the Performance tab"
```

---

### Task 10: Browser verification

**Files:** none (manual verification only)

- [ ] **Step 1: Start the dev server**

Use the `gradio-app` launch config (`.claude/launch.json`, port 7860/auto) via the preview tooling, or run directly:

Run: `python app.py`

Wait for the Gradio startup banner (`Running on local URL: http://127.0.0.1:PORT`). Note this will take longer than before — EasyOCR's models load/download during the startup `evaluate_all_engines` call across all 8 bundled samples.

- [ ] **Step 2: Confirm the Performance tab renders both new tables**

Open the app in a browser, click the "📊 Performance" tab. Confirm:
- The "Batch Evaluation Metrics" table shows 4 rows (Stock TrOCR, TrOCR + Groq, Tesseract, EasyOCR) with WER/CER percentages.
- The per-sample comparison table has 10 columns and 8 data rows, with Tesseract/EasyOCR output text visible.
- Below "Learning From Your Corrections", confirm a "📈 Live Accuracy From Your Corrections" table appears with real percentages for all four engines (populated by Task 7's migration).

- [ ] **Step 3: Log a new correction from the Sample tab and confirm all columns populate**

Click "📋 Sample Lines", pick a sample, click Transcribe, type a deliberately different correction into the correction box, click Explain (or press Enter in the correction box).

Check `corrections_log.csv` — confirm a new row was appended with non-empty `image_path`, `trocr_output`, `tesseract_output`, `easyocr_output` (and `groq_output` non-empty if Explain was clicked before submitting).

- [ ] **Step 4: Confirm the Live Accuracy table updates without a page reload**

After Step 3, re-check the Performance tab in the same browser session (no reload) — the "Live Accuracy" table and the correction count in "Learning From Your Corrections" should reflect the new row immediately (both are re-rendered as the output of `log_correction`'s Gradio event chain).

- [ ] **Step 5: Log a correction from the Upload tab and confirm the image persists**

Click "📤 Upload", upload any handwriting image, transcribe it, type a correction, submit. Confirm a new file appears under `correction_images/`, and the corresponding `corrections_log.csv` row's `image_path` points to it.

---

## Self-review notes

- **Spec coverage:** Section 1 (few-shot loop) → Task 2. Section 2 (schema change) → Tasks 5, 7. Section 3 (OCR engine helpers) → Task 3. Section 4 (Performance tab changes, both bundled and live) → Tasks 6, 8, 9. Section 5 (dependencies) → Task 3, Step 2. Testing section → Task 10 plus the inline verification step in each task.
- **Migration ordering:** Task 7 (migrate the CSV) depends on Task 3 (`ocr_engines.py` must exist) and Task 1 (`extract_corrected_text` must exist) — tasks are ordered so each depends only on earlier tasks.
- **`_AGGREGATE_CER`/`_AGGREGATE_WER` globals:** kept alive after Task 9 only because nothing outside the replaced table block reads them; if a future change removes the table, these two lines can go too.
