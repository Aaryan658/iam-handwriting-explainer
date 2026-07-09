# Supabase Auth, RL Data Logging & Real Performance Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Performance tab's hardcoded static numbers with live CER/WER and a stock-TrOCR-vs-full-pipeline comparison; add Supabase-backed transcription/correction logging; gate the app behind Google OAuth — all without leaving Gradio.

**Architecture:** A new `performance_metrics.py` module computes CER/WER live via `jiwer` against a ground-truth CSV consolidated from files `download_samples.py` already wrote. A new `supabase_client.py` module wraps `supabase-py` inserts, with the client passed as a parameter so it's testable without a live project. FastAPI wraps the existing Gradio `Blocks` app (`gr.mount_gradio_app`) to add three auth routes; nothing about the existing TrOCR/Groq/segmentation pipeline changes.

**Tech Stack:** Gradio (existing), jiwer (existing), supabase-py (new), fastapi (new, gradio already depends on starlette but we add fastapi directly for the auth routes), pytest (new, for the TDD tasks below).

---

## Before you start

Phase 1 (Tasks 1–4) has no external dependencies and can be built and fully verified right now. Phases 2 and 3 need things only you can do:

- **Before Task 5 can be verified:** a Supabase project (free tier is fine) — its Project URL and `service_role` key.
- **Before Task 9 can be verified:** a Google Cloud OAuth 2.0 Client ID/Secret, added as a provider in Supabase Auth → Providers → Google, with the redirect URL set to `<your-app-url>/auth/callback`.

Each blocked task says exactly what's needed and how to hand it to me.

Install pytest once, up front:
```bash
.venv/Scripts/pip install pytest
```

---

## Phase 1: Real Performance Tab (no blockers)

### Task 1: Consolidate ground truth into a CSV

`download_samples.py` already wrote `samples/line_01.txt` through `samples/line_08.txt` — the real IAM-line ground truth for each bundled sample. This task consolidates them into one CSV instead of leaving them scattered.

**Files:**
- Create: `build_ground_truth_csv.py`
- Create (by running the script): `samples/ground_truth.csv`

- [ ] **Step 1: Write the script**

```python
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
```

- [ ] **Step 2: Run it**

Run: `.venv/Scripts/python build_ground_truth_csv.py`
Expected: `Wrote 8 rows to .../samples/ground_truth.csv`, and the file exists with a header row plus 8 data rows.

- [ ] **Step 3: Commit**

```bash
git add build_ground_truth_csv.py samples/ground_truth.csv
git commit -m "feat: consolidate sample ground-truth transcriptions into a CSV"
```

### Task 2: CER/WER computation module

**Files:**
- Create: `performance_metrics.py`
- Test: `test_performance_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
from performance_metrics import load_ground_truth, compute_cer_wer


def test_load_ground_truth_reads_csv(tmp_path):
    csv_path = tmp_path / "gt.csv"
    csv_path.write_text("image_path,text\nline_01.png,hello world\n", encoding="utf-8")
    rows = load_ground_truth(str(csv_path))
    assert rows == [{"image_path": "line_01.png", "text": "hello world"}]


def test_compute_cer_wer_identical_strings_is_zero():
    cer, wer = compute_cer_wer(hypothesis="hello world", reference="hello world")
    assert cer == 0.0
    assert wer == 0.0


def test_compute_cer_wer_detects_errors():
    cer, wer = compute_cer_wer(hypothesis="helo wrold", reference="hello world")
    assert cer > 0.0
    assert wer > 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/pytest test_performance_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'performance_metrics'`

- [ ] **Step 3: Write the minimal implementation**

```python
"""Live performance metrics for the Performance tab: CER/WER against the
bundled ground-truth samples, and a stock-TrOCR-vs-full-pipeline comparison."""
import csv

import jiwer


def load_ground_truth(csv_path="samples/ground_truth.csv"):
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_cer_wer(hypothesis, reference):
    cer = jiwer.cer(reference, hypothesis)
    wer = jiwer.wer(reference, hypothesis)
    return cer, wer
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/pytest test_performance_metrics.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add performance_metrics.py test_performance_metrics.py
git commit -m "feat: add CER/WER computation module for live performance metrics"
```

### Task 3: Stock-vs-pipeline comparison function

**Files:**
- Modify: `performance_metrics.py`
- Test: `test_performance_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
def test_evaluate_stock_vs_pipeline_uses_transcribe_and_explain(monkeypatch):
    from performance_metrics import evaluate_stock_vs_pipeline

    def fake_transcribe(image_path):
        return "helo wrold"

    def fake_explain(ocr_text, confidence_md=""):
        return "### LLM Correction + Confidence\nCorrected: hello world\nConfidence: HIGH\n"

    monkeypatch.setattr("performance_metrics.transcribe", fake_transcribe)
    monkeypatch.setattr("performance_metrics.explain", fake_explain)

    ground_truth = [{"image_path": "line_01.png", "text": "hello world"}]
    results = evaluate_stock_vs_pipeline(ground_truth)

    assert len(results) == 1
    row = results[0]
    assert row["image_path"] == "line_01.png"
    assert row["stock_output"] == "helo wrold"
    assert row["pipeline_output"] == "hello world"
    assert row["stock_cer"] > row["pipeline_cer"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/pytest test_performance_metrics.py -v`
Expected: FAIL with `ImportError: cannot import name 'evaluate_stock_vs_pipeline'`

- [ ] **Step 3: Write the minimal implementation**

Add to `performance_metrics.py`:

```python
import re

from app import transcribe, explain


def _extract_corrected(explain_markdown):
    match = re.search(r"Corrected:\s*(.+)", explain_markdown)
    return match.group(1).strip() if match else ""


def evaluate_stock_vs_pipeline(ground_truth):
    results = []
    for row in ground_truth:
        image_path = f"samples/{row['image_path']}"
        reference = row["text"]

        stock_output = transcribe(image_path)
        stock_cer, stock_wer = compute_cer_wer(stock_output, reference)

        explain_output = explain(stock_output)
        pipeline_output = _extract_corrected(explain_output) or stock_output
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

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/pytest test_performance_metrics.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add performance_metrics.py test_performance_metrics.py
git commit -m "feat: add stock-vs-pipeline comparison evaluation"
```

### Task 4: Wire live metrics into the Performance tab

**Files:**
- Modify: `app.py` — near the top (after `SAMPLES_DIR`/model loading, before `build_ui()`), and the `📊 Performance` tab body (currently hand-typed markdown, see the `gr.Tab("📊 Performance")` block).

- [ ] **Step 1: Compute metrics once at module load**

Add after the model/`SAMPLES_DIR` setup, before `def build_ui():`:

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

- [ ] **Step 2: Replace the hardcoded metrics table**

Inside the `with gr.Tab("📊 Performance"):` block, replace the hardcoded `gr.Markdown("| Metric | Value | Description |\n...")` block with:

```python
gr.Markdown(
    "| Metric | Value |\n"
    "| :--- | :--- |\n"
    f"| **Word Error Rate (WER)** | **{_AGGREGATE_WER * 100:.2f}%** |\n"
    f"| **Character Error Rate (CER)** | **{_AGGREGATE_CER * 100:.2f}%** |\n"
    f"| **Overall Word Accuracy** | **{(1 - _AGGREGATE_WER) * 100:.2f}%** |\n"
    f"| **Overall Character Accuracy** | **{(1 - _AGGREGATE_CER) * 100:.2f}%** |"
)
```

- [ ] **Step 3: Replace the hardcoded comparison table**

Replace the hardcoded "Sample-by-Sample Comparison" table (currently comparing TrOCR vs Florence-2) with one built from `_COMPARISON_ROWS`, comparing stock TrOCR vs the full pipeline:

```python
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

- [ ] **Step 4: Manually verify**

Run: `.venv/Scripts/python app.py`, open the app, go to the Performance tab. Confirm the WER/CER numbers are computed values (not "9.27%"/"2.44%") and the comparison table shows all 8 samples with real stock/pipeline outputs. Confirm no traceback in the terminal.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: replace static Performance tab metrics with live CER/WER and stock-vs-pipeline comparison"
```

---

## Phase 2: Supabase Data Logging

### Task 5: Create the Supabase project and schema — **you do this step**

1. Create a project at supabase.com (free tier).
2. In the SQL Editor, run:

```sql
create table profiles (
    id uuid primary key references auth.users(id),
    email text,
    created_at timestamptz default now()
);

create table transcriptions (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users(id),
    image_path text,
    source text check (source in ('sample', 'upload')),
    line_count integer,
    raw_text text,
    trocr_confidence_pct integer,
    created_at timestamptz default now()
);

create table corrections (
    id uuid primary key default gen_random_uuid(),
    transcription_id uuid references transcriptions(id),
    corrected_text text,
    groq_confidence_label text,
    added_content text,
    unmatched_tokens jsonb,
    user_verified_text text,
    created_at timestamptz default now()
);

create table rl_feedback (
    id uuid primary key default gen_random_uuid(),
    transcription_id uuid references transcriptions(id),
    reward_signal float,
    reward_source text check (reward_source in ('edit_distance', 'token_overlap', 'explicit_rating')),
    created_at timestamptz default now()
);
```

3. Send me the **Project URL** and the **`service_role` key** (Project Settings → API) — needed for Task 7's end-to-end verification. Task 6 below doesn't need them (it's tested against a fake client).

### Task 6: `supabase_client.py` logging module (testable without a live project)

**Files:**
- Create: `supabase_client.py`
- Test: `test_supabase_client.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add the dependency**

Append to `requirements.txt`:
```
supabase
```

Run: `.venv/Scripts/pip install supabase`

- [ ] **Step 2: Write the failing test**

```python
class FakeTable:
    def __init__(self, store):
        self.store = store
        self._pending = None

    def insert(self, row):
        self._pending = row
        return self

    def execute(self):
        self.store.append(self._pending)
        return self._pending


class FakeSupabaseClient:
    def __init__(self):
        self.inserted = {"transcriptions": [], "corrections": []}

    def table(self, name):
        return FakeTable(self.inserted[name])


def test_log_transcription_inserts_expected_row():
    from supabase_client import log_transcription

    fake_client = FakeSupabaseClient()
    log_transcription(
        fake_client,
        user_id="user-123",
        image_path="line_01.png",
        source="sample",
        line_count=1,
        raw_text="put down a resolution on the subject",
        trocr_confidence_pct=98,
    )

    assert len(fake_client.inserted["transcriptions"]) == 1
    row = fake_client.inserted["transcriptions"][0]
    assert row["user_id"] == "user-123"
    assert row["raw_text"] == "put down a resolution on the subject"
    assert row["trocr_confidence_pct"] == 98
```

- [ ] **Step 3: Run to verify it fails**

Run: `.venv/Scripts/pytest test_supabase_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'supabase_client'`

- [ ] **Step 4: Write the minimal implementation**

```python
"""Thin wrapper around the Supabase client for logging transcriptions and
corrections. Functions take the client as a parameter (not a module-level
singleton) so tests can inject a fake client without a live Supabase project."""
import os

from supabase import create_client


def get_client():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    return create_client(url, key)


def log_transcription(client, user_id, image_path, source, line_count, raw_text, trocr_confidence_pct):
    return client.table("transcriptions").insert({
        "user_id": user_id,
        "image_path": image_path,
        "source": source,
        "line_count": line_count,
        "raw_text": raw_text,
        "trocr_confidence_pct": trocr_confidence_pct,
    }).execute()


def log_correction(client, transcription_id, corrected_text, groq_confidence_label, added_content, unmatched_tokens, user_verified_text=None):
    return client.table("corrections").insert({
        "transcription_id": transcription_id,
        "corrected_text": corrected_text,
        "groq_confidence_label": groq_confidence_label,
        "added_content": added_content,
        "unmatched_tokens": unmatched_tokens,
        "user_verified_text": user_verified_text,
    }).execute()
```

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/Scripts/pytest test_supabase_client.py -v`
Expected: `1 passed`

- [ ] **Step 6: Commit**

```bash
git add supabase_client.py test_supabase_client.py requirements.txt
git commit -m "feat: add Supabase logging module for transcriptions and corrections"
```

### Task 7: Wire logging into the app — **blocked on Task 5's credentials**

**Files:**
- Modify: `app.py` — `explain()` function, and the two places that call `transcribe_with_confidence` (`transcribe_sample_and_reset`, `transcribe_upload_and_reset`)

- [ ] **Step 1: Set environment variables**

Once you've sent the Task 5 credentials, they get set as `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` (e.g. in a local `.env` loaded via `python-dotenv`, or exported in the shell before `python app.py`).

- [ ] **Step 2: Insert logging calls**

In `app.py`, import the new module near the other local imports:

```python
import supabase_client as _sb
```

In `explain()`, after successfully parsing `corrected_line`/`confidence`/`added_content`/`unmatched_tokens`, insert:

```python
try:
    client = _sb.get_client()
    t_row = _sb.log_transcription(
        client, user_id=None, image_path="", source="unknown",
        line_count=1, raw_text=ocr_text, trocr_confidence_pct=0,
    )
    transcription_id = t_row[0]["id"] if isinstance(t_row, list) else None
    _sb.log_correction(
        client, transcription_id=transcription_id, corrected_text=corrected_line,
        groq_confidence_label=confidence, added_content=added_content,
        unmatched_tokens=unmatched_tokens,
    )
except Exception as e:
    print(f"Supabase logging failed (non-fatal): {e}")
```

*(Note: `user_id`, `image_path`, `source`, `line_count`, `trocr_confidence_pct` are placeholder-shaped here because `explain()` currently only receives `ocr_text`/`confidence_md` — wiring the real values through requires passing them from the Transcribe step, which is a small follow-up once Task 9's auth work establishes how `user_id` is recovered from the session. Flagging this explicitly rather than pretending it's finished.)*

- [ ] **Step 3: Verify against your live project**

Run: `.venv/Scripts/python app.py`, transcribe and explain a sample, then check the Supabase Table Editor — confirm a row appears in `transcriptions` and `corrections`.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: log transcriptions and corrections to Supabase"
```

---

## Phase 3: FastAPI + Google OAuth Gate

### Task 8: Register the Google OAuth provider — **you do this step**

1. In Google Cloud Console, create an OAuth 2.0 Client ID (Web application), authorized redirect URI: `https://<your-supabase-project>.supabase.co/auth/v1/callback`.
2. In Supabase Dashboard → Authentication → Providers → Google, paste the Client ID and Secret, enable it.
3. What I need from you: confirmation this step is done, plus your Supabase **anon/public key** (safe to share, unlike the service key) for the frontend-facing auth calls.

### Task 9: FastAPI-mounted auth routes — **blocked on Task 8**

**Files:**
- Create: `server.py` (new entrypoint, replaces `python app.py` as the way to run the app)
- Modify: `requirements.txt`
- Modify: `app.py` — extract `build_ui()` as a standalone function that returns `demo` without calling `.launch()` (see note below)

- [ ] **Step 1: Add the dependency**

Append to `requirements.txt`:
```
fastapi
```

- [ ] **Step 2: Write the auth routes**

```python
"""FastAPI shell around the existing Gradio app, adding Google-OAuth-gated
access via Supabase Auth. Run with: python server.py"""
import os

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
import gradio as gr
from supabase import create_client

from app import build_ui

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_ANON_KEY = os.environ["SUPABASE_ANON_KEY"]
COOKIE_NAME = "sb_session"

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
fastapi_app = FastAPI()


@fastapi_app.get("/")
def root(request: Request):
    if request.cookies.get(COOKIE_NAME):
        return RedirectResponse("/app")
    return RedirectResponse("/auth/login")


@fastapi_app.get("/auth/login")
def login():
    result = supabase.auth.sign_in_with_oauth({
        "provider": "google",
        "options": {"redirect_to": f"{os.environ['APP_BASE_URL']}/auth/callback"},
    })
    return RedirectResponse(result.url)


@fastapi_app.get("/auth/callback")
def callback(code: str):
    session = supabase.auth.exchange_code_for_session({"auth_code": code})
    response = RedirectResponse("/app")
    response.set_cookie(COOKIE_NAME, session.session.access_token, httponly=True)
    return response


@fastapi_app.get("/auth/logout")
def logout():
    response = RedirectResponse("/auth/login")
    response.delete_cookie(COOKIE_NAME)
    return response


demo = build_ui()
gr.mount_gradio_app(fastapi_app, demo, path="/app")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
```

*(Note: `build_ui()` doesn't exist as a standalone callable yet — `app.py` currently builds and launches the UI inline under `if __name__ == "__main__":`. Extracting it into a `build_ui()` function that returns the `demo` object without calling `.launch()` is a small, mechanical prerequisite refactor of `app.py` needed before this task compiles — flagging it here rather than glossing over it. The existing `if __name__ == "__main__":` block in `app.py` keeps working unchanged, calling `build_ui().launch(...)`, so `python app.py` still works standalone alongside the new `python server.py` entrypoint.)*

- [ ] **Step 3: Verify against your live project**

Set `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `APP_BASE_URL` (e.g. `http://localhost:7860`) as environment variables. Run: `.venv/Scripts/python server.py`. Visit `http://localhost:7860/` — confirm it redirects to Google's login, and after signing in, lands on `/app` with the Gradio UI visible and a session cookie set.

- [ ] **Step 4: Commit**

```bash
git add server.py app.py requirements.txt
git commit -m "feat: add FastAPI-mounted Google OAuth gate in front of the Gradio app"
```

---

## Self-review notes

- **Spec coverage:** Performance tab (Tasks 1–4, complete, no blockers). Supabase logging (Tasks 5–7, code complete, live verification blocked on your credentials). Google OAuth (Tasks 8–9, code complete, live verification blocked on your OAuth setup). Phase B fine-tuning comparison is explicitly out of scope per the spec's "not built in this phase."
- **Known follow-up, called out inline rather than hidden:** Task 7's `explain()` logging call currently hardcodes `user_id=None`/`source="unknown"` because `explain()` doesn't yet receive those values — real values need threading through once Task 9 establishes how `user_id` is recovered from the session cookie inside Gradio handlers.
