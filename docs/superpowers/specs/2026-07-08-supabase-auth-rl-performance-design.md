# Supabase Auth, RL Data Logging & Real Performance Analysis — Design

*Date: 2026-07-08*
*Status: Approved, pending implementation plan*

## Context

The app is a Gradio-based handwriting transcription tool (`app.py`, `paragraph_pipeline.py`, `segmentation.py`) running TrOCR + Groq (Llama 3.3) correction, with a per-line/paragraph confidence score derived from TrOCR's own token probabilities. It currently has no auth, no durable structured storage (only a flat `corrections_log.csv`), and a "Performance" tab that is 100% hardcoded static markdown.

Three capabilities were requested: (1) authentication + a Supabase backend, (2) storing transcriptions for future RL-based learning, (3) an initial ask to migrate off Gradio to Next.js/Render — **explicitly reversed**: the app stays on Gradio. A fourth capability was added during design: real (not static) performance metrics, including a comparison between stock TrOCR and "our modified TrOCR."

**Decision: stay on Gradio.** No Next.js, no Render, no frontend rewrite. All new capability is added to the existing Python/Gradio stack.

## Non-goals

- No frontend framework migration.
- No live/online RL training loop. "RL-based learning" for this phase means durable, queryable data collection that a future offline fine-tuning run can train on — not a reward-model/policy-update pipeline running today.
- No fine-tuned checkpoint is produced in this phase (see Phase B below) — `finetune_trocr.py` exists but has never been run; there is no training-data CSV and no checkpoint anywhere in the repo today.

## Architecture

Single Python process. A thin FastAPI shell wraps the existing Gradio `Blocks` app via `gr.mount_gradio_app(fastapi_app, demo, path="/app")`. FastAPI owns three new routes for the OAuth flow; every existing Gradio function (`transcribe`, `transcribe_with_confidence`, `explain`, `segment_lines`, `transcribe_paragraph_with_confidence`) is unchanged.

```
Browser → FastAPI (/auth/login, /auth/callback, /auth/logout, mounts /app)
                              │
                    gr.mount_gradio_app
                              │
                         Gradio Blocks (/app)  ──► existing app.py pipeline (TrOCR, Groq, segmentation)
                              │
                          supabase-py  ──► Supabase (Postgres + Auth), Google OAuth provider
```

### Auth flow

1. Unauthenticated request to `/` → FastAPI redirects to `/auth/login`.
2. `/auth/login` redirects to the Google OAuth URL obtained via `supabase.auth.sign_in_with_oauth(provider="google", redirect_to=".../auth/callback")`.
3. Google → Supabase → `/auth/callback` receives an auth code, exchanges it via `supabase.auth.exchange_code_for_session(...)`, sets an HttpOnly session cookie, redirects into `/app`.
4. Gradio event handlers read the cookie via `gr.Request` (available as an optional handler argument in Gradio callbacks) to recover the Supabase user id, attached to every Supabase write below.
5. `/auth/logout` clears the cookie and redirects to `/auth/login`.
6. `/app` (the Gradio mount) is gated behind a dependency/middleware check on the session cookie — no valid session, no access to the transcription UI.

## Supabase schema

- `profiles` — `id` (= `auth.users.id`), `email`, `created_at` *(Supabase convention)*
- `transcriptions` — `id`, `user_id`, `image_path` (Supabase Storage), `source` (`sample`/`upload`), `line_count`, `raw_text`, `trocr_confidence_pct`, `created_at`
- `corrections` — `id`, `transcription_id`, `corrected_text`, `groq_confidence_label`, `added_content`, `unmatched_tokens` (jsonb), `user_verified_text` (nullable — the existing "Was this correct?" field), `created_at`
- `rl_feedback` — `id`, `transcription_id`, `reward_signal` (float), `reward_source` (`edit_distance` / `token_overlap` / `explicit_rating`), `created_at`

`supabase-py` is added to `requirements.txt`. Every `explain()` / `transcribe_with_confidence()` call also writes to `transcriptions`/`corrections` — this replaces `corrections_log.csv` as the durable record, and is exactly the data Phase B fine-tuning will train on.

## Performance tab — from static to real

The current tab (`app.py`, the `📊 Performance` tab) is hand-typed markdown with fixed numbers (WER 9.27%, CER 2.44%, a static TrOCR-vs-Florence-2 table). Replacing with:

- `samples/ground_truth.csv` (`image_path,text`) formalizing correct transcriptions for the 8 bundled IAM line samples (currently only implicit in the static markdown).
- CER/WER computed live via `jiwer` at app startup, cached in a module-level variable (not recomputed per tab view).
- Groq confidence-label distribution and hallucination-override rate computed from real logged Supabase data (`corrections` table) instead of hardcoded "27 HIGH claims, 3 overrides."
- Latency: average ms per TrOCR call and per Groq call, measured and displayed.

## Comparison: stock TrOCR vs "our modified TrOCR" — phased

- **Phase A (this project, no training required):** for each ground-truth sample, run (1) stock `transcribe()` alone and (2) the full pipeline (TrOCR → confidence gate → Groq `explain()` correction). Compute CER/WER for both against ground truth. Table: Sample | Ground Truth | Stock Output | Stock CER/WER | Pipeline Output | Pipeline CER/WER. This quantifies what the Groq correction layer is actually buying, using pieces that already exist.
- **Phase B (later, explicitly deferred, data-gated):** once `corrections` has accumulated enough verified corrections (proposed threshold: 200+), export to a training CSV and run the existing (currently unused) `finetune_trocr.py` to produce a real checkpoint. Extend the comparison table to three columns: stock / fine-tuned / pipeline-with-Groq. Not built in this phase — only the trigger condition and export path are specified now.

## Build order

1. Supabase project: schema (`profiles`, `transcriptions`, `corrections`, `rl_feedback`), Google OAuth provider enabled.
2. `supabase-py` integration: wire logging into the existing `explain()` / `transcribe_with_confidence()` flow.
3. FastAPI-mounted Gradio + `/auth/login`, `/auth/callback`, `/auth/logout`; gate `/app` behind a valid session.
4. `samples/ground_truth.csv` + live CER/WER computation + rebuilt Performance tab, including the Phase A stock-vs-pipeline comparison table.
5. *(Later, gated on data volume — not part of this build)* Phase B: run `finetune_trocr.py` on exported corrections, add the three-way comparison.

## Open risks / honesty notes

- Gradio's `gr.Request` cookie access inside event handlers needs verification against the installed Gradio version's exact API before relying on it for auth propagation — flagged for the implementation plan to confirm early, not assumed.
- Phase B has no committed timeline — it is explicitly gated on real usage data existing, which does not exist yet.
