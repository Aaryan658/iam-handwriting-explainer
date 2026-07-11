# ✍️ IAM Handwriting Explainer

A Gradio app that transcribes handwritten English text and explains/corrects the
result with an LLM — built on `microsoft/trocr-large-handwritten`, benchmarked
against IAM/Bentham handwriting samples, and deployed as a
[Hugging Face Space](https://huggingface.co/spaces/AaryanDharrmik/iam-handwriting-explainer).

## What it does

- **Transcribe** — run a handwriting line (or a full paragraph, auto-segmented
  into lines) through TrOCR and get a transcription with a per-word confidence
  badge.
- **Explain + Correct** — pass the OCR output through an LLM that fixes
  transcription errors, flags low-confidence words with alternatives, and adds
  historical/linguistic context when it's confident. A deterministic
  token-overlap check independently verifies the LLM's own confidence claim
  and downgrades it if the "corrected" text doesn't actually match the OCR
  input.
- **Personality Read / Pen Pal Reply** — two lighter features that riff on the
  transcribed text for fun (handwriting "personality" read from measured
  slant/density stats, and an in-character reply to the letter).
- **Learn from corrections** — user-submitted corrections are logged
  (`corrections_log.csv`) and folded back into future LLM prompts as few-shot
  examples, plus surfaced as a live per-engine accuracy dashboard.
- **Performance tab** — live CER/WER comparison of stock TrOCR vs. the
  TrOCR+LLM pipeline vs. Tesseract vs. EasyOCR, computed at startup against 8
  bundled ground-truth samples from `Teklia/IAM-line`.

## LLM correction backend

`explain()` tries a local **Ollama** model first (`llama3.1:8b` by default,
free and unlimited on your own machine), then falls back to a small chain of
free models on **OpenRouter** if Ollama isn't reachable — which is always the
case on the deployed Space, so it transparently uses OpenRouter there. Both
backends go through the same OpenAI-compatible client code path. Configure via:

```
OPENROUTER_API_KEY=...      # required for the OpenRouter fallback
OLLAMA_BASE_URL=...         # optional, defaults to http://localhost:11434/v1
OLLAMA_MODEL=...            # optional, defaults to llama3.1:8b
```

## Running locally

```
pip install -r requirements.txt
python app.py
```

Tesseract must also be installed and on `PATH` (see `packages.txt` for the
Debian/HF-Space package list).

## Fine-tuning

`finetune_trocr.py` (plus the Colab/Kaggle notebook variants) fine-tunes
`trocr-large-handwritten` on IAM/Bentham line data prepared by
`prepare_iam_data.py` / `prepare_bentham_data.py`. This is a separate,
experimental track from the app above — the deployed app currently uses the
stock pretrained checkpoint, not a fine-tuned one.

## Project layout

| File | Purpose |
|---|---|
| `app.py` | Main Gradio app — transcription, LLM correction, UI |
| `ocr_engines.py` | Tesseract / EasyOCR wrappers used for benchmarking |
| `paragraph_pipeline.py` | Line segmentation + reassembly for multi-line uploads |
| `performance_metrics.py` | CER/WER evaluation used by the Performance tab |
| `finetune_trocr.py` | TrOCR fine-tuning script |
| `prepare_iam_data.py` / `prepare_bentham_data.py` | Dataset prep for fine-tuning |
| `samples/` | Bundled sample line images + ground truth for benchmarking |
