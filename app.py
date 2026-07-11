import os
import glob
import difflib
import string
import sys
import subprocess
import warnings

# When run directly (`python app.py`), this module is registered in
# sys.modules as "__main__", not "app". Sibling modules that do
# `from app import transcribe` (paragraph_pipeline.py, performance_metrics.py)
# then find no "app" entry and re-execute this entire file from scratch under
# the name "app" -- reloading the TrOCR model a second time. Aliasing "app" to
# the already-running "__main__" module avoids that duplicate execution.
if __name__ == "__main__":
    sys.modules.setdefault("app", sys.modules["__main__"])

import gradio as gr
import torch
import numpy as np
from PIL import Image
from openai import OpenAI
import re
from segmentation import segment_lines
import ocr_engines
# --- Hack to force install missing dependencies if HF Spaces caching fails ---
try:
    import sentencepiece
except ImportError:
    print("Force installing sentencepiece...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece", "tiktoken", "protobuf"])

# --- Hack to disable SSL verification on Windows to prevent HF Hub errors ---
import ssl
import httpx
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    
    old_init = httpx.Client.__init__
    def new_init(self, *args, **kwargs):
        kwargs["verify"] = False
        old_init(self, *args, **kwargs)
    httpx.Client.__init__ = new_init
except Exception:
    pass

# --- Hack to bypass Gradio SSRF 403 error on private/gated Spaces ---
try:
    import shutil
    from urllib.parse import urlparse, unquote
    import gradio.processing_utils

    old_download = gradio.processing_utils.async_ssrf_protected_download
    old_sync_download = gradio.processing_utils.ssrf_protected_download

    async def new_download(url: str, cache_dir: str) -> str:
        print(f"[DEBUG SSRF async_download] url={url} cache_dir={cache_dir}")
        try:
            parsed_url = urlparse(url)
            path_part = unquote(parsed_url.path)
            print(f"[DEBUG SSRF async_download] path_part={path_part}")
            
            local_path = None
            if "/file=" in path_part:
                local_path = path_part.split("/file=", 1)[1]
            elif path_part.startswith("/file/"):
                local_path = path_part[6:]
                
            print(f"[DEBUG SSRF async_download] local_path={local_path} exists={os.path.exists(local_path) if local_path else False}")
            if local_path and os.path.exists(local_path):
                temp_dir = os.path.join(cache_dir, gradio.processing_utils.hash_url(url))
                os.makedirs(temp_dir, exist_ok=True)
                filename = os.path.basename(local_path)
                full_temp_file_path = os.path.abspath(os.path.join(temp_dir, filename))
                
                shutil.copy(local_path, full_temp_file_path)
                print(f"Bypassed SSRF download for local file: {local_path} -> {full_temp_file_path}")
                return full_temp_file_path
        except Exception as e:
            print(f"Error in SSRF bypass patch: {e}")
            
        try:
            return await old_download(url, cache_dir)
        except Exception as e:
            print(f"Failed to download external URL {url}: {e}. Returning dummy path as fallback.")
            dummy_file = os.path.join(cache_dir, "dummy_fallback.png")
            if not os.path.exists(dummy_file):
                with open(dummy_file, "wb") as f:
                    f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82')
            return dummy_file

    def new_sync_download(url: str, cache_dir: str) -> str:
        print(f"[DEBUG SSRF sync_download] url={url} cache_dir={cache_dir}")
        try:
            parsed_url = urlparse(url)
            path_part = unquote(parsed_url.path)
            print(f"[DEBUG SSRF sync_download] path_part={path_part}")
            
            local_path = None
            if "/file=" in path_part:
                local_path = path_part.split("/file=", 1)[1]
            elif path_part.startswith("/file/"):
                local_path = path_part[6:]
                
            print(f"[DEBUG SSRF sync_download] local_path={local_path} exists={os.path.exists(local_path) if local_path else False}")
            if local_path and os.path.exists(local_path):
                temp_dir = os.path.join(cache_dir, gradio.processing_utils.hash_url(url))
                os.makedirs(temp_dir, exist_ok=True)
                filename = os.path.basename(local_path)
                full_temp_file_path = os.path.abspath(os.path.join(temp_dir, filename))
                
                shutil.copy(local_path, full_temp_file_path)
                print(f"Bypassed SSRF sync download for local file: {local_path} -> {full_temp_file_path}")
                return full_temp_file_path
        except Exception as e:
            print(f"Error in SSRF sync bypass patch: {e}")
            
        try:
            return old_sync_download(url, cache_dir)
        except Exception as e:
            print(f"Failed to download external URL {url} (sync): {e}. Returning dummy path as fallback.")
            dummy_file = os.path.join(cache_dir, "dummy_fallback.png")
            if not os.path.exists(dummy_file):
                with open(dummy_file, "wb") as f:
                    f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82')
            return dummy_file

    gradio.processing_utils.async_ssrf_protected_download = new_download
    gradio.processing_utils.ssrf_protected_download = new_sync_download
except Exception as e:
    print(f"Failed to apply SSRF bypass patch: {e}")

# ---------------------------------------------------------------------------
# Suppress noisy warnings
# ---------------------------------------------------------------------------
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Secrets — read from HF Spaces or local .env file
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY and os.path.exists(".env"):
    try:
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("OPENROUTER_API_KEY="):
                    OPENROUTER_API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
                    break
    except Exception:
        pass

# Free-tier model on OpenRouter -- separate quota from Groq's, used as the
# LLM correction/explanation backend instead of Groq's API.
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

# Free-tier upstream providers get congested independently of each other, so
# list alternates here -- OpenRouter tries each in order and only moves to
# the next if the previous one is unavailable/rate-limited.
OPENROUTER_FALLBACK_MODELS = [
    OPENROUTER_MODEL,
    "deepseek/deepseek-chat-v3-0324:free",
    "qwen/qwen-2.5-72b-instruct:free",
]


def _openrouter_client():
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# ---------------------------------------------------------------------------
# TrOCR model loading
# ---------------------------------------------------------------------------
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, RobertaTokenizer, ViTImageProcessor

print("Loading TrOCR model and processor...")
# trocr-large-handwritten: 2.89% published CER vs trocr-base's 3.42%, with
# near-identical inference speed per Microsoft's own benchmarks -- a same-day
# accuracy win with no fine-tuning required.
TROCR_MODEL_NAME = "microsoft/trocr-large-handwritten"
# Instantiate RobertaTokenizer and ViTImageProcessor manually to bypass TrOCR Processor bugs
image_processor = ViTImageProcessor.from_pretrained(TROCR_MODEL_NAME)
tokenizer = RobertaTokenizer.from_pretrained(TROCR_MODEL_NAME)
processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME)
print("TrOCR Model loaded successfully.")

# ---------------------------------------------------------------------------
# Discover bundled sample line images  (samples/line_*.png)
# ---------------------------------------------------------------------------
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
SAMPLE_IMAGES = sorted(glob.glob(os.path.join(SAMPLES_DIR, "line_*.png")))


# ---------------------------------------------------------------------------
# Groq system prompt  (matches validated Colab notebook)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an AI assistant correcting OCR output from handwriting recognition.\n"
    "For each sentence you receive, respond with EXACTLY this format:\n\n"
    "Added content: <YES | NO>\n"
    "Corrected: <your best corrected transcription>\n"
    "Uncertain words: <comma-separated list of specific words in the Corrected text that are low-confidence guesses, or \"none\">\n"
    "Alternatives: <word>: <alt1>, <alt2> (ONLY if Uncertain words is not \"none\", one line per uncertain word)\n"
    "Confidence: <HIGH | MEDIUM | LOW>\n"
    "Note: <any notes, or omit this line if confidence is HIGH>\n"
    "Context: <1-2 sentence explanation, ONLY if confidence is HIGH — must reference specific period/historical/linguistic detail if present (e.g. archaic terms, 1950s British political titles), not a generic paraphrase>\n\n"
    "Rules:\n"
    "1. Fix obvious OCR/transcription errors (misspellings, garbled tokens) while preserving the original meaning.\n"
    "2. Before assigning confidence, first answer: \"Did I add, infer, or invent any word, phrase, or meaning not directly present in the OCR input — including completing a sentence fragment, or substituting a different word/phrasing than what was written even if grammatically similar?\" Output this as \"Added content: YES\" or \"Added content: NO\" as the FIRST line of your response, before anything else.\n"
    "3. If Added content = YES, Confidence MUST be MEDIUM or LOW. It cannot be HIGH under any circumstance.\n"
    "4. If Added content = NO, Confidence MAY be HIGH.\n"
    "5. Confidence label definitions:\n"
    "   - HIGH: the corrected sentence clearly reflects the intended meaning, and no words/phrases were added, inferred, or substituted beyond fixing character-level noise (spacing, capitalization, punctuation).\n"
    "   - MEDIUM: some words are uncertain, or minor inference was needed, but the gist is likely correct.\n"
    "   - LOW: more than 2 words were substantially altered, or the corrected sentence's meaning is a guess rather than a clear fix.\n"
    "6. If confidence is MEDIUM or LOW, you MUST include the note: \"This reconstruction may not reflect the original meaning.\" Do NOT provide a contextual explanation paragraph in that case.\n"
    "7. Do NOT invent specific details (times, named actions, first-person narrative, verbs, or events) that are not clearly present in the input tokens.\n"
    "8. Do not rephrase or substitute words that are already correct and clear — only fix genuine OCR noise. Preserve original word choice (\"put down\" must stay \"put down,\" not become \"put forward\").\n"
    "9. Do not output any additional chat or conversational text."
)


# ---------------------------------------------------------------------------
# Helper: tokenize for comparison
# ---------------------------------------------------------------------------
def tokenize(text):
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def transcribe(image):
    """Run TrOCR on one PIL image and return the raw transcription."""
    if image is None:
        return "⚠️ Please select or upload an image first."

    if isinstance(image, str):
        pil_image = Image.open(image).convert("RGB")
    else:
        pil_image = Image.fromarray(image) if not isinstance(image, Image.Image) else image
        pil_image = pil_image.convert("RGB")

    pixel_values = processor(pil_image, return_tensors="pt").pixel_values
    
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=128, num_beams=4)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text.strip()


# Below this, a transcription's TrOCR-native confidence is considered too low
# to spend a Groq call correcting -- the LLM would be "correcting" noise it
# can't actually ground in a legible source.
LOW_CONFIDENCE_GATE = 0.50


def _confidence_label(avg_prob):
    pct = round(avg_prob * 100)
    if avg_prob >= 0.90:
        label = "HIGH"
    elif avg_prob >= 0.70:
        label = "MEDIUM"
    else:
        label = "LOW"
    return pct, label


def format_confidence_badge(avg_prob, prefix="TrOCR model confidence"):
    pct, label = _confidence_label(avg_prob)
    return (
        f"{prefix}: **{pct}%** "
        f"<span class=\"confidence-badge {label.lower()}\">{label}</span>"
    )


def transcribe_with_confidence_score(image):
    """Run TrOCR and return (text, avg_token_prob) -- the raw 0-1 model-grounded
    confidence signal (mean max-softmax probability across generated tokens).

    This is independent of the Groq LLM's self-reported confidence, which only
    judges plausibility of the corrected text and can be overconfident on
    inputs the OCR model itself was unsure about. Exposed as its own function
    (rather than folded into transcribe_with_confidence) so multi-line callers
    can aggregate several lines' raw scores before formatting a single badge.
    """
    if isinstance(image, str):
        pil_image = Image.open(image).convert("RGB")
    else:
        pil_image = Image.fromarray(image) if not isinstance(image, Image.Image) else image
        pil_image = pil_image.convert("RGB")

    pixel_values = processor(pil_image, return_tensors="pt").pixel_values

    with torch.no_grad():
        output = model.generate(
            pixel_values,
            max_new_tokens=128,
            num_beams=4,
            output_scores=True,
            return_dict_in_generate=True,
        )

    generated_text = processor.batch_decode(output.sequences, skip_special_tokens=True)[0].strip()
    token_probs = [torch.softmax(scores, dim=-1).max().item() for scores in output.scores]
    avg_prob = sum(token_probs) / len(token_probs) if token_probs else 0.0
    return generated_text, avg_prob


def transcribe_with_confidence(image):
    """Run TrOCR and also return a model-grounded confidence badge (markdown)."""
    if image is None:
        return "⚠️ Please select or upload an image first.", ""

    generated_text, avg_prob = transcribe_with_confidence_score(image)
    return generated_text, format_confidence_badge(avg_prob)


def _md_table_cell(text):
    """Flatten text for safe interpolation into a single markdown table row.

    A literal newline splits the row into multiple lines and an unescaped
    '|' reads as an extra column delimiter -- either one desyncs Gradio's
    Markdown table parser for every row that follows. Tesseract in
    particular hallucinates multi-line garbage (with stray '|' characters)
    on noisy handwriting crops, so this must run on every OCR/user-text
    value before it goes into a table cell.
    """
    if text is None:
        return ""
    return str(text).replace("\r\n", " ").replace("\n", " ").replace("|", "\\|").strip()


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
        image_id = _md_table_cell(r['image_id'])[:40]
        trocr_output = _md_table_cell(r['trocr_output'])[:40]
        user_correction = _md_table_cell(r['user_correction'])[:40]
        lines.append(f"| {image_id} | {trocr_output} | {user_correction} |")

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


def compute_handwriting_features(image):
    """Compute a few real, simple visual stats from the ink strokes -- slant
    angle, ink density, line-height variance -- to ground the playful
    'graphology' read in actual measurements rather than pure LLM fabrication."""
    import cv2
    if isinstance(image, str):
        pil_image = Image.open(image).convert("L")
    elif isinstance(image, Image.Image):
        pil_image = image.convert("L")
    else:
        pil_image = Image.fromarray(image).convert("L")

    gray = np.array(pil_image)
    ink_mask = (gray < 128).astype(np.uint8) * 255
    ink_density = float(ink_mask.mean() / 255.0)

    coords = np.column_stack(np.where(ink_mask > 0))
    if len(coords) >= 10:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
    else:
        angle = 0.0

    row_sums = ink_mask.sum(axis=1)
    ink_rows = np.where(row_sums > 0)[0]
    height_variance = float(np.std(ink_rows)) if len(ink_rows) > 1 else 0.0

    return {
        "slant_deg": round(float(angle), 1),
        "ink_density": round(ink_density, 3),
        "height_variance": round(height_variance, 1),
    }


def graphology_read(image):
    """Playful, clearly-labeled-as-entertainment 'personality read' grounded
    in real computed slant/density/spacing stats -- not a scientific claim."""
    if image is None:
        return "⚠️ Please select or upload an image first."
    if not OPENROUTER_API_KEY:
        return "⚠️ OPENROUTER_API_KEY not found.\nAdd it under Settings → Repository secrets in your HF Space."

    features = compute_handwriting_features(image)
    client = _openrouter_client()
    prompt = (
        f"Measured handwriting stats: slant={features['slant_deg']} degrees, "
        f"ink density={features['ink_density']}, line-height variance={features['height_variance']}.\n"
        "Write a short, playful, upbeat 'handwriting personality read' (3-4 sentences) "
        "based on these stats, in the style of a fun graphology app. "
        "Be creative and complimentary, not clinical."
    )
    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            extra_body={"models": OPENROUTER_FALLBACK_MODELS},
            messages=[
                {"role": "system", "content": "You write short, fun, lighthearted handwriting personality reads. Never claim scientific validity."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
            max_tokens=200,
        )
        blurb = response.choices[0].message.content
    except Exception as e:
        return f"⚠️ OpenRouter API error: {e}"

    return (
        "### 🖋️ Handwriting Personality Read\n"
        "_For fun only — not a scientific assessment._\n\n"
        f"{blurb}\n\n"
        f"<sub>Measured: slant {features['slant_deg']}°, ink density {features['ink_density']}, "
        f"line variance {features['height_variance']}</sub>"
    )


def pen_pal_reply(ocr_text):
    """Have Groq write an in-character, period-appropriate reply to the
    transcribed letter, as if a contemporary pen pal were responding."""
    if not ocr_text or ocr_text.startswith("⚠️"):
        return "⚠️ Nothing to reply to — run Transcribe first."
    if not OPENROUTER_API_KEY:
        return "⚠️ OPENROUTER_API_KEY not found.\nAdd it under Settings → Repository secrets in your HF Space."

    client = _openrouter_client()
    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            extra_body={"models": OPENROUTER_FALLBACK_MODELS},
            messages=[
                {"role": "system", "content": (
                    "You are a pen pal replying to a handwritten letter, matching its "
                    "apparent time period, register, and tone. Write a short, warm, "
                    "in-character reply (4-6 sentences). Do not break character or "
                    "mention that you are an AI."
                )},
                {"role": "user", "content": f"Letter received:\n{ocr_text}"},
            ],
            temperature=0.8,
            max_tokens=300,
        )
        reply = response.choices[0].message.content
    except Exception as e:
        return f"⚠️ OpenRouter API error: {e}"

    return f"### ✉️ Reply from a Pen Pal\n\n{reply}"


DEFAULT_EXPLANATION = "### LLM Correction + Confidence\n_Run Transcribe and Explain to see results here._"

def reset_explanation():
    return DEFAULT_EXPLANATION


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


def explain(ocr_text, confidence_md=""):
    """Send transcription to Groq, run token-overlap check, return formatted output."""
    if not ocr_text or ocr_text.startswith("⚠️"):
        return "⚠️ Nothing to explain — run Transcribe first."
    if not OPENROUTER_API_KEY:
        return (
            "⚠️ OPENROUTER_API_KEY not found.\n"
            "Add it under Settings → Repository secrets in your HF Space."
        )

    # --- Low-confidence gate: skip the Groq call if TrOCR itself was unsure,
    # rather than spending an API call "correcting" text that was never
    # reliably read in the first place. ---
    conf_match = re.search(r"\*\*(\d+)%\*\*", confidence_md or "")
    if conf_match and int(conf_match.group(1)) / 100 < LOW_CONFIDENCE_GATE:
        return (
            "### LLM Correction + Confidence\n"
            f"⚠️ TrOCR model confidence was only **{conf_match.group(1)}%** — too low to reliably "
            "correct. Skipped the Groq call rather than risk correcting noise. "
            "Try a clearer image or a different sample."
        )

    # --- Sanity check: do not call Groq on degenerate OCR output ---
    words = ocr_text.split()
    
    # Strip punctuation and filter empty tokens for word analysis
    cleaned_words = [w.translate(str.maketrans("", "", string.punctuation)) for w in words]
    cleaned_words = [w for w in cleaned_words if w]

    if len(cleaned_words) < 1:
        return "⚠️ Transcription unclear — this image may not be a supported single-line format. Try a different sample or a clearer single-line upload."
        
    # Count single-char words (excluding legitimate 'a' and 'i') and standalone digits
    single_char_or_digit_words = sum(
        1 for w in cleaned_words 
        if (len(w) == 1 and w.lower() not in ["a", "i"]) or w.isdigit()
    )
    if len(cleaned_words) > 0 and single_char_or_digit_words / len(cleaned_words) >= 0.5:
        return "⚠️ Transcription unclear — this image may not be a supported single-line format. Try a different sample or a clearer single-line upload."
        
    letters = sum(c.isalpha() for c in ocr_text)
    non_whitespace = sum(not c.isspace() for c in ocr_text)
    if non_whitespace > 0 and letters / non_whitespace < 0.4:
        return "⚠️ Transcription unclear — this image may not be a supported single-line format. Try a different sample or a clearer single-line upload."

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

    client = _openrouter_client()

    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            extra_body={"models": OPENROUTER_FALLBACK_MODELS},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"OCR output:\n{corrected_ocr_text}"},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        llm_output = response.choices[0].message.content
    except Exception as e:
        print(f"OpenRouter API error in explain(): {e}")
        return f"⚠️ OpenRouter API error: {e}"

    # --- Parse structured fields from LLM response ---
    lines = llm_output.strip().split("\n")
    added_content = "UNKNOWN"
    corrected_line = ""
    confidence = ""
    uncertain_words_str = "none"
    alternatives = []
    
    notes = []
    context = []

    for line in lines:
        if line.startswith("Added content:"):
            added_content = line.split(":", 1)[1].strip()
        elif line.startswith("Corrected:"):
            corrected_line = line.split(":", 1)[1].strip()
        elif line.startswith("Uncertain words:"):
            uncertain_words_str = line.split(":", 1)[1].strip()
        elif line.startswith("Alternatives:"):
            alternatives.append(line.split(":", 1)[1].strip())
        elif line.startswith("Confidence:"):
            confidence = line.split(":", 1)[1].strip()
        elif line.startswith("Note:"):
            notes.append(line)
        elif line.startswith("Context:"):
            context.append(line)

    # --- Deterministic token-overlap check ---
    ocr_tokens = tokenize(corrected_ocr_text)
    corrected_tokens = tokenize(corrected_line)
    joined_ocr = "".join(ocr_tokens)

    unmatched_tokens = []
    for ct in corrected_tokens:
        cutoff = 0.6 if len(ct) <= 3 else 0.7
        matches = difflib.get_close_matches(ct, ocr_tokens, n=1, cutoff=cutoff)
        if not matches and ct not in joined_ocr:
            unmatched_tokens.append(ct)

    overridden = False
    if len(unmatched_tokens) > 0 and "HIGH" in confidence:
        # Override: downgrade to MEDIUM, strip Context, force disclaimer
        confidence = "MEDIUM"
        context = []
        notes = ["Note: This reconstruction may not reflect the original meaning."]
        overridden = True

    # --- Bold uncertain words in the corrected line ---
    display_corrected = corrected_line
    if uncertain_words_str.lower() != "none":
        words = [w.strip() for w in uncertain_words_str.split(",") if w.strip()]
        for w in words:
            # Use regex to bold the exact word, case-insensitive
            pattern = re.compile(r'\b' + re.escape(w) + r'\b', re.IGNORECASE)
            display_corrected = pattern.sub(f"**{w}**", display_corrected)

    # --- Build display output (Markdown) ---
    display_parts = []
    display_parts.append("### LLM Correction + Confidence")
    
    display_parts.append(f"**Corrected:** {display_corrected}")
    display_parts.append(f"**Confidence:** <span class=\"confidence-badge {confidence.lower().strip()}\">{confidence}</span>")
    
    if overridden:
        display_parts.append("\n⚠️ *Confidence OVERRIDDEN from HIGH → MEDIUM by token check.*")

    for note in notes:
        display_parts.append(f"\n{note}")
    
    for ctx in context:
        display_parts.append(f"\n{ctx}")
        
    if alternatives:
        display_parts.append("")
        display_parts.append("<details><summary>View Alternatives for Uncertain Words</summary>")
        display_parts.append("<ul>")
        for alt in alternatives:
            display_parts.append(f"<li>{alt}</li>")
        display_parts.append("</ul>")
        display_parts.append("</details>")
        display_parts.append("")

    display_parts.append("<hr>")
    display_parts.append("")
    display_parts.append("#### Token-Overlap Check")
    display_parts.append(f"- **Added content:** {added_content}")
    display_parts.append(f"- **Unmatched tokens (code):** `{unmatched_tokens if unmatched_tokens else '[] (none)'}`")

    return "\n".join(display_parts)


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


# ---------------------------------------------------------------------------
# Paragraph pipeline UI wrapper
# ---------------------------------------------------------------------------

def transcribe_upload_and_reset(image):
    """Route an uploaded image to the single-line or paragraph pipeline based
    on how many lines segmentation detects, and return
    (text, confidence_md, reset_verify, reset_explanation) for the UI.

    Single line -> existing transcribe_with_confidence() path, unchanged.
    2+ lines -> paragraph_pipeline, displayed as the reassembled continuous
    text only (no per-line numbered breakdown) -- segmentation and per-line
    transcription still happen internally exactly as before.
    """
    # Deferred import: paragraph_pipeline does `from app import transcribe`
    # at module level; importing it here (after app.py is fully loaded)
    # breaks the circular dependency without changing paragraph_pipeline.py.
    import paragraph_pipeline as _pp

    if image is None:
        return "⚠️ Please upload an image first.", "", "", DEFAULT_EXPLANATION

    pil_image = Image.open(image).convert("RGB") if isinstance(image, str) else image
    lines = segment_lines(pil_image)

    if len(lines) <= 1:
        text, conf_md = transcribe_with_confidence(image)
        return text, conf_md, "", DEFAULT_EXPLANATION

    paragraph_text, conf_md, per_line = _pp.transcribe_paragraph_with_confidence(lines=lines)
    if not per_line:
        return "⚠️ No lines detected. Try an image with clearer line spacing.", "", "", DEFAULT_EXPLANATION

    return paragraph_text, conf_md, "", DEFAULT_EXPLANATION


# ---------------------------------------------------------------------------
# Gradio UI  —  Blocks layout
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
:root {
    --background-fill-primary: #f8fafc;
    --background-fill-secondary: #ffffff;
    --block-background-fill: #ffffff;
    --block-border-color: #e2e8f0;
    --block-border-width: 1px;
    --block-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05);
    --block-title-text-color: #0f172a;
    --body-background-fill: #f8fafc;
    --body-text-color: #334155;
    --border-color-accent: #6366f1;
    --border-color-primary: #e2e8f0;
    --button-primary-background-fill: #4f46e5;
    --button-primary-background-fill-hover: #4338ca;
    --button-primary-text-color: #ffffff;
    --button-secondary-background-fill: #ffffff;
    --button-secondary-background-fill-hover: #f1f5f9;
    --button-secondary-text-color: #0f172a;
    --input-background-fill: #ffffff;
    --input-border-color: #e2e8f0;
    --radius-lg: 12px;
    --radius-md: 8px;
    --radius-sm: 6px;
}

.dark {
    --background-fill-primary: #09090b;
    --background-fill-secondary: #09090b;
    --block-background-fill: #09090b;
    --block-border-color: #27272a;
    --block-border-width: 1px;
    --block-shadow: 0 0 0 1px rgba(255, 255, 255, 0.02);
    --block-title-text-color: #fafafa;
    --body-background-fill: #09090b;
    --body-text-color: #a1a1aa;
    --border-color-accent: #6366f1;
    --border-color-primary: #27272a;
    --button-primary-background-fill: #4f46e5;
    --button-primary-background-fill-hover: #4338ca;
    --button-primary-text-color: #ffffff;
    --button-secondary-background-fill: #18181b;
    --button-secondary-background-fill-hover: #27272a;
    --button-secondary-text-color: #fafafa;
    --input-background-fill: #09090b;
    --input-border-color: #27272a;
}

body, .gradio-container, .gradio-container * { 
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; 
}

.gradio-container {
    max-width: 1040px !important;
    margin: 0 auto !important;
    padding: 32px 16px !important;
    background-color: var(--background-fill-primary) !important;
}

#app-title {
    font-size: 1.875rem !important;
    font-weight: 700 !important;
    color: var(--block-title-text-color) !important;
    text-align: center;
    margin-bottom: 6px !important;
    letter-spacing: -0.025em !important;
}

#app-subtitle {
    font-size: 0.9rem !important;
    color: #64748b !important;
    text-align: center;
    margin-top: 0 !important;
    margin-bottom: 32px !important;
}

.dark #app-subtitle {
    color: #a1a1aa !important;
}

/* Containers / Cards */
.gradio-container .block, .gradio-container .group {
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--block-border-color) !important;
    background-color: var(--block-background-fill) !important;
    box-shadow: var(--block-shadow) !important;
    padding: 20px !important;
}

/* Tabs list container - pill style */
.gradio-container [role="tablist"], .gradio-container .tab-nav {
    display: inline-flex !important;
    gap: 4px !important;
    background: #f1f5f9 !important;
    background-color: #f1f5f9 !important;
    padding: 4px !important;
    border-radius: 8px !important;
    border: 1px solid #e2e8f0 !important;
    margin-bottom: 24px !important;
}

.dark .gradio-container [role="tablist"], .dark .gradio-container .tab-nav {
    background: #18181b !important;
    background-color: #18181b !important;
    border-color: #27272a !important;
}

/* Tab triggers */
.gradio-container [role="tab"], .gradio-container .tab-nav button {
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    padding: 6px 16px !important;
    border-radius: 6px !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
}

.dark .gradio-container [role="tab"], .dark .gradio-container .tab-nav button {
    color: #a1a1aa !important;
}

.gradio-container [role="tab"][aria-selected="true"], .gradio-container .tab-nav button.selected {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #0f172a !important;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.08), 0 1px 2px -1px rgba(0, 0, 0, 0.08) !important;
    border: none !important;
}

.dark .gradio-container [role="tab"][aria-selected="true"], .dark .gradio-container .tab-nav button.selected {
    background: #27272a !important;
    background-color: #27272a !important;
    color: #fafafa !important;
    box-shadow: none !important;
    border: none !important;
}

/* Primary Button Styling */
button.primary-btn, .gradio-container button.primary-btn, button.primary-btn.secondary, button.primary-btn.primary {
    background: #4f46e5 !important;
    background-color: #4f46e5 !important;
    background-image: none !important;
    color: #ffffff !important;
    border: 1px solid #4f46e5 !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
}

button.primary-btn:hover, .gradio-container button.primary-btn:hover {
    background: #4338ca !important;
    background-color: #4338ca !important;
    border-color: #4338ca !important;
}

/* Explain Button Styling */
button.explain-btn, .gradio-container button.explain-btn, button.explain-btn.secondary, button.explain-btn.primary {
    background: #0f172a !important;
    background-color: #0f172a !important;
    background-image: none !important;
    color: #ffffff !important;
    border: 1px solid #0f172a !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
}

button.explain-btn:hover, .gradio-container button.explain-btn:hover {
    background: #1e293b !important;
    background-color: #1e293b !important;
    border-color: #1e293b !important;
}

.dark button.explain-btn, .dark .gradio-container button.explain-btn {
    background: #27272a !important;
    background-color: #27272a !important;
    border-color: #27272a !important;
    color: #fafafa !important;
}

.dark button.explain-btn:hover, .dark .gradio-container button.explain-btn:hover {
    background: #3f3f46 !important;
    background-color: #3f3f46 !important;
    border-color: #3f3f46 !important;
}

/* Inputs & Textareas */
.output-box textarea, input[type="text"], .gradio-dropdown {
    border: 1px solid var(--block-border-color) !important;
    background-color: var(--input-background-fill) !important;
    color: var(--body-text-color) !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
}

/* Markdown box display */
.markdown-box {
    background-color: var(--block-background-fill) !important;
    border: 1px solid var(--block-border-color) !important;
    border-radius: 8px !important;
    padding: 16px !important;
    min-height: 240px !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    color: var(--body-text-color) !important;
}

.upload-caption {
    color: #64748b !important;
    font-size: 0.8rem !important;
    margin-top: 8px !important;
}

.row-top-align {
    align-items: flex-start !important;
}

/* Color-Coded Confidence Badges */
span.confidence-badge, .gradio-container span.confidence-badge, .markdown-box span.confidence-badge {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 2px 8px !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    border-radius: 9999px !important;
    margin-left: 6px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

span.confidence-badge.high, .gradio-container span.confidence-badge.high {
    background-color: #dcfce7 !important;
    color: #15803d !important;
}

span.confidence-badge.medium, .gradio-container span.confidence-badge.medium {
    background-color: #fef9c3 !important;
    color: #a16207 !important;
}

span.confidence-badge.low, .gradio-container span.confidence-badge.low {
    background-color: #fee2e2 !important;
    color: #b91c1c !important;
}

.dark span.confidence-badge.high, .dark .gradio-container span.confidence-badge.high {
    background-color: rgba(21, 128, 61, 0.25) !important;
    color: #4ade80 !important;
}

.dark span.confidence-badge.medium, .dark .gradio-container span.confidence-badge.medium {
    background-color: rgba(161, 98, 7, 0.25) !important;
    color: #facc15 !important;
}

.dark span.confidence-badge.low, .dark .gradio-container span.confidence-badge.low {
    background-color: rgba(185, 28, 28, 0.25) !important;
    color: #f87171 !important;
}

@media (prefers-color-scheme: dark) {
    span.confidence-badge.high, .gradio-container span.confidence-badge.high {
        background-color: rgba(21, 128, 61, 0.25) !important;
        color: #4ade80 !important;
    }
    span.confidence-badge.medium, .gradio-container span.confidence-badge.medium {
        background-color: rgba(161, 98, 7, 0.25) !important;
        color: #facc15 !important;
    }
    span.confidence-badge.low, .gradio-container span.confidence-badge.low {
        background-color: rgba(185, 28, 28, 0.25) !important;
        color: #f87171 !important;
    }
}

/* Clean Tables styling */
table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin-top: 16px !important;
    margin-bottom: 16px !important;
}

th {
    font-weight: 600 !important;
    text-align: left !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid var(--block-border-color) !important;
    color: var(--block-title-text-color) !important;
    background-color: var(--background-fill-primary) !important;
}

td {
    padding: 10px 14px !important;
    border-bottom: 1px solid var(--block-border-color) !important;
    color: var(--body-text-color) !important;
}

tr:hover td {
    background-color: var(--background-fill-primary) !important;
}
"""

# ---------------------------------------------------------------------------
# Live performance metrics (Performance tab) -- computed once at startup
# against the bundled ground-truth samples, replacing hardcoded numbers.
# ---------------------------------------------------------------------------
import performance_metrics as _perf


def _aggregate(rows, field):
    """Average a numeric field across rows, skipping rows where it's None
    (e.g. Groq was rate-limited/unavailable for that sample). Returns None
    if no row has a usable value, rather than raising or averaging in a
    missing correction as if it were a real zero-change result."""
    values = [r[field] for r in rows if r[field] is not None]
    return sum(values) / len(values) if values else None


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


def build_ui():
    with gr.Blocks(theme=gr.themes.Base(), css=CUSTOM_CSS, title="IAM Handwriting Explainer") as demo:
        # ---- Header ----
        gr.Markdown("<h1 id='app-title'>✍️ IAM Handwriting Explainer</h1>")
        gr.Markdown(
            "<p id='app-subtitle'>"
            "Click a sample handwritten line below or upload your own — "
            "TrOCR transcribes, Groq explains."
            "</p>"
        )

        # Defined here (render=False) so Sample/Upload tab event handlers can
        # target it as an output; actually placed in the Performance tab below
        # via .render().
        corrections_dashboard_display = gr.Markdown(value=build_corrections_dashboard(), render=False)

        with gr.Tabs():
            # ==============================================================
            # Tab 1 — Sample Lines  (primary demo path)
            # ==============================================================
            with gr.Tab("📋 Sample Lines"):
                with gr.Row(elem_classes=["row-top-align"]):
                    with gr.Column(scale=1):
                        sample_map = {f"Sample {i+1}: {os.path.basename(p)}": p for i, p in enumerate(SAMPLE_IMAGES)}
                        
                        def transcribe_sample_and_reset(key):
                            if not key or key not in sample_map:
                                return "⚠️ Please select a sample first.", "", "", DEFAULT_EXPLANATION
                            text, conf_md = transcribe_with_confidence(sample_map[key])
                            return text, conf_md, "", DEFAULT_EXPLANATION

                        sample_dropdown = gr.Dropdown(
                            choices=list(sample_map.keys()),
                            label="Bundled IAM Line Samples (select one)",
                            value=None,
                            interactive=True
                        )
                        sample_image = gr.Image(
                            label="Line Image",
                            type="filepath",
                            height=180,
                            interactive=False,
                        )
                        
                        def load_sample(key):
                            if key and key in sample_map:
                                return sample_map[key]
                            return None
                            
                        sample_dropdown.change(
                            fn=load_sample,
                            inputs=[sample_dropdown],
                            outputs=[sample_image]
                        )
                    with gr.Column(scale=1):
                        sample_transcription = gr.Textbox(
                            label="Raw Transcription (TrOCR)",
                            interactive=False,
                            elem_classes=["output-box"],
                            lines=2,
                        )
                        sample_confidence = gr.Markdown(value="")
                        sample_verify = gr.Textbox(
                            label="Was the transcription correct? If not, enter the correct text (optional).",
                            placeholder="Type correct transcription and press Enter, or click Explain to log.",
                            interactive=True,
                        )
                        with gr.Row():
                            sample_transcribe_btn = gr.Button(
                                "Transcribe", elem_classes=["primary-btn"]
                            )
                            sample_explain_btn = gr.Button(
                                "Explain", elem_classes=["explain-btn"]
                            )
                        sample_explanation = gr.Markdown(
                            value="### LLM Correction + Confidence\n_Run Transcribe and Explain to see results here._",
                            elem_classes=["output-box", "markdown-box"],
                        )
                        with gr.Row():
                            sample_graphology_btn = gr.Button("🖋️ Personality Read", elem_classes=["explain-btn"])
                            sample_penpal_btn = gr.Button("✉️ Reply as Pen Pal", elem_classes=["explain-btn"])
                        sample_graphology_output = gr.Markdown(value="", elem_classes=["output-box", "markdown-box"])
                        sample_penpal_output = gr.Markdown(value="", elem_classes=["output-box", "markdown-box"])

                def clear_sample_outputs():
                    return "", "", "", DEFAULT_EXPLANATION, "", ""

                sample_image.change(
                    fn=clear_sample_outputs,
                    inputs=[],
                    outputs=[sample_transcription, sample_confidence, sample_verify, sample_explanation, sample_graphology_output, sample_penpal_output],
                )
                sample_transcribe_btn.click(
                    fn=transcribe_sample_and_reset,
                    inputs=[sample_dropdown],
                    outputs=[sample_transcription, sample_confidence, sample_verify, sample_explanation],
                )
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
                sample_graphology_btn.click(
                    fn=graphology_read,
                    inputs=[sample_image],
                    outputs=[sample_graphology_output],
                )
                sample_penpal_btn.click(
                    fn=pen_pal_reply,
                    inputs=[sample_transcription],
                    outputs=[sample_penpal_output],
                )

            # ==============================================================
            # Tab 2 — Upload  (auto-detects single-line vs multi-line)
            # ==============================================================
            with gr.Tab("📤 Upload"):
                with gr.Row(elem_classes=["row-top-align"]):
                    with gr.Column(scale=1):
                        upload_image = gr.Image(
                            label="Upload a handwritten image",
                            type="filepath",
                            height=220,
                        )
                        gr.Markdown(
                            "<p class='upload-caption'>"
                            "Upload a handwritten line or multi-line passage — the app "
                            "automatically detects and handles both."
                            "</p>"
                        )
                        upload_transcribe_btn = gr.Button(
                            "Transcribe", elem_classes=["primary-btn"]
                        )
                        MULTILINE_EXAMPLE = os.path.join(SAMPLES_DIR, "verified_multiline_test.png")
                        gr.Examples(
                            examples=[[MULTILINE_EXAMPLE]],
                            inputs=[upload_image],
                            label="Bundled multi-line sample (click to load)",
                        )
                    with gr.Column(scale=1):
                        upload_transcription = gr.Textbox(
                            label="Raw Transcription (TrOCR)",
                            interactive=False,
                            elem_classes=["output-box"],
                            lines=4,
                        )
                        upload_confidence = gr.Markdown(value="")
                        upload_verify = gr.Textbox(
                            label="Was the transcription correct? If not, enter the correct text (optional).",
                            placeholder="Type correct transcription and press Enter, or click Explain to log.",
                            interactive=True,
                        )
                        upload_explain_btn = gr.Button(
                            "Explain", elem_classes=["explain-btn"]
                        )
                        upload_explanation = gr.Markdown(
                            value=DEFAULT_EXPLANATION,
                            elem_classes=["output-box", "markdown-box"],
                        )
                        with gr.Row():
                            upload_graphology_btn = gr.Button("🖋️ Personality Read", elem_classes=["explain-btn"])
                            upload_penpal_btn = gr.Button("✉️ Reply as Pen Pal", elem_classes=["explain-btn"])
                        upload_graphology_output = gr.Markdown(value="", elem_classes=["output-box", "markdown-box"])
                        upload_penpal_output = gr.Markdown(value="", elem_classes=["output-box", "markdown-box"])

                def clear_upload_outputs():
                    return "", "", "", DEFAULT_EXPLANATION, "", ""

                upload_image.change(
                    fn=clear_upload_outputs,
                    inputs=[],
                    outputs=[upload_transcription, upload_confidence, upload_verify, upload_explanation, upload_graphology_output, upload_penpal_output],
                )
                upload_transcribe_btn.click(
                    fn=transcribe_upload_and_reset,
                    inputs=[upload_image],
                    outputs=[upload_transcription, upload_confidence, upload_verify, upload_explanation],
                )
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
                upload_graphology_btn.click(
                    fn=graphology_read,
                    inputs=[upload_image],
                    outputs=[upload_graphology_output],
                )
                upload_penpal_btn.click(
                    fn=pen_pal_reply,
                    inputs=[upload_transcription],
                    outputs=[upload_penpal_output],
                )

            # ==============================================================
            # Tab 3 — Performance
            # ==============================================================
            with gr.Tab("📊 Performance"):
                gr.Markdown("## 📊 Model Performance & Validation Analysis")
                corrections_dashboard_display.render()

                with gr.Row():
                    with gr.Column(scale=1):
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
                            if _wer is None or _cer is None:
                                _bundled_metric_lines.append(f"| **{_engine_label}** | _unavailable this run_ | _unavailable this run_ |")
                            else:
                                _bundled_metric_lines.append(f"| **{_engine_label}** | **{_wer*100:.2f}%** | **{_cer*100:.2f}%** |")
                        gr.Markdown("\n".join(_bundled_metric_lines))
                        
                        gr.Markdown(
                            "### 🧠 Why TrOCR Powers This Pipeline\n"
                            "TrOCR (`trocr-large-handwritten`) is fine-tuned specifically for handwriting recognition, "
                            "unlike general-purpose vision-language models (e.g. Florence-2) that target broader document "
                            "understanding rather than character-level handwriting fidelity. This project did not run a "
                            "benchmarked comparison against those alternatives — the choice reflects task specialization, "
                            "not a measured result. The numbers actually measured on this app's own pipeline are the "
                            "engine-comparison tables above and below: stock TrOCR against the TrOCR+Groq pipeline, "
                            "Tesseract, and EasyOCR, computed live against real ground truth."
                        )
                        
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Error Rates & Confidence Distribution Chart")
                        gr.Image(
                            value=os.path.join(SAMPLES_DIR, "confidence_distribution.png"),
                            label="Evaluation Visualizations",
                            interactive=False,
                            show_label=False
                        )
                
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
                    if r['pipeline_cer'] is None:
                        pipeline_output_cell = "_unavailable_"
                        pipeline_cer_cell = "—"
                    else:
                        pipeline_output_cell = _md_table_cell(r['pipeline_output'])
                        pipeline_cer_cell = f"{r['pipeline_cer']*100:.2f}%"
                    _comparison_lines.append(
                        f"| {_md_table_cell(r['image_path'])} | {_md_table_cell(r['reference'])} | "
                        f"{_md_table_cell(r['stock_output'])} | {r['stock_cer']*100:.2f}% | "
                        f"{pipeline_output_cell} | {pipeline_cer_cell} | "
                        f"{_md_table_cell(r['tesseract_output'])} | {r['tesseract_cer']*100:.2f}% | "
                        f"{_md_table_cell(r['easyocr_output'])} | {r['easyocr_cer']*100:.2f}% |"
                    )
                gr.Markdown("\n".join(_comparison_lines))

        # ---- Footer ----
        gr.Markdown(
            "<div style='text-align:center; color:#9ca3af; font-size:0.8rem; "
            "margin-top:1.5rem;'>"
            "Powered by TrOCR · OpenRouter (Llama 3.3) · Gradio  ·  "
            "Samples from <a href='https://huggingface.co/datasets/Teklia/IAM-line' "
            "style='color:#667eea;'>Teklia/IAM-line</a>"
            "</div>"
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_port=int(os.environ["PORT"]) if os.environ.get("PORT") else None)
