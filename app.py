import os
import glob
import difflib
import string
import sys
import subprocess
import warnings

import gradio as gr
import torch
import numpy as np
from PIL import Image
from groq import Groq
import re
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
        try:
            parsed_url = urlparse(url)
            path_part = unquote(parsed_url.path)
            
            local_path = None
            if "/file=" in path_part:
                local_path = path_part.split("/file=", 1)[1]
            elif path_part.startswith("/file/"):
                local_path = path_part[6:]
                
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
            
        return await old_download(url, cache_dir)

    def new_sync_download(url: str, cache_dir: str) -> str:
        try:
            parsed_url = urlparse(url)
            path_part = unquote(parsed_url.path)
            
            local_path = None
            if "/file=" in path_part:
                local_path = path_part.split("/file=", 1)[1]
            elif path_part.startswith("/file/"):
                local_path = path_part[6:]
                
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
            
        return old_sync_download(url, cache_dir)

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
# Secrets — read from HF Spaces (Settings → Repository secrets)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ---------------------------------------------------------------------------
# TrOCR model loading
# ---------------------------------------------------------------------------
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, RobertaTokenizer, ViTImageProcessor

print("Loading TrOCR model and processor...")
# Instantiate RobertaTokenizer and ViTImageProcessor manually to bypass TrOCR Processor bugs
image_processor = ViTImageProcessor.from_pretrained("microsoft/trocr-base-handwritten")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/trocr-base-handwritten")
processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
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
        generated_ids = model.generate(pixel_values, max_new_tokens=128)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text.strip()


DEFAULT_EXPLANATION = "### LLM Correction + Confidence\n_Run Transcribe and Explain to see results here._"

def reset_explanation():
    return DEFAULT_EXPLANATION


def explain(ocr_text):
    """Send transcription to Groq, run token-overlap check, return formatted output."""
    if not ocr_text or ocr_text.startswith("⚠️"):
        return "⚠️ Nothing to explain — run Transcribe first."
    if not GROQ_API_KEY:
        return (
            "⚠️ GROQ_API_KEY not found.\n"
            "Add it under Settings → Repository secrets in your HF Space."
        )

    # --- Sanity check: do not call Groq on degenerate OCR output ---
    words = ocr_text.split()
    if len(words) < 3:
        return "⚠️ Transcription unclear — this image may not be a supported single-line format. Try a different sample or a clearer single-line upload."
        
    single_char_or_digit_words = sum(1 for w in words if len(w) == 1 or w.isdigit())
    if single_char_or_digit_words / len(words) >= 0.5:
        return "⚠️ Transcription unclear — this image may not be a supported single-line format. Try a different sample or a clearer single-line upload."
        
    letters = sum(c.isalpha() for c in ocr_text)
    non_whitespace = sum(not c.isspace() for c in ocr_text)
    if non_whitespace > 0 and letters / non_whitespace < 0.4:
        return "⚠️ Transcription unclear — this image may not be a supported single-line format. Try a different sample or a clearer single-line upload."

    # Pass raw OCR output directly to Groq
    corrected_ocr_text = ocr_text

    client = Groq(api_key=GROQ_API_KEY)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"OCR output:\n{corrected_ocr_text}"},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        llm_output = response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Groq API error: {e}"

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
    display_parts.append(f"**Confidence:** {confidence}")
    
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


# ---------------------------------------------------------------------------
# Gradio UI  —  Blocks layout
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
}

#app-title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0;
}
#app-subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 0.95rem;
    margin-top: 0;
}

.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    transition: opacity 0.2s ease !important;
}
.primary-btn:hover {
    opacity: 0.9 !important;
}

.explain-btn {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    transition: opacity 0.2s ease !important;
}
.explain-btn:hover {
    opacity: 0.9 !important;
}

.output-box textarea {
    font-size: 1rem !important;
    line-height: 1.6 !important;
    border-radius: 8px !important;
}

.markdown-box {
    background-color: var(--block-background-fill) !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 8px !important;
    padding: 16px !important;
    min-height: 320px !important;
    overflow-y: auto !important;
}

.upload-caption {
    color: #6b7280;
    font-size: 0.85rem;
    font-style: italic;
    margin-top: 4px;
}
"""


def build_ui():
    with gr.Blocks(css=CUSTOM_CSS, title="IAM Handwriting Explainer") as demo:
        # ---- Header ----
        gr.Markdown("<h1 id='app-title'>✍️ IAM Handwriting Explainer</h1>")
        gr.Markdown(
            "<p id='app-subtitle'>"
            "Click a sample handwritten line below or upload your own — "
            "TrOCR transcribes, Groq explains."
            "</p>"
        )

        with gr.Tabs():
            # ==============================================================
            # Tab 1 — Sample Lines  (primary demo path)
            # ==============================================================
            with gr.Tab("📋 Sample Lines"):
                with gr.Row():
                    with gr.Column(scale=1):
                        sample_map = {f"Sample {i+1}: {os.path.basename(p)}": p for i, p in enumerate(SAMPLE_IMAGES)}
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

                def clear_sample_outputs():
                    return "", DEFAULT_EXPLANATION

                sample_image.change(
                    fn=clear_sample_outputs,
                    inputs=[],
                    outputs=[sample_transcription, sample_explanation],
                )

                def transcribe_sample(key):
                    if not key or key not in sample_map:
                        return "⚠️ Please select a sample first."
                    return transcribe(sample_map[key])

                sample_transcribe_btn.click(
                    fn=reset_explanation,
                    outputs=[sample_explanation],
                ).then(
                    fn=transcribe_sample,
                    inputs=[sample_dropdown],
                    outputs=[sample_transcription],
                )
                sample_explain_btn.click(
                    fn=explain,
                    inputs=[sample_transcription],
                    outputs=[sample_explanation],
                )

            # ==============================================================
            # Tab 2 — Upload  (optional path)
            # ==============================================================
            with gr.Tab("📤 Upload"):
                with gr.Row():
                    with gr.Column(scale=1):
                        upload_image = gr.Image(
                            label="Upload a handwritten image",
                            type="filepath",
                            height=180,
                        )
                        gr.Markdown(
                            "<p class='upload-caption'>"
                            "Upload a single handwritten line only — not a paragraph or tilted image. "
                            "This app is optimized for clean, single-line handwriting."
                            "</p>"
                        )
                    with gr.Column(scale=1):
                        upload_transcription = gr.Textbox(
                            label="Raw Transcription (TrOCR)",
                            interactive=False,
                            elem_classes=["output-box"],
                            lines=2,
                        )
                        with gr.Row():
                            upload_transcribe_btn = gr.Button(
                                "Transcribe", elem_classes=["primary-btn"]
                            )
                            upload_explain_btn = gr.Button(
                                "Explain", elem_classes=["explain-btn"]
                            )
                        upload_explanation = gr.Markdown(
                            value="### LLM Correction + Confidence\n_Run Transcribe and Explain to see results here._",
                            elem_classes=["output-box", "markdown-box"],
                        )

                def clear_upload_outputs():
                    return "", DEFAULT_EXPLANATION

                upload_image.change(
                    fn=clear_upload_outputs,
                    inputs=[],
                    outputs=[upload_transcription, upload_explanation],
                )

                upload_transcribe_btn.click(
                    fn=reset_explanation,
                    outputs=[upload_explanation],
                ).then(
                    fn=transcribe,
                    inputs=[upload_image],
                    outputs=[upload_transcription],
                )
                upload_explain_btn.click(
                    fn=explain,
                    inputs=[upload_transcription],
                    outputs=[upload_explanation],
                )

        # ---- Footer ----
        gr.Markdown(
            "<div style='text-align:center; color:#9ca3af; font-size:0.8rem; "
            "margin-top:1.5rem;'>"
            "Powered by TrOCR · Groq (Llama 3.3) · Gradio  ·  "
            "Samples from <a href='https://huggingface.co/datasets/Teklia/IAM-line' "
            "style='color:#667eea;'>Teklia/IAM-line</a>"
            "</div>"
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
