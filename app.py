import os
import glob
import difflib
import string
import gradio as gr
import torch
from transformers import AutoProcessor, VisionEncoderDecoderModel, logging as hf_logging
from PIL import Image
from groq import Groq
import warnings

# ---------------------------------------------------------------------------
# Suppress noisy warnings
# ---------------------------------------------------------------------------
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Secrets — read from HF Spaces (Settings → Repository secrets)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ---------------------------------------------------------------------------
# Load TrOCR model once at startup
# ---------------------------------------------------------------------------
print("Loading TrOCR processor and model …")
processor = AutoProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
print("Model loaded.")

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
    "Confidence: <HIGH | MEDIUM | LOW>\n"
    "Note: <any notes, or omit this line if confidence is HIGH>\n"
    "Context: <1-2 sentence explanation, ONLY if confidence is HIGH>\n\n"
    "Rules:\n"
    "1. Fix obvious OCR/transcription errors (misspellings, garbled tokens) "
    "while preserving the original meaning.\n"
    "2. Before assigning confidence, first answer: \"Did I add, infer, or invent "
    "any word, phrase, or meaning not directly present in the OCR input — "
    "including completing a sentence fragment, or substituting a different "
    "word/phrasing than what was written even if grammatically similar?\" "
    "Output this as \"Added content: YES\" or \"Added content: NO\" as the "
    "FIRST line of your response, before anything else.\n"
    "3. If Added content = YES, Confidence MUST be MEDIUM or LOW. It cannot "
    "be HIGH under any circumstance.\n"
    "4. If Added content = NO, Confidence MAY be HIGH.\n"
    "5. Confidence label definitions:\n"
    "   - HIGH: the corrected sentence clearly reflects the intended meaning, "
    "and no words/phrases were added, inferred, or substituted beyond fixing "
    "character-level noise (spacing, capitalization, punctuation).\n"
    "   - MEDIUM: some words are uncertain, or minor inference was needed, "
    "but the gist is likely correct.\n"
    "   - LOW: more than 2 words were substantially altered, or the corrected "
    "sentence's meaning is a guess rather than a clear fix.\n"
    "6. If confidence is MEDIUM or LOW, you MUST include the note: "
    "\"This reconstruction may not reflect the original meaning.\" "
    "Do NOT provide a contextual explanation paragraph in that case.\n"
    "7. Do NOT invent specific details (times, named actions, first-person "
    "narrative, verbs, or events) that are not clearly present in the input tokens.\n"
    "8. Do not rephrase or substitute words that are already correct and clear "
    "— only fix genuine OCR noise. Preserve original word choice "
    "(\"put down\" must stay \"put down,\" not become \"put forward\").\n"
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

    pil_image = Image.fromarray(image) if not isinstance(image, Image.Image) else image
    pil_image = pil_image.convert("RGB")

    pixel_values = processor(pil_image, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=64)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def explain(ocr_text):
    """Send transcription to Groq, run token-overlap check, return formatted output."""
    if not ocr_text or ocr_text.startswith("⚠️"):
        return "⚠️ Nothing to explain — run Transcribe first."
    if not GROQ_API_KEY:
        return (
            "⚠️ GROQ_API_KEY not found.\n"
            "Add it under Settings → Repository secrets in your HF Space."
        )

    client = Groq(api_key=GROQ_API_KEY)

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"OCR output:\n{ocr_text}"},
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

    for line in lines:
        if line.startswith("Added content:"):
            added_content = line.split(":", 1)[1].strip()
        elif line.startswith("Corrected:"):
            corrected_line = line.split(":", 1)[1].strip()
        elif line.startswith("Confidence:"):
            confidence = line.split(":", 1)[1].strip()

    # --- Deterministic token-overlap check ---
    ocr_tokens = tokenize(ocr_text)
    corrected_tokens = tokenize(corrected_line)

    unmatched_tokens = []
    for ct in corrected_tokens:
        matches = difflib.get_close_matches(ct, ocr_tokens, n=1, cutoff=0.7)
        if not matches:
            unmatched_tokens.append(ct)

    overridden = False
    if len(unmatched_tokens) > 0 and "HIGH" in confidence:
        # Override: downgrade to MEDIUM, strip Context, force disclaimer
        new_lines = []
        for line in lines:
            if line.startswith("Confidence:"):
                new_lines.append("Confidence: MEDIUM")
            elif line.startswith("Note:") or line.startswith("Context:"):
                continue
            else:
                new_lines.append(line)
        # Clean trailing blanks
        while new_lines and not new_lines[-1].strip():
            new_lines.pop()
        new_lines.append(
            "Note: This reconstruction may not reflect the original meaning."
        )
        llm_output = "\n".join(new_lines)
        overridden = True

    # --- Build display output ---
    display_parts = [llm_output]

    display_parts.append("")
    display_parts.append("─── Token-Overlap Check ───")
    display_parts.append(f"Model claimed Added content: {added_content}")
    display_parts.append(f"Unmatched tokens (code):     {unmatched_tokens if unmatched_tokens else '[] (none)'}")
    if overridden:
        display_parts.append("⚠️  Confidence OVERRIDDEN from HIGH → MEDIUM by token check.")

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
                        sample_image = gr.Image(
                            label="Line Image",
                            type="pil",
                            height=180,
                        )
                        gr.Examples(
                            examples=SAMPLE_IMAGES,
                            inputs=sample_image,
                            label="Bundled IAM Line Samples (click one)",
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
                        sample_explanation = gr.Textbox(
                            label="LLM Correction + Confidence",
                            interactive=False,
                            elem_classes=["output-box"],
                            lines=10,
                        )

                sample_transcribe_btn.click(
                    fn=transcribe,
                    inputs=[sample_image],
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
                            label="Upload a handwritten line image",
                            type="pil",
                            height=180,
                        )
                        gr.Markdown(
                            "<p class='upload-caption'>"
                            "For best results, upload a single handwritten line "
                            "(not a full page or paragraph)."
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
                        upload_explanation = gr.Textbox(
                            label="LLM Correction + Confidence",
                            interactive=False,
                            elem_classes=["output-box"],
                            lines=10,
                        )

                upload_transcribe_btn.click(
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
            "Powered by TrOCR · Groq (Llama 3.1) · Gradio  ·  "
            "Samples from <a href='https://huggingface.co/datasets/Teklia/IAM-line' "
            "style='color:#667eea;'>Teklia/IAM-line</a>"
            "</div>"
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
