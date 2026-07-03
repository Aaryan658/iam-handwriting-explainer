import os
import glob
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
# Discover bundled sample line groups   (samples/<line-group-id>/*.png)
# ---------------------------------------------------------------------------
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
LINE_GROUPS = {}

for group_dir in sorted(glob.glob(os.path.join(SAMPLES_DIR, "*"))):
    if os.path.isdir(group_dir):
        group_id = os.path.basename(group_dir)
        images = sorted(glob.glob(os.path.join(group_dir, "*.png")))
        if images:
            LINE_GROUPS[group_id] = images

LINE_GROUP_CHOICES = list(LINE_GROUPS.keys())

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def transcribe_single_image(image):
    """Run TrOCR on one PIL image and return the predicted text."""
    pixel_values = processor(image.convert("RGB"), return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=21)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def transcribe_uploaded_image(image):
    """Handle a user-uploaded single word image."""
    if image is None:
        return "⚠️ Please upload an image first.", ""
    pil_image = Image.fromarray(image) if not isinstance(image, Image.Image) else image
    word = transcribe_single_image(pil_image)
    return word, word


def transcribe_line_group(group_id):
    """Reconstruct a sentence from a bundled line group."""
    if not group_id or group_id not in LINE_GROUPS:
        return "⚠️ Please select a valid line group.", [], ""
    image_paths = LINE_GROUPS[group_id]
    pred_words = []
    gallery_images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        gallery_images.append(path)
        pred_words.append(transcribe_single_image(img))
    sentence = " ".join(pred_words)
    return sentence, gallery_images, sentence


def explain_with_groq(sentence):
    """Send reconstructed sentence to Groq for correction + explanation."""
    if not sentence or sentence.startswith("⚠️"):
        return "⚠️ Nothing to explain — run Transcribe first."
    if not GROQ_API_KEY:
        return (
            "⚠️ GROQ_API_KEY not found.\n"
            "Add it under Settings → Repository secrets in your HF Space."
        )

    client = Groq(api_key=GROQ_API_KEY)

    system_prompt = (
        "You are an AI assistant correcting OCR output from handwriting recognition. "
        "For each sentence you receive, respond with EXACTLY this format:\n\n"
        "Corrected: <your best corrected transcription>\n"
        "Confidence: <HIGH | MEDIUM | LOW>\n"
        "Note: <any notes, or omit this line if confidence is HIGH>\n"
        "Context: <1-2 sentence explanation, ONLY if confidence is HIGH>\n\n"
        "Rules:\n"
        "1. Fix obvious OCR/transcription errors (misspellings, garbled tokens) "
        "while preserving the original meaning.\n"
        "2. Assign a confidence label for the correction as a whole:\n"
        "   - HIGH: the corrected sentence clearly reflects the intended meaning.\n"
        "   - MEDIUM: some words are uncertain but the gist is likely correct.\n"
        "   - LOW: more than 2 words were substantially altered, or the corrected "
        "sentence's meaning is a guess rather than a clear fix.\n"
        "3. If confidence is MEDIUM or LOW, you MUST include the note: "
        "'This reconstruction may not reflect the original meaning.' "
        "Do NOT provide a contextual explanation paragraph in that case.\n"
        "4. Do NOT invent specific details (times, named actions, first-person narrative) "
        "that are not clearly present in the input tokens.\n"
        "5. Do not output any additional chat or conversational text."
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "OCR output:\n" + sentence},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        return "⚠️ Groq API error: " + str(e)


# ---------------------------------------------------------------------------
# Gradio UI
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

.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
.tab-nav button.selected {
    border-color: #667eea !important;
    color: #667eea !important;
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
"""


def build_ui():
    with gr.Blocks(css=CUSTOM_CSS, title="IAM Handwriting Explainer") as demo:
        # ---- Header ----
        gr.Markdown("<h1 id='app-title'>✍️ IAM Handwriting Explainer</h1>")
        gr.Markdown(
            "<p id='app-subtitle'>"
            "Upload a handwritten word image or pick a bundled IAM line group — "
            "TrOCR transcribes, Groq explains."
            "</p>"
        )

        with gr.Tabs():
            # ============================================================
            # Tab 1 — Single Word Upload
            # ============================================================
            with gr.Tab("🖼️ Single Word"):
                with gr.Row():
                    with gr.Column(scale=1):
                        upload_image = gr.Image(
                            label="Upload a handwritten word image",
                            type="pil",
                            height=220,
                        )
                        transcribe_upload_btn = gr.Button(
                            "Transcribe", elem_classes=["primary-btn"]
                        )
                    with gr.Column(scale=1):
                        upload_result = gr.Textbox(
                            label="Transcribed Word",
                            interactive=False,
                            elem_classes=["output-box"],
                            lines=2,
                        )
                        # Hidden textbox to hold sentence for Explain button
                        upload_sentence_hidden = gr.Textbox(
                            visible=False,
                        )
                        explain_upload_btn = gr.Button(
                            "Explain with LLM", elem_classes=["explain-btn"]
                        )
                        upload_explanation = gr.Textbox(
                            label="LLM Explanation",
                            interactive=False,
                            elem_classes=["output-box"],
                            lines=8,
                        )

                transcribe_upload_btn.click(
                    fn=transcribe_uploaded_image,
                    inputs=[upload_image],
                    outputs=[upload_result, upload_sentence_hidden],
                )
                explain_upload_btn.click(
                    fn=explain_with_groq,
                    inputs=[upload_sentence_hidden],
                    outputs=[upload_explanation],
                )

            # ============================================================
            # Tab 2 — Line Group Reconstruction
            # ============================================================
            with gr.Tab("📝 Line Group"):
                with gr.Row():
                    with gr.Column(scale=1):
                        group_dropdown = gr.Dropdown(
                            choices=LINE_GROUP_CHOICES,
                            label="Select a line group",
                            value=LINE_GROUP_CHOICES[0] if LINE_GROUP_CHOICES else None,
                        )
                        transcribe_group_btn = gr.Button(
                            "Transcribe Line Group", elem_classes=["primary-btn"]
                        )
                        word_gallery = gr.Gallery(
                            label="Word Images",
                            columns=4,
                            height=180,
                        )
                    with gr.Column(scale=1):
                        group_result = gr.Textbox(
                            label="Reconstructed Sentence",
                            interactive=False,
                            elem_classes=["output-box"],
                            lines=3,
                        )
                        # Hidden textbox to hold sentence for Explain button
                        group_sentence_hidden = gr.Textbox(
                            visible=False,
                        )
                        explain_group_btn = gr.Button(
                            "Explain with LLM", elem_classes=["explain-btn"]
                        )
                        group_explanation = gr.Textbox(
                            label="LLM Explanation",
                            interactive=False,
                            elem_classes=["output-box"],
                            lines=8,
                        )

                transcribe_group_btn.click(
                    fn=transcribe_line_group,
                    inputs=[group_dropdown],
                    outputs=[group_result, word_gallery, group_sentence_hidden],
                )
                explain_group_btn.click(
                    fn=explain_with_groq,
                    inputs=[group_sentence_hidden],
                    outputs=[group_explanation],
                )

        # ---- Footer ----
        gr.Markdown(
            "<div style='text-align:center; color:#9ca3af; font-size:0.8rem; "
            "margin-top:1.5rem;'>"
            "Powered by TrOCR · Groq (Llama 3.1) · Gradio"
            "</div>"
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
