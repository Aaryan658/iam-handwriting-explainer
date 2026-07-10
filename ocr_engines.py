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
