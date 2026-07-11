"""Comparison OCR engines (Tesseract, EasyOCR) for benchmarking against
TrOCR, both on the bundled ground-truth set and on logged corrections."""
import os
import platform
import threading

import pytesseract
from PIL import Image

if platform.system() == "Windows":
    # Local dev only -- the tesseract-ocr apt package (installed via
    # packages.txt on HF Spaces / other Linux hosts) puts the binary on
    # PATH, so pytesseract's default lookup already finds it there.
    _WINDOWS_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_WINDOWS_TESSERACT):
        pytesseract.pytesseract.tesseract_cmd = _WINDOWS_TESSERACT

_easyocr_reader = None
_easyocr_reader_lock = threading.Lock()


def tesseract_transcribe(image_path):
    """Run Tesseract OCR on one image and return the raw transcription."""
    image = Image.open(image_path).convert("RGB")
    return pytesseract.image_to_string(image).strip()


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is not None:
        return _easyocr_reader

    with _easyocr_reader_lock:
        if _easyocr_reader is None:
            import ssl
            import easyocr

            # This machine's network runs Sophos TLS inspection, whose intercepting
            # CA cert has a malformed Basic Constraints extension (not marked
            # critical). Python 3.14 enables ssl.VERIFY_X509_STRICT by default,
            # which makes OpenSSL reject that cert on github.com /
            # raw.githubusercontent.com (where EasyOCR fetches its model weights via
            # urllib.request.urlretrieve, which sources its default HTTPS context
            # from ssl._create_default_https_context) even though the cert chain is
            # otherwise trusted. Relax only that one strict structural check -- not
            # hostname verification or CA trust-chain verification -- and only for
            # the duration of this download, then restore the original default so
            # the rest of the app (TrOCR/huggingface.co downloads, Groq API calls)
            # keeps full default strict verification.
            original_https_context = ssl._create_default_https_context

            def _relaxed_context(*args, **kwargs):
                ctx = ssl.create_default_context(*args, **kwargs)
                ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
                return ctx

            ssl._create_default_https_context = _relaxed_context
            try:
                _easyocr_reader = easyocr.Reader(["en"], gpu=False)
            finally:
                ssl._create_default_https_context = original_https_context
    return _easyocr_reader


def easyocr_transcribe(image_path):
    """Run EasyOCR on one image and return the raw transcription."""
    reader = _get_easyocr_reader()
    results = reader.readtext(image_path, detail=0)
    return " ".join(results).strip()
