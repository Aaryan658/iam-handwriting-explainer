"""Live performance metrics for the Performance tab: CER/WER against the
bundled ground-truth samples, and a stock-TrOCR-vs-full-pipeline comparison."""
import csv

import jiwer

from app import transcribe, explain, extract_corrected_text
import ocr_engines as _engines


def load_ground_truth(csv_path="samples/ground_truth.csv"):
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_cer_wer(hypothesis, reference):
    cer = jiwer.cer(reference, hypothesis)
    wer = jiwer.wer(reference, hypothesis)
    return cer, wer


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
        corrected_text = extract_corrected_text(explain_output)
        if corrected_text:
            pipeline_output = corrected_text
            pipeline_cer, pipeline_wer = compute_cer_wer(pipeline_output, reference)
        else:
            # explain() couldn't produce a correction (Groq error, rate limit,
            # missing key, etc.) -- keep its raw message for transparency and
            # leave CER/WER as None rather than silently scoring stock_output
            # as if Groq had confirmed it unchanged.
            pipeline_output = explain_output
            pipeline_cer = None
            pipeline_wer = None

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
