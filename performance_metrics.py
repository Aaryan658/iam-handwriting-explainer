"""Live performance metrics for the Performance tab: CER/WER against the
bundled ground-truth samples, and a stock-TrOCR-vs-full-pipeline comparison."""
import csv
import re

import jiwer

from app import transcribe, explain


def load_ground_truth(csv_path="samples/ground_truth.csv"):
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_cer_wer(hypothesis, reference):
    cer = jiwer.cer(reference, hypothesis)
    wer = jiwer.wer(reference, hypothesis)
    return cer, wer


def _extract_corrected(explain_markdown):
    match = re.search(r"Corrected:\s*(.+)", explain_markdown)
    return match.group(1).strip() if match else ""


def evaluate_stock_vs_pipeline(ground_truth):
    """For each ground-truth row, run (1) stock transcribe() alone and (2) the
    full pipeline (transcribe -> explain()'s Groq correction), and compute
    CER/WER for both against the reference text."""
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
