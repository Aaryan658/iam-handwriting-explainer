"""Live performance metrics for the Performance tab: CER/WER against the
bundled ground-truth samples, and a stock-TrOCR-vs-full-pipeline comparison."""
import csv

import jiwer


def load_ground_truth(csv_path="samples/ground_truth.csv"):
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_cer_wer(hypothesis, reference):
    cer = jiwer.cer(reference, hypothesis)
    wer = jiwer.wer(reference, hypothesis)
    return cer, wer
