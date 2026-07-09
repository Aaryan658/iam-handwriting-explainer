from performance_metrics import load_ground_truth, compute_cer_wer


def test_load_ground_truth_reads_csv(tmp_path):
    csv_path = tmp_path / "gt.csv"
    csv_path.write_text("image_path,text\nline_01.png,hello world\n", encoding="utf-8")
    rows = load_ground_truth(str(csv_path))
    assert rows == [{"image_path": "line_01.png", "text": "hello world"}]


def test_compute_cer_wer_identical_strings_is_zero():
    cer, wer = compute_cer_wer(hypothesis="hello world", reference="hello world")
    assert cer == 0.0
    assert wer == 0.0


def test_compute_cer_wer_detects_errors():
    cer, wer = compute_cer_wer(hypothesis="helo wrold", reference="hello world")
    assert cer > 0.0
    assert wer > 0.0


def test_evaluate_stock_vs_pipeline_uses_transcribe_and_explain(monkeypatch):
    from performance_metrics import evaluate_stock_vs_pipeline

    def fake_transcribe(image_path):
        return "helo wrold"

    def fake_explain(ocr_text, confidence_md=""):
        # Mirrors app.explain()'s real formatting: a bold label and bolded
        # "uncertain words" inline -- both must be stripped, not treated as
        # literal transcribed characters.
        return "**Corrected:** **hello** world\nConfidence: HIGH\n"

    monkeypatch.setattr("performance_metrics.transcribe", fake_transcribe)
    monkeypatch.setattr("performance_metrics.explain", fake_explain)

    ground_truth = [{"image_path": "line_01.png", "text": "hello world"}]
    results = evaluate_stock_vs_pipeline(ground_truth)

    assert len(results) == 1
    row = results[0]
    assert row["image_path"] == "line_01.png"
    assert row["stock_output"] == "helo wrold"
    assert row["pipeline_output"] == "hello world"
    assert row["stock_cer"] > row["pipeline_cer"]
