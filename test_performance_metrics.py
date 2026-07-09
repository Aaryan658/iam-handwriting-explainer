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
