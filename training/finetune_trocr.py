"""
Phase 4: fine-tune microsoft/trocr-base-handwritten on new line-level
handwriting data (e.g. historical manuscripts), following the recipe from
the project's TrOCR-for-historical-handwriting research:

  - data: line-image crop + transcription pairs (CSV: image_path,text)
  - augmentation: random rotation (+/- 3deg) and elastic distortion
  - optimizer: AdamW via Seq2SeqTrainer, lr=5e-5, weight_decay=0.01
  - encoder is the brittle part: freeze only its early layers (default 3);
    the decoder tolerates freezing more (default 6)
  - evaluation: CER/WER via jiwer, on a held-out split

Usage:
    python finetune_trocr.py --data-csv data/train.csv --output-dir out/
    python finetune_trocr.py --data-csv data/train.csv --eval-csv data/val.csv --output-dir out/

The CSV must have a header with columns "image_path,text". image_path is
resolved relative to --image-root (default: the CSV's own directory).
"""

import argparse
import csv
import os
import random

# --- Keep all HF downloads/caches inside the project directory (D:), not
# the default C:\Users\...\.cache -- must be set before importing
# transformers so it picks this up. ---
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_HF_CACHE_DIR = os.path.join(_PROJECT_ROOT, ".hf_cache")
os.environ.setdefault("HF_HOME", _HF_CACHE_DIR)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_HF_CACHE_DIR, "hub"))

# --- huggingface_hub's httpx client hits "Cannot send a request, as the
# client has been closed" in this environment (same issue app.py works
# around) -- forcing verify=False on every httpx.Client avoids it. ---
import httpx as _httpx
_old_httpx_init = _httpx.Client.__init__
def _patched_httpx_init(self, *a, **kw):
    kw["verify"] = False
    _old_httpx_init(self, *a, **kw)
_httpx.Client.__init__ = _patched_httpx_init

import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    RobertaTokenizer,
    ViTImageProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import jiwer


def elastic_distort(image_np, alpha=34, sigma=4, seed=None):
    """Random elastic distortion, the augmentation with the most consistent
    gains for historical-manuscript TrOCR fine-tunes in the research."""
    rng = np.random.RandomState(seed)
    shape = image_np.shape[:2]
    dx = cv2.GaussianBlur((rng.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((rng.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(image_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderValue=(255, 255, 255))


def random_rotate(image_np, max_angle=3.0):
    angle = random.uniform(-max_angle, max_angle)
    h, w = image_np.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image_np, matrix, (w, h), borderValue=(255, 255, 255))


def augment(pil_image):
    arr = np.array(pil_image.convert("RGB"))
    arr = random_rotate(arr)
    if random.random() < 0.5:
        arr = elastic_distort(arr)
    return Image.fromarray(arr)


class LineOCRDataset(Dataset):
    """Reads (image_path, text) pairs from a CSV for TrOCR fine-tuning."""

    def __init__(self, csv_path, processor, image_root=None, max_target_length=128, train=True):
        self.processor = processor
        self.max_target_length = max_target_length
        self.train = train
        self.image_root = image_root or os.path.dirname(os.path.abspath(csv_path))

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "image_path" not in reader.fieldnames or "text" not in reader.fieldnames:
                raise ValueError(
                    f"{csv_path} must have a header row with columns 'image_path,text' "
                    f"(found: {reader.fieldnames})"
                )
            self.rows = list(reader)

        if not self.rows:
            raise ValueError(f"{csv_path} has no data rows.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_path = row["image_path"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.image_root, image_path)

        image = Image.open(image_path).convert("RGB")
        if self.train:
            image = augment(image)

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            row["text"], padding="max_length", max_length=self.max_target_length, truncation=True
        ).input_ids
        labels = [l if l != self.processor.tokenizer.pad_token_id else -100 for l in labels]

        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


def freeze_layers(model, freeze_encoder_layers, freeze_decoder_layers):
    """
    Freeze only the early layers of each stack -- research found the ViT
    encoder degrades quickly if frozen past ~layer 3, while the RoBERTa-style
    decoder tolerates freezing up to ~layer 6. Leaving the rest trainable
    keeps fine-tuning capacity where it matters most.
    """
    if freeze_encoder_layers > 0:
        for param in model.encoder.embeddings.parameters():
            param.requires_grad = False
        for layer in model.encoder.layers[:freeze_encoder_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    if freeze_decoder_layers > 0:
        decoder_layers = model.decoder.model.decoder.layers
        for layer in decoder_layers[:freeze_decoder_layers]:
            for param in layer.parameters():
                param.requires_grad = False


def build_compute_metrics(processor):
    def compute_metrics(eval_pred):
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids

        # Both label_ids and pred_ids can be padded with -100/-1 when batch
        # elements have different generated lengths -- replace with pad_token_id
        # before decoding, or the tokenizer chokes on negative ids.
        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
        pred_ids = np.where(pred_ids >= 0, pred_ids, processor.tokenizer.pad_token_id)
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = [p if p.strip() else " " for p in pred_str]
        label_str = [l if l.strip() else " " for l in label_str]

        cer = jiwer.cer(label_str, pred_str)
        wer = jiwer.wer(label_str, pred_str)
        return {"cer": cer, "wer": wer}

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-csv", required=True, help="Training CSV with columns image_path,text")
    parser.add_argument("--eval-csv", default=None, help="Held-out (ideally cross-document) eval CSV, same format")
    parser.add_argument("--image-root", default=None, help="Base dir for relative image_path entries (default: CSV's dir)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--freeze-encoder-layers", type=int, default=3)
    parser.add_argument("--freeze-decoder-layers", type=int, default=6)
    args = parser.parse_args()

    print(f"Loading base model {args.base_model} ...")
    image_processor = ViTImageProcessor.from_pretrained(args.base_model)
    tokenizer = RobertaTokenizer.from_pretrained(args.base_model)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = VisionEncoderDecoderModel.from_pretrained(args.base_model)

    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    freeze_layers(model, args.freeze_encoder_layers, args.freeze_decoder_layers)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    train_dataset = LineOCRDataset(args.data_csv, processor, image_root=args.image_root, train=True)
    eval_dataset = (
        LineOCRDataset(args.eval_csv, processor, image_root=args.image_root, train=False)
        if args.eval_csv
        else None
    )
    print(f"Train lines: {len(train_dataset)}" + (f" | Eval lines: {len(eval_dataset)}" if eval_dataset else " | No eval set provided"))

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        predict_with_generate=True,
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        logging_steps=10,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics(processor) if eval_dataset else None,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Fine-tuned model saved to {args.output_dir}")

    if eval_dataset:
        metrics = trainer.evaluate()
        print(f"Final eval CER: {metrics.get('eval_cer'):.4f}  WER: {metrics.get('eval_wer'):.4f}")


if __name__ == "__main__":
    main()
