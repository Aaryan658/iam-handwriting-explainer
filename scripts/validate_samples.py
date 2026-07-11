import os
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass
import jiwer
from PIL import Image

# app.py lives at the repo root, one level up from scripts/ -- add it to
# sys.path so this bare import keeps resolving after the move.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app

# Define the samples to validate
SAMPLES = [
    "samples/line_01.png",
    "samples/line_02.png",
    "samples/line_03.png",
    "samples/line_05.png",
    "samples/education_paragraph.png" # Full multi-line paragraph
]

def main():
    print("=== Starting Validation on 5 Samples ===")
    
    for sample_path in SAMPLES:
        if not os.path.exists(sample_path):
            print(f"Sample not found: {sample_path}")
            continue
            
        print(f"\n--- Validating: {sample_path} ---")
        
        # 1. Load image and Ground Truth
        img = Image.open(sample_path).convert("RGB")
        txt_path = sample_path.replace(".png", ".txt")
        ground_truth = ""
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                ground_truth = f.read().strip()
                
        # 2. Run TrOCR
        print("Running TrOCR...")
        raw_ocr = app.transcribe(img)
        print(f"[RAW OCR]: {raw_ocr}")
        
        # Space correction is deprecated (reverted wordninja), using raw ocr directly
        corrected_ocr = raw_ocr
        
        # 4. Print Ground Truth and calculate WER
        if ground_truth:
            print(f"[GROUND TRUTH]: {ground_truth}")
            # Compute Word Level Accuracy (1 - WER)
            try:
                error = jiwer.wer(ground_truth.lower(), corrected_ocr.lower())
                accuracy = max(0.0, 1.0 - error)
                print(f"[WORD-LEVEL ACCURACY]: {accuracy:.2%}")
            except Exception as e:
                print(f"[WER CALCULATION ERROR]: {e}")
        else:
            print("[GROUND TRUTH]: None provided for this sample.")
            
        # 5. Run Groq Explanation (includes the substring check internally now)
        print("Running Groq explanation + Token overlap check...")
        explanation_result = app.explain(raw_ocr)
        print("\n[GROQ RESULT]:\n" + explanation_result)

if __name__ == "__main__":
    main()
