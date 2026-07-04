import os
import jiwer
from PIL import Image
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
                
        # 2. Run Florence-2 OCR
        print("Running Florence-2...")
        raw_ocr = app.transcribe(img)
        print(f"[RAW OCR]: {raw_ocr}")
        
        # 3. Run space correction
        corrected_ocr = app.correct_spaces(raw_ocr)
        print(f"[SPACE CORRECTED]: {corrected_ocr}")
        
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
