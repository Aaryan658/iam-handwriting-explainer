import os
import httpx
import torch
from transformers import AutoProcessor, VisionEncoderDecoderModel, logging
from PIL import Image
import warnings

# Suppress warnings for clean output
logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# Monkey-patch httpx to disable SSL verify for the Hugging Face hub on Windows
old_init = httpx.Client.__init__
def new_init(self, *args, **kwargs):
    kwargs["verify"] = False
    old_init(self, *args, **kwargs)
httpx.Client.__init__ = new_init

def main():
    words_file = os.path.join("data", "words.txt")
    groups = {}

    print("Parsing words.txt...")
    # 1. Parse words.txt
    with open(words_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.strip().split()
            if len(parts) >= 9:
                word_id = parts[0]
                status = parts[1]
                
                # Filter to rows where status == "ok" only
                if status != "ok":
                    continue
                
                transcription = " ".join(parts[8:]) 
                
                # wordID format: formID-lineNum-wordPos (e.g. a01-000u-00-00)
                id_parts = word_id.split('-')
                if len(id_parts) >= 4:
                    form_id = f"{id_parts[0]}-{id_parts[1]}"
                    line_num = id_parts[2]
                    word_pos = int(id_parts[3])
                    
                    # 2. Group entries by form+line prefix
                    line_prefix = f"{form_id}-{line_num}"
                    
                    if line_prefix not in groups:
                        groups[line_prefix] = []
                    
                    groups[line_prefix].append({
                        "word_id": word_id,
                        "form_id": form_id,
                        "word_pos": word_pos,
                        "transcription": transcription
                    })

    # Pick 3 different line groups to reconstruct
    selected_groups = []
    for prefix, words in groups.items():
        if len(words) >= 5: # Pick lines that have at least 5 words for a good sentence
            selected_groups.append(prefix)
            if len(selected_groups) == 3:
                break
                
    if len(selected_groups) < 3:
        print("Could not find 3 suitable line groups.")
        return

    print("Loading TrOCR processor and model...")
    processor = AutoProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    
    # Process the 3 selected line groups
    for prefix in selected_groups:
        words = groups[prefix]
        
        # 3. Within each group, sort by word position number
        words.sort(key=lambda x: x['word_pos'])
        
        gt_words = []
        pred_words = []
        
        for w in words:
            gt_words.append(w['transcription'])
            
            # Construct image path: data/words/a01/a01-000u/a01-000u-00-00.png
            dir1 = w['form_id'].split('-')[0]
            dir2 = w['form_id']
            img_path = os.path.join("data", "words", dir1, dir2, w['word_id'] + ".png")
            
            try:
                # 4. Load each word's image file and run TrOCR
                image = Image.open(img_path).convert("RGB")
                pixel_values = processor(image, return_tensors="pt").pixel_values
                generated_ids = model.generate(pixel_values, max_new_tokens=21)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                pred_words.append(generated_text)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                pred_words.append("[ERROR]")
                
        # 4 & 5. Join words to form reconstructed and reference sentences
        gt_sentence = " ".join(gt_words)
        pred_sentence = " ".join(pred_words)
        
        print(f"\nDiagnostic: len(gt_words)={len(gt_words)}, len(pred_words)={len(pred_words)}")
        if prefix == "a01-000u-01":
            print(f"Raw GT List  : {gt_words}")
            print(f"Raw Pred List: {pred_words}")
        
        # 6. Compute word-level accuracy (case-insensitive match count / total words)
        matches = 0
        for gw, pw in zip(gt_words, pred_words):
            # Evaluate using case-insensitive string matching
            if gw.lower() == pw.lower():
                matches += 1
        accuracy = (matches / len(gt_words)) * 100 if gt_words else 0.0
        
        print(f"\n--- Line Group: {prefix} ---")
        print(f"Ground Truth : {gt_sentence}")
        print(f"Prediction   : {pred_sentence}")
        print(f"Accuracy     : {matches}/{len(gt_words)} ({accuracy:.1f}%)")

if __name__ == "__main__":
    main()
