import os
import httpx

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# Monkey-patch httpx to disable SSL verify
old_init = httpx.Client.__init__
def new_init(self, *args, **kwargs):
    kwargs["verify"] = False
    old_init(self, *args, **kwargs)
httpx.Client.__init__ = new_init

import torch
from transformers import AutoProcessor, VisionEncoderDecoderModel, logging
from PIL import Image
import warnings

# Suppress huggingface warnings about missing pooler weights
logging.set_verbosity_error()
# Suppress general python warnings (like the generation max_length warning)
warnings.filterwarnings("ignore")

def main():
    print("Loading TrOCR processor and model...")
    processor = AutoProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    # Sample images from the IAM dataset
    image_paths = [
        r"data\words\a01\a01-000u\a01-000u-00-00.png",
        r"data\words\a01\a01-000u\a01-000u-00-01.png",
        r"data\words\a01\a01-000u\a01-000u-00-02.png",
    ]

    print("\nRunning inference on sample images...")
    for path in image_paths:
        try:
            # Load and convert image to RGB as required by TrOCR
            image = Image.open(path).convert("RGB")
            
            # Preprocess the image
            pixel_values = processor(image, return_tensors="pt").pixel_values
            
            # Generate text (set max_new_tokens to avoid the warning)
            generated_ids = model.generate(pixel_values, max_new_tokens=21)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            print(f"Image: {path}\nPredicted Text: '{generated_text}'\n")
        except Exception as e:
            print(f"Error processing {path}: {e}\n")

if __name__ == "__main__":
    main()
