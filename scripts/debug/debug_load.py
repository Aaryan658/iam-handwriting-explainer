from transformers import TrOCRProcessor, RobertaTokenizer

print("Trying with use_fast=False")
try:
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten", use_fast=False)
    print("Success without tokenizer kwarg, use_fast=False!")
except Exception as e:
    print(e)

print("Trying with tokenizer=tokenizer and use_fast=False")
try:
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/trocr-base-handwritten")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten", tokenizer=tokenizer, use_fast=False)
    print("Success with tokenizer kwarg and use_fast=False!")
except Exception as e:
    print(e)
