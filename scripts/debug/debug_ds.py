from datasets import load_dataset
ds = load_dataset("Teklia/IAM-line", split="train", streaming=True)
count = 0
for item in ds:
    print(item.keys())
    print(item['text'])
    count += 1
    if count >= 3:
        break
