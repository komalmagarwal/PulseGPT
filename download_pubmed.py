from datasets import load_dataset
import os

os.makedirs("data/pubmed", exist_ok=True)
total = 0

# Dataset 1 - PubMedQA
print("Downloading PubMedQA...")
ds1 = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
print(f"Got {len(ds1)} examples")
with open("data/pubmed/input.txt", "w") as f:
    for item in ds1:
        context = " ".join(item["context"]["contexts"])
        f.write(f"Question: {item['question']}\nContext: {context}\nAnswer: {item['long_answer']}\n\n")
        total += 1

# Dataset 2 - Medical meadow flashcards
print("Downloading medical flashcards...")
ds2 = load_dataset("medalpaca/medical_meadow_flashcards", split="train")
print(f"Got {len(ds2)} examples")
with open("data/pubmed/input.txt", "a") as f:
    for item in ds2:
        f.write(f"Question: {item['input']}\nAnswer: {item['output']}\n\n")
        total += 1

# Dataset 3 - Medical meadow wikidoc
print("Downloading medical wikidoc...")
ds3 = load_dataset("medalpaca/medical_meadow_wikidoc", split="train")
print(f"Got {len(ds3)} examples")
with open("data/pubmed/input.txt", "a") as f:
    for item in ds3:
        f.write(f"Question: {item['input']}\nAnswer: {item['output']}\n\n")
        total += 1

print(f"\nDone! Total examples: {total}")
print("Saved to data/pubmed/input.txt")

import os
size_mb = os.path.getsize("data/pubmed/input.txt") / 1024 / 1024
print(f"File size: {size_mb:.1f} MB")
