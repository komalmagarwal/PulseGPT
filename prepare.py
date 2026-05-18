import os
import numpy as np
import tiktoken

# Load the text
print("Loading text...")
with open("data/pubmed/input.txt", "r") as f:
    data = f.read()

print(f"Total characters: {len(data):,}")

# Tokenize
print("Tokenizing...")
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode_ordinary(data)
print(f"Total tokens: {len(tokens):,}")

# Split 90% train, 10% validation
split = int(0.9 * len(tokens))
train_tokens = tokens[:split]
val_tokens = tokens[split:]
print(f"Train tokens: {len(train_tokens):,}")
print(f"Val tokens:   {len(val_tokens):,}")

# Save as binary files
train_arr = np.array(train_tokens, dtype=np.uint16)
val_arr = np.array(val_tokens, dtype=np.uint16)
train_arr.tofile("data/pubmed/train.bin")
val_arr.tofile("data/pubmed/val.bin")

print("Done! Saved train.bin and val.bin")
