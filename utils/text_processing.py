import pandas as pd
from transformers import CLIPTokenizer
import torch
import numpy as np  # import numpy

# Load your CSV
csv_path = ""  # prompt list file path
df = pd.read_csv(csv_path)

# Get unique text prompts in order from the second column, drop NaN
prompts = df.iloc[:, 1].dropna().tolist()

# Load CLIP tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Tokenize prompts into numpy arrays of shape (1, 77) and dtype int64
tokenized_texts = []
attention_masks = []
for prompt in prompts:
    encoded = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt"
    )
    tokens = encoded["input_ids"].to(torch.int64)
    mask = encoded["attention_mask"].to(torch.int64)
    tokenized_texts.append(tokens.numpy())
    attention_masks.append(mask.numpy())

# Example: check first element
print(tokenized_texts[0].shape)  # (1, 77)
print(tokenized_texts[0].dtype)  # int64
print(attention_masks[0].dtype)  # int64

# Optional: check total number of prompts
print(len(tokenized_texts))  # batch size