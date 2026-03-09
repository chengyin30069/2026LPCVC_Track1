import pandas as pd
from transformers import CLIPTokenizer, AutoTokenizer
import torch
import numpy as np  # import numpy

BLIP_MODEL_NAME = "Salesforce/blip-itm-base-coco"

# Load your CSV
csv_path = ""  # prompt list file path
df = pd.read_csv(csv_path)

# Get unique text prompts in order from the second column, drop NaN
prompts = df.iloc[:, 1].dropna().tolist()

# Load tokenizers
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
blip_tokenizer = AutoTokenizer.from_pretrained(BLIP_MODEL_NAME)

# Map CLIP tokens to BLIP ids by decode + retokenize.
tokenized_texts = []
attention_masks = []
for prompt in prompts:
    clip_encoded = clip_tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt"
    )

    recovered_text = clip_tokenizer.decode(
        clip_encoded["input_ids"][0],
        skip_special_tokens=True,
    )
    blip_encoded = blip_tokenizer(
        recovered_text,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt"
    )

    tokens = blip_encoded["input_ids"].to(torch.int32)
    mask = blip_encoded["attention_mask"].to(torch.int32)
    tokenized_texts.append(tokens.numpy())
    attention_masks.append(mask.numpy())

# Example: check first element
print(tokenized_texts[0].shape)  # (1, 77)
print(tokenized_texts[0].dtype)  # int32
print(attention_masks[0].dtype)  # int32

# Optional: check total number of prompts
print(len(tokenized_texts))  # batch size