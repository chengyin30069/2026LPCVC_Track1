import torch
from torch import nn

from PIL import Image
import requests
import os

from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Data

def process_image(image_path, target_size=(224, 224)):
    """Loads and processes an image to the required input shape (C, H, W)."""
    image = Image.open(image_path).convert('RGB').resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize
    return np.transpose(image_array, (2, 0, 1))[np.newaxis, :]  # Convert to (1, C, H, W)

def load_images_from_folder(folder_path, target_size=(224, 224)):
    """Loads and processes all images in a folder, sorted by name."""
    image_paths = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))
    ])
    return [process_image(path, target_size) for path in image_paths]

def get_data(csv_path, image_folder):
    input_image = load_images_from_folder(image_folder)
    df = pd.read_csv(csv_path)
    prompts = df.iloc[:, 1].dropna().tolist()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    tokenized_texts = []
    for prompt in prompts:
        tokens = tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )["input_ids"].to(torch.int32)
        tokenized_texts.append(tokens.squeeze(0).numpy())
    
    tokenized_texts = np.array(tokenized_texts, dtype=np.int32)   # [N, 77]
    tokenized_texts = torch.from_numpy(tokenized_texts)           # int32 tensor

    input_image = np.concatenate(input_image, axis=0)             # [N, 3, 224, 224]
    input_image = torch.from_numpy(input_image).to(torch.float32)

    return tokenized_texts, input_image

# Evaluate

def parse_ground_truth(txt_list, img_list):
    # Load your CSV
    df_img = pd.read_csv(img_list)
    df_txt = pd.read_csv(txt_list)

    df_img = df_img.sort_values(by='Image_names')

    # Get unique text prompts in order from the second column
    txt_id = df_txt.iloc[:, 0].dropna().astype(np.int16).tolist()
    gt = df_img.iloc[:, 1].dropna().tolist() # list of txt id for each image
    return txt_id, gt

def evaluate_track1(img_output, txt_output, txt_list, img_list, k=10):
    """
    Compute Recall@K between image and text embeddings.

    Args:
        img_output (np.ndarray): Image encoder output, shape (N, D)
        txt_output (np.ndarray): Text encoder output, shape (M, D)
        ground_truth_dir (str): Path to ground truth JSON file
        k (int): Top-K for recall computation

    Returns:
        float: Mean recall@K (accuracy)
    """

    # Stack them into a single 2D array: [batch, D]
    img_embeds = np.vstack([x for x in img_output])  # shape: [N, D]
    txt_embeds = np.vstack([x for x in txt_output])  # shape: [M, D]

    # Normalize
    img_embeds = img_embeds / np.linalg.norm(img_embeds, axis=1, keepdims=True)
    txt_embeds = txt_embeds / np.linalg.norm(txt_embeds, axis=1, keepdims=True)

    # Now similarity will work
    sim_matrix = cosine_similarity(img_embeds, txt_embeds)

    # Load ground truth
    txt_id, gt = parse_ground_truth(txt_list, img_list)

    recalls = []

    # print(len(img_embeds))

    for i in range(len(img_embeds)):

        # print(txt_id[i])
        # print(gt[i])
        gt_ids = [int(x) for x in gt[i].split(';')]

        # Top-K text indices by similarity
        k = 10
        # Top-K text indices by similarity
        top_k = np.argsort(-sim_matrix[i])[:k]

        # Map to real text IDs
        predicted_txt_ids = [txt_id[idx] for idx in top_k]

        # Fractional recall: how many GTs are in top-K
        # print(predicted_txt_ids)
        # print(gt_ids)
        matched = len(set(predicted_txt_ids) & set(gt_ids))
        recall_i = matched / len(gt_ids)
        # print(recall_i)
        recalls.append(recall_i)

    return np.mean(recalls)

# Model

class TextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text_model = model.text_model
        self.text_projection = model.text_projection
    
    def forward(self, x):
        output = self.text_model(x)
        x = output.pooler_output
        x = self.text_projection(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x

class ImageEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision_model = model.vision_model
        self.visual_projection = model.visual_projection
    
    def forward(self, x):
        output = self.vision_model(x)
        x = output.pooler_output
        x = self.visual_projection(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x
    
if __name__ == '__main__':
    
    model_id = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
    # model_id = "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)

    text_encoder = TextEncoder(model).eval()
    image_encoder = ImageEncoder(model).eval()

    csv_path = 'dataset/txt_list.csv'
    image_folder = 'dataset/images'
    text_input, image_input = get_data(csv_path, image_folder)

    with torch.no_grad():
        text_output = text_encoder(text_input)
        image_output = image_encoder(image_input)


    text_output = text_output.detach().numpy()
    image_output = image_output.detach().numpy()

    txt_list = 'dataset/txt_list.csv'
    img_list = 'dataset/img_list.csv'

    result = evaluate_track1(image_output, text_output, txt_list, img_list)
    
    print(result)
    # 0.8922619047619048