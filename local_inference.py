import torch
import open_clip
from PIL import Image
import pandas as pd
import numpy as np
import os
from mobileclip.modules.common.mobileone import reparameterize_model
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPTokenizer
import torch
import open_clip




def parse_ground_truth(txt_list, img_list):
    # Load your CSV
    df_img = pd.read_csv(img_list)
    df_txt = pd.read_csv(txt_list)

    # Get unique text prompts in order from the second column
    txt_id = df_txt.iloc[:, 0].dropna().astype(np.int16).tolist()
    gt = df_img.iloc[:, 1].dropna().tolist() # list of txt id for each image
    return txt_id, gt

def process_image(image_path, target_size=(224, 224)):
    """Loads and processes an image to the required input shape (C, H, W)."""
    # image = Image.open(image_path)  # shape: (1, C, H, W)
    # # image = preprocess(image).unsqueeze(0)  # add batch dimension, shape: (1, C, H, W)
    # return image.numpy()  # numpy array, shape: (1, C, H, W)
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

def debug_one_sample(img_output, txt_output, txt_list, img_list, sample_i=0, k=10):
    df_img = pd.read_csv(img_list)
    df_txt = pd.read_csv(txt_list)
    df_txt = df_txt.iloc[:, :2].dropna()
    txt_ids = df_txt.iloc[:, 0].astype(np.int64).tolist()
    txt_prompts = df_txt.iloc[:, 1].tolist()

    img_embeds = np.vstack(img_output)
    txt_embeds = np.vstack(txt_output)

    img_embeds = img_embeds / np.clip(np.linalg.norm(img_embeds, axis=1, keepdims=True), 1e-12, None)
    txt_embeds = txt_embeds / np.clip(np.linalg.norm(txt_embeds, axis=1, keepdims=True), 1e-12, None)
    sim = img_embeds @ txt_embeds.T

    top_k = np.argsort(-sim[sample_i])[:k]
    pred_ids = [txt_ids[j] for j in top_k]

    gt_raw = df_img.iloc[sample_i, 1]
    gt_ids = [int(x) for x in str(gt_raw).split(';') if str(x).strip()]

    print("sample_i:", sample_i)
    print("GT ids:", gt_ids)
    print("Pred top-k ids:", pred_ids)
    print("Matched:", set(pred_ids) & set(gt_ids))
    print("\nTop-k prompts:")
    for rank, j in enumerate(top_k, 1):
        print(f"{rank:2d}. txt_id={txt_ids[j]}, sim={sim[sample_i, j]:.4f}, prompt={txt_prompts[j]}")

# model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP2-S0', pretrained='./mobileclip2_s0.pt')
# tokenizer = open_clip.get_tokenizer('MobileCLIP2-S0')

# # Model needs to be in eval mode for inference because of batchnorm layers unlike ViTs
# model.eval()

# # For inference/model exporting purposes, please reparameterize first
# model = reparameterize_model(model)

# ============================================================

device = torch.device("cpu")


model_name = "MobileCLIP2-S0"
model_kwargs = {}
if not (model_name.endswith("S3") or model_name.endswith("S4") or model_name.endswith("L-14")):
    model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="./mobileclip2_s0.pt", **model_kwargs)
tokenizer = open_clip.get_tokenizer(model_name)

# Model needs to be in eval mode for inference because of batchnorm layers unlike ViTs
model.eval()

# For inference/model exporting purposes, please reparameterize first
model = reparameterize_model(model)

clip_model = model
clip_model = clip_model.to(device=device, dtype=torch.float32) # convert all model params to float32 type, consistent with input type in compiling and profiling via AIHub
clip_model.eval()

class ImageEncoderWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, images):
        return self.visual(images)

class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.text = clip_model.text
        # self.token_embedding = clip_model.token_embedding
        # self.positional_embedding = clip_model.positional_embedding
        # self.transformer = clip_model.transformer
        # self.ln_final = clip_model.ln_final
        # self.text_projection = clip_model.text_projection

    def forward(self, token_ids):
        is_eot = (token_ids == 49407)
        eot_cumsum = torch.cumsum(is_eot.long(), dim=1)
        is_extra_padding = is_eot & (eot_cumsum > 1)
        corrected_tokens = torch.where(
            is_extra_padding,
            torch.zeros(77, device=token_ids.device, dtype=token_ids.dtype),
            token_ids
        )
        x = self.text(corrected_tokens)
        # x = self.token_embedding(token_ids)
        # x = x + self.positional_embedding
        # x = x.permute(1, 0, 2)
        # x = self.transformer(x)
        # x = x.permute(1, 0, 2)
        # x = self.ln_final(x)
        # eos_index = token_ids.argmax(dim=-1)
        # x = x[torch.arange(x.shape[0]), eos_index]
        # x = x @ self.text_projection
        return x

# -----------------------------
# 4. Create wrapper instances
# -----------------------------
image_encoder = ImageEncoderWrapper(clip_model)
text_encoder = TextEncoderWrapper(clip_model)
image_encoder.eval()
text_encoder.eval()

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
mobileclip_tokenizer = open_clip.get_tokenizer(model_name)

# 建立映射表：mapping[openai_id] = mobileclip_id
mapping = torch.zeros(49408, dtype=torch.long)

# 遍歷 OpenAI 的所有詞位 (0 到 49407)
# 注意：這裡需要取得 OpenAI 的原始詞表文字
# openai_decoder = {v: k for k, v in tokenizer.encoder.items()}
# print(openai_decoder)
# mobile_decoder = {v: k for k, v in mobileclip_tokenizer.encoder.items()}
# print(mobile_decoder)
# for i in range(49408):
#     openai_word = openai_decoder[i]
#     mobile_word = mobile_decoder[i]
#     if(openai_word != mobile_word):
#         print(f"{i} : {openai_word} != {mobile_word}")
    # 使用 MobileCLIP 重新編碼（需處理特殊符號與空格）
    # 這裡的邏輯需根據 MobileCLIP 實作微調
    # mobile_id = mobileclip_tokenizer(word)[0, 1] # 通常 [0,0] 是 Start Token, [0,1] 是內容
    # mapping[i] = mobile_id

# print(mobileclip_tokenizer.vocab_size)

# ============================================================

image_folder = "dataset/images"  # change to your folder

input_image = load_images_from_folder(image_folder)
print(len(input_image))

csv_path = "dataset/txt_list.csv"
df = pd.read_csv(csv_path)

prompts = df.iloc[:, 1].dropna().tolist()
tokenized_texts = []
for prompt in prompts:

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77,   # CLIP 常用 context length
    )
    input_ids = enc["input_ids"]            # shape [1, 77], torch.Tensor
    input_ids = input_ids.to(torch.long)    # 建議先用 long；若模型要求 int32 再改
    # print(enc)
    tokenized_texts.append(input_ids.numpy())
    mobile_enc = mobileclip_tokenizer(
        prompt,
        # return_tensors="pt",
        # padding="max_length",
        # truncation=True,
        # max_length=77,   # CLIP 常用 context length
    )


txt_id, gt = parse_ground_truth("dataset/txt_list.csv", "dataset/img_list.csv")

output_image = []
for img in input_image:
    with torch.no_grad():
        img_tensor = torch.from_numpy(img).to(torch.float32)
        image_features = image_encoder(img_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        output_image.append(image_features.cpu().numpy())

output_text = []
for text in tokenized_texts:
    with torch.no_grad():
        text_tensor = torch.from_numpy(text).to(torch.int32)
        text_features = text_encoder(text_tensor)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        output_text.append(text_features.cpu().numpy())

result = evaluate_track1(output_image, output_text, "dataset/txt_list.csv", "dataset/img_list.csv", k=10)

debug_one_sample(output_image, output_text, "dataset/txt_list.csv", "dataset/img_list.csv", sample_i=0, k=10)

print(result)