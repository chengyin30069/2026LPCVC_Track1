from aimet_onnx import QuantizationSimModel, apply_seq_mse
import onnx
import torch
import numpy as np
from PIL import Image
import os
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

IMAGE_DATASET_DIR = "/home/yyh603/lpcv/coco2017/train2017"
TEST_IMAGES_DIR = "/home/yyh603/lpcv/dataset/images"

MODEL_PATH = 'exported_onnx/image_encoder.onnx'

def process_image(image_path, target_size=(224, 224)):
    """Loads and processes an image to the required input shape (C, H, W)."""
    image = Image.open(image_path).convert('RGB').resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize
    return np.transpose(image_array, (2, 0, 1))[np.newaxis, :]  # Convert to (1, C, H, W)

def load_calibration_images(dataset_dir, max_samples=None):
    """Loads calibration images from the dataset directory."""
    image_paths = sorted([
        os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)
        if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))
    ])
    
    if max_samples:
        image_paths = image_paths[:max_samples]
    
    return [process_image(path) for path in image_paths]

model = onnx.load(MODEL_PATH)
sim = QuantizationSimModel(model,
                            param_type="int8",
                            activation_type="int16",
                            config_file='htp_v69')

device = torch.device("cpu")

# Load calibration dataset
calibration_images = load_calibration_images(IMAGE_DATASET_DIR, max_samples=1)  # Use first 100 images for calibration

# Load calibration dataset as an iterable
def calibration_data_generator():
    for image in calibration_images:
        yield {"image": image}
        # yield {"image_tensor": image}

calibration_dataset = calibration_data_generator()

# Create calibration data list
calibration_data = [{"image": img} for img in calibration_images]
# calibration_data = [{"image_tensor": img} for img in calibration_images]

# apply_seq_mse(sim, calibration_data)

sim.compute_encodings(calibration_data)

os.makedirs('exported_onnx/image_encoder.aimet', exist_ok=True)
sim.export('exported_onnx/image_encoder.aimet', filename_prefix="image_encoder")
onnx_qdq = sim.to_onnx_qdq()
onnx.save(onnx_qdq, 'exported_onnx/image_encoder_qdq.onnx')

def compute_mse(original_output, quantized_output):
    """Compute Mean Squared Error between original and quantized outputs."""
    return np.mean((original_output - quantized_output) ** 2)

def compute_cosine_similarity(original_output, quantized_output):
    """Compute cosine similarity between original and quantized outputs."""
    # Flatten the outputs for cosine similarity computation
    orig_flat = original_output.flatten().reshape(1, -1)
    quant_flat = quantized_output.flatten().reshape(1, -1)
    return cosine_similarity(orig_flat, quant_flat)[0][0]

# Load test images (different from calibration images)
test_images = load_calibration_images(TEST_IMAGES_DIR, max_samples=2)  # Use 10 test images

# Create ONNX Runtime sessions for both models
original_session = ort.InferenceSession(MODEL_PATH)
qdq_session = ort.InferenceSession("exported_onnx/image_encoder_qdq.onnx")

mse_scores = []
cosine_scores = []

print("Comparing original vs quantized model outputs...")
print("=" * 50)

for i, test_img in enumerate(test_images):
    # Run inference on both models
    original_output = original_session.run(None, {"image": test_img})[0]
    quantized_output = sim.session.run(None, {"image": test_img})[0]
    # original_output = original_session.run(None, {"image_tensor": test_img})[0]
    # quantized_output = qdq_session.run(None, {"image_tensor": test_img})[0]
    
    # Compute metrics
    mse = compute_mse(original_output, quantized_output)
    cosine_sim = compute_cosine_similarity(original_output, quantized_output)
    
    mse_scores.append(mse)
    cosine_scores.append(cosine_sim)
    
    print(f"Test Image {i+1}:")
    print(f"  MSE: {mse:.6f}")
    print(f"  Cosine Similarity: {cosine_sim:.6f}")
    print()

# Print average metrics
avg_mse = np.mean(mse_scores)
avg_cosine = np.mean(cosine_scores)

print("=" * 50)
print("Average Metrics:")
print(f"  MSE: {avg_mse:.6f}")
print(f"  Cosine Similarity: {avg_cosine:.6f}")

text_encoder = onnx.load('exported_onnx/text_encoder.onnx')
text_session = ort.InferenceSession('exported_onnx/text_encoder.onnx')

# ============= Local Inference Test =============

import pandas as pd

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

def load_images_from_folder(folder_path, target_size=(224, 224)):
    """Loads and processes all images in a folder, sorted by name."""
    image_paths = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))
    ])
    return [process_image(path, target_size) for path in image_paths]

input_image = load_images_from_folder('../2026LPCVC_Track1/dataset_sample/images')
print(f"Number of test images: {len(input_image)}")

csv_path = "../2026LPCVC_Track1/dataset_sample/txt_list.csv"
df = pd.read_csv(csv_path)

prompts = df.iloc[:, 1].dropna().tolist()

from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

tokenized_texts = []
for prompt in prompts:
    tokens = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt"
    )["input_ids"].to(torch.int32)  # torch tensor [1, 77], int32
    tokenized_texts.append(tokens.numpy())  # convert to numpy array

print(f"Number of text prompts: {len(tokenized_texts)}")

image_output = []
text_output = []

for img in tqdm(input_image):
    img_out_original = original_session.run(None, {"image": img})[0]
    img_out = qdq_session.run(None, {"image": img})[0]
    print(f"Cosine similarity between original and quantized image encoder outputs: {compute_cosine_similarity(img_out_original, img_out):.6f}")
    image_output.append(img_out)

for txt in tqdm(tokenized_texts):
    txt_out = text_session.run(None, {"text": txt})[0]
    text_output.append(txt_out)

result = evaluate_track1(image_output, text_output, "../2026LPCVC_Track1/dataset_sample/txt_list.csv", "../2026LPCVC_Track1/dataset_sample/img_list.csv")
print(f"Recall@10: {result:.4f}")