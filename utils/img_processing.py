import os
import numpy as np
import qai_hub
from PIL import Image

BLIP_IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
BLIP_IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

def process_image(image_path, target_size=(224, 224)):
    """Loads and processes an image to the required input shape (C, H, W)."""
    image = Image.open(image_path).convert('RGB').resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = (image_array - BLIP_IMAGE_MEAN) / BLIP_IMAGE_STD
    return np.transpose(image_array, (2, 0, 1))[np.newaxis, :]  # Convert to (1, C, H, W)

def load_images_from_folder(folder_path, target_size=(224, 224)):
    """Loads and processes all images in a folder, sorted by name."""
    image_paths = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))
    ])
    return [process_image(path, target_size) for path in image_paths]

# TODO: Define image folder path
image_folder = ""

# Process images
input_data = load_images_from_folder(image_folder)
print(len(input_data))

# Check dataset properties
if input_data:
    print(f"Processed {len(input_data)} images.")
    print(f"First image shape: {input_data[0].shape}")  # Should be (1, 3, 224, 224)

# Upload dataset
print(qai_hub.upload_dataset({"image": input_data}))