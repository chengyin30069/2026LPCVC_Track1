import torch
import os
from qai_hub_models.models.openai_clip.model import OpenAIClip
import torch.nn as nn

from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel

# --- Configuration for File Saving ---
ONNX_DIR = "exported_onnx"
device = torch.device("cpu") # use CPU to export onnx model to avoid GPU device issues
# -----------------------------------

# -----------------------------
# 1. Prepare Environment
# -----------------------------
os.makedirs(ONNX_DIR, exist_ok=True)
print(f"Saving ONNX files to directory: {os.path.abspath(ONNX_DIR)}")

# -----------------------------
# 2. Dummy inputs
# -----------------------------
DUMMY_IMAGE_INPUT = torch.rand(1, 3, 224, 224, dtype=torch.float32, device=device)
DUMMY_TEXT_INPUT = torch.randint(0, 49408, (1, 77), dtype=torch.int32, device=device)

# -----------------------------
# 3. Load OpenAIClip wrapper and define encoders
# -----------------------------
print("Loading OpenAIClip wrapper model...")


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

# -----------------------------
# 4. Create wrapper instances
# -----------------------------


model = CLIPModel.from_pretrained("wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M")

image_encoder = ImageEncoder(model).eval()
text_encoder = TextEncoder(model).eval()
image_encoder.eval()
text_encoder.eval()


# -----------------------------
# 5. Export Image Encoder
# -----------------------------
image_onnx_path = os.path.join(ONNX_DIR, "image_encoder.onnx")
print(f"\nExporting Image Encoder to {image_onnx_path}...")

torch.onnx.export(
    image_encoder,
    DUMMY_IMAGE_INPUT,
    image_onnx_path,
    input_names=["image"],
    output_names=["embedding"],
    opset_version=18,
    do_constant_folding=True,
    dynamic_axes=None,
    verbose=False,
    export_params=True,
    training=torch.onnx.TrainingMode.EVAL,
    dynamo=True,
)

# -----------------------------
# 6. Export Text Encoder
# -----------------------------
text_onnx_path = os.path.join(ONNX_DIR, "text_encoder.onnx")
print(f"\nExporting Text Encoder to {text_onnx_path}...")

torch.onnx.export(
    text_encoder,
    DUMMY_TEXT_INPUT,
    text_onnx_path,
    input_names=["text"],
    output_names=["text_embedding"],
    opset_version=18,
    do_constant_folding=True,
    dynamic_axes=None,
    verbose=False,
    export_params=True,
    training=torch.onnx.TrainingMode.EVAL,
    dynamo=True,
)

print("\nExport complete.")
