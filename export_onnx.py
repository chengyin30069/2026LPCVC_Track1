import torch
import os
from transformers import BlipModel

# --- Configuration for File Saving ---
ONNX_DIR = "exported_onnx"
BLIP_MODEL_NAME = "Salesforce/blip-itm-base-coco"
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
DUMMY_TEXT_INPUT = torch.randint(0, 49408, (1, 77), dtype=torch.int64, device=device)
DUMMY_ATTENTION_MASK = torch.ones((1, 77), dtype=torch.int64, device=device)

# -----------------------------
# 3. Load BLIP wrapper and define encoders
# -----------------------------
print(f"Loading BLIP model: {BLIP_MODEL_NAME}...")
blip_model = BlipModel.from_pretrained(BLIP_MODEL_NAME).to(device)
blip_model = blip_model.to(torch.float32)
blip_model.eval()

class ImageEncoderWrapper(torch.nn.Module):
    def __init__(self, blip_model):
        super().__init__()
        self.blip = blip_model

    def forward(self, images):
        return self.blip.get_image_features(pixel_values=images)

class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, blip_model):
        super().__init__()
        self.blip = blip_model
        self.vocab_size = blip_model.text_model.config.vocab_size

    def forward(self, token_ids, attention_mask):
        mapped_token_ids = torch.remainder(token_ids.to(torch.int64), self.vocab_size)
        mapped_attention_mask = attention_mask.to(torch.int64)
        return self.blip.get_text_features(
            input_ids=mapped_token_ids,
            attention_mask=mapped_attention_mask,
        )

# -----------------------------
# 4. Create wrapper instances
# -----------------------------
image_encoder = ImageEncoderWrapper(blip_model)
text_encoder = TextEncoderWrapper(blip_model)
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
    (DUMMY_TEXT_INPUT, DUMMY_ATTENTION_MASK),
    text_onnx_path,
    input_names=["text", "attention_mask"],
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
