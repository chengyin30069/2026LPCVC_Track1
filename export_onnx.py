import torch
import os
from qai_hub_models.models.openai_clip.model import OpenAIClip
import open_clip
from mobileclip.modules.common.mobileone import reparameterize_model
from PIL import Image


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
DUMMY_IMAGE_INPUT = torch.rand(1, 3, 256, 256, dtype=torch.float32, device=device)
DUMMY_TEXT_INPUT = torch.randint(0, 49408, (1, 77), dtype=torch.int32, device=device)

# -----------------------------
# 3. Load OpenAIClip wrapper and define encoders
# -----------------------------
print("Loading OpenAIClip wrapper model...")

model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP2-S0', pretrained='./mobileclip2_s0.pt')
model.to(device)
tokenizer = open_clip.get_tokenizer('MobileCLIP2-S0')
model.eval()
model = reparameterize_model(model)
clip_model = model.to(torch.float32) # convert all model params to float32 type, consistent with input type in compiling and profiling via AIHub
clip_model.eval()

# clip_wrapper_model = OpenAIClip.from_pretrained().to(device)
# clip_wrapper_model.eval()

# clip_model = clip_wrapper_model.clip.to(device)
# clip_model = clip_model.to(torch.float32) # convert all model params to float32 type, consistent with input type in compiling and profiling via AIHub
# clip_model.eval()

class ImageEncoderWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.preprocess = preprocess
        self.visual = clip_model.visual

    def forward(self, images):
        images = self.preprocess(images)
        
        return self.visual(images)

class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.text = clip_model.text

    def forward(self, token_ids):
        is_eot = (token_ids == 49407)
        eot_cumsum = torch.cumsum(is_eot.long(), dim=1)
        is_extra_padding = is_eot & (eot_cumsum > 1)
        corrected_tokens = torch.where(
            is_extra_padding,
            torch.tensor(0, device=token_ids.device, dtype=token_ids.dtype),
            token_ids
        )
        x = self.text(corrected_tokens)
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