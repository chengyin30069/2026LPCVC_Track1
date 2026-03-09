import torch
import os
from transformers import BlipModel, CLIPTokenizer, AutoTokenizer

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
DUMMY_TEXT_INPUT = torch.randint(0, 49408, (1, 77), dtype=torch.int32, device=device)
DUMMY_ATTENTION_MASK = torch.ones((1, 77), dtype=torch.int32, device=device)

# -----------------------------
# 3. Load BLIP wrapper and define encoders
# -----------------------------
print(f"Loading BLIP model: {BLIP_MODEL_NAME}...")
blip_model = BlipModel.from_pretrained(BLIP_MODEL_NAME).to(device)
blip_model = blip_model.to(torch.float32)
blip_model.eval()
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
blip_tokenizer = AutoTokenizer.from_pretrained(BLIP_MODEL_NAME)


def build_clip_to_blip_lookup(clip_tok, blip_tok):
    """Build a deterministic CLIP-vocab -> BLIP-vocab lookup table."""
    clip_vocab_size = clip_tok.vocab_size
    unk_id = blip_tok.unk_token_id if blip_tok.unk_token_id is not None else 0
    lookup = torch.full((clip_vocab_size,), fill_value=unk_id, dtype=torch.int64)

    for clip_id in range(clip_vocab_size):
        piece_text = clip_tok.decode(
            [clip_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if not piece_text:
            continue

        blip_ids = blip_tok(piece_text, add_special_tokens=False)["input_ids"]
        if blip_ids:
            lookup[clip_id] = int(blip_ids[0])

    if clip_tok.bos_token_id is not None and blip_tok.cls_token_id is not None:
        lookup[clip_tok.bos_token_id] = int(blip_tok.cls_token_id)
    if clip_tok.eos_token_id is not None and blip_tok.sep_token_id is not None:
        lookup[clip_tok.eos_token_id] = int(blip_tok.sep_token_id)
    if clip_tok.pad_token_id is not None and blip_tok.pad_token_id is not None:
        lookup[clip_tok.pad_token_id] = int(blip_tok.pad_token_id)

    return lookup

class ImageEncoderWrapper(torch.nn.Module):
    def __init__(self, blip_model):
        super().__init__()
        self.blip = blip_model

    def forward(self, images):
        return self.blip.get_image_features(pixel_values=images)

class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, blip_model, clip_tok, blip_tok):
        super().__init__()
        self.blip = blip_model
        self.register_buffer(
            "clip_to_blip_lookup",
            build_clip_to_blip_lookup(clip_tok, blip_tok),
            persistent=True,
        )

    def forward(self, token_ids, attention_mask):
        clip_ids = token_ids.to(torch.int64)
        max_id = self.clip_to_blip_lookup.shape[0] - 1
        clip_ids = torch.clamp(clip_ids, min=0, max=max_id)
        mapped_token_ids = self.clip_to_blip_lookup[clip_ids].to(torch.int32)
        mapped_attention_mask = attention_mask.to(torch.int32)
        return self.blip.get_text_features(
            input_ids=mapped_token_ids,
            attention_mask=mapped_attention_mask,
        )

# -----------------------------
# 4. Create wrapper instances
# -----------------------------
image_encoder = ImageEncoderWrapper(blip_model)
text_encoder = TextEncoderWrapper(blip_model, clip_tokenizer, blip_tokenizer)
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
