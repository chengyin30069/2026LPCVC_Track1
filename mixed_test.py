import os
import copy
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import open_clip

from aimet_torch import QuantizationSimModel
from aimet_torch.model_preparer import prepare_model
import aimet_torch

from utils.Dataset import CalibrationDataset


# ============================================================
# Config
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

COCO_ROOT = "../coco2017"
CALIBRATION_DIR = os.path.join(COCO_ROOT, "train2017")

BATCH_SIZE = 4
NUM_WORKERS = 8

BLOCK_ID = 0

DEFAULT_OUTPUT_BW = 16
DEFAULT_PARAM_BW = 8
CONFIG_FILE = "htp_v69"

ONNX_OUTPUT = f"single_vit_block_{BLOCK_ID}_aimet.onnx"
OPSET_VERSION = 21

USE_DYNAMO_EXPORT = True
EXPORT_EXTERNAL_DATA = True
EXPORT_INT32_BIAS = True

MAX_CALIBRATION_BATCHES: Optional[int] = 32


# ============================================================
# Optional debug helpers
# ============================================================

def dump_attention_status(model: nn.Module, title: str):
    print(f"\n========== {title} ==========")
    found = False

    for name, mod in model.named_modules():
        cls = mod.__class__
        cls_fullname = f"{cls.__module__}.{cls.__name__}"

        if (
            isinstance(mod, nn.MultiheadAttention)
            or "MultiheadAttention" in cls_fullname
            or "Attention" in cls_fullname
            or "attn" in name.lower()
        ):
            found = True
            print(f"{name}: {cls_fullname}")

            if hasattr(mod, "param_quantizers"):
                try:
                    print("  param_quantizers:", list(mod.param_quantizers.keys()))
                except Exception:
                    print("  param_quantizers:", mod.param_quantizers)

            if hasattr(mod, "input_quantizers"):
                print("  has input_quantizers")

            if hasattr(mod, "output_quantizers"):
                print("  has output_quantizers")

    if not found:
        print("No attention-like modules found.")


def dump_linear_transpose_patterns(onnx_path: str):
    """
    Simple ONNX graph inspection:
    prints MatMul inputs and nearby names.
    This does not do full graph tracing, but is useful for quick inspection.
    """
    try:
        import onnx
    except ImportError:
        print("onnx is not installed; skip ONNX inspection.")
        return

    print(f"\n========== Inspect ONNX MatMul nodes: {onnx_path} ==========")

    model = onnx.load(onnx_path, load_external_data=False)

    producer = {}
    for node in model.graph.node:
        for out in node.output:
            producer[out] = node

    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue

        print(f"\nMatMul: {node.name}")
        for idx, inp in enumerate(node.input):
            print(f"  input[{idx}]: {inp}")

            p = producer.get(inp, None)
            if p is not None:
                print(f"    produced by: {p.op_type} / {p.name}")
                for pidx, pinp in enumerate(p.input):
                    pp = producer.get(pinp, None)
                    if pp is not None:
                        print(f"      parent input[{pidx}] produced by: {pp.op_type} / {pp.name}")


# ============================================================
# Single block wrapper
# ============================================================

class SingleTransformerBlock(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor):
        return self.block(x)


# ============================================================
# OpenCLIP ViT preprocessing into transformer hidden states
# ============================================================

@torch.no_grad()
def get_visual_transformer_input(visual: nn.Module, image: torch.Tensor) -> torch.Tensor:
    """
    Convert image input into the input of visual.transformer.resblocks[0].

    For OpenCLIP ViT:
      image: [N, 3, H, W]
      output: usually [seq_len, N, width]

    For ViT-H/14 with 224x224:
      seq_len = 257
      width = 1280
    """

    x = visual.conv1(image)                         # [N, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)       # [N, width, grid**2]
    x = x.permute(0, 2, 1)                          # [N, grid**2, width]

    class_embedding = visual.class_embedding.to(dtype=x.dtype, device=x.device)
    class_token = class_embedding + torch.zeros(
        x.shape[0],
        1,
        x.shape[-1],
        dtype=x.dtype,
        device=x.device,
    )

    x = torch.cat([class_token, x], dim=1)           # [N, grid**2 + 1, width]
    x = x + visual.positional_embedding.to(dtype=x.dtype, device=x.device)

    x = visual.ln_pre(x)                             # [N, seq_len, width]
    x = x.permute(1, 0, 2)                           # [seq_len, N, width]

    return x


@torch.no_grad()
def get_block_input(visual: nn.Module, image: torch.Tensor, block_id: int) -> torch.Tensor:
    """
    Get input activation for visual.transformer.resblocks[block_id].

    block_id = 0:
      return input after patch embedding + class token + pos embedding + ln_pre.

    block_id > 0:
      run previous transformer blocks first.
    """

    x = get_visual_transformer_input(visual, image)

    for i in range(block_id):
        x = visual.transformer.resblocks[i](x)

    return x


# ============================================================
# Main
# ============================================================

def main():
    print(f"Using device: {DEVICE}")
    print(f"Loading OpenCLIP model: {MODEL_NAME}")

    clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(MODEL_NAME)

    clip_model = clip_model.to(DEVICE).eval()
    visual = clip_model.visual.to(DEVICE).eval()

    num_blocks = len(visual.transformer.resblocks)
    print(f"Number of transformer blocks: {num_blocks}")

    if not (0 <= BLOCK_ID < num_blocks):
        raise ValueError(f"Invalid BLOCK_ID={BLOCK_ID}; expected 0 <= BLOCK_ID < {num_blocks}")

    print(f"Extracting transformer block: {BLOCK_ID}")
    block = copy.deepcopy(visual.transformer.resblocks[BLOCK_ID]).to(DEVICE).eval()
    single_block_model = SingleTransformerBlock(block).to(DEVICE).eval()

    dump_attention_status(single_block_model, "Before prepare_model")

    # ------------------------------------------------------------
    # Create calibration dataloader
    # ------------------------------------------------------------

    if not os.path.exists(CALIBRATION_DIR):
        raise FileNotFoundError(f"Calibration directory not found: {CALIBRATION_DIR}")

    cali_set = CalibrationDataset(CALIBRATION_DIR, preprocess_val)
    cali_loader = DataLoader(
        cali_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Loaded calibration images: {len(cali_set)}")
    print(f"Calibration batch size: {BATCH_SIZE}")

    # ------------------------------------------------------------
    # Build dummy input for this block
    # ------------------------------------------------------------

    dummy_image = torch.randn(1, 3, 224, 224, device=DEVICE)

    with torch.no_grad():
        dummy_block_input = get_block_input(visual, dummy_image, BLOCK_ID)

    print("Dummy block input shape:", tuple(dummy_block_input.shape))

    with torch.no_grad():
        dummy_out = single_block_model(dummy_block_input)

    print("Dummy block output shape:", tuple(dummy_out.shape))

    # ------------------------------------------------------------
    # Prepare model for AIMET
    # ------------------------------------------------------------

    single_block_model = prepare_model(single_block_model)
    single_block_model = single_block_model.to(DEVICE).eval()

    dump_attention_status(single_block_model, "After prepare_model")

    # ------------------------------------------------------------
    # Create QuantSim
    # ------------------------------------------------------------

    sim = QuantizationSimModel(
        single_block_model,
        dummy_block_input,
        default_output_bw=DEFAULT_OUTPUT_BW,
        default_param_bw=DEFAULT_PARAM_BW,
        in_place=True,
        config_file=CONFIG_FILE,
    )

    dump_attention_status(sim.model, "After QuantizationSimModel")

    # ------------------------------------------------------------
    # Calibration callback
    # ------------------------------------------------------------

    @torch.no_grad()
    def pass_calibration_data(model: nn.Module) -> None:
        model.eval()
        visual.eval()

        for batch_idx, images in enumerate(cali_loader):
            if MAX_CALIBRATION_BATCHES is not None and batch_idx >= MAX_CALIBRATION_BATCHES:
                break

            images = images.to(DEVICE, non_blocking=True)

            block_input = get_block_input(visual, images, BLOCK_ID)
            _ = model(block_input)

            if batch_idx % 10 == 0:
                print(f"Calibration batch {batch_idx}")

    print("\nComputing AIMET encodings...")
    sim.compute_encodings(pass_calibration_data)
    print("Encoding computation done.")

    # ------------------------------------------------------------
    # Optional numerical sanity check: FP block vs QuantSim block
    # ------------------------------------------------------------

    print("\nRunning numerical sanity check...")

    with torch.no_grad():
        test_image = torch.randn(1, 3, 224, 224, device=DEVICE)
        test_block_input = get_block_input(visual, test_image, BLOCK_ID)

        fp_out = single_block_model(test_block_input)
        q_out = sim.model(test_block_input)

        diff = (fp_out - q_out).abs()
        print("fp_out shape:", tuple(fp_out.shape))
        print("q_out shape:", tuple(q_out.shape))
        print("max abs diff:", float(diff.max()))
        print("mean abs diff:", float(diff.mean()))

    # ------------------------------------------------------------
    # Export ONNX
    # ------------------------------------------------------------

    print(f"\nExporting ONNX to: {ONNX_OUTPUT}")

    aimet_torch.onnx.export(
        sim,
        dummy_block_input,
        f=ONNX_OUTPUT,
        input_names=["block_input"],
        output_names=["block_output"],
        opset_version=OPSET_VERSION,
        dynamo=USE_DYNAMO_EXPORT,
        export_int32_bias=EXPORT_INT32_BIAS,
        external_data=EXPORT_EXTERNAL_DATA,
    )

    print("ONNX export done.")

    # ------------------------------------------------------------
    # Inspect ONNX MatMul / Transpose patterns
    # ------------------------------------------------------------

    dump_linear_transpose_patterns(ONNX_OUTPUT)

    print("\nDone.")


if __name__ == "__main__":
    main()
