import copy

import open_clip
from open_clip import model
from open_clip.transformer import LayerNorm as OpenCLIPLayerNorm, LayerNormFp32
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.fx as fx
import operator
import numpy as np
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch import QuantizationSimModel
from aimet_torch.quant_analyzer import QuantAnalyzer
from aimet_torch.common.defs import QuantScheme
from aimet_torch.common.utils import CallbackFunc
from aimet_torch.v2.mixed_precision import MixedPrecisionConfigurator

from aimet_torch.model_preparer import prepare_model

import open_clip
from utils.Dataset import CocoClipDataset, CalibrationDataset

from functools import partial
import aimet_torch

import math

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = "cuda"   # Force CPU for analysis to avoid GPU-specific issues

COCO_ROOT = "../coco2017"

TRAIN_IMG_DIR = os.path.join(COCO_ROOT, "train2017")
TRAIN_CAP_JSON = os.path.join(COCO_ROOT, "annotations", "captions_train2017.json")

BATCH_SIZE = 32
NUM_WORKERS = 8

class ScaleMul(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class AttentionHead(nn.Module):
    def __init__(self, embed_dim: int, head_dim: int, dropout: float):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, head_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, head_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, head_dim, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.scale_mul = ScaleMul(1.0 / math.sqrt(head_dim))
        self.dropout = float(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = self.scale_mul(attn_scores)
        attn_probs = self.softmax(attn_scores)

        if self.training and self.dropout > 0:
            attn_probs = F.dropout(attn_probs, p=self.dropout)

        out = torch.matmul(attn_probs, v)
        return out, attn_probs


class ExplicitQKVAttention(nn.Module):
    """
    Replace nn.MultiheadAttention with explicit per-head q_proj/k_proj/v_proj.

    Supports both:
      - batch_first=False: [L, N, C]
      - batch_first=True : [N, L, C]
    """
    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()

        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = float(mha.dropout)
        self.batch_first = mha.batch_first

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})."
            )

        self.heads = nn.ModuleList(
            AttentionHead(self.embed_dim, self.head_dim, self.dropout)
            for _ in range(self.num_heads)
        )
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        self._load_from_mha(mha)

    def _load_from_mha(self, mha: nn.MultiheadAttention):
        with torch.no_grad():
            if mha.in_proj_weight is None:
                raise ValueError("Expected mha.in_proj_weight to exist.")

            q_w, k_w, v_w = mha.in_proj_weight.chunk(3, dim=0)

            if mha.in_proj_bias is not None:
                q_b, k_b, v_b = mha.in_proj_bias.chunk(3, dim=0)
            else:
                q_b = k_b = v_b = None

            for head_idx, head in enumerate(self.heads):
                start = head_idx * self.head_dim
                end = start + self.head_dim

                head.q_proj.weight.copy_(q_w[start:end])
                head.k_proj.weight.copy_(k_w[start:end])
                head.v_proj.weight.copy_(v_w[start:end])

                if q_b is not None:
                    head.q_proj.bias.copy_(q_b[start:end])
                    head.k_proj.bias.copy_(k_b[start:end])
                    head.v_proj.bias.copy_(v_b[start:end])
                else:
                    nn.init.zeros_(head.q_proj.bias)
                    nn.init.zeros_(head.k_proj.bias)
                    nn.init.zeros_(head.v_proj.bias)

            self.out_proj.weight.copy_(mha.out_proj.weight)
            if mha.out_proj.bias is not None:
                self.out_proj.bias.copy_(mha.out_proj.bias)
            else:
                nn.init.zeros_(self.out_proj.bias)

    def _to_batch_first(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            return x
        return x.transpose(0, 1)

    def _from_batch_first(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            return x
        return x.transpose(0, 1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights: bool = False,
        average_attn_weights: bool = True,
    ):
        query = self._to_batch_first(query)
        key = self._to_batch_first(key)
        value = self._to_batch_first(value)

        head_outputs = []
        head_weights = []
        for head in self.heads:
            head_out, head_attn = head(query, key, value)
            head_outputs.append(head_out)
            head_weights.append(head_attn)

        out = torch.cat(head_outputs, dim=-1)
        out = self.out_proj(out)
        out = self._from_batch_first(out)

        if need_weights:
            weights = torch.stack(head_weights, dim=1)
            if average_attn_weights:
                weights = weights.mean(dim=1)
            return out, weights

        return out, None


def replace_mha_with_explicit_qkv(module: nn.Module):
    """
    Recursively replace all nn.MultiheadAttention with ExplicitQKVAttention.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, ExplicitQKVAttention(child))
        else:
            replace_mha_with_explicit_qkv(child)


def replace_layernorm_with_torch(model):
    """Replace openclip LayerNorm with torch.nn.LayerNorm recursively."""
    for name, module in model.named_modules():
        if module.__class__.__name__ in ('LayerNorm', 'LayerNormFp32'):
            # Get parent module and attribute name
            parts = name.rsplit('.', 1)
            parent = model
            if len(parts) == 2:
                parent_name, attr_name = parts
                for p in parent_name.split('.'):
                    parent = getattr(parent, p)
            else:
                attr_name = parts[0] if parts else name

            # Create new torch LayerNorm with same config
            new_ln = torch.nn.LayerNorm(
                module.normalized_shape,
                eps=module.eps,
                bias=module.bias is not None,
                dtype=module.weight.dtype,
                device=module.weight.device
            )
            # Copy weights and bias
            new_ln.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new_ln.bias.data.copy_(module.bias.data)

            setattr(parent, attr_name, new_ln)

    return model



def cosine_sim(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def evaluate_similarity(golden_model, clip_model, train_loader, device, max_batches=10):

    similarities = []
    with torch.no_grad():
        # Sample a few batches from the training set for comparison
        for it, (images, tokens) in enumerate(train_loader):
            if it >= 5:  # Use first 5 batches
                break

            images = images.to(DEVICE, non_blocking=True)
            tokens = tokens.to(DEVICE, non_blocking=True)

            # Get image embeddings
            img_feat_golden = golden_model(images)
            img_feat_quant = clip_model(images)


            # Compute cosine similarities
            for i in range(img_feat_golden.shape[0]):
                img_sim = cosine_sim(img_feat_golden[i].cpu().numpy(), img_feat_quant[i].cpu().numpy())
                similarities.append({"image": img_sim})

    avg_img_sim = np.mean([s["image"] for s in similarities])

    print(f"Average Image Embedding Cosine Similarity: {avg_img_sim:.6f}")


def main():
    device = "cuda"
    print("▶ Loading ViT-L-14 (datacomp_xl_s13b_b90k) ...")
    model_name = "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"

    clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)

    # Replace openclip LayerNorm with torch.nn.LayerNorm
    print("▶ Replacing openclip LayerNorm with torch.nn.LayerNorm...")
    clip_model = replace_layernorm_with_torch(clip_model)

    golden_model = copy.deepcopy(clip_model)
    golden_model.to(DEVICE).eval()
    clip_model.to(DEVICE).eval()
    print(f"Using device: {DEVICE}")

    model = clip_model.visual
    replace_mha_with_explicit_qkv(model)
    model = replace_layernorm_with_torch(model)
    model = prepare_model(model)
    model.to(DEVICE).eval()

    train_set = CocoClipDataset(TRAIN_IMG_DIR, TRAIN_CAP_JSON, preprocess_train, tokenizer)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    CALIBRATION_DIR = "../coco2017/train2017"
    cali_data = None
    cali_set = None
    if os.path.exists(CALIBRATION_DIR):
        cali_set = CalibrationDataset(CALIBRATION_DIR, preprocess_val)
        cali_data = DataLoader(
            cali_set,
            batch_size=4,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        print(f"▶ Loaded {len(cali_set)} calibration images from {CALIBRATION_DIR}")
    else:
        print(f"⚠ Calibration directory not found at {CALIBRATION_DIR}")
        cali_data = None

    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    @torch.no_grad()
    def pass_calibration_data(model: torch.nn.Module) -> None:
        device = "cuda"
        # Pass N batches of calibration data through the model
        for images in cali_data:
            _ = model(images.to(device))

    sim = QuantizationSimModel(
        model,
        dummy_input,
        default_output_bw=16,
        default_param_bw=8,
        in_place=True,
        config_file="htp_v69"
        )
    from aimet_torch.v2.quantization.affine.quantizer import QuantizeDequantize

    for name, mod in sim.model.named_modules():
        if name.endswith(".softmax"):
            mod.input_quantizers[0] = QuantizeDequantize(
                shape=(),          # scalar encoding
                bitwidth=16,
                symmetric=False
            ).to(DEVICE)
    sim.compute_encodings(pass_calibration_data)

    # Compare golden_model and quantized clip_model before training
    print("\n▶ Computing cosine similarity between golden and quantized models (before training)...")
    clip_model.eval()
    golden_model.eval()
    evaluate_similarity(golden_model.visual, model, train_loader, device)

    DUMMY_INPUT = torch.randn(1, 3, 224, 224, device=DEVICE)
    aimet_torch.onnx
    aimet_torch.onnx.export(sim, DUMMY_INPUT,
                             f = "aimet_mixed_quant_image.onnx",
                             input_names=["input"], output_names=["output"],
                            opset_version=21,
                            dynamo=True,
                            export_int32_bias=True,
                            external_data=False,)


if __name__ == "__main__":
    main()
