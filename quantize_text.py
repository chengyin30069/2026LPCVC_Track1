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
from utils.Dataset import CocoClipDataset, CalibrationDataset, TextCalibrationDataset

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

class ExplicitQKVAttention(nn.Module):
    """
    Replace nn.MultiheadAttention with explicit q_proj/k_proj/v_proj/out_proj.

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
        self.softmax = nn.Softmax(dim=-1)
        self.scale_mul = ScaleMul(1.0 / math.sqrt(self.head_dim))

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})."
            )

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        self._load_from_mha(mha)

    def _load_from_mha(self, mha: nn.MultiheadAttention):
        with torch.no_grad():
            if mha.in_proj_weight is None:
                raise ValueError("Expected mha.in_proj_weight to exist.")

            q_w, k_w, v_w = mha.in_proj_weight.chunk(3, dim=0)
            self.q_proj.weight.copy_(q_w)
            self.k_proj.weight.copy_(k_w)
            self.v_proj.weight.copy_(v_w)

            if mha.in_proj_bias is not None:
                q_b, k_b, v_b = mha.in_proj_bias.chunk(3, dim=0)
                self.q_proj.bias.copy_(q_b)
                self.k_proj.bias.copy_(k_b)
                self.v_proj.bias.copy_(v_b)
            else:
                nn.init.zeros_(self.q_proj.bias)
                nn.init.zeros_(self.k_proj.bias)
                nn.init.zeros_(self.v_proj.bias)

            self.out_proj.weight.copy_(mha.out_proj.weight)
            if mha.out_proj.bias is not None:
                self.out_proj.bias.copy_(mha.out_proj.bias)
            else:
                nn.init.zeros_(self.out_proj.bias)

    def _reshape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        batch_first=False: [L, N, C] -> [N, H, L, D]
        batch_first=True : [N, L, C] -> [N, H, L, D]
        """
        if self.batch_first:
            N, L, C = x.shape
            x = x.reshape(N, L, self.num_heads, self.head_dim)
            x = x.permute(0, 2, 1, 3)  # [N, H, L, D]
        else:
            L, N, C = x.shape
            x = x.reshape(L, N, self.num_heads, self.head_dim)
            x = x.permute(1, 2, 0, 3)  # [N, H, L, D]
        return x

    def _reshape_from_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        [N, H, L, D] ->
          batch_first=False: [L, N, C]
          batch_first=True : [N, L, C]
        """
        N, H, L, D = x.shape
        if self.batch_first:
            x = x.permute(0, 2, 1, 3).reshape(N, L, H * D)
        else:
            x = x.permute(2, 0, 1, 3).reshape(L, N, H * D)
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask=None,
        need_weights: bool = False,
        attn_mask=None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        if key_padding_mask is not None:
            raise NotImplementedError("key_padding_mask is not implemented in this wrapper.")

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = self._reshape_to_heads(q)  # [N, H, Lq, D]
        k = self._reshape_to_heads(k)  # [N, H, Lk, D]
        v = self._reshape_to_heads(v)  # [N, H, Lv, D]

        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = self.scale_mul(attn_scores)
        # if attn_mask is not None:
        #     attn_scores = attn_scores + attn_mask

        # if is_causal:
        #     Lq = attn_scores.size(-2)
        #     Lk = attn_scores.size(-1)
        #     causal_mask = torch.triu(
        #         torch.full((Lq, Lk), float("-inf"), device=attn_scores.device, dtype=attn_scores.dtype),
        #         diagonal=1,
        #     )
        #     attn_scores = attn_scores + causal_mask

        attn_probs = self.softmax(attn_scores)

        if self.training and self.dropout > 0:
            attn_probs = F.dropout(attn_probs, p=self.dropout)

        out = torch.matmul(attn_probs, v)  # [N, H, Lq, D]
        out = self._reshape_from_heads(out)
        out = self.out_proj(out)

        if need_weights:
            if average_attn_weights:
                weights = attn_probs.mean(dim=1)  # [N, Lq, Lk]
            else:
                weights = attn_probs             # [N, H, Lq, Lk]
            return out, weights

        return out, None

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(self, model, normalize=True):
        super().__init__()
        self.model = model
        self.normalize = normalize

    def get_transformer_cast_dtype(self):
        transformer = self.model.transformer
        if hasattr(transformer, "get_cast_dtype"):
            return transformer.get_cast_dtype()

        for _, param in transformer.named_parameters():
            if param.is_floating_point():
                return param.dtype

        for _, buf in transformer.named_buffers():
            if buf.is_floating_point():
                return buf.dtype

        return self.model.token_embedding.weight.dtype

    def build_transformer_input(self, input_ids):
        text = input_ids

        cast_dtype = self.get_transformer_cast_dtype()

        # [B, L, C]
        x = self.model.token_embedding(text).to(cast_dtype)
        x = x + self.model.positional_embedding.to(cast_dtype)
        return x

    def forward(self, input_ids):
        text = input_ids
        x = self.build_transformer_input(text)

        # 不要 permute，照原本 encode_text 直接餵 transformer
        x = self.model.transformer(x, attn_mask=self.model.attn_mask)

        # [B, L, C]
        x = self.model.ln_final(x)

        # 只替換掉原本 text_global_pool 的 advanced indexing
        if self.model.text_pool_type == "first":
            x = x[:, 0]
        elif self.model.text_pool_type == "last":
            x = x[:, -1]
        elif self.model.text_pool_type == "argmax":
            idx = text.argmax(dim=-1)                           # [B]
            idx = idx.view(-1, 1, 1).expand(-1, 1, x.shape[-1])
            x = x.gather(dim=1, index=idx).squeeze(1)
        elif self.model.text_pool_type == "eos":
            eos_token_id = getattr(self.model, "text_eos_id", None)
            if eos_token_id is None:
                raise RuntimeError("text_eos_id is required when text_pool_type='eos'")
            idx = (text == eos_token_id).int().argmax(dim=-1)
            idx = idx.view(-1, 1, 1).expand(-1, 1, x.shape[-1])
            x = x.gather(dim=1, index=idx).squeeze(1)
        else:
            # 保持和原本函式一致
            pass

        if self.model.text_projection is not None:
            if isinstance(self.model.text_projection, nn.Linear):
                x = self.model.text_projection(x)
            else:
                x = x @ self.model.text_projection

        return F.normalize(x, dim=-1) if self.normalize else x

class TextEncoderOnlyPool(nn.Module):
    def __init__(self, model, normalize=False):
        super().__init__()
        self.model = model
        self.normalize = normalize

    def forward(self, input_ids):
        text = input_ids
        if hasattr(self.model.transformer, "get_cast_dtype"):
            cast_dtype = self.model.transformer.get_cast_dtype()
        else:
            cast_dtype = self.model.token_embedding.weight.dtype

        x = self.model.token_embedding(text).to(cast_dtype)
        x = x + self.model.positional_embedding.to(cast_dtype)
        x = self.model.transformer(x, attn_mask=self.model.attn_mask)
        x = self.model.ln_final(x)

        if self.model.text_pool_type == "first":
            x = x[:, 0]
        elif self.model.text_pool_type == "last":
            x = x[:, -1]
        elif self.model.text_pool_type == "argmax":
            idx = text.argmax(dim=-1)
            idx = idx.view(-1, 1, 1).expand(-1, 1, x.shape[-1])
            x = x.gather(dim=1, index=idx).squeeze(1)
        elif self.model.text_pool_type == "eos":
            eos_token_id = getattr(self.model, "text_eos_id", None)
            idx = (text == eos_token_id).int().argmax(dim=-1)
            idx = idx.view(-1, 1, 1).expand(-1, 1, x.shape[-1])
            x = x.gather(dim=1, index=idx).squeeze(1)

        if self.model.text_projection is not None:
            if isinstance(self.model.text_projection, nn.Linear):
                x = self.model.text_projection(x)
            else:
                x = x @ self.model.text_projection

        return F.normalize(x, dim=-1) if self.normalize else x


class TransformerWithFixedAttnMask(nn.Module):
    def __init__(self, transformer: nn.Module, attn_mask: torch.Tensor | None):
        super().__init__()
        self.transformer = transformer
        if attn_mask is None:
            self.attn_mask = None
        else:
            self.register_buffer("attn_mask", attn_mask, persistent=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        mask = attn_mask if attn_mask is not None else self.attn_mask
        return self.transformer(x, attn_mask=mask)

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
    a = a.flatten().cpu().detach().numpy() if isinstance(a, torch.Tensor) else a
    b = b.flatten().cpu().detach().numpy() if isinstance(b, torch.Tensor) else b
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def evaluate_similarity(golden_model, clip_model, train_loader, device, max_batches=10):
    similarities = []
    with torch.no_grad():
        for it, (images, tokens) in enumerate(train_loader):
            if it >= max_batches:
                break

            tokens = tokens.to(device, non_blocking=True)

            text_feat_golden = golden_model.encode_text(tokens)
            text_feat_quant = clip_model(tokens)

            for i in range(text_feat_golden.shape[0]):
                sim = cosine_sim(
                    text_feat_golden[i].cpu().numpy(),
                    text_feat_quant[i].cpu().numpy()
                )
                similarities.append(sim)

    avg_sim = np.mean(similarities)
    print(f"Average Text Embedding Cosine Similarity: {avg_sim:.6f}")


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

    model = TextEncoder(clip_model)

    replace_mha_with_explicit_qkv(model.model.transformer)
    model.model.transformer = replace_layernorm_with_torch(model.model.transformer)
    model.model.transformer = prepare_model(model.model.transformer)
    model.model.transformer = TransformerWithFixedAttnMask(
        model.model.transformer,
        model.model.attn_mask,
    )
    model.to(DEVICE).eval()

    dummy_tokens = torch.randint(0, 49408, (1, 77), device=DEVICE)
    transformer_dummy_input = model.build_transformer_input(dummy_tokens)

    train_set = CocoClipDataset(TRAIN_IMG_DIR, TRAIN_CAP_JSON, preprocess_train, tokenizer)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    CALIBRATION_DIR = "../coco2017"
    cali_data = None
    if os.path.exists(CALIBRATION_DIR):
        # Use COCO captions for text calibration
        cali_set = CocoClipDataset(TRAIN_IMG_DIR, TRAIN_CAP_JSON, preprocess_train, tokenizer, max_samples=20)
        cali_data = DataLoader(
            cali_set,
            batch_size=4,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        print(f"▶ Loaded {len(cali_set)} calibration text samples from COCO captions")
    else:
        print(f"⚠ Calibration directory not found at {CALIBRATION_DIR}")
        cali_data = None

    @torch.no_grad()
    def pass_calibration_data(transformer: torch.nn.Module) -> None:
        device = "cuda"
        # Only calibrate the transformer. Token embedding/pooling stay in float.
        for images, tokens in cali_data:
            transformer_input = model.build_transformer_input(tokens.to(device))
            _ = transformer(transformer_input)

    print("=== prepared model ===")
    evaluate_similarity(golden_model, model, train_loader, device)

    sim = QuantizationSimModel(
        model.model.transformer,
        transformer_dummy_input,
        default_output_bw=16,
        default_param_bw=8,
        in_place=True,
        config_file="htp_v69"
        )
    from aimet_torch.v2.quantization.affine.quantizer import QuantizeDequantize

    for name, mod in sim.model.named_modules():
        if name.endswith(".attn.softmax"):
            mod.input_quantizers[0] = QuantizeDequantize(
                shape=(),          # scalar encoding
                bitwidth=16,
                symmetric=False
            ).to(DEVICE)



    # print("=== quantsim before compute_encodings ===")
    # evaluate_similarity(golden_model, sim.model, train_loader, device)

    sim.compute_encodings(pass_calibration_data)

    print("=== quantsim after compute_encodings ===")
    evaluate_similarity(golden_model, model, train_loader, device)

    # Compare golden_model and quantized clip_model before training
    print("\n▶ Computing cosine similarity between golden and quantized models (before training)...")
    clip_model.eval()
    golden_model.eval()

    evaluate_similarity(golden_model, model, train_loader, device)

    DUMMY_INPUT = transformer_dummy_input
    aimet_torch.onnx.export(sim, DUMMY_INPUT,
                             f = "aimet_mixed_quant_text_transformer.onnx",
                             input_names=["transformer_input"], output_names=["transformer_output"],
                            opset_version=21,
                            dynamo=True,
                            export_int32_bias=True,
                            external_data=False,)


if __name__ == "__main__":
    main()
