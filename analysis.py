import copy

import open_clip
from open_clip import model
from open_clip.transformer import LayerNorm as OpenCLIPLayerNorm, LayerNormFp32
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
import os
import json
from tqdm import tqdm
import numpy as np
import torch
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch import QuantizationSimModel
from aimet_torch.quant_analyzer import QuantAnalyzer
from aimet_torch.common.defs import QuantScheme
from aimet_torch.common.utils import CallbackFunc
from aimet_torch.model_preparer import prepare_model

import open_clip
from utils.Dataset import CocoClipDataset, CalibrationDataset

from functools import partial

import math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Force CPU for analysis to avoid GPU-specific issues

COCO_ROOT = "../coco2017"

TRAIN_IMG_DIR = os.path.join(COCO_ROOT, "train2017")
TRAIN_CAP_JSON = os.path.join(COCO_ROOT, "annotations", "captions_train2017.json")

BATCH_SIZE = 32
NUM_WORKERS = 8
QUANT_FRIENDLY_MASK_VALUE = -100.0

def replace_attn_mask_neg_inf(module: nn.Module, mask_value: float = QUANT_FRIENDLY_MASK_VALUE) -> int:
    """Replace -inf entries in attention-mask buffers with a finite value."""
    replaced = 0
    for name, buffer in module.named_buffers():
        if "attn_mask" not in name or not torch.is_floating_point(buffer):
            continue

        neg_inf_mask = torch.isneginf(buffer)
        if neg_inf_mask.any():
            buffer.masked_fill_(neg_inf_mask, mask_value)
            replaced += int(neg_inf_mask.sum().item())

    return replaced

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
            img_feat_golden = golden_model.encode_image(images)
            img_feat_quant = clip_model.encode_image(images)

            # Get text embeddings
            txt_feat_golden = golden_model.encode_text(tokens)
            txt_feat_quant = clip_model.encode_text(tokens)

            # Compute cosine similarities
            for i in range(img_feat_golden.shape[0]):
                img_sim = cosine_sim(img_feat_golden[i].cpu().numpy(), img_feat_quant[i].cpu().numpy())
                txt_sim = cosine_sim(txt_feat_golden[i].cpu().numpy(), txt_feat_quant[i].cpu().numpy())
                similarities.append({"image": img_sim, "text": txt_sim})

    avg_img_sim = np.mean([s["image"] for s in similarities])
    avg_txt_sim = np.mean([s["text"] for s in similarities])

    print(f"Average Image Embedding Cosine Similarity: {avg_img_sim:.6f}")
    print(f"Average Text Embedding Cosine Similarity: {avg_txt_sim:.6f}")
    print(f"Overall Average Cosine Similarity: {(avg_img_sim + avg_txt_sim) / 2:.6f}")

# class CallbackFunc:
#     """
#     Class encapsulating call back function and it's arguments
#     """

#     def __init__(self, func):
#         """
#         :param func: Callable Function
#         :param func_callback_args: Arguments passed to the callable function
#         """
#         self.func = func
#         self.args = None

#     def __call__(self, arg):
#         return self.func(arg)
class ImageEncoderWrapper(nn.Module):
    def __init__(self, visual_model):
        super().__init__()
        self.visual = visual_model

    def forward(self, image):
        out = self.visual(image)
        return out

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

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [N, H, Lq, Lk]

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)

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


def replace_mha_with_explicit_qkv(module: nn.Module):
    """
    Recursively replace all nn.MultiheadAttention with ExplicitQKVAttention.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, ExplicitQKVAttention(child))
        else:
            replace_mha_with_explicit_qkv(child)


class TextEncoderWrapper(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, tokens):
        out = self.clip_model.encode_text(tokens)
        return out

class CocoTextCalibrationDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, tokenizer, max_samples=1000):
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.captions = []

        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                coco_data = json.load(f)
            for annotation in coco_data.get('annotations', [])[: self.max_samples]:
                caption = annotation.get('caption', '').strip()
                if caption:
                    self.captions.append(caption)
        else:
            raise FileNotFoundError(f"COCO annotation file not found: {annotation_file}")

        if len(self.captions) == 0:
            raise ValueError(f"No captions loaded from {annotation_file}")

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        tokens = self.tokenizer([caption])[0]
        return tokens

def main():
    print("▶ Loading ViT-L-14 (datacomp_xl_s13b_b90k) ...")
    model_name = "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"

    clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)

    # Replace openclip LayerNorm with torch.nn.LayerNorm
    print("▶ Replacing openclip LayerNorm with torch.nn.LayerNorm...")
    clip_model = replace_layernorm_with_torch(clip_model)

    golden_model = copy.deepcopy(clip_model)

    print(f"Using device: {DEVICE}")
    model = TextEncoderWrapper(clip_model).to(DEVICE).eval()
    replaced_mask_values = replace_attn_mask_neg_inf(model)
    print(
        f"▶ Replaced {replaced_mask_values} -inf attention-mask values "
        f"with {QUANT_FRIENDLY_MASK_VALUE} for quantization."
    )
    replace_mha_with_explicit_qkv(model)
    golden_model.to(DEVICE).eval()
    clip_model.to(DEVICE).eval()

    # model = ImageEncoderWrapper(clip_model).to(DEVICE).eval()
    # print(model)
    model = replace_layernorm_with_torch(model)

    model = prepare_model(model)
    dummy_input = torch.randint(0, 49408, (1, 77), dtype=torch.int32, device=DEVICE)

    train_set = CocoClipDataset(TRAIN_IMG_DIR, TRAIN_CAP_JSON, preprocess_train, tokenizer)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    cali_set = CocoTextCalibrationDataset(TRAIN_CAP_JSON, tokenizer, max_samples=1000)
    cali_data = DataLoader(
        cali_set,
        batch_size=32,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    print(f"▶ Loaded {len(cali_set)} calibration text captions from {TRAIN_CAP_JSON}")

    # fold_all_batch_norms(model, dummy_input.shape)

    @torch.no_grad()
    def pass_calibration_data(model: torch.nn.Module) -> None:
        # Pass N batches of calibration text through the model
        for tokens in cali_data:
            _ = model(tokens.to(DEVICE))

    def evaluate_callback_func(model: torch.nn.Module, train_loader):
        similarities = []
        with torch.no_grad():
            # Sample a few batches from the training set for comparison
            for it, (_, tokens) in enumerate(train_loader):
                if it >= 5:  # Use first 5 batches
                    break

                tokens = tokens.to(DEVICE, non_blocking=True)

                txt_feat_golden = golden_model.encode_text(tokens)
                txt_feat_quant = model(tokens)

                for i in range(txt_feat_golden.shape[0]):
                    txt_sim = cosine_sim(txt_feat_golden[i].cpu().numpy(), txt_feat_quant[i].cpu().numpy())
                    similarities.append({"text": txt_sim})

        avg_txt_sim = np.mean([s["text"] for s in similarities])
        return avg_txt_sim

    def pass_analysis(model: torch.nn.Module, calibration_data) -> None:
        # Pass N batches of calibration text through the model
        for tokens in calibration_data:
            _ = model(tokens.to(DEVICE))


    pass_analysis_callback = CallbackFunc(pass_analysis, cali_data);
    evaluate_callback = CallbackFunc(evaluate_callback_func, train_loader)

    print(pass_analysis_callback.args)

    quant_analyzer = QuantAnalyzer(model, dummy_input, pass_analysis_callback, evaluate_callback)
    unlabeled_dataset_iterable = cali_set
    # quant_analyzer.enable_per_layer_mse_loss(
    #     unlabeled_dataset_iterable, num_batches=4
    # )
    quant_analyzer.analyze(
                        default_param_bw=8,
                        default_output_bw=16,
                        config_file="default_config.json",
                        results_dir="./reports/quant_analysis/text/")

    sim = QuantizationSimModel(
        model,
        dummy_input,
        default_output_bw=8,
        default_param_bw=8,
        in_place=True,
        config_file="htp_v69"
        )
    sim.compute_encodings(pass_calibration_data)



    # Compare golden_model and quantized clip_model before training
    print("\n▶ Computing cosine similarity between golden and quantized models (before training)...")
    clip_model.eval()
    golden_model.eval()
    evaluate_similarity(golden_model, clip_model, train_loader, DEVICE)


if __name__ == "__main__":
    main()
