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
from utils.Dataset import CalibrationDataset, Track1SampleClipDataset

from functools import partial
import aimet_torch

import math
from benchmark import evaluate_coco
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = "cuda"

COCO_ROOT = "../coco2017"
DATASET_SAMPLE_ROOT = "dataset_sample"
DATASET_SAMPLE_IMG_DIR = os.path.join(DATASET_SAMPLE_ROOT, "images")

TRAIN_IMG_DIR = os.path.join(COCO_ROOT, "train2017")
TRAIN_CAP_JSON = os.path.join(COCO_ROOT, "annotations", "captions_train2017.json")

BATCH_SIZE = 8
NUM_WORKERS = 8
QAT_EPOCHS = 1
QAT_LR = 1e-5
QAT_MAX_STEPS_PER_EPOCH = 500
QAT_CHECKPOINT = "qat_clip_visual.pth"

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
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        if is_causal:
            Lq = attn_scores.size(-2)
            Lk = attn_scores.size(-1)
            causal_mask = torch.triu(
                torch.full((Lq, Lk), float("-inf"), device=attn_scores.device, dtype=attn_scores.dtype),
                diagonal=1,
            )
            attn_scores = attn_scores + causal_mask

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

def clip_contrastive_loss(image_features, text_features, logit_scale):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    logits = logit_scale * (image_features @ text_features.t())
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.t(), targets)
    return (loss_i + loss_t) / 2




def run_training(clip_model, golden_model, train_loader, epochs=1, freeze_text_encoder=True,
                 lr=1e-5,
                 amp=True,
                 wd=0.05,
                 max_steps_per_epoch=None,
                 checkpoint_path=QAT_CHECKPOINT,
                 evaluate_after_epoch=False,
                ) -> torch.nn.Module:
    clip_model.train()
    golden_model.eval()
    for param in golden_model.parameters():
        param.requires_grad = False

    if freeze_text_encoder:
        for param in clip_model.parameters():
            param.requires_grad = False

    for param in clip_model.visual.parameters():
        param.requires_grad = True

    # Only train the last 2 layers of visual transformer
    # if hasattr(clip_model, 'visual') and hasattr(clip_model.visual, 'transformer'):
    #     blocks = clip_model.visual.transformer.resblocks
    #     for block in blocks[-2:]:
    #         for param in block.parameters():
    #             param.requires_grad = True

    params = [p for p in clip_model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found for QAT.")
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and DEVICE.startswith("cuda")))


    best_metric = -float("inf")
    # 7) train
    loss_func = nn.CosineEmbeddingLoss()
    for epoch in range(epochs):
        clip_model.train()
        epoch_loss = 0.0
        step_count = 0
        for it, (images, _tokens) in tqdm(enumerate(train_loader), total=len(train_loader)):
            images = images.to(DEVICE, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with torch.no_grad():
                img_golden = golden_model.encode_image(images).detach()


            with torch.amp.autocast("cuda", enabled=(amp and DEVICE.startswith("cuda"))):
                img_feat = clip_model.encode_image(images)
                y_ones = torch.ones(img_feat.size(0), device=DEVICE)
                loss = loss_func(img_feat, img_golden, y_ones)


                # txt_feat = clip_model.encode_text(tokens)
                # logit_scale = clip_model.logit_scale.exp().clamp(max=100)
                # loss = clip_contrastive_loss(img_feat, txt_feat, logit_scale)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            epoch_loss += loss.item()
            step_count += 1
            if it % 10 == 0:
                print(f"epoch {epoch} iter {it} loss {loss.item():.4f}")
            if max_steps_per_epoch is not None and step_count >= max_steps_per_epoch:
                break

        avg_loss = epoch_loss / max(step_count, 1)
        metric = -avg_loss
        if evaluate_after_epoch:
            try:
                recall = evaluate_coco(clip_model)
                if recall is not None:
                    metric = recall
                    print(f"epoch {epoch} recall@10: {recall:.4f}")
                else:
                    print(f"epoch {epoch} avg loss: {avg_loss:.4f}")
            except Exception as exc:
                print(f"⚠ evaluate_coco failed after epoch {epoch}: {exc}")
                print(f"epoch {epoch} avg loss: {avg_loss:.4f}")
        else:
            print(f"epoch {epoch} avg loss: {avg_loss:.4f}")

        if metric > best_metric:
            best_metric = metric
            torch.save(clip_model.state_dict(), checkpoint_path)
            print(f"saved QAT checkpoint to {checkpoint_path}")
    return clip_model

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



    train_set = Track1SampleClipDataset(DATASET_SAMPLE_ROOT, preprocess_train, tokenizer)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    CALIBRATION_DIR = os.path.join(COCO_ROOT, "train2017")
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
    # fold_all_batch_norms(model, dummy_input.shape)

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
        if name.endswith(".attn.softmax"):
            # 在 Softmax 輸入前加一顆 activation quantizer
            mod.input_quantizers[0] = QuantizeDequantize(
                shape=(),          # scalar encoding
                bitwidth=8,
                symmetric=False
            ).to(DEVICE)

    mp_configurator = MixedPrecisionConfigurator(sim)
    # quant_sub_name = ['mlp.gelu', 'mlp.c_fc', 'mlp.c_proj', 'ln_1', 'ln_2', 'attn.q_proj', 'attn.k_proj', 'attn.v_proj', 'attn.out_proj']
    quant_sub_name = ['mlp.gelu', 'mlp.c_fc', 'mlp.c_proj', 'attn.k_proj', 'attn.out_proj']
    quant_matmul_id = [0, 1]
    quant_add_id = [1, 2]
    # print(sim.model.get_submodule(f"transformer.resblocks.0.attn.softmax"))
    # mp_configurator.set_precision(sim.model.get_submodule(f"transformer.resblocks.0.mlp.c_proj"), activation='int8', param={'weight': 'int8'})

    # mp_configurator.set_precision(sim.model.get_submodule(f"transformer.resblocks.0.mlp.c_fc"), activation='int8', param={'weight': 'int8'})
    # mp_configurator.set_precision(sim.model.get_submodule(f"transformer.resblocks.0.attn.module_matmul_1"), activation='int8', param={'weight': 'int8'})
    # mp_configurator.set_precision(sim.model.get_submodule(f"transformer.resblocks.0.attn.out_proj"), activation='int8', param={'weight': 'int8'})
    # mp_configurator.set_precision(sim.model.get_submodule(f"transformer.resblocks.0.attn.k_proj"), activation='int8', param={'weight': 'int8'})
    mp_configurator.set_precision(sim.model.get_submodule(f"transformer.resblocks.0.attn.module_matmul"), activation='int8', param={'weight': 'int8'})
    mp_configurator.set_precision(sim.model.get_submodule(f"transformer.resblocks.0.mlp.gelu"), activation='int8', param={'weight': 'int8'})



    # for i in range(0, 24):
    #     for j in quant_add_id:
    #         mp_configurator.set_precision(sim.model.get_submodule(f"transformer.resblocks.{i}.module_add_{i * 2 + j}"), activation='int8', param={'weight': 'int8'})



    for i in range(1, 24):
        # for j in quant_add_id:
        #     mp_configurator.set_precision(sim.model.get_submodule(f"transformer.resblocks.{i}.module_add_{i * 2 + j}"), activation='int8', param={'weight': 'int8'})
        for matmul_id in quant_matmul_id:
            if(i == 0 and matmul_id == 0):
                mp_configurator.set_precision(sim.model.get_submodule(f"transformer.resblocks.{i}.attn.module_matmul"), activation='int8', param={'weight': 'int8'})
            else:
                mp_configurator.set_precision(sim.model.get_submodule(f"transformer.resblocks.{i}.attn.module_matmul_{i * 2 + matmul_id}"), activation='int8', param={'weight': 'int8'})

        for sub_module in quant_sub_name:
            mp_configurator.set_precision(sim.model.get_submodule(f"transformer.resblocks.{i}.{sub_module}"), activation='int8', param={'weight': 'int8'})

    mp_configurator.apply()

    mp_configurator = MixedPrecisionConfigurator(sim)
    # quant_sub_name = ['attn.q_proj']

    # for i in range(1, 10):
    #     mp_configurator.set_precision(sim.model.get_submodule(f"transformer.resblocks.{i}.ln_1"), activation='int8', param={'weight': 'int8'})
    #     mp_configurator.set_precision(sim.model.get_submodule(f"transformer.resblocks.{i}.ln_2"), activation='int8', param={'weight': 'int8'})

    # mp_configurator.apply()

    sim.compute_encodings(pass_calibration_data)



    # Compare golden_model and quantized clip_model before training
    print("\n▶ Computing cosine similarity between golden and quantized models (before training)...")
    clip_model.eval()
    golden_model.eval()
    evaluate_similarity(golden_model.visual, sim.model, train_loader, device)

    print("\n▶ Running QAT training...")
    clip_model.visual = sim.model
    clip_model = run_training(
        clip_model,
        golden_model,
        train_loader,
        epochs=QAT_EPOCHS,
        lr=QAT_LR,
        max_steps_per_epoch=QAT_MAX_STEPS_PER_EPOCH,
        evaluate_after_epoch=False,
    )
    sim.model = clip_model.visual

    print("\n▶ Computing cosine similarity between golden and quantized models (after training)...")
    clip_model.eval()
    golden_model.eval()
    evaluate_similarity(golden_model.visual, sim.model, train_loader, device)

    # evaluate_coco(golden_model)
    # evaluate_coco(clip_model, root_dir = '..')

    DUMMY_INPUT = torch.randn(1, 3, 224, 224, device=DEVICE)

    aimet_torch.onnx.export(sim, DUMMY_INPUT,
                             f = "aimet_mixed_quant_image.onnx",
                             input_names=["image"], output_names=["embedding"],
                            opset_version=21,
                            dynamo=True,
                            export_int32_bias=True,
                            external_data=False,)


if __name__ == "__main__":
    main()
