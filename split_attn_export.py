import os
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import create_model_from_pretrained

ONNX_DIR = "exported_onnx"
device = torch.device("cpu")  # export on CPU

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

        if is_causal:
            Lq = attn_scores.size(-2)
            Lk = attn_scores.size(-1)
            causal_mask = torch.triu(
                torch.full((Lq, Lk), float("-inf"), device=attn_scores.device, dtype=attn_scores.dtype),
                diagonal=1,
            )
            attn_scores = attn_scores + causal_mask

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


class ImageEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        x = self.model.encode_image(pixel_values)
        x = F.normalize(x, dim=-1)
        return x


def main():
    os.makedirs(ONNX_DIR, exist_ok=True)

    model, _ = create_model_from_pretrained(
        "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
        # "hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K"
    )

    # Deep copy if you want to keep original model untouched
    model = copy.deepcopy(model).to(device).eval()

    # Replace packed MHA with explicit q/k/v projections
    replace_mha_with_explicit_qkv(model)

    image_encoder = ImageEncoder(model).to(device).eval()

    image_onnx_path = os.path.join(ONNX_DIR, "image_encoder.onnx")
    dummy_image_input = torch.rand(1, 3, 224, 224, dtype=torch.float32, device=device)

    with torch.no_grad():
        torch.onnx.export(
            image_encoder,
            dummy_image_input,
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

    print(f"Exported to: {image_onnx_path}")


if __name__ == "__main__":
    main()