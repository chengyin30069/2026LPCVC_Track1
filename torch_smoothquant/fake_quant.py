import copy
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoCaptions
from PIL import Image

import open_clip

import os
from pathlib import Path
import numpy as np

from pathlib import Path

from tqdm import tqdm

BATCH_SIZE = 32
EPOCHS = 1
NUM_WORKERS = 8
LR = 2e-5
WD = 0.05
AMP = True




# =========================
# 1. fake quant functions
# =========================

def fake_quantize_per_tensor_symmetric(x: torch.Tensor, num_bits: int = 8, eps: float = 1e-8):
    qmax = 2 ** (num_bits - 1) - 1
    max_abs = x.detach().abs().max().clamp_min(eps)
    scale = max_abs / qmax
    x_int = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return x_int * scale


def fake_quantize_per_channel_symmetric(
    x: torch.Tensor,
    channel_axis: int,
    num_bits: int = 8,
    eps: float = 1e-8,
):
    qmax = 2 ** (num_bits - 1) - 1
    reduce_dims = [i for i in range(x.ndim) if i != channel_axis]
    max_abs = x.detach().abs().amax(dim=reduce_dims, keepdim=True).clamp_min(eps)
    scale = max_abs / qmax
    x_int = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return x_int * scale


# =========================
# 2. SmoothQuant fake Linear
# =========================

class SmoothQuantLinearFakeQuant(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        act_bits: int = 8,
        weight_bits: int = 8,
        alpha: float = 0.5,
        quantize_output: bool = False,
    ):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.act_bits = act_bits
        self.weight_bits = weight_bits
        self.alpha = alpha
        self.quantize_output = quantize_output

        self.weight = nn.Parameter(linear.weight.detach().clone())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        else:
            self.bias = None

        self.calibrated = False

        self.register_buffer("smooth_scale", torch.ones(self.in_features))
        self.register_buffer("act_channel_max", torch.zeros(self.in_features))
        # self.register_buffer("calibrated", torch.tensor(False))

    @torch.no_grad()
    def update_activation_stats(self, x: torch.Tensor):
        # x: [B, C] or [B, N, C]
        x_absmax = x.detach().abs().reshape(-1, x.shape[-1]).amax(dim=0)
        self.act_channel_max = self.act_channel_max.to('cuda')
        self.act_channel_max.copy_(torch.maximum(self.act_channel_max, x_absmax))

    @torch.no_grad()
    def compute_smooth_scale(self, eps: float = 1e-8):
        w_channel_max = self.weight.detach().abs().amax(dim=0).clamp_min(eps)
        a_channel_max = self.act_channel_max.detach().clamp_min(eps)
        s = (a_channel_max ** self.alpha) / (w_channel_max ** (1.0 - self.alpha))
        s = torch.clamp(s, min=eps)
        self.smooth_scale.copy_(s)
        self.calibrated = True

    def forward(self, x: torch.Tensor):
        # calibration 階段：先走原始 FP linear
        if not bool(self.calibrated):
            return F.linear(x, self.weight, self.bias)

        s = self.smooth_scale.to(dtype=x.dtype, device=x.device)

        x_smooth = x / s
        w_smooth = self.weight * s.unsqueeze(0)

        x_q = fake_quantize_per_tensor_symmetric(x_smooth, num_bits=self.act_bits)
        w_q = fake_quantize_per_channel_symmetric(w_smooth, channel_axis=0, num_bits=self.weight_bits)

        y = F.linear(x_q, w_q, self.bias)

        if self.quantize_output:
            y = fake_quantize_per_tensor_symmetric(y, num_bits=self.act_bits)

        return y

class SmoothLinear(nn.Module):
    def __init__(self, linear: nn.Linear, smooth_scale: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = nn.Parameter(linear.bias.detach().clone()) if linear.bias is not None else None
        self.register_buffer("smooth_scale", smooth_scale.detach().clone())

    def forward(self, x):
        s = self.smooth_scale.to(dtype=x.dtype, device=x.device)
        x = x / s
        w = self.weight * s.unsqueeze(0)
        return F.linear(x, w, self.bias)


class CocoClipDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, cap_json, img_tf, tokenizer):
        self.ds = CocoCaptions(root=img_dir, annFile=cap_json)
        self.img_tf = img_tf
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, caps = self.ds[idx]
        cap = caps[torch.randint(low=0, high=len(caps), size=(1,)).item()]
        img = self.img_tf(img)
        tokens = self.tokenizer([cap])[0]
        return img, tokens


# =========================
# 3. replace all Linear
# =========================

def replace_linear_with_smoothquant(
    module: nn.Module,
    act_bits: int = 8,
    weight_bits: int = 8,
    alpha: float = 0.5,
    quantize_output: bool = False,
    skip_names: Optional[List[str]] = None,
    prefix: str = "",
):
    if skip_names is None:
        skip_names = ["visual_projection"]

    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear) and full_name not in skip_names:
            setattr(
                module,
                name,
                SmoothQuantLinearFakeQuant(
                    child,
                    act_bits=act_bits,
                    weight_bits=weight_bits,
                    alpha=alpha,
                    quantize_output=quantize_output,
                ),
            )
        else:
            replace_linear_with_smoothquant(
                child,
                act_bits=act_bits,
                weight_bits=weight_bits,
                alpha=alpha,
                quantize_output=quantize_output,
                skip_names=skip_names,
                prefix=full_name,
            )

def replace_linear_with_smooth(
    module: nn.Module,
    skip_names: Optional[List[str]] = None,
    prefix: str = "",
):
    if skip_names is None:
        skip_names = ["visual_projection"]

    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, SmoothQuantLinearFakeQuant) and full_name not in skip_names:
            setattr(
                module,
                name,
                SmoothLinear(
                    linear=child,
                    smooth_scale=child.smooth_scale,
                ),
            )
        else:
            replace_linear_with_smooth(
                child,
                skip_names=skip_names,
                prefix=full_name,
            )

# =========================
# 4. calibration hooks
# =========================

@torch.no_grad()
def attach_smoothquant_calibration_hooks(model: nn.Module):
    hooks = []

    def make_hook(layer: SmoothQuantLinearFakeQuant):
        def hook(module, inputs):
            x = inputs[0]
            layer.update_activation_stats(x)
        return hook

    for m in model.modules():
        if isinstance(m, SmoothQuantLinearFakeQuant):
            h = m.register_forward_pre_hook(make_hook(m))
            hooks.append(h)

    return hooks


@torch.no_grad()
def finalize_smoothquant_scales(model: nn.Module):
    for m in model.modules():
        if isinstance(m, SmoothQuantLinearFakeQuant):
            m.compute_smooth_scale()


# =========================
# 5. build quant model
# =========================

@torch.no_grad()
def prepare_smoothquant_fake_model(
    image_encoder: nn.Module,
    calibration_dataloader,
    forward_fn,
    device: str = "cuda",
    act_bits: int = 8,
    weight_bits: int = 8,
    alpha: float = 0.5,
    quantize_output: bool = False,
    max_calib_batches: Optional[int] = None,
):
    qmodel = copy.deepcopy(image_encoder).to(device)
    qmodel.eval()

    replace_linear_with_smoothquant(
        qmodel,
        act_bits=act_bits,
        weight_bits=weight_bits,
        alpha=alpha,
        quantize_output=quantize_output,
    )

    hooks = attach_smoothquant_calibration_hooks(qmodel)

    for i, batch in enumerate(calibration_dataloader):
        forward_fn(qmodel, batch)
        if max_calib_batches is not None and (i + 1) >= max_calib_batches:
            break

    for h in hooks:
        h.remove()

    qmodel.to(device)

    finalize_smoothquant_scales(qmodel)
    return qmodel

@torch.no_grad()
def convert_smoothquant_fake_to_smooth(model: nn.Module):
    new_model = copy.deepcopy(model)
    replace_linear_with_smooth(new_model)
    return new_model


# =========================
# 6. demo dataset
# =========================

def process_image(image_path, target_size=(224, 224)):
    """Loads and processes an image to the required input shape (C, H, W)."""
    image = Image.open(image_path).convert('RGB').resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize
    return np.transpose(image_array, (2, 0, 1)) # Convert to (C, H, W)

class ImageFolderListDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        pixel_values = process_image(self.image_paths[idx])
        return {"pixel_values": pixel_values}


# =========================
# 7. compare fp vs quant
# =========================

@torch.no_grad()
def get_embedding(model, batch, device):
    pixel_values = batch["pixel_values"].to(device)
    out = model(pixel_values)
    emb = out
    emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return emb


@torch.no_grad()
def compare_models(fp_model, q_model, dataloader, device="cuda", max_batches=10):
    fp_model.eval()
    q_model.eval()

    cos_list = []
    for i, batch in enumerate(dataloader):
        fp = get_embedding(fp_model, batch, device)
        q = get_embedding(q_model, batch, device)
        cos = (fp * q).sum(dim=-1).mean().item()
        cos_list.append(cos)

        print(f"[batch {i}] cosine = {cos:.6f}")

        if (i + 1) >= max_batches:
            break

    print("mean cosine:", sum(cos_list) / len(cos_list))


def clip_contrastive_loss(image_features, text_features, logit_scale):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    logits = logit_scale * (image_features @ text_features.t())
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.t(), targets)
    return (loss_i + loss_t) / 2

def  run_training(clip_model, train_loader, epoch):
    clip_model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = [p for p in clip_model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda", enabled=(AMP and device == "cuda"))

    # 7) train
    for epoch in range(EPOCHS):
        for it, (images, tokens) in tqdm(enumerate(train_loader), total=len(train_loader)):
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(AMP and device == "cuda")):
                img_feat = clip_model.encode_image(images)
                txt_feat = clip_model.encode_text(tokens)
                logit_scale = clip_model.logit_scale.exp().clamp(max=100)
                loss = clip_contrastive_loss(img_feat, txt_feat, logit_scale)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            if it % 50 == 0:
                print(f"epoch {epoch} iter {it} loss {loss.item():.4f} logit_scale {logit_scale.item():.2f}")
    return clip_model



# =========================
# 8. main
# =========================

ONNX_DIR = "exported_onnx"
CALIBRATION_DATASET_PATH = "./coco2017/train2017"



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K"

    # 你要改成自己的圖片路徑
    image_paths = [Path(CALIBRATION_DATASET_PATH) / f for f in os.listdir(CALIBRATION_DATASET_PATH)][:1024]

    clip_model, preprocess_train, _ = open_clip.create_model_and_transforms(model_name)
    clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    dataset = ImageFolderListDataset(image_paths)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    def forward_fn(model, batch):
        pixel_values = batch["pixel_values"].to(device)
        return model(pixel_values)

    print("Calibrating SmoothQuant fake model...")
    q_model = prepare_smoothquant_fake_model(
        image_encoder=clip_model.visual,
        calibration_dataloader=loader,
        forward_fn=forward_fn,
        device=device,
        act_bits=8,
        weight_bits=8,
        alpha=0.5,
        quantize_output=False,
        max_calib_batches=1024,
    )
    
    print("Comparing FP vs SmoothQuant fake quant...")
    compare_models(clip_model.visual, q_model, loader, device=device)
    golden_model = copy.deepcopy(clip_model.visual)

    clip_model.visual = copy.deepcopy(q_model)

    q_model = convert_smoothquant_fake_to_smooth(q_model)

    for p in clip_model.transformer.parameters():
        p.requires_grad = False
    clip_model.logit_scale.requires_grad = True


    # 5) data
    train_set = CocoClipDataset('./coco2017/train2017', './coco2017/annotations/captions_train2017.json', preprocess_train, tokenizer)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    run_training(clip_model, train_loader, 1)

    compare_models(clip_model.visual, golden_model, loader, device=device)




    save_path = Path(ONNX_DIR) / "smoothquant_image_encoder.onnx"
    torch.onnx.export(
        clip_model.visual,
        torch.randn(1, 3, 224, 224).to(device),
        save_path,
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

if __name__ == "__main__":
    main()