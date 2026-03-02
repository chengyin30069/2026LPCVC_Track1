import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
import open_clip
import torch_pruning as tp

from mobileclip.modules.common.mobileone import reparameterize_model

# -----------------------
# Config
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "MobileCLIP2-S0"
PRETRAINED_PATH = "./mobileclip2_s0.pt"

COCO_ROOT = "./coco2017"
TRAIN_IMG_DIR = os.path.join(COCO_ROOT, "train2017")
TRAIN_CAP_JSON = os.path.join(COCO_ROOT, "annotations", "captions_train2017.json")

PRUNING_RATIO = 0.30
ROUND_TO = 8
ITER_STEPS = 1

BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5
WD = 0.05
NUM_WORKERS = 8
AMP = True


# -----------------------
# Load / Utils
# -----------------------
def load_full_mobileclip(device="cuda"):
    clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED_PATH
    )
    clip_model = clip_model.to(device).eval()

    # reparameterize MobileOne blocks in visual
    clip_model = reparameterize_model(clip_model)

    # fp32
    clip_model = clip_model.to(torch.float32).eval()

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return clip_model, preprocess_train, preprocess_val, tokenizer


def build_ignored_layers_for_visual(visual_only: nn.Module):
    ignored = []
    for name, m in visual_only.named_modules():
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            ignored.append(m)
            continue

        # avoid FastViT stage3 (MHSA)
        if "trunk.stages.3" in name or "trunk.stage3" in name:
            ignored.append(m)
            continue

        cls = m.__class__.__name__.lower()
        if ("token_mixer" in name) or any(k in cls for k in ["attention", "mhsa"]):
            ignored.append(m)
            continue

        if "layer_scale" in name:
            ignored.append(m)
            continue
        if "trunk.head" in name or "head.fc" in name:
            ignored.append(m)
            continue
        if "final_conv" in name:
            ignored.append(m)
            continue

    return ignored


def clip_contrastive_loss(image_features, text_features, logit_scale):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    logits = logit_scale * (image_features @ text_features.t())
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.t(), targets)
    return (loss_i + loss_t) / 2


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


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# -----------------------
# Main
# -----------------------
def main():
    # 1) load FULL CLIP
    clip_model, preprocess_train, _, tokenizer = load_full_mobileclip(DEVICE)
    print(f"[Init] params = {count_params(clip_model)/1e6:.3f}M")

    # 2) build a visual-only wrapper FOR PRUNING TRACE (avoid None outputs)
    class VisualOnly(nn.Module):
        def __init__(self, visual):
            super().__init__()
            self.visual = visual
        def forward(self, x):
            return self.visual(x)

    visual_only = VisualOnly(clip_model.visual).to(DEVICE).eval()

    example_inputs = (torch.randn(1, 3, 224, 224, device=DEVICE, dtype=torch.float32),)
    ignored_layers = build_ignored_layers_for_visual(visual_only)

    # 3) prune VISUAL ONLY (in-place)
    importance = tp.importance.MagnitudeImportance(p=2)
    pruner = tp.pruner.MagnitudePruner(
        visual_only,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=PRUNING_RATIO,
        ignored_layers=ignored_layers,
        iterative_steps=ITER_STEPS,
        round_to=ROUND_TO,
    )
    with torch.no_grad():
        pruner.step()

    # clip_model.visual has been pruned (same module)
    print(f"[Pruned] params = {count_params(clip_model)/1e6:.3f}M")

    # 4) freeze TEXT tower; train only VISUAL (+ optionally logit_scale)
    for p in clip_model.text.parameters():
        p.requires_grad = False
    clip_model.logit_scale.requires_grad = True

    clip_model.train()

    # 5) data
    train_set = CocoClipDataset(TRAIN_IMG_DIR, TRAIN_CAP_JSON, preprocess_train, tokenizer)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # 6) optim
    params = [p for p in clip_model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda", enabled=(AMP and DEVICE.startswith("cuda")))

    # 7) train
    for epoch in range(EPOCHS):
        for it, (images, tokens) in enumerate(train_loader):
            images = images.to(DEVICE, non_blocking=True)
            tokens = tokens.to(DEVICE, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(AMP and DEVICE.startswith("cuda"))):
                img_feat = clip_model.encode_image(images)
                txt_feat = clip_model.encode_text(tokens)
                logit_scale = clip_model.logit_scale.exp().clamp(max=100)
                loss = clip_contrastive_loss(img_feat, txt_feat, logit_scale)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            if it % 50 == 0:
                print(f"epoch {epoch} iter {it} loss {loss.item():.4f} logit_scale {logit_scale.item():.2f}")

    # 8) save
    out = "mobileclip2_s0_pruned_finetuned_coco.pth"
    torch.save(clip_model.state_dict(), out)
    print("Saved:", out)


if __name__ == "__main__":
    main()