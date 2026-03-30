import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
import open_clip
import torch_pruning as tp
from tqdm import tqdm

from utils.benchmark import evaluate_coco

# -----------------------
# Config
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
COCO_ROOT = "./coco2017"
PRUNING_LOG = "pruning_log.txt"
TRAIN_IMG_DIR = os.path.join(COCO_ROOT, "train2017")
TRAIN_CAP_JSON = os.path.join(COCO_ROOT, "annotations", "captions_train2017.json")

PRUNING_RATIO = 0.9
ROUND_TO = 4
ITER_STEPS = 10

BATCH_SIZE = 32
EPOCHS = 1
LR = 2e-5
WD = 0.05
NUM_WORKERS = 8
AMP = True


# -----------------------
# Load / Utils
# -----------------------
def load_full_clip(device="cuda"):
    clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        MODEL_NAME
    )
    clip_model = clip_model.to(device).eval()
    # fp32
    clip_model = clip_model.to(torch.float32).eval()

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return clip_model, preprocess_train, preprocess_val, tokenizer


def build_ignored_layers_for_visual(visual_only: nn.Module):
    ignored = []
    print(visual_only)
    for name, m in visual_only.named_modules():
        cls = m.__class__.__name__.lower()
        
        # Ignore Patch Embedding to prevent changing the base token dimension
        if name == "conv1":
            ignored.append(m)
        # Ignore ONLY the MultiheadAttention layer (not the ResidualAttentionBlock)
        elif name.endswith("attn") or cls == "multiheadattention":
            ignored.append(m)
        # Ignore LayerNorms to keep the main embedding dimension intact
        elif "layernorm" in cls or "ln_" in name:
            ignored.append(m)

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

def run_training(clip_model, train_loader, epoch):
    clip_model.train()
    params = [p for p in clip_model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda", enabled=(AMP and DEVICE.startswith("cuda")))

    # 7) train
    for epoch in range(EPOCHS):
        for it, (images, tokens) in tqdm(enumerate(train_loader), total=len(train_loader)):
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
    return clip_model




# -----------------------
# Main
# -----------------------
def main():
    # 1) load FULL CLIP
    clip_model, preprocess_train, _, tokenizer = load_full_clip(DEVICE)
    print(f"[Init] params = {count_params(clip_model)/1e6:.3f}M")

    macs, nparams = tp.utils.count_ops_and_params(clip_model.visual, example_inputs=(torch.randn(1, 3, 224, 224, device=DEVICE),))
    print(f"Visual-only: {nparams/1e6:.3f}M, {macs/1e9:.3f}GMACs")

    example_inputs = (torch.randn(1, 3, 224, 224, device=DEVICE, dtype=torch.float32),)
    ignored_layers = build_ignored_layers_for_visual(clip_model.visual)

    # 3) prune VISUAL ONLY (in-place)
    importance = tp.importance.MagnitudeImportance(p=2)
    pruner = tp.pruner.MagnitudePruner(
        clip_model.visual,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=PRUNING_RATIO,
        ignored_layers=ignored_layers,
        iterative_steps=ITER_STEPS,
        round_to=ROUND_TO,
    )

    # clip_model.visual has been pruned (same module)
    macs, nparams = tp.utils.count_ops_and_params(clip_model.visual, example_inputs=(torch.randn(1, 3, 224, 224, device=DEVICE),))
    print(f"Visual-only: {nparams/1e6:.3f}M, {macs/1e9:.3f}GMACs")
    # print(clip_model)

    # 4) freeze TEXT tower; train only VISUAL (+ optionally logit_scale)
    for p in clip_model.transformer.parameters():
        p.requires_grad = False
    clip_model.logit_scale.requires_grad = True

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
    
    for prune_epoch in range(ITER_STEPS):
        print(f"=== Pruning iteration {prune_epoch+1}/{ITER_STEPS} ===")
        pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(clip_model.visual, example_inputs=(torch.randn(1, 3, 224, 224, device=DEVICE),))
        print(f"Visual-only: {nparams/1e6:.3f}M, {macs/1e9:.3f}GMACs")
        # 6) finetune
        clip_model = run_training(clip_model, train_loader, EPOCHS)
        recall10 = evaluate_coco(clip_model)
        with open(PRUNING_LOG, "a") as f:
            f.write(f"Iter {prune_epoch+1}/{ITER_STEPS}: params={nparams/1e6:.3f}M, macs={macs/1e9:.3f}GMACs, R@10={recall10:.4f}\n")
        out = f"./laion_pruned_finetuned_coco_{prune_epoch + 1}_{ITER_STEPS}.pth"
        torch.save(clip_model, out)
        print("Saved:", out)



    

    # 8) save



if __name__ == "__main__":
    main()