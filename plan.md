# plan.md

## Goal

Achieve **≥0.60 accuracy** in LPCV 2026 Track1 while:

* **Not increasing inference latency**
* **Training feasible on a single RTX 3060Ti (8GB VRAM)**

Current model:

```
CLIP-ViT-B-16-DataComp.XL-s13B-b90K
```

Task:

```
Retrieve the most relevant text descriptions for a given image
from a candidate pool
```

Constraints:

| Constraint        | Value           |
| ----------------- | --------------- |
| Inference latency | cannot increase |
| Training GPU      | RTX 3060Ti      |
| VRAM              | ~8GB            |
| Dataset           | moderate scale  |

Therefore training must use:

```
mixed precision
small batch sizes
gradient accumulation
frozen components
offline teacher features
```

---

# 1. Key Design Principle

To fit training on a **3060Ti**, we avoid training large models directly.

Instead:

```
Large model → offline feature extraction
Small model → train with distillation
```

This keeps training memory small.

---

# 2. Final Target Architecture (Inference)

The deployed model must remain extremely close to baseline.

```
Image
 ↓
ViT-B/16
 ↓
Projection head (512 → 512)
 ↓
Normalize
```

Extra parameters:

```
≈ 0.26M
```

Latency impact:

```
~0 ms (negligible)
```

---

# 3. Teacher Distillation (3060Ti-friendly)

Instead of running the teacher model during training (too expensive), **precompute teacher embeddings**.

Teacher model options:

* OpenCLIP ViT-L/14
* OpenCLIP ViT-H/14

Teacher only runs **once offline**.

Pipeline:

```
Dataset images
     ↓
Teacher CLIP
     ↓
save image embeddings
```

Stored format:

```
float16
512 or 768 dim
```

Memory estimate:

Example:

```
40k images × 768 × 2 bytes
≈ 60MB
```

Very small.

Training then becomes:

```
student(image) → embedding
teacher_embedding → loaded from disk
```

Loss:

```
L = contrastive_loss + λ * MSE(student_embed, teacher_embed)
```

Suggested weight:

```
λ = 0.3
```

---

# 4. Freeze Text Encoder

Text encoder does not need training.

Instead:

```
encode all candidate texts once
```

Save embeddings:

```
text_embeddings.npy
```

Benefits:

```
0 GPU cost during training
```

---

# 5. Hard Negative Mining (Low Compute)

Instead of random negatives, compute **offline nearest neighbors**.

Procedure:

1. Encode all text embeddings.
2. Compute cosine similarity.
3. For each text, keep top-k neighbors.

Example:

```
k = 10
```

Training pair:

```
(image_i, text_i)   positive
(image_i, text_j)   hard negative
```

This dramatically improves ranking quality without GPU cost.

---

# 6. Prompt Optimization

Prompt tuning is extremely cheap.

Two options:

## Option A — Prompt Ensemble

Use multiple templates:

```
"a photo of {}"
"an image of {}"
"a picture of {}"
"this is {}"
```

Compute:

```
mean(text_embedding_i)
```

Cost:

```
offline only
```

## Option B — CoOp Prompt Learning

Train small prompt tokens.

Parameters:

```
~8k
```

Memory footprint:

```
negligible
```

This is fully feasible on a 3060Ti.

---

# 7. Projection Head

Add a tiny projection head after the image encoder.

Architecture:

```
Linear(512 → 512)
LayerNorm
normalize
```

Parameter count:

```
512×512 ≈ 262k
```

Training cost:

```
minimal
```

This head allows dataset-specific adaptation.

---

# 8. Training Configuration (3060Ti Safe)

Recommended settings:

```
precision: fp16
batch_size: 32
grad_accumulation: 8
effective_batch: 256
```

Memory usage estimate:

```
~6–7GB VRAM
```

Framework:

```
PyTorch + AMP
```

Optimizer:

```
AdamW
lr = 1e-5
weight_decay = 1e-4
```

Epochs:

```
10–15
```

---

# 9. Memory Optimization Tricks

To ensure training fits 8GB:

### Freeze most layers

Train only:

```
projection head
last transformer block
```

This reduces gradients drastically.

---

### Gradient checkpointing

Enable:

```
torch.utils.checkpoint
```

Memory reduction:

```
~30%
```

---

### Smaller image resolution during finetune

CLIP default:

```
224×224
```

Use:

```
192×192
```

during training if memory is tight.

Inference still uses:

```
224×224
```

---

# 10. Quantization-Aware Training (Optional)

If deployment requires INT8:

```
QAT for last 2 epochs
```

Training setup:

```
fp32 → QAT fine-tune
```

Benefit:

```
recover accuracy lost from INT8 quantization
```

---

# 11. Expected Accuracy Gain

Estimated improvements:

| Technique       | Gain  |
| --------------- | ----- |
| Prompt tuning   | +3%   |
| Hard negatives  | +3–5% |
| Distillation    | +5–7% |
| Projection head | +1–2% |

Expected final accuracy:

```
0.52 → 0.62–0.66
```

---

# 12. Implementation Order (Fastest Path)

Follow this order.

### Phase 1 (1–2 days)

```
Prompt ensemble
Hard negative mining
```

Expected:

```
0.52 → ~0.56
```

---

### Phase 2 (3–5 days)

```
Teacher embedding distillation
```

Expected:

```
~0.56 → 0.60+
```

---

### Phase 3 (optional improvement)

```
Projection head
CoOp prompts
```

Expected:

```
0.60 → 0.63+
```

---

# 13. Training Pipeline Summary

Final training pipeline:

```
images
  ↓
student CLIP (ViT-B/16)
  ↓
projection head
  ↓
embedding

teacher embedding (precomputed)
text embedding (precomputed)

loss:
contrastive + distillation
```

GPU load:

```
only student model
```

Fits comfortably on a **3060Ti**.

---

# 14. Success Criteria

Primary goal:

Benchmark using https://github.com/LAION-AI/CLIP_benchmark
Compare the performance with CLIP-ViT-B-16-DataComp.XL-s13B-b90K base on this benchmark

```
Accuracy ≥ 0.60
```

Constraints satisfied:

```
Inference latency unchanged
Training feasible on 3060Ti
```

Stretch goal:

```
Accuracy ≥ 0.63
```
