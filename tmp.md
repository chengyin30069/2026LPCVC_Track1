# ============================================================
# CLIP-KD Distillation Training Commands
# Target: Recall@10 ≥ 45.2 on benchmark.py
# ============================================================
# Teacher: laion/CLIP-ViT-bigG-14-laion2B-39B-b160k (1280-dim)
# Student: laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K (512-dim)
# Dataset: 1,345,289 image-text pairs from datacomp_small
# Hardware: RTX 3090Ti (24GB VRAM)
# Conda env: lpcv
# Total time: ~30-35 hours (precompute + training)
# ============================================================

# ============================================================
# STEP 0: Activate Environment
# ============================================================
conda activate lpcv

# Verify environment
python -V
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Verify dataset
wc -l dataset/img_list.csv dataset/txt_list.csv
ls -lh dataset/images/ | head -5

# ============================================================
# STEP 1: Precompute bigG-14 Teacher Image Embeddings
# ============================================================
# Time: ~7.5 hours on single 3090Ti
# Output: 1,345,289 × 1280 × float16 = ~3.4GB
# Memory: ~18GB peak VRAM

python precompute_teacher_embeddings.py \
  --model-id "hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k" \
  --backend open_clip \
  --img-list dataset/img_list.csv \
  --image-folder dataset/images \
  --batch-size 32 \
  --output artifacts/teacher_image_embeddings_bigG14.npz \
  --output-dtype float16 \
  --channels-last \
  --device cuda:0

# Verify output
ls -lh artifacts/teacher_image_embeddings_bigG14.npz
python -c "import numpy as np; d=np.load('artifacts/teacher_image_embeddings_bigG14.npz'); print(f'Shape: {d[\"embeddings\"].shape}, dtype: {d[\"embeddings\"].dtype}')"

# ============================================================
# STEP 2: Precompute bigG-14 Teacher Text Embeddings
# ============================================================
# Time: ~25 minutes
# Output: 1,249,355 × 1280 × float16 = ~3.2GB
# Memory: ~8GB peak VRAM

python cache_text_embeddings.py \
  --model-id "hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k" \
  --backend open_clip \
  --txt-list dataset/txt_list.csv \
  --batch-size 512 \
  --output artifacts/text_embeddings_teacher_bigG14.npz \
  --device cuda:0

# Verify output
ls -lh artifacts/text_embeddings_teacher_bigG14.npz
python -c "import numpy as np; d=np.load('artifacts/text_embeddings_teacher_bigG14.npz'); print(f'Shape: {d[\"embeddings\"].shape}')"

# ============================================================
# STEP 3: Precompute Student Text Embeddings (for ICL loss)
# ============================================================
# Time: ~15 minutes
# Output: 1,249,355 × 512 × float16 = ~1.3GB
# Memory: ~6GB peak VRAM

python cache_text_embeddings.py \
  --model-id "hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K" \
  --backend open_clip \
  --txt-list dataset/txt_list.csv \
  --batch-size 512 \
  --output artifacts/text_embeddings.npz \
  --device cuda:0

# Verify output
ls -lh artifacts/text_embeddings.npz

# ============================================================
# STEP 3.5: Build cleaner CLIP-KD subset (recommended)
# ============================================================
# Filters noisy e-commerce / non-natural captions while preserving original IDs.
# This usually improves retrieval transfer stability on benchmark_legacy.
python prepare_clipkd_filtered_dataset.py \
  --img-list dataset/img_list.csv \
  --txt-list dataset/txt_list.csv \
  --output-dir dataset_clipkd \
  --min-chars 12 \
  --max-chars 120 \
  --min-words 4 \
  --max-words 28 \
  --min-letter-ratio 0.55 \
  --max-nonlatin-ratio 0.20 \
  --max-digit-ratio 0.35 \
  --min-kept-images 100000

# ============================================================
# STEP 4: Train Student with CLIP-KD bigG14 Recipe
# ============================================================
# Strategy: Projected FD + ICL + Contrastive
# Time: ~22-26 hours (40 epochs)
# Memory: ~16-18GB VRAM (safe for 24GB)
#
# Loss weights (from bigG14-fd-icl preset):
#   - projected_fd: 3.0 (MSE between proj(student_512→1280) and teacher_1280)
#   - icl: 0.15 → 0.08 (interactive contrastive learning, annealing)
#   - contrastive: 0.1 (standard CLIP InfoNCE loss)
#   - anchor: 0.005 → 0.001 (baseline drift prevention, annealing)
#
# Training config:
#   - Epochs: 40
#   - Batch size: 128 (effective: 128 × 8 = 1024 with grad accumulation)
#   - Learning rate: 5e-6 with cosine schedule
#   - Warmup: 2000 steps
#   - Weight decay: 0.05
#   - Trainable: last 2 ViT blocks + projection head + teacher projector (~19M params)
#   - Early stopping: patience=6 epochs on val_recall_at_10

python train_student_distill.py \
  --clipkd-preset bigG14-fd-icl \
  --img-list dataset/img_list.csv \
  --image-folder dataset/images \
  --teacher-embeddings artifacts/teacher_image_embeddings_bigG14.npz \
  --text-embeddings artifacts/text_embeddings.npz \
  --teacher-text-embeddings artifacts/text_embeddings_teacher_bigG14.npz \
  --output-dir artifacts/student-bigG14-CLIPKD \
  --device cuda:0

# Upstream CLIP-KD-style recipe (recommended if Recall@10 keeps dropping):
# - CE ICL + CKD + projected FD (image + text) with online student text encoding
# - partial unfreeze for both image/text towers
python train_student_distill.py \
  --clipkd-preset clipkd-upstream-bigg14 \
  --img-list dataset_clipkd/img_list.csv \
  --image-folder dataset/images \
  --teacher-embeddings artifacts/teacher_image_embeddings_bigG14.npz \
  --text-embeddings artifacts/text_embeddings.npz \
  --teacher-text-embeddings artifacts/text_embeddings_teacher_bigG14.npz \
  --output-dir artifacts/student-bigG14-CLIPKD-upstream \
  --device cuda:0

# Resume after interruption/kill (current latest: epoch 4) - FAST & LOW-RISK
# IMPORTANT: --epochs means "additional epochs", not total epochs.
# To continue from epoch 4 and end at epoch 40, set --epochs 36.
# Speed/safety settings:
#   - --num-workers 2 + --prefetch-factor 2: faster than 0 workers, but conservative RAM use
#   - --val-split 0: disable in-training validation to remove heavy recall/val overhead
#   - checkpointing per epoch still enabled by preset (for safe recovery)
python train_student_distill.py \
  --clipkd-preset bigG14-fd-icl \
  --img-list dataset/img_list.csv \
  --image-folder dataset/images \
  --teacher-embeddings artifacts/teacher_image_embeddings_bigG14.npz \
  --text-embeddings artifacts/text_embeddings.npz \
  --teacher-text-embeddings artifacts/text_embeddings_teacher_bigG14.npz \
  --output-dir artifacts/student-bigG14-CLIPKD \
  --resume-checkpoint artifacts/student-bigG14-CLIPKD/student_checkpoint_epoch_04.pt \
  --resume-optimizer-state \
  --epochs 36 \
  --num-workers 6 \
  --prefetch-factor 2 \
  --val-split 0 \
  --device cuda:0 2>&1 | tee -a artifacts/student-bigG14-CLIPKD/train_resume.log

# If interrupted again, continue from latest checkpoint automatically:
LATEST_CKPT=$(ls -1 artifacts/student-bigG14-CLIPKD/student_checkpoint_epoch_*.pt | sort | tail -n 1)
LAST_EPOCH=$(basename "$LATEST_CKPT" | sed -E 's/.*epoch_([0-9]+)\.pt/\1/')
REMAIN_EPOCHS=$((40 - LAST_EPOCH))
if [ "$REMAIN_EPOCHS" -le 0 ]; then
  echo "No remaining epochs (last=$LAST_EPOCH)."
  exit 0
fi
python train_student_distill.py \
  --clipkd-preset bigG14-fd-icl \
  --img-list dataset/img_list.csv \
  --image-folder dataset/images \
  --teacher-embeddings artifacts/teacher_image_embeddings_bigG14.npz \
  --text-embeddings artifacts/text_embeddings.npz \
  --teacher-text-embeddings artifacts/text_embeddings_teacher_bigG14.npz \
  --output-dir artifacts/student-bigG14-CLIPKD \
  --resume-checkpoint "$LATEST_CKPT" \
  --resume-optimizer-state \
  --epochs "$REMAIN_EPOCHS" \
  --num-workers 2 \
  --prefetch-factor 2 \
  --val-split 0 \
  --device cuda:0 2>&1 | tee -a artifacts/student-bigG14-CLIPKD/train_resume.log

# ============================================================
# STEP 5: Monitor Training Progress
# ============================================================

# In a separate terminal, watch GPU usage
watch -n 2 nvidia-smi

# Monitor training log in real-time
tail -f artifacts/student-bigG14-CLIPKD/train_resume.log

# Check latest training metrics
grep "\"total_loss\"" artifacts/student-bigG14-CLIPKD/train_resume.log | tail -10

# Note: with --val-split 0, validation is disabled during training for speed.
# Run benchmark sweep after training to select the best checkpoint.

# ============================================================
# STEP 6: Benchmark Evaluation - Sweep All Checkpoints
# ============================================================

# After training completes, evaluate all epoch checkpoints
# to find the one with highest Recall@10

mkdir -p artifacts/benchmark_results

echo "=== Sweeping all checkpoints for best Recall@10 ===" > artifacts/benchmark_results/sweep_log.txt

for epoch in {1..40}; do
  ckpt="artifacts/student-bigG14-CLIPKD/student_checkpoint_epoch_${epoch}.pt"
  if [ -f "$ckpt" ]; then
    echo "Evaluating epoch $epoch..."
    python benchmark.py --student-checkpoint "$ckpt" --device cuda:0 2>&1 \
      | tee -a artifacts/benchmark_results/sweep_log.txt
  fi
done

# Extract and sort results
echo "=== Top 10 checkpoints by Recall@10 ===" | tee artifacts/benchmark_results/top_checkpoints.txt
grep "student.*:" artifacts/benchmark_results/sweep_log.txt \
  | awk '/epoch/{epoch=$0} /student.*:/{print epoch, $3}' \
  | sort -k3 -nr \
  | head -10 \
  | tee -a artifacts/benchmark_results/top_checkpoints.txt

# Also evaluate the best_loss_checkpoint
python benchmark.py \
  --student-checkpoint artifacts/student-bigG14-CLIPKD/best_loss_checkpoint.pt \
  --device cuda:0

# ============================================================
# STEP 7: Verify Target Achievement
# ============================================================

# Check if best Recall@10 >= 45.2
best_score=$(grep "student.*:" artifacts/benchmark_results/sweep_log.txt \
  | awk '{print $3}' | sort -nr | head -1)

echo "Best Recall@10 achieved: $best_score"

# If score >= 45.2, proceed to export
# If score < 45.0, proceed to fallback strategy (STEP 8)

# ============================================================
# STEP 8: Fallback Strategy (if Recall@10 < 45.0)
# ============================================================

# Option 1: Retrain with enhanced CRD loss and higher FD weight
python train_student_distill.py \
  --clipkd-preset bigG14-fd-icl \
  --img-list dataset/img_list.csv \
  --image-folder dataset/images \
  --teacher-embeddings artifacts/teacher_image_embeddings_bigG14.npz \
  --text-embeddings artifacts/text_embeddings.npz \
  --teacher-text-embeddings artifacts/text_embeddings_teacher_bigG14.npz \
  --output-dir artifacts/student-bigG14-CLIPKD-v2 \
  --projected-fd-weight 4.0 \
  --projected-fd-final-weight 4.0 \
  --crd-weight 2.0 \
  --crd-final-weight 2.0 \
  --epochs 50 \
  --lr 3e-6 \
  --device cuda:0

# Option 2: Fine-tune best checkpoint with lower learning rate
python train_student_distill.py \
  --clipkd-preset bigG14-fd-icl \
  --img-list dataset/img_list.csv \
  --image-folder dataset/images \
  --teacher-embeddings artifacts/teacher_image_embeddings_bigG14.npz \
  --text-embeddings artifacts/text_embeddings.npz \
  --teacher-text-embeddings artifacts/text_embeddings_teacher_bigG14.npz \
  --output-dir artifacts/student-bigG14-CLIPKD-finetune \
  --resume-checkpoint artifacts/student-bigG14-CLIPKD/best_loss_checkpoint.pt \
  --epochs 20 \
  --lr 1e-6 \
  --projected-fd-weight 4.0 \
  --device cuda:0

# Option 3: Unfreeze more layers (3 blocks instead of 2)
python train_student_distill.py \
  --clipkd-preset bigG14-fd-icl \
  --img-list dataset/img_list.csv \
  --image-folder dataset/images \
  --teacher-embeddings artifacts/teacher_image_embeddings_bigG14.npz \
  --text-embeddings artifacts/text_embeddings.npz \
  --teacher-text-embeddings artifacts/text_embeddings_teacher_bigG14.npz \
  --output-dir artifacts/student-bigG14-CLIPKD-v3 \
  --unfreeze-last-n-blocks 3 \
  --epochs 40 \
  --device cuda:0

# ============================================================
# STEP 9: Export Best Model to ONNX (if target achieved)
# ============================================================

# After achieving Recall@10 >= 45.2, identify best checkpoint
best_ckpt=$(grep "student.*:" artifacts/benchmark_results/sweep_log.txt \
  | awk '/epoch/{epoch=$0} /student.*:/{print epoch}' \
  | sort -k3 -nr | head -1 \
  | awk '{print $2}' | tr -d ':')

echo "Exporting checkpoint: artifacts/student-bigG14-CLIPKD/student_checkpoint_epoch_${best_ckpt}.pt"

python export_onnx.py \
  --student-checkpoint "artifacts/student-bigG14-CLIPKD/student_checkpoint_epoch_${best_ckpt}.pt" \
  --output exported_onnx/student_bigG14_r45.onnx \
  --device cuda:0

# Verify ONNX model matches PyTorch performance
python benchmark.py \
  --student-onnx exported_onnx/student_bigG14_r45.onnx \
  --device cuda:0

# ============================================================
# Utility Scripts & Diagnostics
# ============================================================

# Plot training history
python plot_training_history.py \
  --checkpoint artifacts/student-bigG14-CLIPKD/best_loss_checkpoint.pt \
  --output artifacts/training_history.png

# Check checkpoint metadata
python -c "
import torch
ckpt = torch.load('artifacts/student-bigG14-CLIPKD/best_loss_checkpoint.pt')
print('Epoch:', ckpt.get('epoch', 'N/A'))
print('Best metric:', ckpt.get('best_metric_name', 'N/A'), '=', ckpt.get('best_metric_value', 'N/A'))
print('Training history length:', len(ckpt.get('history', [])))
if 'history' in ckpt and len(ckpt['history']) > 0:
    last = ckpt['history'][-1]
    print('Last epoch metrics:', {k: f'{v:.4f}' for k, v in last.items() if isinstance(v, float)})
"

# Compare multiple checkpoints side-by-side
python -c "
import sys
import torch
import numpy as np

checkpoints = [
    'artifacts/student-bigG14-CLIPKD/student_checkpoint_epoch_30.pt',
    'artifacts/student-bigG14-CLIPKD/student_checkpoint_epoch_35.pt',
    'artifacts/student-bigG14-CLIPKD/student_checkpoint_epoch_40.pt',
    'artifacts/student-bigG14-CLIPKD/best_loss_checkpoint.pt',
]

print(f'{'Checkpoint':<50} {'Val Loss':>10} {'Val R@10':>10}')
print('-' * 72)

for ckpt_path in checkpoints:
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        history = ckpt.get('history', [])
        if history:
            last = history[-1]
            val_loss = last.get('val_loss', float('nan'))
            val_r10 = last.get('val_recall_at_10', float('nan'))
            name = ckpt_path.split('/')[-1]
            print(f'{name:<50} {val_loss:>10.4f} {val_r10:>10.4f}')
    except Exception as e:
        print(f'{ckpt_path:<50} ERROR: {e}')
"

# ============================================================
# Troubleshooting
# ============================================================

# If training crashes with OOM (Out of Memory):
# 1. Reduce batch size: --batch-size 96 (instead of 128)
# 2. Increase grad accumulation: --grad-accumulation 12 (instead of 8)
# 3. Disable epoch checkpoints: remove --save-epoch-checkpoints (or use preset "none" and set args manually)

# If training is too slow:
# 1. Use cuda:1 (2nd 3090Ti) if available
# 2. Reduce validation frequency: --val-every-epochs 2
# 3. Use channels-last memory format: already enabled in preset

# If val_recall_at_10 plateaus early:
# 1. Increase projected_fd_weight: --projected-fd-weight 4.0
# 2. Add CRD loss: --crd-weight 2.0
# 3. Reduce learning rate: --lr 3e-6

# If loss becomes NaN:
# 1. Reduce learning rate: --lr 3e-6 or --lr 1e-6
# 2. Increase warmup: --warmup-steps 3000
# 3. Check teacher embeddings are valid (no NaN/Inf)

# ============================================================
# Expected Final Results
# ============================================================
# Target: Recall@10 >= 45.2 on COCO validation
# Baseline (without distillation): ~37-40
# With bigG-14 CLIP-KD: ~44-47 (expected +5-7% gain)
#
# If achieved:
#   - Student learned bigG-14 teacher knowledge successfully
#   - Cross-dimensional distillation (512→1280→512) worked
#   - FD+ICL losses were effective
#   - Ready for deployment (ONNX export)
#
# Timeline summary:
#   Step 1: 7.5h  (teacher image embeddings)
#   Step 2: 0.5h  (teacher text embeddings)
#   Step 3: 0.3h  (student text embeddings)
#   Step 4: 22-26h (training 40 epochs)
#   Step 5-7: 2h  (evaluation & benchmarking)
#   Total: ~33-37 hours
# ============================================================
