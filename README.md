# LPCVC 2026 Track1 Distillation (Retrieval-Focused Refactor)

This repository is refactored into a modular distillation stack aimed at improving local Image-to-Text retrieval Recall@10 for:

- Student: `laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K`
- Teacher: `google/siglip2-so400m-patch16-512`

The benchmark entrypoint stays `benchmark.py` unchanged.

## What Was Refactored

- `train_student_distill.py` remains available as the direct training entrypoint.
- Core training logic moved into `distill/` modules:
  - `distill/args.py`: CLI + validation
  - `distill/data.py`: dataset, filtering, workers
  - `distill/losses.py`: retrieval-oriented loss functions
  - `distill/features.py`: intermediate block distillation hooks
  - `distill/optim.py`: schedule utilities
  - `distill/trainer.py`: training/eval/checkpoint loop
- `run_two_stage_distill.py` now supports recipe presets to reduce repeated manual tuning.

## Implemented Techniques

The training stack directly includes the requested advanced distillation strategies:

1. Temperature tuning
- Teacher and student temperatures are decoupled:
  - `--distill-teacher-temperature`
  - `--distill-student-temperature`
- Contrastive temperature can be learnable:
  - `--learnable-temperature`

2. Cross-batch / memory bank distillation
- Enabled by:
  - `--memory-bank-size`
  - `--memory-bank-distill-weight`
  - `--memory-bank-distill-temperature`

3. Hard-negative emphasis
- Margin objective over mined negatives with softmax hardness weighting:
  - `--hard-negative-weight`
  - `--num-hard-negatives`
  - `--hard-negative-weighting softmax`

4. Intermediate feature distillation
- ViT block-level relation matching between teacher and student:
  - `--intermediate-distill-weight`
  - `--intermediate-distill-num-blocks`
  - `--intermediate-distill-frequency`

## Reference Paper Alignment

The refactor follows the retrieval-oriented and affinity/relation distillation spirit from:
- TinyCLIP and related CLIP distillation literature
- Your requested reference: https://arxiv.org/abs/2307.12732

## Workflow

### 0) Environment (conda lpcv)

```bash
conda activate lpcv
python -V
```

If you prefer not to activate first, replace `python ...` with ` python ...` as shown below.

### 1) Build datasets

```bash
 python prepare_track1_dataset.py --output-dir dataset
 python prepare_image_only_distill_dataset.py --output-dir dataset_image_only --overwrite
```

### 2) Precompute caches

```bash
 python precompute_teacher_embeddings.py \
  --img-list dataset_image_only/img_list.csv \
  --image-folder dataset_image_only/images \
  --output artifacts/teacher_image_embeddings_image_only.npz

 python precompute_teacher_embeddings.py \
  --img-list dataset/img_list.csv \
  --image-folder dataset/images \
  --output artifacts/teacher_image_embeddings.npz

 python cache_text_embeddings.py \
  --txt-list dataset/txt_list.csv \
  --output artifacts/text_embeddings.npz

 python cache_text_embeddings.py \
  --backend transformers \
  --model-id google/siglip2-so400m-patch16-512 \
  --txt-list dataset/txt_list.csv \
  --output artifacts/text_embeddings_teacher_siglip2.npz

 python mine_hard_negatives.py \
  --embeddings artifacts/text_embeddings.npz \
  --output artifacts/hard_negatives.npz \
  --csv-output artifacts/hard_negatives.csv \
  --top-k 10

# SigLIP2 teacher image cache (used by stage2 when --teacher-preset siglip2)
 python precompute_teacher_embeddings.py \
  --backend transformers \
  --model-id google/siglip2-so400m-patch16-512 \
  --img-list dataset/img_list.csv \
  --image-folder dataset/images \
  --output artifacts/teacher_image_embeddings_siglip2.npz
```

### 3) Launch two-stage distillation

```bash
 python run_two_stage_distill.py --recipe clipkd --device cuda:1

# Run stage2 with SigLIP2 teacher preset (expects the two SigLIP2 caches above)
 python run_two_stage_distill.py --recipe clipkd --teacher-preset siglip2 --device cuda:1

# Disable ICL if teacher text cache is unavailable
 python run_two_stage_distill.py --recipe clipkd --teacher-preset siglip2 --no-stage2-enable-icl --device cuda:1

# Enable intermediate block distill only with an OpenCLIP-compatible intermediate teacher
 python run_two_stage_distill.py --recipe clipkd --teacher-preset siglip2 --stage2-intermediate-teacher-forward --device cuda:1
```

Available stage2 presets:
- `--recipe clipkd` (paper-guided default: FD+ICL+CRD style)
- `--recipe stable`
- `--recipe strong`
- `--recipe aggressive`

Use dry-run to inspect full commands:

```bash
 python run_two_stage_distill.py --recipe clipkd --dry-run
```

## Notes

- Do not modify benchmark scripts if you need consistent local scoring.
- For retrieval quality, select the best checkpoint by `val_recall_at_10` whenever available.
- If VRAM is tight, reduce batch size and increase `--grad-accumulation`.
