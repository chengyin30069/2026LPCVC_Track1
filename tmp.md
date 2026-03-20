```bash
python prepare_track1_dataset.py --output-dir dataset --include-coco2017-train --max-coco-images 80000 --max-coco2017-images 118000 --max-vg-images 70000 --max-coco-captions-per-image 10 --max-vg-regions-per-image 10 --max-unique-texts 250000 --overwrite
```

```bash
python train_student_distill.py \
--img-list dataset_image_only/img_list.csv \
--image-folder dataset_image_only/images \
--teacher-embeddings artifacts/teacher_image_embeddings_image_only.npz \
--output-dir artifacts/student_distill_v4 \
--resume-checkpoint best_loss_checkpoint.pt \
--epochs 20 \
--batch-size 32 \
--grad-accumulation 4 \
--lr 1e-5 \
--weight-decay 1e-4 \
--contrastive-weight 0.0 \
--distill-weight 1.0 \
--distill-loss-type cosine \
--baseline-anchor-weight 0.1 \
--baseline-anchor-final-weight 0.0 \
--hard-negative-weight 0.0 \
--lr-scheduler cosine \
--warmup-steps 500 \
--min-lr 1e-7 \
--val-split 0.05 \
--best-checkpoint-metric val-loss \
--grad-clip-norm 1.0 \
--unfreeze-last-n-blocks 2 \
--gradient-checkpointing \
--save-epoch-checkpoints
```

```bash
python precompute_teacher_embeddings.py --img-list dataset_image_only/img_list.csv --image-folder dataset_image_only/images --model-id hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K --output artifacts/teacher_image_embeddings_image_only.npz --batch-size 128 --channels-last --failed-images-log artifacts/teacher_image_embeddings_image_only.skipped.tsv
```

```bash
python train_student_distill.py --img-list dataset_image_only/img_list.csv --image-folder dataset_image_only/images --teacher-embeddings artifacts/teacher_image_embeddings_image_only.npz --output-dir artifacts/student_distill_image_only_v1 --epochs 20 --batch-size 128 --grad-accumulation 2 --contrastive-weight 0 --distill-weight 1.0 --distill-loss-type cosine --baseline-anchor-weight 0.2 --baseline-anchor-final-weight 0.0 --hard-negative-weight 0 --lr-scheduler cosine --warmup-steps 500 --min-lr 1e-7 --val-split 0.05 --best-checkpoint-metric val-loss --grad-clip-norm 1.0 --unfreeze-last-n-blocks 1 --gradient-checkpointing --missing-teacher-log artifacts/student_distill_image_only_v1/missing_teacher_images.txt
```

```bash
python prepare_track1_dataset.py --output-dir dataset --include-coco2017-train --max-coco-images 80000 --max-coco2017-images 118000 --max-vg-images 70000 --max-coco-captions-per-image 10 --max-vg-regions-per-image 10 --max-unique-texts 250000 --overwrite
```

```bash
python precompute_teacher_embeddings.py --img-list dataset_image_only/img_list.csv --image-folder dataset_image_only/images --model-id hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K --output teacher_image_embeddings_image_only.npz --batch-size 128 --channels-last --failed-images-log teacher_image_embeddings_image_only.skipped.tsv
```

```bash
python train_student_distill.py --img-list dataset_image_only/img_list.csv --image-folder dataset_image_only/images --teacher-embeddings teacher_image_embeddings_image_only.npz --output-dir artifacts/student_distill_image_only_v2 --epochs 20 --batch-size 128 --grad-accumulation 2 --contrastive-weight 0 --distill-weight 1.0 --distill-loss-type cosine --baseline-anchor-weight 0.2 --baseline-anchor-final-weight 0.0 --hard-negative-weight 0 --lr-scheduler cosine --warmup-steps 500 --min-lr 1e-7 --val-split 0.05 --best-checkpoint-metric val-loss --grad-clip-norm 1.0 --unfreeze-last-n-blocks 1 --gradient-checkpointing --missing-teacher-log artifacts/student_distill_image_only_v2/missing_teacher_images.txt
```

```bash
python train_student_distill.py --img-list dataset_image_only/img_list.csv --image-folder dataset_image_only/images --teacher-embeddings artifacts/teacher_image_embeddings_image_only.npz --output-dir artifacts/student_stage1_image_only --epochs 12 --batch-size 128 --grad-accumulation 2 --lr 5e-6 --weight-decay 1e-4 --contrastive-weight 0 --distill-weight 0.7 --distill-loss-type cosine --baseline-anchor-weight 0.3 --baseline-anchor-final-weight 0.1 --hard-negative-weight 0 --lr-scheduler cosine --warmup-steps 500 --min-lr 1e-7 --val-split 0.05 --best-checkpoint-metric val-loss --grad-clip-norm 1.0 --unfreeze-last-n-blocks 0 --gradient-checkpointing --missing-teacher-log artifacts/student_stage1_image_only/missing_teacher_images.txt
```

```bash
python train_student_distill.py --img-list dataset/img_list.csv --image-folder dataset/images --text-embeddings artifacts/text_embeddings.npz --teacher-embeddings artifacts/teacher_image_embeddings.npz --hard-negatives artifacts/hard_negatives.npz --output-dir artifacts/student_stage2_text_align --resume-checkpoint artifacts/student_stage1_image_only/best_loss_checkpoint.pt --epochs 8 --batch-size 96 --grad-accumulation 2 --lr 5e-6 --weight-decay 1e-4 --contrastive-weight 1.0 --distill-weight 0.2 --distill-loss-type cosine --baseline-anchor-weight 0.05 --baseline-anchor-final-weight 0.0 --hard-negative-weight 0.03 --num-hard-negatives 4 --lr-scheduler cosine --warmup-steps 500 --min-lr 1e-7 --val-split 0.05 --best-checkpoint-metric val-recall@10 --val-recall-max-texts 0 --val-recall-image-chunk-size 256 --val-recall-text-chunk-size 8192 --grad-clip-norm 1.0 --unfreeze-last-n-blocks 2 --gradient-checkpointing --save-epoch-checkpoints
```

```bash
python run_two_stage_distill.py --gradient-checkpointing --device cuda
```

python precompute_teacher_embeddings.py --img-list dataset/img_list.csv --image-folder dataset/images --output artifacts/teacher_image_embeddings.npz

python cache_text_embeddings.py --txt-list dataset/txt_list.csv --output artifacts/text_embeddings.npz

python mine_hard_negatives.py --embeddings artifacts/text_embeddings.npz --output artifacts/hard_negatives.npz --csv-output artifacts/hard_negatives.csv --top-k 10

run_two_stage_distill.py --skip-stage1 --gradient-checkpointing --device cuda:1