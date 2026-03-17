```bash
python train_student_distill.py \
--img-list img_list.csv \
--image-folder dataset/images \
--text-embeddings text_embeddings.npz \
--teacher-embeddings teacher_image_embeddings.npz \
--hard-negatives hard_negatives.npz \
--output-dir artifacts/student_distill_v4 \
--resume-checkpoint best_loss_checkpoint.pt \
--epochs 20 \
--batch-size 32 \
--grad-accumulation 4 \
--lr 1e-5 \
--weight-decay 1e-4 \
--distill-weight 0.3 \
--baseline-anchor-weight 0.1 \
--hard-negative-weight 0.05 \
--num-hard-negatives 4 \
--unfreeze-last-n-blocks 2 \
--gradient-checkpointing \
--save-epoch-checkpoints
```