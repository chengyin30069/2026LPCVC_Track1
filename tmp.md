python cache_text_embeddings.py --txt-list dataset/txt_list.csv --output artifacts/text_embeddings.npz --model-id hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K --backend open_clip --device cuda:0

python cache_text_embeddings.py --txt-list dataset/txt_list.csv --output artifacts/text_embeddings_teacher_siglip2.npz --model-id google/siglip2-so400m-patch16-512 --backend transformers --device cuda:1

python precompute_teacher_embeddings.py \
--img-list dataset/img_list.csv \
--image-folder dataset/images \
--output artifacts/teacher_image_embeddings_siglip2.npz \
--model-id google/siglip2-so400m-patch16-512 \
--backend transformers \
--devices cuda:0,cuda:1 \
--batch-size 320 \
--num-workers 12 \
--prefetch-factor 4 \
--max-loader-buffer-gb 12 \
--decoder auto \
--channels-last \
--attn-impl sdpa \
--model-dtype float16 \
--output-dtype float32



python mine_hard_negatives.py --embeddings artifacts/text_embeddings.npz --output artifacts/hard_negatives.npz --csv-output artifacts/hard_negatives.csv --top-k 10

python train_student_distill.py --img-list dataset/img_list.csv --image-folder dataset/images --text-embeddings artifacts/text_embeddings.npz --teacher-embeddings artifacts/teacher_image_embeddings_siglip2.npz --teacher-text-embeddings artifacts/text_embeddings_teacher_siglip2.npz --hard-negatives artifacts/hard_negatives.npz --output-dir artifacts/student_stage2_text_align_30m --device cuda:0,cuda:1