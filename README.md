# LPCVC 2026 Track 1 - Image-to-Text Retrieval Sample Solution

## For Submissions

Check out [this repo](https://github.com/lpcvai/25LPCVC_AIHub_Guide) for more details on how to run models on AIHub.

## Overview

This repository contains Python scripts designed to extract, compile, and profile the OpenAI-CLIP's image and text encoders using the `qai_hub` library. It also includes scripts for uploading datasets and running inference with evaluation metrics such as Recall@10.

The current training path extends the baseline OpenCLIP image tower with a lightweight student head for LPCVC Track1 while keeping deployment latency effectively unchanged. The working recipe is:

* Baseline backbone: `ViT-B-16` with `datacomp_xl_s13b_b90k` weights.
* Student head: `Linear(512 -> 512) + LayerNorm + L2 normalize`.
* Trainable scope: projection head plus the last visual transformer block by default.
* Frozen scope: text tower stays frozen and all candidate texts are cached offline.
* Distillation: teacher image embeddings are precomputed once offline, then loaded from disk during training.
* Negatives: hard negatives are mined offline from cached text embeddings.

## **Table of Contents**

1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Current Architecture](#current-architecture)
5. [Training Flow](#training-flow)
6. [Benchmark with CLIP_benchmark](#benchmark-with-clip_benchmark)
7. [Usage](#usage)

---

## **Features**

* **Preprocessing Scripts**: Includes resizing and normalization for image inputs, and tokenization for text inputs.
* Extract CLIP Encoders: Extract image and text encoders from OpenAI-CLIP model and export as ONNX models.
* **Model Compilation**: Supports compiling the model for a specific target device using QAI Hub.
* **Model Profiling**: Submit and retrieve profiling results via QAI Hub.
* **Dataset Upload**: Upload image and text datasets to AI Hub for inference.
* **Inference & Evaluation**: Run inference on datasets and compute metrics such as Recall@10.

---

## **Requirements**

* Python 3.9+
* Torch and torchvision
* QAI Hub
* Required packages listed in `requirements.txt`
* `clip-benchmark` for public zero-shot benchmark runs

---

## **Installation**

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/lpcvai/26LPCVC_Track1_Sample_Solution.git
cd 26LPCVC_Track1_Sample_Solution
```

### **Step 2: Install Dependencies**

Ensure you have Python 3.9+ installed. Install the required Python packages:

```bash
pip install --no-build-isolation -r requirements.txt
```

If you also need the AI Hub compile / profile flow, install the QAI-only dependencies separately:

```bash
pip install --no-build-isolation -r requirements-qai.txt
```

For public retrieval benchmark runs with [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark):

```bash
pip install clip-benchmark
```


## uv Installation

### Init

```bash
uv init --python 3.10
```

### Package

```bash
uv add -r requirements.txt --no-build-isolation
```

For QAI Hub tooling in the same environment:

```bash
uv add -r requirements-qai.txt --no-build-isolation
```

For CLIP_benchmark in the same environment:

```bash
uv add clip-benchmark
```



---

## **Current Architecture**

The current Track1 training setup keeps the deployed backbone close to the baseline and moves all expensive operations offline.

```text
image
	-> OpenCLIP ViT-B-16 backbone
	-> projection head: Linear(512 -> 512)
	-> LayerNorm
	-> L2 normalize
	-> image embedding
```

Training uses the following side inputs:

```text
cached teacher image embeddings -> required distillation target
cached text embeddings          -> optional contrastive positives
cached hard-negative ids        -> optional ranking-aware negatives
```

Loss used by `train_student_distill.py`:

```text
total_loss = contrastive_weight * contrastive_loss
					 + distill_weight * distill_loss
					 + baseline_anchor_weight * anchor_loss
					 + hard_negative_weight * margin_loss
```

Default training characteristics:

* student backbone: `hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K`
* teacher backbone: `hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K`
* mixed precision on CUDA
* `batch_size=64`
* `grad_accumulation=4`
* `epochs=14`
* `lr=1e-5`
* `weight_decay=1e-4`

## **Training Flow**

The current end-to-end training flow is:

1. Build a Track1-style dataset from COCO and Visual Genome.
2. (Recommended) build a larger image-only distillation dataset from all local image sources.
3. Precompute teacher image embeddings once offline.
4. Train the student image model with image-only distillation as the default.
5. Optionally add text-cache contrastive / hard-negative losses later.
6. Evaluate the trained checkpoint on the local LPCVC Track1 metric.

### **Step 0. Build a Low-Friction Training Dataset**

The recommended first dataset build uses official sources for:

* COCO 2014 train captions + train images
* Visual Genome region descriptions + image metadata + image archives

This avoids Hugging Face loader issues and directly generates the Track1 files expected by the training scripts.

```bash
python prepare_track1_dataset.py \
	--output-dir dataset \
	--include-coco2017-train \
	--max-coco-images 80000 \
	--max-coco2017-images 118000 \
	--max-vg-images 70000 \
	--max-coco-captions-per-image 10 \
	--max-vg-regions-per-image 10 \
	--max-unique-texts 250000
```

By default the script downloads the needed COCO and Visual Genome archives into `raw_datasets/` and then creates symlinks into `dataset/images/` so image storage is not duplicated.

Generated files:

```text
dataset/images/
dataset/txt_list.csv
dataset/img_list.csv
dataset/summary.json
```

### **Step 1. (Recommended) Build a Larger Image-Only Distillation Set**

```bash
python prepare_image_only_distill_dataset.py \
	--output-dir dataset_image_only \
	--shuffle \
	--max-images 190000 \
	--overwrite
```

Default source folders are:

```text
raw_datasets/coco2014/images/train2014
raw_datasets/visual_genome/VG_100K/VG_100K
raw_datasets/visual_genome/VG_100K_2/VG_100K_2
coco2017/val2017
```

The script creates:

```text
dataset_image_only/images/
dataset_image_only/img_list.csv
dataset_image_only/summary.json
```

### **Step 2. Precompute Teacher Image Embeddings**

```bash
python precompute_teacher_embeddings.py \
	--img-list dataset_image_only/img_list.csv \
	--image-folder dataset_image_only/images \
	--output artifacts/teacher_image_embeddings_image_only.npz
```

Default teacher is now `hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K`.
To switch teacher models, override `--model-id`.

### **Step 3. Train the Student (Image-Only Distillation First)**

```bash
python train_student_distill.py \
  --img-list dataset_image_only/img_list.csv \
  --image-folder dataset_image_only/images \
  --teacher-embeddings artifacts/teacher_image_embeddings_image_only.npz \
  --output-dir artifacts/student_distill_image_only_v1 \
	--epochs 12 \
  --batch-size 256 \
  --grad-accumulation 1 \
	--lr 5e-6 \
  --weight-decay 1e-4 \
  --contrastive-weight 0.0 \
	--distill-weight 0.7 \
	--distill-loss-type cosine \
	--baseline-anchor-weight 0.3 \
	--baseline-anchor-final-weight 0.1 \
  --hard-negative-weight 0.0 \
	--lr-scheduler cosine \
	--warmup-steps 500 \
	--min-lr 1e-7 \
	--val-split 0.05 \
	--best-checkpoint-metric val-loss \
	--grad-clip-norm 1.0 \
	--unfreeze-last-n-blocks 0 \
  --gradient-checkpointing \
  --save-epoch-checkpoints
```

For single-GPU 3090Ti, start with `--batch-size 128` and `--grad-accumulation 2` if you see unstable training.

### **Step 4. (Optional) Enable Text-Based Losses Later**

```bash
python cache_text_embeddings.py \
	--txt-list dataset/txt_list.csv \
	--output artifacts/text_embeddings.npz

python mine_hard_negatives.py \
	--embeddings artifacts/text_embeddings.npz \
	--output artifacts/hard_negatives.npz \
	--csv-output artifacts/hard_negatives.csv \
	--top-k 10

python train_student_distill.py \
	--img-list dataset/img_list.csv \
	--image-folder dataset/images \
	--text-embeddings artifacts/text_embeddings.npz \
	--teacher-embeddings artifacts/teacher_image_embeddings.npz \
	--hard-negatives artifacts/hard_negatives.npz \
	--contrastive-weight 1.0 \
	--distill-weight 0.2 \
	--hard-negative-weight 0.03 \
	--num-hard-negatives 4 \
	--val-split 0.05 \
	--best-checkpoint-metric val-recall@10 \
	--val-recall-max-texts 0 \
	--val-recall-image-chunk-size 256 \
	--val-recall-text-chunk-size 8192
```

`val-recall@10` is available only when text supervision is enabled and a validation split exists.
If not available, the trainer automatically falls back to loss-based checkpoint selection.

### **Step 4.5. Run Stage1 + Stage2 with One Command**

```bash
python run_two_stage_distill.py \
	--gradient-checkpointing \
	--device cuda
```

Dry-run mode (print commands without executing):

```bash
python run_two_stage_distill.py --dry-run
```

### **Step 5. Evaluate the Trained Checkpoint Locally**

```bash
python local_inference_openclip.py \
	--txt-list dataset/txt_list.csv \
	--img-list dataset/img_list.csv \
	--image-folder dataset/images \
	--student-checkpoint artifacts/student_distill_image_only_v1/best_loss_checkpoint.pt
```

This path evaluates the exact student checkpoint used in this repository on the Track1 retrieval metric.

## **Benchmark with CLIP_benchmark**

Use [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark) to benchmark the public OpenCLIP backbones used by this project on standard zero-shot retrieval datasets, then compare them against the Track1-local result from `local_inference_openclip.py`.

### **Benchmark the Current Baseline Backbone**

The repository baseline corresponds to:

* `--model ViT-B-16`
* `--pretrained datacomp_xl_s13b_b90k`

Example on Flickr30k retrieval:

```bash
clip_benchmark eval \
	--model ViT-B-16 \
	--pretrained datacomp_xl_s13b_b90k \
	--model_type open_clip \
	--dataset flickr30k \
	--task zeroshot_retrieval \
	--language en \
	--dataset_root benchmark_datasets/{dataset} \
	--output benchmark/flickr30k_vitb16_datacomp_xl_s13b_b90k.json \
	--batch_size 64
```

Example on COCO retrieval:

```bash
clip_benchmark eval \
	--model ViT-B-16 \
	--pretrained datacomp_xl_s13b_b90k \
	--model_type open_clip \
	--dataset mscoco_captions \
	--task zeroshot_retrieval \
	--language en \
	--dataset_root benchmark_datasets/{dataset} \
	--output benchmark/mscoco_vitb16_datacomp_xl_s13b_b90k.json \
	--batch_size 64
```

### **Benchmark Multiple Public Backbones Together**

Create `benchmark/models.txt` with one model per line:

```text
ViT-B-16,datacomp_xl_s13b_b90k
ViT-L-14,datacomp_xl_s13b_b90k
```

Then run a retrieval benchmark sweep:

```bash
clip_benchmark eval \
	--pretrained_model benchmark/models.txt \
	--dataset retrieval \
	--dataset_root "benchmark_datasets/{dataset}" \
	--output "benchmark/{dataset}_{pretrained}_{model}_{language}_{task}.json" \
	--batch_size 64 \
	--skip_existing
```

Aggregate the JSON outputs into one CSV table:

```bash
clip_benchmark build benchmark/ --output benchmark/summary.csv
```

### **Important Note About the Student Checkpoint**

`train_student_distill.py` saves a repository-specific student checkpoint containing the OpenCLIP image tower plus the custom projection head. That checkpoint is **not** a drop-in OpenCLIP pretrained weight file, so `clip_benchmark eval` cannot load it directly without adding a custom model loader inside CLIP_benchmark.

For now, use this split:

* `clip_benchmark`: benchmark the public baseline / teacher backbones on standard datasets.
* `local_inference_openclip.py`: evaluate the trained student checkpoint on LPCVC Track1 data.

---

## **Usage**

### **1. Export ONNX Models**

Execute the script to export the encoders as ONNX models:

```bash
python export_onnx.py
```

### **2. Compile and Profile**

```bash
python compile_and_profile.py
```

This script will:

* Upload the ONNX models to AI Hub and submit a compile job.
* Submit a profiling job with the compiled models.

### **3. Upload Dataset**

Before running inference, datasets must be uploaded to AI Hub using `upload_dataset.py`. This script handles:

* Formatting images and text data into the structure expected by QAI Hub. (image: (1,3,224,224), txt: (1,77))
* Uploading the dataset and returning a dataset ID to be used in inference scripts.

```bash
python upload_dataset.py
```

This will print a `dataset_id` that you can use in `inference.py`.

### **4. Run Inference and Evaluate**

The `inference.py` script runs the compiled models on the uploaded datasets:

1. Retrieves the compiled image and text encoders from AI Hub.
2. Runs inference on the uploaded datasets.
3. Collects output embeddings for images and text.
4. Computes evaluation metrics, such as **Recall@10**, which measures how often the correct text is among the top-10 retrieved results for each image.

```bash
python inference.py
```

After completion, the script prints the Recall@10 score for the dataset.

### **5. Build an Image-Only Distillation Dataset**

Use this when you want to maximize distillation image count without requiring paired text labels.

```bash
python prepare_image_only_distill_dataset.py \
	--output-dir dataset_image_only \
	--shuffle \
	--max-images 190000 \
	--overwrite
```

### **6. Precompute Teacher Image Embeddings**

```bash
python precompute_teacher_embeddings.py \
	--img-list dataset_image_only/img_list.csv \
	--image-folder dataset_image_only/images \
	--output artifacts/teacher_image_embeddings_image_only.npz
```

### **7. Train the Student (Image-Only First)**

```bash
python train_student_distill.py \
	--img-list dataset_image_only/img_list.csv \
	--image-folder dataset_image_only/images \
	--teacher-embeddings artifacts/teacher_image_embeddings_image_only.npz \
	--output-dir artifacts/student_distill_image_only_v1 \
	--contrastive-weight 0.0 \
	--distill-weight 1.0 \
	--hard-negative-weight 0.0
```

### **8. Optional: Add Text Cache + Hard Negatives Later**

If image-only distillation is stable and you want extra retrieval gains, re-enable text-side losses.

```bash
python cache_text_embeddings.py \
	--txt-list dataset/txt_list.csv \
	--output artifacts/text_embeddings.npz
```

By default this uses the baseline OpenCLIP ViT-B/16 model together with a small prompt ensemble:

```text
{}
a photo of {}
an image of {}
a picture of {}
this is {}
```

Then mine hard negatives and train with text supervision:

```bash
python mine_hard_negatives.py \
	--embeddings artifacts/text_embeddings.npz \
	--output artifacts/hard_negatives.npz \
	--csv-output artifacts/hard_negatives.csv \
	--top-k 10

python train_student_distill.py \
	--img-list dataset/img_list.csv \
	--image-folder dataset/images \
	--text-embeddings artifacts/text_embeddings.npz \
	--teacher-embeddings artifacts/teacher_image_embeddings.npz \
	--hard-negatives artifacts/hard_negatives.npz \
	--contrastive-weight 1.0 \
	--distill-weight 0.3 \
	--hard-negative-weight 0.05 \
	--num-hard-negatives 4
```

The implementation uses chunked cosine search, so it avoids materializing the full $N \times N$ similarity matrix in memory.

### **9. Evaluate a Trained Student Checkpoint**

```bash
python local_inference_openclip.py \
	--student-checkpoint artifacts/student_distill/student_checkpoint.pt
```

This evaluates the trained projection head and any unfrozen vision-block updates using the same local Track1 metric.
