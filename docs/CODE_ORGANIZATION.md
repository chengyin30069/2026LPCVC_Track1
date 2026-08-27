# Code Organization

This index follows the methods described in the two final presentations. It classifies the code into the final pipeline, deployment and validation, issue-specific fixes, and archived experiments.

## 1. Final Pipelines

### Distillation: H/14 Teacher → B/16 Student

| File | Purpose |
| --- | --- |
| `train_student_distill.py` | Entry point for distillation training. |
| `distill/args.py` | Training arguments, teacher/student settings, and loss weights. |
| `distill/data.py` | Dataset and DataLoader construction, including embedding-cache loading. |
| `distill/features.py` | Teacher/student feature extraction and projection. |
| `distill/losses.py` | Feature-distillation, contrastive, and ICL-style losses. |
| `distill/optim.py` | Optimizer and learning-rate scheduler setup. |
| `distill/trainer.py` | Main training, validation, checkpointing, and history loop. |
| `prepare_datacomp_track1_dataset.py` | Prepares DataComp and Track 1 data for distillation. |
| `prepare_image_only_distill_dataset.py` | Builds an image-only distillation dataset. |
| `prepare_track1_dataset.py` | Prepares competition-format data and metadata. |
| `precompute_teacher_embeddings.py` | Precomputes H/14 teacher image embeddings. |
| `cache_text_embeddings.py` | Precomputes text embeddings to reduce training cost. |
| `mine_hard_negatives.py` | Builds hard negatives for contrastive training. |
| `plot_training_history.py` | Plots training and validation history. |
| `utils/student_model.py` | Defines the student model and projection head. |
| `utils/track1_utils.py` | Shared Track 1 data and evaluation utilities. |
| `export_onnx.py` | Loads a distilled checkpoint and exports the student image encoder and text encoder. |

`student_checkpoint_epoch_13.pt` and the embedding caches are related artifacts, but they are too large for the source repository and are excluded from Git.

### Quantization: L/14 W8A16 → Mixed Precision → QAT

| File | Purpose |
| --- | --- |
| `mixed_qat.py` | Final L/14 mixed-precision AIMET QAT pipeline. Selected attention/MLP operators use A8 while the remaining operators retain A16. |
| `export_quantized_onnx.py` | Exports the L/14 W8A16 QDQ ONNX model, encodings, and explicit-QKV graph. |
| `quantize_image.py` | Image-encoder W8A16 PTQ baseline. |
| `quantize_text.py` | Text-encoder W8A16 PTQ and attention-mask handling. |
| `analysis.py` | AIMET per-layer sensitivity and quantization analysis. |
| `reports/mixed_precision/mmp_log.txt` | Mixed-precision configuration recorded for each block of the final L/14 model. |
| `reports/quant_analysis/text/` | Per-layer text-quantization analysis reports. |

`qat_clip_visual.pth` is the final L/14 QAT checkpoint, but it is excluded from Git.

## 2. QNN Deployment and Retrieval Evaluation

| File | Purpose |
| --- | --- |
| `compile_and_profile.py` | Validates ONNX, submits QAI Hub compilation for a QNN runtime target, and profiles on XR2 Gen 2. |
| `upload_dataset.py` | Prepares image/text inputs and uploads them as QAI Hub datasets. |
| `inference.py` | Runs remote inference, collects embeddings, and calculates Recall@10. |
| `benchmark.py` | Shared retrieval-similarity and Recall metric functions. |
| `local_inference.py` | Local baseline and exported-model validation. |
| `local_inference_openclip.py` | OpenCLIP reference-output validation. |
| `utils/img_processing.py` | Image preprocessing. |
| `utils/text_processing.py` | Tokenization and text preprocessing. |
| `utils/Dataset.py` | Dataset helpers used by quantization and evaluation. |

## 3. QNN Graph Fixes and Diagnostics

| File | Purpose |
| --- | --- |
| `remove_transpose.py` | Folds constant transposes applied to Linear weights so QDQ nodes do not break QNN Fully Connected pattern matching. |
| `add_zero_bias.py` | Adds zero biases to graph nodes that need a bias for QNN pattern matching. |
| `fix_input_rank.py` | Fixes input-rank and shape issues after export. |
| `compare_text_openclip_qdq.py` | Numerically compares the OpenCLIP and QDQ text encoders. |
| `mixed_test.py` | Reproduces mixed-quantization and graph issues with a single ViT block. |
| `split_attn_export.py` | Exports attention with explicit Q/K/V projections for QNN compatibility experiments. |

These files are not primary training entry points, but they directly address the QNN compilation and attention-mask issues described in the final results, so they remain part of the repository.

## 4. Archived Experiments Excluded from the Final Repository

The following items were moved rather than deleted and remain available at:

`../unused_experiments/2026LPCVC_Track1/`

- `code/`: early AIMET PTQ, manually generated encodings, H/14 mixed-quantization experiments, and one-off test scripts.
- `artifacts/H14-quant/`: the H/14 path that failed to compile, as reported in the presentation.
- `artifacts/aimet_mixed_quant_image/`: older output overwritten by an H/14 experiment; it is not the final L/14 mixed-QAT model.
- `artifacts/artifacts/`: early L/14-teacher/COCO distillation caches and checkpoints, not the final H/14 + CC3M path.
- `artifacts/torch_smoothquant/`: SmoothQuant experiments not used by the final method.
- Single-block ViT ONNX files and large external tensor data: debugging artifacts whose generating code is retained in the main repository.

Third-party repositories and large datasets outside the workspace were not moved, to avoid breaking the existing environment or duplicating disk usage. They are not part of this Git repository.
