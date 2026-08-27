# 2026 LPCVC Track 1 — Efficient Image–Text Retrieval

This repository contains the code used for the final project: mixed-precision quantization and quantization-aware training (QAT) for a CLIP image encoder, H/14-teacher-to-B/16-student distillation, and compilation, profiling, and Recall@10 evaluation on Qualcomm QAI Hub and the QNN HTP backend.

## Final Pipelines

1. **L/14 mixed-precision QAT:** Start from W8A16, assign A8 to sensitive operators, and fine-tune with a cosine feature loss.
2. **H/14 → B/16 distillation:** Train the student with precomputed teacher image and text embeddings using feature-distillation and contrastive/ICL-style losses.
3. **QNN deployment:** Export ONNX, fix QDQ graph patterns, compile to a QNN DLC through QAI Hub, and profile on the XR2 Gen 2 (Proxy) device.

See [docs/CODE_ORGANIZATION.md](docs/CODE_ORGANIZATION.md) for a complete classification of the primary pipeline, supporting tools, and archived experiments.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The quantization scripts also require a Qualcomm AIMET release compatible with the installed PyTorch version. AIMET is intentionally not pinned in `requirements.txt` to avoid installing an incompatible build; install it separately according to the official AIMET instructions.

## Common Entry Points

```bash
# Distillation training
python train_student_distill.py --help

# Export the distilled student and text encoder to ONNX
python export_onnx.py --help

# Export the L/14 W8A16 QDQ ONNX model and encodings
python export_quantized_onnx.py --help

# L/14 mixed-precision QAT
python mixed_qat.py

# Compile and profile through QAI Hub
python compile_and_profile.py --help

# Upload evaluation inputs and calculate retrieval metrics
python upload_dataset.py
python inference.py
```

`upload_dataset.py` and `inference.py` currently use configuration values defined in the source files. Set the local data paths and QAI Hub job/dataset IDs before running them.

## Models and Data

Checkpoints, ONNX/DLC files, teacher-embedding caches, and datasets are intentionally excluded from Git. The main artifacts related to the final result that may be retained locally include:

- `qat_clip_visual.pth`: L/14 mixed-precision QAT checkpoint.
- `student_checkpoint_epoch_13.pt`: B/16 student checkpoint distilled from an H/14 teacher.
- `dataset/` and `dataset_sample/images/`: local training or evaluation data.

To publish checkpoints, use a GitHub Release or Git LFS and record the model name, dataset, source commit, and quantization configuration in the release notes.

## Notes

- `remove_transpose.py` folds the `Transpose` applied to a Linear weight after QDQ insertion so that QNN can recognize the Fully Connected pattern.
- The text encoder's causal attention mask should not quantize `-inf` directly. See `analysis.py` and `quantize_text.py` for the related analysis and validation.
- `reports/` contains small quantization-analysis outputs suitable for source control. Larger generated artifacts remain excluded by `.gitignore`.
