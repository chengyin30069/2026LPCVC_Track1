# 2026 LPCVC Track 1 — Efficient Image–Text Retrieval

這個 repository 整理本專案最後成果所使用的程式：CLIP image encoder 的混合精度量化與 QAT、以 H/14 teacher 蒸餾 B/16 student，以及在 Qualcomm QAI Hub / QNN HTP 上的編譯、效能量測與 Recall@10 評估。

## 最終方法主線

1. **L/14 mixed-precision QAT**：以 W8A16 為基礎，對敏感算子配置 A8，再以 cosine feature loss 做 QAT。
2. **H/14 → B/16 distillation**：使用預先計算的 teacher image embeddings 與 text embeddings，搭配 feature distillation、contrastive / ICL 類損失訓練 student。
3. **QNN deployment**：匯出 ONNX、修正 QDQ graph、送至 QAI Hub 編譯為 QNN DLC，並在 XR2 Gen 2 (Proxy) profile。

完整檔案分工與「主線／輔助／封存」分類請看 [docs/CODE_ORGANIZATION.md](docs/CODE_ORGANIZATION.md)。

## 環境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

量化程式另外需要與目前 PyTorch 版本相容的 Qualcomm AIMET，請依 AIMET 官方安裝方式建立環境；它未直接固定在 `requirements.txt`，避免 pip 裝到不相容版本。

## 常用入口

```bash
# 蒸餾訓練
python train_student_distill.py --help

# 將蒸餾 student / text encoder 匯出成 ONNX
python export_onnx.py --help

# 匯出 L/14 W8A16 QDQ ONNX 與 encoding
python export_quantized_onnx.py --help

# L/14 mixed-precision QAT
python mixed_qat.py

# QAI Hub 編譯與 profile
python compile_and_profile.py --help

# 上傳資料並計算 retrieval 指標
python upload_dataset.py
python inference.py
```

`upload_dataset.py` 與 `inference.py` 目前仍使用檔案內設定值；執行前先填入本機資料路徑與 QAI Hub job / dataset IDs。

## 模型與資料

Checkpoint、ONNX/DLC、teacher embedding cache 與資料集都刻意排除在 Git 之外。與最終成果直接相關、但只保留在本機的主要檔案包括：

- `qat_clip_visual.pth`：L/14 mixed-precision QAT checkpoint。
- `student_checkpoint_epoch_13.pt`：H/14 teacher 蒸餾的 B/16 student checkpoint。
- `dataset/`、`dataset_sample/images/`：本機訓練或評估資料。

若要公開 checkpoint，建議使用 GitHub Release 或 Git LFS，並在 release 註明模型名稱、資料集、commit 與量化設定。

## 注意事項

- `remove_transpose.py` 處理 Linear weight QDQ 後的 `Transpose`，讓 QNN 能辨識 Fully Connected pattern。
- text encoder 的 causal attention mask 不應直接量化 `-inf`；相關分析與驗證在 `analysis.py`、`quantize_text.py`。
- `reports/` 是小型、可提交的量化分析結果；大型生成物仍由 `.gitignore` 排除。
