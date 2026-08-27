# Code organization

這份索引以兩份最終簡報中的方法為準，將程式分成「最終主線」、「部署／驗證」、「問題修正」與「已封存實驗」。

## 1. 最終主線

### Distillation：H/14 teacher → B/16 student

| 檔案 | 作用 |
| --- | --- |
| `train_student_distill.py` | 蒸餾訓練入口。 |
| `distill/args.py` | 訓練參數與 teacher/student、loss 權重設定。 |
| `distill/data.py` | Dataset、DataLoader 與 embedding cache 載入。 |
| `distill/features.py` | Teacher/student feature extraction 與投影。 |
| `distill/losses.py` | Feature distillation、contrastive 與 ICL 類損失。 |
| `distill/optim.py` | Optimizer 與 learning-rate scheduler。 |
| `distill/trainer.py` | 訓練、驗證、checkpoint 與 history 主迴圈。 |
| `prepare_datacomp_track1_dataset.py` | 準備 DataComp / Track 1 蒸餾資料。 |
| `prepare_image_only_distill_dataset.py` | 建立只需 image side 的蒸餾資料。 |
| `prepare_track1_dataset.py` | 準備比賽格式資料與 metadata。 |
| `precompute_teacher_embeddings.py` | 預先計算 H/14 teacher image embeddings。 |
| `cache_text_embeddings.py` | 預先計算 text embeddings，降低訓練成本。 |
| `mine_hard_negatives.py` | 建立 contrastive training 的 hard negatives。 |
| `plot_training_history.py` | 畫 training / validation history。 |
| `utils/student_model.py` | Student model 與 projection head 定義。 |
| `utils/track1_utils.py` | Track 1 共用資料與評估工具。 |
| `export_onnx.py` | 載入蒸餾 checkpoint，匯出 student image encoder 與 text encoder。 |

`student_checkpoint_epoch_13.pt` 與 embedding caches 是相關產物，但體積太大，不進 Git。

### Quantization：L/14 W8A16 → mixed precision → QAT

| 檔案 | 作用 |
| --- | --- |
| `mixed_qat.py` | 最終 L/14 mixed-precision AIMET QAT；對選定 attention/MLP 算子配置 A8，其餘保留 A16。 |
| `export_quantized_onnx.py` | 匯出 L/14 W8A16 QDQ ONNX、encoding 與 explicit-QKV graph。 |
| `quantize_image.py` | Image encoder W8A16 PTQ / baseline。 |
| `quantize_text.py` | Text encoder W8A16 PTQ 與 attention-mask 處理。 |
| `analysis.py` | AIMET per-layer sensitivity / quantization analysis。 |
| `reports/mixed_precision/mmp_log.txt` | 最終 L/14 各 block 的 mixed-precision 配置紀錄。 |
| `reports/quant_analysis/text/` | Text quantization per-layer 分析報告。 |

`qat_clip_visual.pth` 是最終 L/14 QAT checkpoint，但不進 Git。

## 2. QNN deployment 與 retrieval 評估

| 檔案 | 作用 |
| --- | --- |
| `compile_and_profile.py` | 驗證 ONNX，送 QAI Hub 編譯成 QNN runtime target，並在 XR2 Gen 2 profile。 |
| `upload_dataset.py` | 將 image/text 輸入整理並上傳到 QAI Hub dataset。 |
| `inference.py` | 執行遠端 inference、收集 embeddings、計算 Recall@10。 |
| `benchmark.py` | Retrieval similarity 與 Recall 指標共用函式。 |
| `local_inference.py` | 本機 baseline / exported model 驗證。 |
| `local_inference_openclip.py` | OpenCLIP reference output 驗證。 |
| `utils/img_processing.py` | Image preprocessing。 |
| `utils/text_processing.py` | Tokenization / text preprocessing。 |
| `utils/Dataset.py` | 量化與評估用 dataset helper。 |

## 3. QNN graph 問題修正與診斷

| 檔案 | 作用 |
| --- | --- |
| `remove_transpose.py` | 將 Linear weight 的常數 transpose fold 掉，避免 QNN FC pattern 被 QDQ graph 打斷。 |
| `add_zero_bias.py` | 對缺少 bias 的 graph 補零 bias，便於 QNN pattern matching。 |
| `fix_input_rank.py` | 修正匯出後 input rank / shape 問題。 |
| `compare_text_openclip_qdq.py` | 比較 OpenCLIP 與 QDQ text encoder 數值。 |
| `mixed_test.py` | 以單一 ViT block 重現 mixed quant / graph 問題。 |
| `split_attn_export.py` | 將 attention 拆成 explicit Q/K/V 的匯出實驗；保留作為 QNN 相容性工具。 |

這些不是訓練主入口，但直接對應最終成果中提到的 QNN compilation 與 attention-mask 問題，因此保留。

## 4. 已封存、不進最終 repository 的實驗

所有項目都只是搬移，沒有刪除，位置在：

`../unused_experiments/2026LPCVC_Track1/`

- `code/`：早期 AIMET PTQ、手寫 encoding、H/14 mixed quant 與一次性測試程式。
- `artifacts/H14-quant/`：簡報指出無法成功 compile 的 H/14 路線。
- `artifacts/aimet_mixed_quant_image/`：被 H/14 實驗覆寫的舊輸出，不是最終 L/14 mixed-QAT 模型。
- `artifacts/artifacts/`：早期 L/14 teacher / COCO 蒸餾 cache 與 checkpoint，不是最終 H/14 + CC3M 路線。
- `artifacts/torch_smoothquant/`：SmoothQuant 實驗，未出現在最終方法。
- 單一 ViT block ONNX 與大型外部 tensor data：只用於除錯，程式保留、生成物封存。

工作區外層的第三方 repositories 與大型資料集沒有搬動，以免破壞既有環境或重複佔用磁碟；它們也不在此 Git repository 內。
