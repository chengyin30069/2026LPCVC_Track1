```bash
python export_onnx.py --student-checkpoint artifacts/student-h14-cc3m-CLIPKD-retune-v6/student_checkpoint_epoch_13.pt --output-dir exported_onnx_v6_e13_32 --export-original-clip-arch --text-input-dtype int32 --opset-version 16 --no-dynamo-export
```


```bash
python compile_and_profile.py      --onnx-dir exported_onnx_v6_e13_32/      --metadata exported_onnx_v6_e13_32/export_metadata.json      --target-runtime precompiled_qnn_onnx
```