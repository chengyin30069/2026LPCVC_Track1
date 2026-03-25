# quantize_to_qdq.py
from pathlib import Path
import numpy as np
from PIL import Image

import onnx
from onnx.external_data_helper import uses_external_data
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

# 新版 ORT 常見 preprocess API
from onnxruntime.quantization.shape_inference import quant_pre_process

def process_image(image_path, target_size=(224, 224)):
    """Loads and processes an image to the required input shape (C, H, W)."""
    image = Image.open(image_path).convert('RGB').resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize
    return np.transpose(image_array, (2, 0, 1))[np.newaxis, :] # Convert to (1, C, H, W)

class ImageDataReader(CalibrationDataReader):
    def __init__(self, image_dir: str, input_name: str, size=(224, 224), limit: int | None = None):
        self.input_name = input_name
        self.size = size

        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            paths.extend(Path(image_dir).glob(ext))
        paths = sorted(paths)
        print(f"Found {len(paths)} images in {image_dir}")
        if limit is not None:
            paths = paths[:limit]

        self.paths = paths
        self.idx = 0

    def _load_image(self, path: Path) -> np.ndarray:
        return process_image(path, target_size=self.size)

    def get_next(self):
        if self.idx >= len(self.paths):
            return None

        x = self._load_image(self.paths[self.idx])
        self.idx += 1
        return {self.input_name: x}

    def rewind(self):
        self.idx = 0


def get_input_name(onnx_path: str) -> str:
    model = onnx.load(onnx_path, load_external_data=False)
    if not model.graph.input:
        raise ValueError(f"Model has no graph inputs: {onnx_path}")
    return model.graph.input[0].name


def model_uses_external_data(onnx_path: str) -> bool:
    model = onnx.load(onnx_path, load_external_data=False)
    return any(uses_external_data(initializer) for initializer in model.graph.initializer)


def save_model_like_source(model: onnx.ModelProto, source_onnx_path: str, output_onnx_path: str) -> None:
    if model_uses_external_data(source_onnx_path):
        output_path = Path(output_onnx_path)
        external_data_path = output_path.with_name(f"{output_path.name}.data")
        if output_path.exists():
            output_path.unlink()
        if external_data_path.exists():
            external_data_path.unlink()
        onnx.save_model(
            model,
            output_path.as_posix(),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{output_path.name}.data",
            size_threshold=1024,
            convert_attribute=False,
        )
    else:
        onnx.save(model, output_onnx_path)


def preprocess_for_quantization(fp_onnx_path: str, preprocessed_onnx: str) -> str:
    uses_external = model_uses_external_data(fp_onnx_path)
    preprocess_attempts = []

    if uses_external:
        print("[INFO] Model uses external data; using direct ONNX shape inference fallback.")
    else:
        preprocess_attempts.append(
            {
                "name": "full preprocessing",
                "kwargs": {
                    "skip_optimization": False,
                    "skip_symbolic_shape": False,
                    "skip_onnx_shape": False,
                },
            }
        )
        preprocess_attempts.extend(
            [
                {
                    "name": "ONNX shape inference only",
                    "kwargs": {
                        "skip_optimization": True,
                        "skip_symbolic_shape": True,
                        "skip_onnx_shape": False,
                    },
                },
                {
                    "name": "symbolic shape with guessed ranks",
                    "kwargs": {
                        "skip_optimization": True,
                        "skip_symbolic_shape": False,
                        "skip_onnx_shape": True,
                        "guess_output_rank": True,
                        "auto_merge": True,
                    },
                },
            ]
        )

    last_error = None
    for attempt in preprocess_attempts:
        try:
            quant_pre_process(
                input_model_path=fp_onnx_path,
                output_model_path=preprocessed_onnx,
                **attempt["kwargs"],
            )
            print(f"Preprocessed ONNX saved to: {preprocessed_onnx} ({attempt['name']})")
            return preprocessed_onnx
        except Exception as exc:
            last_error = exc
            print(f"[WARN] quant_pre_process failed during {attempt['name']}: {exc}")

    if uses_external:
        try:
            model = onnx.load(fp_onnx_path)
            model = onnx.shape_inference.infer_shapes(model)
            save_model_like_source(model, fp_onnx_path, preprocessed_onnx)
            print(f"Preprocessed ONNX saved to: {preprocessed_onnx} (direct ONNX shape inference)")
            return preprocessed_onnx
        except Exception as exc:
            last_error = exc
            print(f"[WARN] direct ONNX shape inference failed: {exc}")

    print("[WARN] Falling back to the original ONNX model without preprocessing.")
    if last_error is not None:
        print(f"[WARN] Last preprocessing error: {last_error}")
    return fp_onnx_path


def convert_fp_onnx_to_qdq(
    fp_onnx_path: str,
    qdq_onnx_path: str,
    calib_image_dir: str,
    calib_limit: int = 32,
):
    preprocessed_onnx = fp_onnx_path.replace(".onnx", ".pre.onnx")

    # 1) preprocess
    model_for_quant = preprocess_for_quantization(fp_onnx_path, preprocessed_onnx)

    # 2) data reader
    input_name = get_input_name(model_for_quant)
    reader = ImageDataReader(
        image_dir=calib_image_dir,
        input_name=input_name,
        size=(224, 224),
        limit=calib_limit,
    )

    # 3) static quant -> QDQ
    quantize_static(
        model_input=model_for_quant,
        model_output=qdq_onnx_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        op_types_to_quantize=["MatMul", "Gemm"]
    )
    print(f"QDQ ONNX saved to: {qdq_onnx_path}")

ONNX_PATH = "exported_onnx"
CALIBRATION_DATASET_PATH = "../coco2017/train2017"

if __name__ == "__main__":
    convert_fp_onnx_to_qdq(
        fp_onnx_path=f"{ONNX_PATH}/smoothquant_image_encoder.onnx",
        qdq_onnx_path=f"{ONNX_PATH}/image_encoder_qdq.onnx",
        calib_image_dir=CALIBRATION_DATASET_PATH,
        calib_limit=1024,
    )
