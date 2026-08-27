import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
DEFAULT_QDQ_ONNX = ROOT / "exported_onnx" / "text_encoder_qdq.onnx"
DEFAULT_FLOAT_ONNX = ROOT / "exported_onnx" / "text_encoder.onnx"
DEFAULT_PROMPTS = [
    "a diagram",
    "a dog",
    "a cat",
    "a red car",
    "a person riding a bicycle",
]


class TextEncoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.model.encode_text(input_ids)
        x = F.normalize(x, dim=-1)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare OpenCLIP text encoder outputs against float/QDQ ONNX outputs."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="OpenCLIP model id.")
    parser.add_argument(
        "--qdq-onnx",
        type=Path,
        default=DEFAULT_QDQ_ONNX,
        help="Path to the QDQ ONNX model.",
    )
    parser.add_argument(
        "--float-onnx",
        type=Path,
        default=DEFAULT_FLOAT_ONNX,
        help="Optional float ONNX model path. If missing, float ONNX comparison is skipped.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help="Prompt list. If omitted, built-in example prompts are used unless --prompt-file is set.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Optional text file containing one prompt per line.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to save the comparison results as JSON.",
    )
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_file is not None:
        if not args.prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")
        prompts = [line.strip() for line in args.prompt_file.read_text().splitlines() if line.strip()]
        if not prompts:
            raise ValueError(f"No valid prompts found in: {args.prompt_file}")
        return prompts

    if args.prompts:
        return args.prompts

    return DEFAULT_PROMPTS


def ensure_dependencies():
    try:
        import open_clip  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "open_clip is required. Install it with `pip install open_clip_torch`."
        ) from exc

    try:
        import onnxruntime  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required. Install it with `pip install onnxruntime`."
        ) from exc


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom == 0.0:
        return 1.0 if np.allclose(a_flat, b_flat) else 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def compare_tensors(reference: np.ndarray, candidate: np.ndarray) -> dict:
    ref = reference.astype(np.float64)
    cand = candidate.astype(np.float64)
    diff = cand - ref
    abs_diff = np.abs(diff)

    return {
        "shape": list(reference.shape),
        "cosine_similarity": cosine_similarity(reference, candidate),
        "mse": float(np.mean(diff ** 2)),
        "mae": float(np.mean(abs_diff)),
        "max_abs": float(np.max(abs_diff)),
        "reference_min": float(np.min(ref)),
        "reference_max": float(np.max(ref)),
        "candidate_min": float(np.min(cand)),
        "candidate_max": float(np.max(cand)),
    }


def summarize_metric(results: list[dict], key: str) -> float:
    return float(np.mean([item[key] for item in results]))


def run_onnx(session, input_name: str, token_ids: np.ndarray) -> np.ndarray:
    outputs = session.run(None, {input_name: token_ids})
    if not outputs:
        raise RuntimeError("ONNX session returned no outputs.")
    return outputs[0]


def print_results(title: str, results: list[dict]):
    print(f"\n{title}")
    print("=" * 120)
    print(
        f"{'idx':>3}  {'prompt':<40}  {'shape':<12}  {'cosine':>10}  {'mse':>12}  {'mae':>12}  {'max_abs':>12}"
    )
    print("-" * 120)

    for item in results:
        prompt_preview = item["prompt"]
        if len(prompt_preview) > 40:
            prompt_preview = prompt_preview[:37] + "..."
        shape_str = "x".join(str(v) for v in item["shape"])
        print(
            f"{item['index']:>3}  "
            f"{prompt_preview:<40}  "
            f"{shape_str:<12}  "
            f"{item['cosine_similarity']:>10.6f}  "
            f"{item['mse']:>12.6e}  "
            f"{item['mae']:>12.6e}  "
            f"{item['max_abs']:>12.6e}"
        )

    print("-" * 120)
    print(
        "avg"
        f"{'':>2}  "
        f"{'':<40}  "
        f"{'':<12}  "
        f"{summarize_metric(results, 'cosine_similarity'):>10.6f}  "
        f"{summarize_metric(results, 'mse'):>12.6e}  "
        f"{summarize_metric(results, 'mae'):>12.6e}  "
        f"{summarize_metric(results, 'max_abs'):>12.6e}"
    )


def main():
    ensure_dependencies()

    import onnxruntime as ort
    import open_clip

    args = parse_args()
    prompts = load_prompts(args)

    if not args.qdq_onnx.exists():
        raise FileNotFoundError(
            f"QDQ ONNX not found: {args.qdq_onnx}\n"
            "You can update the path with --qdq-onnx."
        )

    compare_float_onnx = args.float_onnx.exists()

    print(f"Loading OpenCLIP model: {args.model_name}")
    model, _ = open_clip.create_model_from_pretrained(args.model_name)
    tokenizer = open_clip.get_tokenizer(args.model_name)
    text_encoder = TextEncoder(model).eval()

    token_ids = tokenizer(prompts).to(torch.int32)
    token_ids_np = token_ids.cpu().numpy()

    print(f"Number of prompts: {len(prompts)}")
    print(f"Token tensor shape: {tuple(token_ids_np.shape)}")

    with torch.no_grad():
        torch_output = text_encoder(token_ids).cpu().numpy()

    qdq_session = ort.InferenceSession(str(args.qdq_onnx), providers=["CPUExecutionProvider"])
    qdq_input_name = qdq_session.get_inputs()[0].name
    qdq_output = run_onnx(qdq_session, qdq_input_name, token_ids_np)

    qdq_results = []
    for idx, prompt in enumerate(prompts):
        metrics = compare_tensors(torch_output[idx], qdq_output[idx])
        metrics["index"] = idx
        metrics["prompt"] = prompt
        qdq_results.append(metrics)

    print_results("OpenCLIP vs QDQ ONNX", qdq_results)

    output_payload = {
        "model_name": args.model_name,
        "qdq_onnx": str(args.qdq_onnx),
        "prompts": prompts,
        "qdq_results": qdq_results,
    }

    if compare_float_onnx:
        float_session = ort.InferenceSession(str(args.float_onnx), providers=["CPUExecutionProvider"])
        float_input_name = float_session.get_inputs()[0].name
        float_output = run_onnx(float_session, float_input_name, token_ids_np)

        float_results = []
        for idx, prompt in enumerate(prompts):
            metrics = compare_tensors(torch_output[idx], float_output[idx])
            metrics["index"] = idx
            metrics["prompt"] = prompt
            float_results.append(metrics)

        print_results("OpenCLIP vs Float ONNX", float_results)
        output_payload["float_onnx"] = str(args.float_onnx)
        output_payload["float_results"] = float_results
    else:
        print(f"\nSkipping float ONNX comparison because the file does not exist: {args.float_onnx}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output_payload, indent=2))
        print(f"\nSaved comparison JSON to: {args.output_json}")


if __name__ == "__main__":
    main()
