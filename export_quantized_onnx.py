import argparse
import copy
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import ModelProto, version_converter
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import (
    get_qnn_qdq_config,
    qnn_preprocess_model,
)
from PIL import Image
from open_clip import create_model_from_pretrained

import aimet_torch
from aimet_torch.common import quantsim as aimet_quantsim
from aimet_torch.common.defs import QuantScheme


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "exported_onnx"


class ExplicitQKVAttention(nn.Module):
    """
    Replace nn.MultiheadAttention with explicit q_proj/k_proj/v_proj/out_proj.
    This keeps the graph easier to export and easier to map in ONNX.
    """

    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = float(mha.dropout)
        self.batch_first = mha.batch_first

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})."
            )

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self._load_from_mha(mha)

    def _load_from_mha(self, mha: nn.MultiheadAttention) -> None:
        with torch.no_grad():
            q_w, k_w, v_w = mha.in_proj_weight.chunk(3, dim=0)
            self.q_proj.weight.copy_(q_w)
            self.k_proj.weight.copy_(k_w)
            self.v_proj.weight.copy_(v_w)

            if mha.in_proj_bias is not None:
                q_b, k_b, v_b = mha.in_proj_bias.chunk(3, dim=0)
                self.q_proj.bias.copy_(q_b)
                self.k_proj.bias.copy_(k_b)
                self.v_proj.bias.copy_(v_b)
            else:
                nn.init.zeros_(self.q_proj.bias)
                nn.init.zeros_(self.k_proj.bias)
                nn.init.zeros_(self.v_proj.bias)

            self.out_proj.weight.copy_(mha.out_proj.weight)
            if mha.out_proj.bias is not None:
                self.out_proj.bias.copy_(mha.out_proj.bias)
            else:
                nn.init.zeros_(self.out_proj.bias)

    def _reshape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            batch, length, channels = x.shape
            x = x.reshape(batch, length, self.num_heads, self.head_dim)
            return x.permute(0, 2, 1, 3)

        length, batch, channels = x.shape
        x = x.reshape(length, batch, self.num_heads, self.head_dim)
        return x.permute(1, 2, 0, 3)

    def _reshape_from_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, length, head_dim = x.shape
        if self.batch_first:
            return x.permute(0, 2, 1, 3).reshape(batch, length, heads * head_dim)
        return x.permute(2, 0, 1, 3).reshape(length, batch, heads * head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask=None,
        need_weights: bool = False,
        attn_mask=None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        if key_padding_mask is not None:
            raise NotImplementedError("key_padding_mask is not implemented in this wrapper.")

        q = self._reshape_to_heads(self.q_proj(query))
        k = self._reshape_to_heads(self.k_proj(key))
        v = self._reshape_to_heads(self.v_proj(value))

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        if is_causal:
            q_len = attn_scores.size(-2)
            k_len = attn_scores.size(-1)
            causal_mask = torch.triu(
                torch.full(
                    (q_len, k_len),
                    float("-inf"),
                    device=attn_scores.device,
                    dtype=attn_scores.dtype,
                ),
                diagonal=1,
            )
            attn_scores = attn_scores + causal_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)

        if self.training and self.dropout > 0:
            attn_probs = F.dropout(attn_probs, p=self.dropout)

        out = torch.matmul(attn_probs, v)
        out = self._reshape_from_heads(out)
        out = self.out_proj(out)

        if need_weights:
            if average_attn_weights:
                weights = attn_probs.mean(dim=1)
            else:
                weights = attn_probs
            return out, weights

        return out, None


def replace_mha_with_explicit_qkv(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, ExplicitQKVAttention(child))
        else:
            replace_mha_with_explicit_qkv(child)


def replace_custom_layernorms_with_torch_layernorm(module: nn.Module) -> None:
    """
    AIMET v2 does not have quantized module registrations for open_clip's custom
    LayerNorm subclasses. Convert them back to plain torch.nn.LayerNorm so the
    model can be wrapped by QuantizationSimModel.
    """
    for name, child in list(module.named_children()):
        if type(child).__module__ == "open_clip.transformer" and type(child).__name__ in {
            "LayerNorm",
            "LayerNormFp32",
        }:
            replacement = nn.LayerNorm(
                normalized_shape=child.normalized_shape,
                eps=child.eps,
                elementwise_affine=child.elementwise_affine,
                bias=child.bias is not None,
                device=child.weight.device if child.weight is not None else None,
                dtype=child.weight.dtype if child.weight is not None else None,
            )
            with torch.no_grad():
                if child.weight is not None:
                    replacement.weight.copy_(child.weight)
                if child.bias is not None:
                    replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            replace_custom_layernorms_with_torch_layernorm(child)


class ImageEncoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        embedding = self.model.encode_image(image)
        return F.normalize(embedding, dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export OpenCLIP image encoder to ONNX with either AIMET or ORT QNN mixed-precision quantization."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the exported ONNX and encodings.",
    )
    parser.add_argument(
        "--output-prefix",
        default="clip_datacomp_xl_s13b_b90k_image_encoder_w8a16",
        help="Output filename prefix without suffix.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help="Optional directory of calibration images. If omitted, random inputs are used.",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=8,
        help="Number of calibration samples to use.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Input image size used for dummy input and random calibration.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used for loading/calibration/export.",
    )
    parser.add_argument(
        "--param-bw",
        type=int,
        default=8,
        help="Weight bitwidth.",
    )
    parser.add_argument(
        "--output-bw",
        type=int,
        default=16,
        help="Activation/output bitwidth.",
    )
    parser.add_argument(
        "--aimet-config",
        default="htp_v69",
        help="AIMET config file or built-in config alias.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=21,
        help="ONNX opset version for export.",
    )
    parser.add_argument(
        "--use-embedded-encodings",
        action="store_true",
        help="Also export an ONNX with embedded fake-quant nodes.",
    )
    parser.add_argument(
        "--encoding-version",
        default="0.6.1",
        choices=["0.6.1", "1.0.0"],
        help="AIMET encoding JSON version. 0.6.1 matches legacy activation_encodings/param_encodings layout.",
    )
    parser.add_argument(
        "--quant-scheme",
        default="tf_enhanced",
        choices=["tf_enhanced", "min_max"],
        help="AIMET quant scheme. tf_enhanced matches the legacy default before aimet-torch 2.0.",
    )
    parser.add_argument(
        "--pipeline",
        default="aimet",
        choices=["aimet", "ort_mixed_qnn"],
        help="aimet=export AIMET ONNX+encodings; ort_mixed_qnn=export float ONNX then quantize to QDQ W8A16 with one attention MatMul input converted to A8.",
    )
    parser.add_argument(
        "--target-matmul",
        default=None,
        help="Exact attention MatMul node.name to patch in ort_mixed_qnn mode. Defaults to the first attention act-act MatMul.",
    )
    parser.add_argument(
        "--patched-input-index",
        type=int,
        default=0,
        choices=[0, 1],
        help="Which target MatMul input to convert to A8 in ort_mixed_qnn mode.",
    )
    parser.add_argument(
        "--aimet-patch-attn-matmul-inputs-a8",
        action="store_true",
        help="After AIMET export, add encodings for the selected attention MatMul input tensor and requantize that encoding to A8.",
    )
    parser.add_argument(
        "--aimet-save-encodings-only",
        action="store_true",
        help="Only compute AIMET encodings and save them with sim.save_encodings_to_json(...), without calling sim.export().",
    )
    return parser.parse_args()


def load_model(model_name: str, device: torch.device) -> tuple[nn.Module, object]:
    model, preprocess = create_model_from_pretrained(model_name)
    model = copy.deepcopy(model).to(device).eval()
    replace_mha_with_explicit_qkv(model)
    replace_custom_layernorms_with_torch_layernorm(model)
    image_encoder = ImageEncoder(model).to(device).eval()
    return image_encoder, preprocess


def iter_calibration_images(
    calibration_dir: Path,
    preprocess,
    num_samples: int,
) -> Iterator[torch.Tensor]:
    image_paths = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        image_paths.extend(sorted(calibration_dir.rglob(pattern)))

    if not image_paths:
        raise FileNotFoundError(f"No calibration images found in: {calibration_dir}")

    for image_path in image_paths[:num_samples]:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            yield preprocess(rgb).unsqueeze(0)


def iter_random_calibration_tensors(
    num_samples: int,
    input_size: int,
) -> Iterator[torch.Tensor]:
    for _ in range(num_samples):
        yield torch.rand(1, 3, input_size, input_size, dtype=torch.float32)


def calibration_batches(
    args: argparse.Namespace,
    preprocess,
) -> Iterable[torch.Tensor]:
    if args.calibration_dir is not None:
        return iter_calibration_images(
            args.calibration_dir.expanduser().resolve(),
            preprocess,
            args.num_calibration_samples,
        )
    return iter_random_calibration_tensors(args.num_calibration_samples, args.input_size)


class ORTCalibrationDataReader(CalibrationDataReader):
    def __init__(self, input_name: str, batches: List[torch.Tensor]):
        self.input_name = input_name
        self.data = [
            {self.input_name: batch.detach().cpu().numpy().astype("float32")}
            for batch in batches
        ]
        self.iterator = iter(self.data)

    def get_next(self):
        return next(self.iterator, None)

    def rewind(self):
        self.iterator = iter(self.data)


def export_float_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
    opset: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["image"],
            output_names=["embedding"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=None,
            verbose=False,
            export_params=True,
            training=torch.onnx.TrainingMode.EVAL,
            dynamo=True,
        )


def get_ai_onnx_opset_version(model: ModelProto) -> int:
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return opset.version
    raise RuntimeError("No ai.onnx opset found.")


def ensure_ai_onnx_opset_21(model: ModelProto) -> ModelProto:
    opset = get_ai_onnx_opset_version(model)
    if opset >= 21:
        return model
    return version_converter.convert_version(model, 21)


def get_initializer_names(model: ModelProto) -> Set[str]:
    return {init.name for init in model.graph.initializer}


def is_attention_act_act_matmul(node: onnx.NodeProto, initializer_names: Set[str]) -> bool:
    if node.op_type != "MatMul" or len(node.input) < 2:
        return False

    a, b = node.input[0], node.input[1]
    if a in initializer_names or b in initializer_names:
        return False

    haystacks = [
        (node.name or "").lower(),
        a.lower(),
        b.lower(),
        " ".join(out.lower() for out in node.output),
    ]

    attention_markers = (
        "attn",
        "attention",
        "softmax",
        "permute",
        "transpose",
    )
    return any(marker in text for text in haystacks for marker in attention_markers)


def find_target_matmul(
    model: ModelProto,
    explicit_target: Optional[str] = None,
) -> onnx.NodeProto:
    initializer_names = get_initializer_names(model)
    candidates = [
        node
        for node in model.graph.node
        if is_attention_act_act_matmul(node, initializer_names)
    ]
    if explicit_target:
        for node in candidates:
            if node.name == explicit_target:
                return node
        raise RuntimeError(f"Target MatMul not found: {explicit_target}")

    if not candidates:
        raise RuntimeError("No attention act-act MatMul found in model.")
    for node in candidates:
        if "softmax" not in " ".join(inp.lower() for inp in node.input):
            return node
    return candidates[0]


def build_fixed_tensor_overrides(
    target_node: onnx.NodeProto,
    patched_input_index: int,
) -> Dict[str, List[dict]]:
    tensor_name = target_node.input[patched_input_index]
    return {
        tensor_name: [{
            "quant_type": QuantType.QUInt16,
            "convert": {
                "quant_type": QuantType.QUInt8,
                "recv_nodes": {target_node.name},
            },
        }]
    }


def run_ort_mixed_qnn_export(
    args: argparse.Namespace,
    image_encoder: nn.Module,
    preprocess,
    dummy_input: torch.Tensor,
    device: torch.device,
) -> None:
    float_onnx_path = args.output_dir / f"{args.output_prefix}.float.onnx"
    mixed_onnx_path = args.output_dir / f"{args.output_prefix}.mixed_a8_w8a16.onnx"
    preproc_path = args.output_dir / f"{args.output_prefix}.preproc.onnx"
    opset21_path = args.output_dir / f"{args.output_prefix}.preproc.opset21.onnx"

    print(f"Exporting float ONNX to: {float_onnx_path}")
    export_float_onnx(image_encoder, dummy_input, float_onnx_path, args.opset)

    model_changed = qnn_preprocess_model(str(float_onnx_path), str(preproc_path))
    model_to_quantize_path = preproc_path if model_changed else float_onnx_path

    model = onnx.load(str(model_to_quantize_path))
    model = ensure_ai_onnx_opset_21(model)
    onnx.save_model(model, str(opset21_path), save_as_external_data=False)
    model_to_quantize = str(opset21_path)

    model = onnx.load(model_to_quantize)
    target_node = find_target_matmul(model, args.target_matmul)
    print(
        f"Target MatMul: {target_node.name} | patched_input_index={args.patched_input_index} "
        f"| patched_tensor={target_node.input[args.patched_input_index]}"
    )

    calib_batches = [batch.to(device) for batch in calibration_batches(args, preprocess)]
    reader = ORTCalibrationDataReader("image", calib_batches)

    qnn_config = get_qnn_qdq_config(
        model_to_quantize,
        reader,
        activation_type=QuantType.QUInt16,
        weight_type=QuantType.QUInt8,
        per_channel=False,
        activation_symmetric=False,
        weight_symmetric=False,
    )
    qnn_config.extra_options["TensorQuantOverrides"] = build_fixed_tensor_overrides(
        target_node,
        args.patched_input_index,
    )
    qnn_config.extra_options["UseQDQContribOps"] = False

    print(f"Quantizing mixed QNN ONNX to: {mixed_onnx_path}")
    quantize(
        model_input=model_to_quantize,
        model_output=str(mixed_onnx_path),
        quant_config=qnn_config,
    )

    print(f"Float ONNX  : {float_onnx_path}")
    print(f"Mixed QDQ   : {mixed_onnx_path}")


PASS_THROUGH_OPS = {"Transpose", "Reshape", "Identity", "Cast", "Squeeze", "Unsqueeze"}


def requantize_encoding_entry(entry: dict, bitwidth: int) -> dict:
    updated = copy.deepcopy(entry)
    enc_min = float(updated["min"])
    enc_max = float(updated["max"])
    qmax = (1 << bitwidth) - 1
    scale = (enc_max - enc_min) / qmax if enc_max != enc_min else 1.0
    offset = int(round(enc_min / scale)) if scale != 0 else 0
    updated["bitwidth"] = bitwidth
    updated["scale"] = scale
    updated["offset"] = offset
    return updated


def patch_aimet_attention_matmul_input_encodings(
    onnx_path: Path,
    encodings_path: Path,
    explicit_target: Optional[str],
    patched_input_index: int,
) -> None:
    model = onnx.load(str(onnx_path))
    raw_text = encodings_path.read_text(encoding="utf-8", errors="replace")
    repaired_text = re.sub(
        r'("bitwidth"\s*:\s*\d+),(/[^\n]+)',
        r"\1",
        raw_text,
    )
    encodings = json.loads(repaired_text)

    activation_encodings = encodings["activation_encodings"]
    producer = {}
    for node in model.graph.node:
        for output in node.output:
            producer[output] = node

    target_node = find_target_matmul(model, explicit_target)
    target_tensor = target_node.input[patched_input_index]

    source_tensor = target_tensor
    hops = []
    while source_tensor not in activation_encodings and source_tensor in producer:
        node = producer[source_tensor]
        hops.append((node.op_type, node.name, source_tensor))
        if node.op_type not in PASS_THROUGH_OPS or not node.input:
            break
        source_tensor = node.input[0]

    if source_tensor not in activation_encodings:
        raise RuntimeError(
            f"Could not find an upstream encoded tensor for MatMul input {target_tensor}. "
            f"Trace: {hops}"
        )

    activation_encodings[target_tensor] = [
        requantize_encoding_entry(activation_encodings[source_tensor][0], bitwidth=8)
    ]

    backup_path = encodings_path.with_suffix(".encodings.bak")
    if not backup_path.exists():
        backup_path.write_text(encodings_path.read_text(encoding="utf-8"), encoding="utf-8")

    if repaired_text != raw_text:
        print(f"Repaired malformed AIMET encodings JSON: {encodings_path}")

    with open(encodings_path, "w", encoding="utf-8") as handle:
        json.dump(encodings, handle, indent=4, sort_keys=True)

    print(
        f"Patched AIMET encodings: target_matmul={target_node.name} "
        f"input_index={patched_input_index} target_tensor={target_tensor} "
        f"source_tensor={source_tensor} bitwidth=8"
    )
    print(f"Updated encodings: {encodings_path}")
    print(f"Backup encodings : {backup_path}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    image_encoder, preprocess = load_model(args.model_name, device)

    dummy_input = torch.rand(1, 3, args.input_size, args.input_size, dtype=torch.float32, device=device)

    if args.pipeline == "ort_mixed_qnn":
        run_ort_mixed_qnn_export(
            args=args,
            image_encoder=image_encoder,
            preprocess=preprocess,
            dummy_input=dummy_input,
            device=device,
        )
        return

    # Match legacy AIMET .encodings layout such as mobilenetv2_v75.encodings.
    aimet_quantsim.encoding_version = args.encoding_version
    quant_scheme = {
        "tf_enhanced": QuantScheme.post_training_tf_enhanced,
        "min_max": QuantScheme.min_max,
    }[args.quant_scheme]

    sim = aimet_torch.QuantizationSimModel(
        image_encoder,
        dummy_input,
        default_param_bw=args.param_bw,
        default_output_bw=args.output_bw,
        quant_scheme=quant_scheme,
        in_place=False,
        config_file=args.aimet_config,
    )

    print(
        f"Computing encodings with W{args.param_bw}A{args.output_bw} "
        f"using {args.num_calibration_samples} calibration samples..."
    )

    def forward_pass_callback(model: nn.Module, _unused) -> None:
        model.eval()
        with torch.no_grad():
            for batch in calibration_batches(args, preprocess):
                _ = model(batch.to(device))

    sim.compute_encodings(forward_pass_callback, None)

    if args.aimet_save_encodings_only:
        print(f"Saving AIMET torch encodings only to: {args.output_dir}")
        sim.save_encodings_to_json(str(args.output_dir), args.output_prefix)
        print(f"Encodings  : {args.output_dir / (args.output_prefix + '.json')}")
        return

    export_args = {
        "opset_version": args.opset,
        "input_names": ["image"],
        "output_names": ["embedding"],
    }

    print(f"Exporting ONNX + encodings to: {args.output_dir}")
    sim.export(
        path=str(args.output_dir),
        filename_prefix=args.output_prefix,
        dummy_input=dummy_input.cpu(),
        onnx_export_args=export_args,
        use_embedded_encodings=args.use_embedded_encodings,
    )

    aimet_export_dir = args.output_dir / f"{args.output_prefix}.aimet"
    aimet_onnx_path = aimet_export_dir / f"{args.output_prefix}.onnx"
    aimet_encodings_path = aimet_export_dir / f"{args.output_prefix}.encodings"

    if args.aimet_patch_attn_matmul_inputs_a8:
        patch_aimet_attention_matmul_input_encodings(
            onnx_path=aimet_onnx_path,
            encodings_path=aimet_encodings_path,
            explicit_target=args.target_matmul,
            patched_input_index=args.patched_input_index,
        )

    print(f"ONNX       : {aimet_onnx_path}")
    print(f"Encodings  : {aimet_encodings_path}")
    if args.use_embedded_encodings:
        print(
            "Embedded ONNX: "
            f"{aimet_export_dir / (args.output_prefix + '_embedded' + '.onnx')}"
        )


if __name__ == "__main__":
    main()
