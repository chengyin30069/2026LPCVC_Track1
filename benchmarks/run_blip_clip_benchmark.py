import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


CLIP_BENCHMARK_REPO = "https://github.com/LAION-AI/CLIP_benchmark.git"
DEFAULT_MODEL = "Salesforce/blip-itm-base-coco"


def run_command(command, cwd=None):
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def ensure_clip_benchmark(repo_dir: Path):
    if repo_dir.exists():
        print(f"Using existing CLIP_benchmark checkout at: {repo_dir}")
        return

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run_command(["git", "clone", CLIP_BENCHMARK_REPO, str(repo_dir)])


def write_custom_loader(repo_dir: Path):
    loader_path = repo_dir / "clip_benchmark" / "models" / "blip_clip_tokenizer.py"
    loader_content = '''import torch
from transformers import BlipImageProcessor, BlipModel, CLIPTokenizer


class BlipClipTokenizerWrapper(torch.nn.Module):
    def __init__(self, blip_model):
        super().__init__()
        self.blip_model = blip_model
        self.vocab_size = blip_model.text_model.config.vocab_size
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def encode_image(self, images):
        return self.blip_model.get_image_features(pixel_values=images)

    def encode_text(self, tokenized_texts):
        input_ids = tokenized_texts["input_ids"].to(torch.int64)
        attention_mask = tokenized_texts.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
        else:
            attention_mask = attention_mask.to(torch.int64)

        mapped_ids = torch.remainder(input_ids, self.vocab_size)
        return self.blip_model.get_text_features(input_ids=mapped_ids, attention_mask=attention_mask)


class ClipTokenizerForBlip:
    def __init__(self, tokenizer_name, max_length, vocab_size):
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.vocab_size = vocab_size

    def __call__(self, texts):
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["input_ids"] = torch.remainder(encoded["input_ids"].to(torch.int64), self.vocab_size)
        encoded["attention_mask"] = encoded["attention_mask"].to(torch.int64)
        return encoded


def _transform_from_processor(processor):
    def transform(image):
        return processor(images=image, return_tensors="pt")["pixel_values"][0]

    return transform


def load_blip_clip_tokenizer(model_name, pretrained, cache_dir, device, jit=False):
    model_id = pretrained if pretrained else model_name
    blip_model = BlipModel.from_pretrained(model_id, cache_dir=cache_dir).to(device)
    blip_model.eval()

    model = BlipClipTokenizerWrapper(blip_model).to(device)
    processor = BlipImageProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    transform = _transform_from_processor(processor)
    tokenizer = ClipTokenizerForBlip(
        tokenizer_name="openai/clip-vit-base-patch32",
        max_length=77,
        vocab_size=blip_model.text_model.config.vocab_size,
    )
    return model, transform, tokenizer
'''
    loader_path.write_text(loader_content, encoding="utf-8")
    print(f"Wrote custom model loader: {loader_path}")

    init_path = repo_dir / "clip_benchmark" / "models" / "__init__.py"
    init_content = init_path.read_text(encoding="utf-8")

    import_line = "from .blip_clip_tokenizer import load_blip_clip_tokenizer"
    if import_line not in init_content:
        init_content = f"{import_line}\n" + init_content

    marker = "TYPE2FUNC = {"
    if marker not in init_content:
        raise RuntimeError("Failed to find TYPE2FUNC in clip_benchmark/models/__init__.py")

    entry = '    "blip_clip_tokenizer": load_blip_clip_tokenizer,\n'
    if '"blip_clip_tokenizer"' not in init_content:
        init_content = init_content.replace(marker, marker + "\n" + entry)

    init_path.write_text(init_content, encoding="utf-8")
    print("Registered model type: blip_clip_tokenizer")


def run_benchmark(args):
    workspace_root = Path(__file__).resolve().parents[1]
    repo_dir = workspace_root / "third_party" / "CLIP_benchmark"
    output_dir = workspace_root / "benchmark_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_clip_benchmark(repo_dir)
    write_custom_loader(repo_dir)

    output_json = output_dir / "clip_benchmark_blip_result.json"

    benchmark_cmd = [
        sys.executable,
        "clip_benchmark/cli.py",
        "eval",
        "--model_type",
        "blip_clip_tokenizer",
        "--model",
        args.model,
        "--pretrained",
        args.model,
        "--dataset",
        args.dataset,
        "--task",
        args.task,
        "--batch_size",
        str(args.batch_size),
        "--dataset_root",
        str(args.dataset_root),
        "--output",
        str(output_json),
    ]

    started = time.perf_counter()
    run_command(benchmark_cmd, cwd=repo_dir)
    elapsed = time.perf_counter() - started

    timing_json = output_dir / "clip_benchmark_blip_timing.json"
    timing_payload = {
        "model": args.model,
        "dataset": args.dataset,
        "task": args.task,
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed,
        "result_file": str(output_json),
    }
    timing_json.write_text(json.dumps(timing_payload, indent=2), encoding="utf-8")

    print("\nBenchmark complete.")
    print(f"Result JSON: {output_json}")
    print(f"Timing JSON: {timing_json}")
    print(f"Elapsed seconds: {elapsed:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run CLIP_benchmark with a BLIP model using openai-clip tokenizer compatibility adapter."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--task", default="zeroshot_classification")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dataset_root", default=".clip_benchmark_data")
    args = parser.parse_args()

    run_benchmark(args)


if __name__ == "__main__":
    main()
