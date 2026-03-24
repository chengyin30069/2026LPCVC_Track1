import argparse
from contextlib import nullcontext
from pathlib import Path

import open_clip
import torch
from tqdm.auto import tqdm

from utils.track1_utils import (
    DEFAULT_PROMPT_TEMPLATES,
    apply_prompt_templates,
    load_track1_texts,
    save_text_embedding_cache,
)


def resolve_backend(model_id: str, requested_backend: str) -> str:
    if requested_backend in {"open_clip", "transformers"}:
        return requested_backend
    lowered = model_id.lower()
    if lowered.startswith("google/siglip") or lowered.startswith("google/siglip2"):
        return "transformers"
    return "open_clip"


def _coerce_feature_tensor(output_obj) -> torch.Tensor | None:
    if isinstance(output_obj, torch.Tensor):
        return output_obj
    for candidate_name in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
        if hasattr(output_obj, candidate_name):
            candidate = getattr(output_obj, candidate_name)
            if isinstance(candidate, torch.Tensor):
                if candidate.ndim == 3:
                    return candidate[:, 0, :]
                return candidate
    return None


def extract_transformers_text_features(model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    if hasattr(model, "get_text_features"):
        candidate = _coerce_feature_tensor(model.get_text_features(**inputs))
        if candidate is not None:
            return candidate

    outputs = model(**inputs)
    candidate = _coerce_feature_tensor(outputs)
    if candidate is not None:
        return candidate

    for candidate_name in ("text_embeds", "pooler_output"):
        if hasattr(outputs, candidate_name):
            candidate = getattr(outputs, candidate_name)
            if isinstance(candidate, torch.Tensor):
                return candidate
    if isinstance(outputs, torch.Tensor):
        return outputs
    raise RuntimeError("Unable to extract text features from transformers model output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache Track1 text embeddings with prompt ensembling.")
    parser.add_argument("--txt-list", default="dataset/txt_list.csv", help="Path to Track1 text list CSV.")
    parser.add_argument(
        "--output",
        default="artifacts/text_embeddings.npz",
        help="Output NPZ path for cached text embeddings.",
    )
    parser.add_argument(
        "--model-id",
        default="hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
        help="Model identifier (OpenCLIP or Hugging Face transformers).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "open_clip", "transformers"],
        default="auto",
        help="Text encoder backend. auto chooses transformers for google/siglip* models.",
    )
    parser.add_argument(
        "--template",
        action="append",
        dest="templates",
        help="Prompt template containing {}. Can be passed multiple times.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Text batch size for encoding.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for encoding.",
    )
    return parser.parse_args()


def encode_prompt_batch(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    batch_size: int,
    device: str,
    progress_desc: str,
) -> torch.Tensor:
    encoded_batches: list[torch.Tensor] = []
    autocast_enabled = device.startswith("cuda")
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for start in tqdm(
            range(0, len(prompts), batch_size),
            total=total_batches,
            desc=progress_desc,
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        ):
            prompt_batch = prompts[start:start + batch_size]
            token_batch = tokenizer(prompt_batch).to(device)
            autocast_context = (
                torch.autocast(device_type="cuda", enabled=True)
                if autocast_enabled
                else nullcontext()
            )
            with autocast_context:
                features = model.encode_text(token_batch)
                features = torch.nn.functional.normalize(features, dim=-1)
            encoded_batches.append(features.cpu())

    return torch.cat(encoded_batches, dim=0)


def main() -> None:
    args = parse_args()
    backend = resolve_backend(args.model_id, args.backend)
    templates = args.templates or list(DEFAULT_PROMPT_TEMPLATES)
    text_ids, texts = load_track1_texts(args.txt_list)

    if not texts:
        raise ValueError(f"No texts found in {args.txt_list}")

    rendered_prompts = apply_prompt_templates(texts, templates)

    template_embeddings: list[torch.Tensor] = []
    if backend == "open_clip":
        model, _, _ = open_clip.create_model_and_transforms(args.model_id)
        tokenizer = open_clip.get_tokenizer(args.model_id)
        model = model.to(args.device).eval()

        for template_index, (template, prompts) in enumerate(zip(templates, rendered_prompts), start=1):
            print(f"Encoding template {template_index}/{len(templates)}: {template}")
            template_embeddings.append(
                encode_prompt_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    batch_size=args.batch_size,
                    device=args.device,
                    progress_desc=f"Template {template_index}/{len(templates)}",
                )
            )
    else:
        from transformers import AutoModel, AutoProcessor

        model = AutoModel.from_pretrained(args.model_id).to(args.device).eval()
        processor = AutoProcessor.from_pretrained(args.model_id, use_fast=False)
        autocast_enabled = args.device.startswith("cuda")

        with torch.no_grad():
            for template_index, (template, prompts) in enumerate(zip(templates, rendered_prompts), start=1):
                print(f"Encoding template {template_index}/{len(templates)}: {template}")
                encoded_batches: list[torch.Tensor] = []
                total_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
                for start in tqdm(
                    range(0, len(prompts), args.batch_size),
                    total=total_batches,
                    desc=f"Template {template_index}/{len(templates)}",
                    unit="batch",
                    dynamic_ncols=True,
                    leave=False,
                ):
                    prompt_batch = prompts[start:start + args.batch_size]
                    inputs = processor(text=prompt_batch, return_tensors="pt", padding=True, truncation=True)
                    model_inputs = {
                        key: value.to(args.device, non_blocking=True)
                        for key, value in inputs.items()
                        if isinstance(value, torch.Tensor)
                    }
                    autocast_context = (
                        torch.autocast(device_type="cuda", enabled=autocast_enabled)
                        if autocast_enabled
                        else nullcontext()
                    )
                    with autocast_context:
                        features = extract_transformers_text_features(model, model_inputs)
                        features = torch.nn.functional.normalize(features, dim=-1)
                    encoded_batches.append(features.float().cpu())

                template_embeddings.append(torch.cat(encoded_batches, dim=0))

    stacked_embeddings = torch.stack(template_embeddings, dim=0)
    mean_embeddings = torch.nn.functional.normalize(stacked_embeddings.mean(dim=0), dim=-1)

    output_path = Path(args.output)
    save_text_embedding_cache(
        output_path,
        text_ids=text_ids,
        texts=texts,
        embeddings=mean_embeddings.numpy(),
        templates=templates,
        model_name=args.model_id,
    )
    print(f"Saved {len(texts)} text embeddings to {output_path}")


if __name__ == "__main__":
    main()