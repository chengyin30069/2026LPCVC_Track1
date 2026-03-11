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
        help="OpenCLIP model identifier.",
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
    templates = args.templates or list(DEFAULT_PROMPT_TEMPLATES)
    text_ids, texts = load_track1_texts(args.txt_list)

    if not texts:
        raise ValueError(f"No texts found in {args.txt_list}")

    rendered_prompts = apply_prompt_templates(texts, templates)

    model, _, _ = open_clip.create_model_and_transforms(args.model_id)
    tokenizer = open_clip.get_tokenizer(args.model_id)
    model = model.to(args.device).eval()

    template_embeddings: list[torch.Tensor] = []
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