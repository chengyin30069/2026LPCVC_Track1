import argparse
from pathlib import Path

import numpy as np

from utils.track1_utils import (
    batched_topk_neighbors,
    build_hard_negative_frame,
    load_text_embedding_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine hard-negative texts from cached embeddings.")
    parser.add_argument(
        "--embeddings",
        default="artifacts/text_embeddings.npz",
        help="Input NPZ created by cache_text_embeddings.py.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/hard_negatives.npz",
        help="Output NPZ containing hard-negative indices and scores.",
    )
    parser.add_argument(
        "--csv-output",
        default="artifacts/hard_negatives.csv",
        help="Optional CSV summary for inspection.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of hard negatives per text.")
    parser.add_argument("--query-chunk-size", type=int, default=1024)
    parser.add_argument("--key-chunk-size", type=int, default=4096)
    parser.add_argument(
        "--device",
        default="auto",
        help="Neighbor search device: auto, cpu, cuda, cuda:0, ...",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = load_text_embedding_cache(args.embeddings)
    text_ids = np.asarray(cache["text_ids"], dtype=np.int64)
    embeddings = np.asarray(cache["embeddings"], dtype=np.float32)

    device = args.device
    if device == "auto":
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is an expected dependency in this repo
            torch = None
        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

    query_chunk_size = args.query_chunk_size
    key_chunk_size = args.key_chunk_size
    if device.startswith("cuda"):
        if query_chunk_size == 1024:
            query_chunk_size = 2048
        if key_chunk_size == 4096:
            key_chunk_size = 8192

    print(
        f"Mining hard negatives on {device} with query_chunk_size={query_chunk_size}, "
        f"key_chunk_size={key_chunk_size}, text_count={len(text_ids)}"
    )

    neighbor_indices, neighbor_scores = batched_topk_neighbors(
        embeddings,
        top_k=args.top_k,
        query_chunk_size=query_chunk_size,
        key_chunk_size=key_chunk_size,
        device=device,
        show_progress=True,
        progress_desc=f"Mining hard negatives ({device})",
    )

    neighbor_text_ids = text_ids[neighbor_indices]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        text_ids=text_ids,
        neighbor_indices=neighbor_indices,
        neighbor_text_ids=neighbor_text_ids,
        neighbor_scores=neighbor_scores.astype(np.float16),
    )

    if args.csv_output:
        csv_path = Path(args.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        build_hard_negative_frame(text_ids, neighbor_indices, neighbor_scores).to_csv(csv_path, index=False)
        print(f"Saved CSV summary to {csv_path}")

    print(f"Saved hard negatives for {len(text_ids)} texts to {output_path}")


if __name__ == "__main__":
    main()