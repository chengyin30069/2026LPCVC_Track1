from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    import torch
except ImportError:  # pragma: no cover - torch is an expected dependency in this repo
    torch = None


DEFAULT_PROMPT_TEMPLATES = (
    "{}",
    "a photo of {}",
    "an image of {}",
    "a picture of {}",
    "this is {}",
)


def process_image(image_path: str | Path, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    from PIL import Image

    image = Image.open(image_path).convert("RGB").resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.transpose(image_array, (2, 0, 1))[np.newaxis, :]


def load_images_from_folder(
    folder_path: str | Path,
    target_size: tuple[int, int] = (224, 224),
) -> list[np.ndarray]:
    folder = Path(folder_path)
    image_paths = sorted(
        path
        for path in folder.iterdir()
        if path.suffix.lower() in {".jpg", ".png", ".jpeg", ".webp"}
    )
    return [process_image(path, target_size=target_size) for path in image_paths]


def load_track1_texts(csv_path: str | Path) -> tuple[np.ndarray, list[str]]:
    dataframe = pd.read_csv(csv_path).iloc[:, :2].dropna().copy()
    text_ids = dataframe.iloc[:, 0].astype(np.int64).to_numpy()
    texts = dataframe.iloc[:, 1].astype(str).tolist()
    return text_ids, texts


def load_track1_ground_truth(
    txt_list_path: str | Path,
    img_list_path: str | Path,
) -> tuple[list[int], list[str]]:
    text_ids, _ = load_track1_texts(txt_list_path)
    dataframe = pd.read_csv(img_list_path)

    if "Image_names" in dataframe.columns:
        dataframe = dataframe.sort_values(by="Image_names")

    ground_truth = dataframe.iloc[:, 1].dropna().astype(str).tolist()
    return text_ids.astype(int).tolist(), ground_truth


def parse_text_id_list(raw_value: str) -> list[int]:
    return [int(value) for value in str(raw_value).split(";") if value]


def load_track1_image_table(
    img_list_path: str | Path,
    image_folder: str | Path | None = None,
) -> pd.DataFrame:
    dataframe = pd.read_csv(img_list_path).iloc[:, :2].dropna().copy()
    image_column = dataframe.columns[0]
    label_column = dataframe.columns[1]

    if image_column == "Image_names":
        dataframe = dataframe.sort_values(by=image_column)

    dataframe = dataframe.rename(
        columns={
            image_column: "image_name",
            label_column: "positive_text_ids_raw",
        }
    )
    dataframe["image_name"] = dataframe["image_name"].astype(str)
    dataframe["positive_text_ids"] = dataframe["positive_text_ids_raw"].map(parse_text_id_list)

    if image_folder is not None:
        folder = Path(image_folder)
        dataframe["image_path"] = dataframe["image_name"].map(lambda name: str(folder / name))

    return dataframe[[column for column in ["image_name", "image_path", "positive_text_ids"] if column in dataframe.columns]]


def stack_embeddings(embeddings: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
    array = np.asarray(embeddings)
    if array.ndim == 2:
        return array.astype(np.float32, copy=False)
    if array.ndim == 3 and array.shape[1] == 1:
        return array[:, 0, :].astype(np.float32, copy=False)
    return np.vstack([np.asarray(item) for item in embeddings]).astype(np.float32, copy=False)


def normalize_embeddings(embeddings: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
    array = stack_embeddings(embeddings)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return array / norms


def apply_prompt_templates(texts: Sequence[str], templates: Sequence[str]) -> list[list[str]]:
    rendered_prompts: list[list[str]] = []
    for template in templates:
        if "{}" not in template:
            raise ValueError(f"Prompt template must contain '{{}}': {template}")
        rendered_prompts.append([template.format(text) for text in texts])
    return rendered_prompts


def evaluate_track1(
    img_output: np.ndarray | Sequence[np.ndarray],
    txt_output: np.ndarray | Sequence[np.ndarray],
    txt_list_path: str | Path,
    img_list_path: str | Path,
    k: int = 10,
    device: str = "auto",
    query_chunk_size: int = 256,
    key_chunk_size: int = 16384,
    show_progress: bool = False,
) -> float:
    img_embeds = normalize_embeddings(img_output)
    txt_embeds = normalize_embeddings(txt_output)
    text_ids, ground_truth = load_track1_ground_truth(txt_list_path, img_list_path)
    top_k_indices, _ = batched_topk_similarity(
        img_embeds,
        txt_embeds,
        top_k=k,
        query_chunk_size=query_chunk_size,
        key_chunk_size=key_chunk_size,
        device=device,
        show_progress=show_progress,
        progress_desc=f"Evaluating recall@{k}",
    )

    recalls: list[float] = []
    predicted_text_ids = np.asarray(text_ids, dtype=np.int64)[top_k_indices]

    for row_index, labels in enumerate(ground_truth):
        gt_ids = [int(value) for value in labels.split(";")]
        matched = len(set(predicted_text_ids[row_index].tolist()) & set(gt_ids))
        recalls.append(matched / len(gt_ids))

    return float(np.mean(recalls))


def batched_topk_similarity(
    query_embeddings: np.ndarray | Sequence[np.ndarray],
    key_embeddings: np.ndarray | Sequence[np.ndarray],
    *,
    top_k: int,
    query_chunk_size: int = 256,
    key_chunk_size: int = 16384,
    device: str = "cpu",
    show_progress: bool = False,
    progress_desc: str = "Searching top-k",
) -> tuple[np.ndarray, np.ndarray]:
    queries = normalize_embeddings(query_embeddings).astype(np.float32, copy=False)
    keys = normalize_embeddings(key_embeddings).astype(np.float32, copy=False)

    query_count = queries.shape[0]
    key_count = keys.shape[0]
    if query_count == 0 or key_count == 0:
        raise ValueError("Top-k similarity search requires non-empty query and key embeddings.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    effective_top_k = min(top_k, key_count)
    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

    if resolved_device != "cpu":
        if torch is None:
            raise RuntimeError("Torch is required for non-CPU top-k similarity search.")
        if resolved_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device '{resolved_device}', but CUDA is not available.")
        return _batched_topk_similarity_torch(
            queries,
            keys,
            top_k=effective_top_k,
            query_chunk_size=query_chunk_size,
            key_chunk_size=key_chunk_size,
            device=resolved_device,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )

    return _batched_topk_similarity_numpy(
        queries,
        keys,
        top_k=effective_top_k,
        query_chunk_size=query_chunk_size,
        key_chunk_size=key_chunk_size,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )


def _batched_topk_similarity_numpy(
    queries: np.ndarray,
    keys: np.ndarray,
    *,
    top_k: int,
    query_chunk_size: int,
    key_chunk_size: int,
    show_progress: bool,
    progress_desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    query_count = queries.shape[0]
    key_count = keys.shape[0]
    top_indices = np.full((query_count, top_k), -1, dtype=np.int32)
    top_scores = np.full((query_count, top_k), -np.inf, dtype=np.float32)

    query_starts = range(0, query_count, query_chunk_size)
    if show_progress:
        query_starts = tqdm(
            query_starts,
            total=(query_count + query_chunk_size - 1) // query_chunk_size,
            desc=progress_desc,
            unit="chunk",
            dynamic_ncols=True,
        )

    for query_start in query_starts:
        query_end = min(query_start + query_chunk_size, query_count)
        query_chunk = queries[query_start:query_end]
        chunk_size = query_end - query_start

        best_scores = np.full((chunk_size, top_k), -np.inf, dtype=np.float32)
        best_indices = np.full((chunk_size, top_k), -1, dtype=np.int32)

        for key_start in range(0, key_count, key_chunk_size):
            key_end = min(key_start + key_chunk_size, key_count)
            similarities = query_chunk @ keys[key_start:key_end].T
            candidate_indices = np.broadcast_to(
                np.arange(key_start, key_end, dtype=np.int32),
                similarities.shape,
            )

            merged_scores = np.concatenate([best_scores, similarities], axis=1)
            merged_indices = np.concatenate([best_indices, candidate_indices], axis=1)
            top_positions = np.argpartition(merged_scores, -top_k, axis=1)[:, -top_k:]
            row_indices = np.arange(chunk_size)[:, None]
            best_scores = merged_scores[row_indices, top_positions]
            best_indices = merged_indices[row_indices, top_positions]

            order = np.argsort(-best_scores, axis=1)
            best_scores = best_scores[row_indices, order]
            best_indices = best_indices[row_indices, order]

        top_indices[query_start:query_end] = best_indices
        top_scores[query_start:query_end] = best_scores

    return top_indices, top_scores


def _batched_topk_similarity_torch(
    queries: np.ndarray,
    keys: np.ndarray,
    *,
    top_k: int,
    query_chunk_size: int,
    key_chunk_size: int,
    device: str,
    show_progress: bool,
    progress_desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    query_count = queries.shape[0]
    key_count = keys.shape[0]
    tensor_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    query_tensor = torch.as_tensor(queries, dtype=tensor_dtype, device=device)
    key_tensor = torch.as_tensor(keys, dtype=tensor_dtype, device=device)

    top_indices = np.full((query_count, top_k), -1, dtype=np.int32)
    top_scores = np.full((query_count, top_k), -np.inf, dtype=np.float32)

    query_starts = range(0, query_count, query_chunk_size)
    if show_progress:
        query_starts = tqdm(
            query_starts,
            total=(query_count + query_chunk_size - 1) // query_chunk_size,
            desc=progress_desc,
            unit="chunk",
            dynamic_ncols=True,
        )

    for query_start in query_starts:
        query_end = min(query_start + query_chunk_size, query_count)
        query_chunk = query_tensor[query_start:query_end]
        chunk_size = query_end - query_start

        best_scores = torch.full((chunk_size, top_k), float("-inf"), dtype=tensor_dtype, device=device)
        best_indices = torch.full((chunk_size, top_k), -1, dtype=torch.long, device=device)

        for key_start in range(0, key_count, key_chunk_size):
            key_end = min(key_start + key_chunk_size, key_count)
            similarities = query_chunk @ key_tensor[key_start:key_end].T
            candidate_indices = torch.arange(key_start, key_end, dtype=torch.long, device=device)
            candidate_indices = candidate_indices.expand(chunk_size, -1)

            merged_scores = torch.cat([best_scores, similarities], dim=1)
            merged_indices = torch.cat([best_indices, candidate_indices], dim=1)
            best_scores, top_positions = torch.topk(merged_scores, k=top_k, dim=1)
            best_indices = torch.gather(merged_indices, dim=1, index=top_positions)

        top_indices[query_start:query_end] = best_indices.cpu().numpy().astype(np.int32, copy=False)
        top_scores[query_start:query_end] = best_scores.float().cpu().numpy()

    return top_indices, top_scores


def save_text_embedding_cache(
    output_path: str | Path,
    *,
    text_ids: np.ndarray,
    texts: Sequence[str],
    embeddings: np.ndarray,
    templates: Sequence[str],
    model_name: str,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        text_ids=np.asarray(text_ids, dtype=np.int64),
        texts=np.asarray(list(texts), dtype=str),
        embeddings=normalize_embeddings(embeddings).astype(np.float16),
        templates=np.asarray(list(templates), dtype=str),
        model_name=np.asarray(model_name, dtype=str),
    )


def load_text_embedding_cache(cache_path: str | Path) -> dict[str, np.ndarray | str | list[str]]:
    with np.load(cache_path) as cache:
        return {
            "text_ids": cache["text_ids"].astype(np.int64),
            "texts": cache["texts"].astype(str).tolist(),
            "embeddings": cache["embeddings"].astype(np.float32),
            "templates": cache["templates"].astype(str).tolist(),
            "model_name": str(cache["model_name"].item()),
        }


def save_image_embedding_cache(
    output_path: str | Path,
    *,
    image_names: Sequence[str],
    embeddings: np.ndarray,
    model_name: str,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        image_names=np.asarray(list(image_names), dtype=str),
        embeddings=normalize_embeddings(embeddings).astype(np.float16),
        model_name=np.asarray(model_name, dtype=str),
    )


def load_image_embedding_cache(cache_path: str | Path) -> dict[str, np.ndarray | str | list[str]]:
    with np.load(cache_path) as cache:
        return {
            "image_names": cache["image_names"].astype(str).tolist(),
            "embeddings": cache["embeddings"].astype(np.float32),
            "model_name": str(cache["model_name"].item()),
        }


def load_hard_negative_cache(cache_path: str | Path) -> dict[str, np.ndarray]:
    with np.load(cache_path) as cache:
        return {
            "text_ids": cache["text_ids"].astype(np.int64),
            "neighbor_indices": cache["neighbor_indices"].astype(np.int32),
            "neighbor_text_ids": cache["neighbor_text_ids"].astype(np.int64),
            "neighbor_scores": cache["neighbor_scores"].astype(np.float32),
        }


def batched_topk_neighbors(
    embeddings: np.ndarray | Sequence[np.ndarray],
    *,
    top_k: int,
    query_chunk_size: int = 1024,
    key_chunk_size: int = 4096,
    device: str = "cpu",
    show_progress: bool = False,
    progress_desc: str = "Mining hard negatives",
) -> tuple[np.ndarray, np.ndarray]:
    normalized = normalize_embeddings(embeddings).astype(np.float32, copy=False)
    item_count = normalized.shape[0]

    if item_count < 2:
        raise ValueError("Hard-negative mining requires at least two text embeddings.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    effective_top_k = max(1, min(top_k, item_count - 1))

    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

    if resolved_device != "cpu":
        if torch is None:
            raise RuntimeError("Torch is required for non-CPU hard-negative mining.")
        if resolved_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device '{resolved_device}', but CUDA is not available.")
        return _batched_topk_neighbors_torch(
            normalized,
            top_k=effective_top_k,
            query_chunk_size=query_chunk_size,
            key_chunk_size=key_chunk_size,
            device=resolved_device,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )

    neighbor_indices = np.full((item_count, effective_top_k), -1, dtype=np.int32)
    neighbor_scores = np.full((item_count, effective_top_k), -np.inf, dtype=np.float32)

    query_starts = range(0, item_count, query_chunk_size)
    if show_progress:
        query_starts = tqdm(
            query_starts,
            total=(item_count + query_chunk_size - 1) // query_chunk_size,
            desc=progress_desc,
            unit="chunk",
            dynamic_ncols=True,
        )

    for query_start in query_starts:
        query_end = min(query_start + query_chunk_size, item_count)
        query_chunk = normalized[query_start:query_end]
        chunk_size = query_end - query_start

        best_scores = np.full((chunk_size, effective_top_k), -np.inf, dtype=np.float32)
        best_indices = np.full((chunk_size, effective_top_k), -1, dtype=np.int32)

        for key_start in range(0, item_count, key_chunk_size):
            key_end = min(key_start + key_chunk_size, item_count)
            similarities = query_chunk @ normalized[key_start:key_end].T

            overlap_start = max(query_start, key_start)
            overlap_end = min(query_end, key_end)
            if overlap_start < overlap_end:
                diagonal_rows = np.arange(overlap_start, overlap_end) - query_start
                diagonal_cols = np.arange(overlap_start, overlap_end) - key_start
                similarities[diagonal_rows, diagonal_cols] = -np.inf

            candidate_indices = np.broadcast_to(
                np.arange(key_start, key_end, dtype=np.int32),
                similarities.shape,
            )

            merged_scores = np.concatenate([best_scores, similarities], axis=1)
            merged_indices = np.concatenate([best_indices, candidate_indices], axis=1)

            top_positions = np.argpartition(merged_scores, -effective_top_k, axis=1)[:, -effective_top_k:]
            row_indices = np.arange(chunk_size)[:, None]
            best_scores = merged_scores[row_indices, top_positions]
            best_indices = merged_indices[row_indices, top_positions]

            order = np.argsort(-best_scores, axis=1)
            best_scores = best_scores[row_indices, order]
            best_indices = best_indices[row_indices, order]

        neighbor_indices[query_start:query_end] = best_indices
        neighbor_scores[query_start:query_end] = best_scores

    return neighbor_indices, neighbor_scores


def _batched_topk_neighbors_torch(
    normalized_embeddings: np.ndarray,
    *,
    top_k: int,
    query_chunk_size: int,
    key_chunk_size: int,
    device: str,
    show_progress: bool,
    progress_desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    item_count = normalized_embeddings.shape[0]
    tensor_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    embeddings = torch.as_tensor(normalized_embeddings, dtype=tensor_dtype, device=device)

    neighbor_indices = np.full((item_count, top_k), -1, dtype=np.int32)
    neighbor_scores = np.full((item_count, top_k), -np.inf, dtype=np.float32)

    query_starts = range(0, item_count, query_chunk_size)
    if show_progress:
        query_starts = tqdm(
            query_starts,
            total=(item_count + query_chunk_size - 1) // query_chunk_size,
            desc=progress_desc,
            unit="chunk",
            dynamic_ncols=True,
        )

    for query_start in query_starts:
        query_end = min(query_start + query_chunk_size, item_count)
        query_chunk = embeddings[query_start:query_end]
        chunk_size = query_end - query_start

        best_scores = torch.full((chunk_size, top_k), float("-inf"), dtype=tensor_dtype, device=device)
        best_indices = torch.full((chunk_size, top_k), -1, dtype=torch.long, device=device)

        for key_start in range(0, item_count, key_chunk_size):
            key_end = min(key_start + key_chunk_size, item_count)
            similarities = query_chunk @ embeddings[key_start:key_end].T

            overlap_start = max(query_start, key_start)
            overlap_end = min(query_end, key_end)
            if overlap_start < overlap_end:
                diagonal_rows = torch.arange(overlap_start, overlap_end, device=device) - query_start
                diagonal_cols = torch.arange(overlap_start, overlap_end, device=device) - key_start
                similarities[diagonal_rows, diagonal_cols] = float("-inf")

            candidate_indices = torch.arange(key_start, key_end, dtype=torch.long, device=device)
            candidate_indices = candidate_indices.expand(chunk_size, -1)

            merged_scores = torch.cat([best_scores, similarities], dim=1)
            merged_indices = torch.cat([best_indices, candidate_indices], dim=1)
            best_scores, top_positions = torch.topk(merged_scores, k=top_k, dim=1)
            best_indices = torch.gather(merged_indices, dim=1, index=top_positions)

        neighbor_indices[query_start:query_end] = best_indices.cpu().numpy().astype(np.int32, copy=False)
        neighbor_scores[query_start:query_end] = best_scores.float().cpu().numpy()

    return neighbor_indices, neighbor_scores


def build_hard_negative_frame(
    text_ids: Sequence[int],
    neighbor_indices: np.ndarray,
    neighbor_scores: np.ndarray,
) -> pd.DataFrame:
    text_id_array = np.asarray(text_ids, dtype=np.int64)
    neighbor_text_ids = text_id_array[neighbor_indices]
    return pd.DataFrame(
        {
            "text_id": text_id_array,
            "hard_negative_ids": [";".join(map(str, row)) for row in neighbor_text_ids],
            "hard_negative_scores": [
                ";".join(f"{score:.6f}" for score in row) for row in neighbor_scores
            ],
        }
    )