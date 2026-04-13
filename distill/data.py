from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from tqdm.auto import tqdm

from utils.track1_utils import (
    load_hard_negative_cache,
    load_track1_image_table,
)


def build_image_name_aliases(image_name: str) -> list[str]:
    aliases = [image_name]
    basename = Path(image_name).name
    if basename not in aliases:
        aliases.append(basename)

    if basename.startswith("coco_"):
        coco2014_name = "coco2014_" + basename[len("coco_") :]
        if coco2014_name not in aliases:
            aliases.append(coco2014_name)
    if basename.startswith("coco2014_"):
        coco_name = "coco_" + basename[len("coco2014_") :]
        if coco_name not in aliases:
            aliases.append(coco_name)

    return aliases


def matches_source_filter(image_name: str, source_filter: str) -> bool:
    if source_filter == "all":
        return True
    if source_filter == "coco":
        return image_name.startswith("coco2014_") or image_name.startswith("coco2017_")
    if source_filter == "coco2014":
        return image_name.startswith("coco2014_")
    if source_filter == "coco2017":
        return image_name.startswith("coco2017_")
    if source_filter == "vg":
        return image_name.startswith("vg_")
    return True


class Track1DistillDataset(Dataset):
    def __init__(
        self,
        *,
        img_list_path: str,
        image_folder: str,
        preprocess,
        text_ids: torch.Tensor | None,
        text_embeddings: torch.Tensor | None,
        texts: list[str] | None,
        teacher_text_ids: torch.Tensor | None,
        teacher_text_embeddings: torch.Tensor | None,
        teacher_image_names: list[str],
        teacher_embeddings: torch.Tensor,
        hard_negative_lookup: dict[int, list[int]] | None,
        num_hard_negatives: int,
        strict_teacher_coverage: bool,
        positive_pooling: str,
    ):
        self.preprocess = preprocess
        self.text_embeddings = text_embeddings
        self.teacher_text_embeddings = teacher_text_embeddings
        self.text_lookup = (
            {int(text_id): str(text) for text_id, text in zip(text_ids.tolist(), texts)}
            if text_ids is not None and texts is not None
            else {}
        )
        self.text_index = (
            {int(text_id): index for index, text_id in enumerate(text_ids.tolist())}
            if text_ids is not None and text_embeddings is not None
            else {}
        )
        self.teacher_text_index = (
            {int(text_id): index for index, text_id in enumerate(teacher_text_ids.tolist())}
            if teacher_text_ids is not None and teacher_text_embeddings is not None
            else {}
        )
        self.text_supervision_enabled = text_embeddings is not None and len(self.text_index) > 0
        self.teacher_text_enabled = teacher_text_embeddings is not None and len(self.teacher_text_index) > 0
        self.teacher_embeddings = teacher_embeddings
        self.teacher_index: dict[str, int] = {}
        for index, image_name in enumerate(teacher_image_names):
            for alias in build_image_name_aliases(image_name):
                self.teacher_index.setdefault(alias, index)

        self.hard_negative_lookup = hard_negative_lookup or {}
        self.num_hard_negatives = max(0, num_hard_negatives)
        self.positive_pooling = positive_pooling
        self.teacher_embedding_dim = int(teacher_embeddings.shape[1])
        self.text_embedding_dim = int(text_embeddings.shape[1]) if text_embeddings is not None else None
        self.teacher_text_embedding_dim = int(teacher_text_embeddings.shape[1]) if teacher_text_embeddings is not None else None
        self.missing_teacher_images: list[str] = []
        self.missing_positive_images: list[str] = []
        self.skipped_missing_teacher_count = 0
        self.skipped_missing_positive_count = 0

        image_table = load_track1_image_table(img_list_path, image_folder)
        self.total_rows = int(len(image_table))
        self.samples: list[dict[str, object]] = []

        for row in tqdm(
            image_table.itertuples(index=False),
            total=len(image_table),
            desc="Preparing training samples",
            unit="image",
            dynamic_ncols=True,
        ):
            teacher_index = self.teacher_index.get(row.image_name)
            if teacher_index is None:
                self.missing_teacher_images.append(row.image_name)
                self.skipped_missing_teacher_count += 1
                if strict_teacher_coverage:
                    raise ValueError(f"Missing teacher embedding for image: {row.image_name}")
                continue

            positive_indices: list[int] = []
            positive_ids: list[int] = []
            hard_negative_ids: list[int] = []
            if self.text_supervision_enabled:
                positive_ids = [text_id for text_id in row.positive_text_ids if text_id in self.text_index]
                if not positive_ids:
                    self.skipped_missing_positive_count += 1
                    self.missing_positive_images.append(row.image_name)
                    continue

                positive_indices = [self.text_index[text_id] for text_id in positive_ids]
                if self.num_hard_negatives > 0:
                    seen_ids = set(positive_ids)
                    for text_id in positive_ids:
                        for candidate_id in self.hard_negative_lookup.get(text_id, []):
                            if candidate_id in seen_ids or candidate_id not in self.text_index:
                                continue
                            seen_ids.add(candidate_id)
                            hard_negative_ids.append(candidate_id)
                            if len(hard_negative_ids) >= self.num_hard_negatives:
                                break
                        if len(hard_negative_ids) >= self.num_hard_negatives:
                            break

            self.samples.append(
                {
                    "image_path": row.image_path,
                    "image_name": row.image_name,
                    "positive_indices": positive_indices,
                    "positive_text_ids": positive_ids,
                    "teacher_index": int(teacher_index),
                    "hard_negative_ids": hard_negative_ids,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        with Image.open(sample["image_path"]) as image:
            pixel_values = self.preprocess(image.convert("RGB"))

        positive_dtype = self.text_embeddings.dtype if self.text_embeddings is not None else torch.float32
        positive_dim = self.text_embedding_dim or self.teacher_embedding_dim
        positive_embedding = torch.zeros(positive_dim, dtype=positive_dtype)

        if self.text_supervision_enabled:
            positive_indices = sample["positive_indices"]
            if self.positive_pooling == "mean":
                positive_embedding = self.text_embeddings[positive_indices].mean(dim=0)
            else:
                selected_index = positive_indices[torch.randint(len(positive_indices), (1,)).item()]
                positive_embedding = self.text_embeddings[selected_index]

        teacher_positive_dim = self.teacher_text_embedding_dim or self.teacher_embedding_dim
        teacher_positive_dtype = self.teacher_text_embeddings.dtype if self.teacher_text_embeddings is not None else torch.float32
        teacher_positive_embedding = torch.zeros(teacher_positive_dim, dtype=teacher_positive_dtype)
        if self.teacher_text_enabled:
            teacher_text_ids = [text_id for text_id in sample["positive_text_ids"] if text_id in self.teacher_text_index]
            if teacher_text_ids:
                mapped_indices = [self.teacher_text_index[text_id] for text_id in teacher_text_ids]
                if self.positive_pooling == "mean":
                    teacher_positive_embedding = self.teacher_text_embeddings[mapped_indices].mean(dim=0)
                else:
                    selected_teacher_index = mapped_indices[torch.randint(len(mapped_indices), (1,)).item()]
                    teacher_positive_embedding = self.teacher_text_embeddings[selected_teacher_index]

        negative_dim = self.text_embedding_dim or self.teacher_embedding_dim
        negative_embeddings = torch.zeros(self.num_hard_negatives, negative_dim, dtype=positive_dtype)
        negative_mask = torch.zeros(self.num_hard_negatives, dtype=torch.bool)

        if self.text_supervision_enabled and self.num_hard_negatives > 0:
            for neg_index, text_id in enumerate(sample["hard_negative_ids"]):
                negative_embeddings[neg_index] = self.text_embeddings[self.text_index[text_id]]
                negative_mask[neg_index] = True

        positive_text = ""
        if sample["positive_text_ids"]:
            selected_text_id = sample["positive_text_ids"][torch.randint(len(sample["positive_text_ids"]), (1,)).item()]
            positive_text = self.text_lookup.get(int(selected_text_id), "")

        return {
            "pixel_values": pixel_values,
            "positive_embedding": positive_embedding,
            "positive_text": positive_text,
            "teacher_embedding": self.teacher_embeddings[int(sample["teacher_index"])],
            "teacher_positive_embedding": teacher_positive_embedding,
            "negative_embeddings": negative_embeddings,
            "negative_mask": negative_mask,
            "sample_index": torch.tensor(index, dtype=torch.long),
        }


def build_hard_negative_lookup(cache_path: str | None) -> dict[int, list[int]] | None:
    if not cache_path:
        return None
    cache = load_hard_negative_cache(cache_path)
    text_ids = cache["text_ids"].tolist()
    neighbor_text_ids = cache["neighbor_text_ids"].tolist()
    return {
        int(text_id): [int(candidate_id) for candidate_id in candidate_ids]
        for text_id, candidate_ids in zip(text_ids, neighbor_text_ids)
    }


def resolve_num_workers(requested_num_workers: int | None, device: str) -> int:
    if requested_num_workers is not None:
        return max(0, requested_num_workers)
    if not device.startswith("cuda"):
        return 0
    cpu_count = os.cpu_count() or 1
    return min(8, max(2, cpu_count // 2))


def configure_cuda_runtime(*, device: str, allow_tf32: bool) -> None:
    if not device.startswith("cuda"):
        return
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = allow_tf32


def build_train_val_subsets(
    dataset: Dataset,
    *,
    val_split: float,
    seed: int,
) -> tuple[Dataset, Dataset | None]:
    total_size = len(dataset)
    if total_size <= 1 or val_split <= 0:
        return dataset, None

    val_size = max(1, min(total_size - 1, int(total_size * val_split)))
    if val_size <= 0:
        return dataset, None

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total_size, generator=generator).tolist()
    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)
