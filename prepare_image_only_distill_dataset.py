import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_SOURCE_DIRS = [
    "raw_datasets/coco2014/images/train2014",
    "raw_datasets/visual_genome/VG_100K/VG_100K",
    "raw_datasets/visual_genome/VG_100K_2/VG_100K_2",
    "coco2017/val2017",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an image-only dataset for teacher-student distillation."
    )
    parser.add_argument("--output-dir", default="dataset_image_only")
    parser.add_argument("--img-list-name", default="img_list.csv")
    parser.add_argument("--image-folder-name", default="images")
    parser.add_argument(
        "--source-dir",
        action="append",
        dest="source_dirs",
        help="Image source directory (can be passed multiple times).",
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=696)
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only scan direct children of each source directory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sanitize_tag(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower() or "source"


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def iter_images(root: Path, recursive: bool):
    iterator = root.rglob("*") if recursive else root.glob("*")
    for path in iterator:
        if is_image_file(path):
            yield path


def link_or_copy_image(source_path: Path, output_path: Path, mode: str) -> None:
    if output_path.exists() or output_path.is_symlink():
        output_path.unlink()

    if mode == "symlink":
        output_path.symlink_to(source_path.resolve())
    elif mode == "hardlink":
        output_path.hardlink_to(source_path.resolve())
    else:
        shutil.copy2(source_path, output_path)


def unique_output_name(tag: str, file_name: str, seen_names: set[str]) -> str:
    base_name = f"{tag}_{file_name}"
    if base_name not in seen_names:
        seen_names.add(base_name)
        return base_name

    stem = Path(file_name).stem
    suffix = Path(file_name).suffix
    index = 1
    while True:
        candidate = f"{tag}_{stem}_{index}{suffix}"
        if candidate not in seen_names:
            seen_names.add(candidate)
            return candidate
        index += 1


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / args.image_folder_name
    img_list_path = output_dir / args.img_list_name
    summary_path = output_dir / "summary.json"

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already exists and is not empty. Use --overwrite to replace it."
        )

    images_dir.mkdir(parents=True, exist_ok=True)

    source_dir_args = args.source_dirs if args.source_dirs is not None else DEFAULT_SOURCE_DIRS
    source_dirs = [Path(path) for path in source_dir_args]
    existing_source_dirs = [path for path in source_dirs if path.exists()]
    if not existing_source_dirs:
        raise FileNotFoundError("No source directories exist. Pass --source-dir with valid paths.")

    resolved_seen: set[str] = set()
    image_records: list[tuple[Path, Path]] = []

    for source_dir in existing_source_dirs:
        discovered = 0
        for image_path in iter_images(source_dir, recursive=not args.non_recursive):
            resolved = str(image_path.resolve())
            if resolved in resolved_seen:
                continue
            resolved_seen.add(resolved)
            image_records.append((source_dir, image_path))
            discovered += 1
        print(f"Indexed {discovered} images from {source_dir}")

    if not image_records:
        raise RuntimeError("No images discovered from source directories.")

    if args.shuffle:
        random.Random(args.seed).shuffle(image_records)

    if args.max_images is not None:
        image_records = image_records[: args.max_images]

    rows: list[dict[str, str]] = []
    seen_output_names: set[str] = set()
    per_source_counts = defaultdict(int)

    for source_dir, source_path in tqdm(
        image_records,
        total=len(image_records),
        desc="Linking images",
        unit="image",
        dynamic_ncols=True,
    ):
        tag = sanitize_tag(source_dir.name)
        output_name = unique_output_name(tag, source_path.name, seen_output_names)
        output_path = images_dir / output_name
        link_or_copy_image(source_path, output_path, args.link_mode)
        rows.append({"Image_names": output_name})
        per_source_counts[str(source_dir)] += 1

    pd.DataFrame(rows).to_csv(img_list_path, index=False)

    summary = {
        "total_images": len(rows),
        "output_dir": str(output_dir.resolve()),
        "images_dir": str(images_dir.resolve()),
        "img_list": str(img_list_path.resolve()),
        "link_mode": args.link_mode,
        "recursive": not args.non_recursive,
        "shuffle": args.shuffle,
        "sources": dict(sorted(per_source_counts.items())),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved image list to {img_list_path}")


if __name__ == "__main__":
    main()
