import argparse
import json
import shutil
import urllib.request
import zipfile
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm.auto import tqdm


def is_valid_zip_archive(path: Path) -> bool:
    return path.exists() and zipfile.is_zipfile(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Track1-style dataset from low-friction public image-text sources."
    )
    parser.add_argument("--output-dir", default="dataset", help="Output Track1 dataset directory.")
    parser.add_argument("--raw-root", default="raw_datasets", help="Directory for downloaded raw datasets.")
    parser.add_argument("--skip-coco", action="store_true", help="Do not include COCO.")
    parser.add_argument("--skip-visual-genome", action="store_true", help="Do not include Visual Genome.")
    parser.add_argument("--skip-download", action="store_true", help="Use only already-downloaded raw files.")
    parser.add_argument(
        "--coco-captions-json",
        default=None,
        help="Optional path to captions_train2014.json.",
    )
    parser.add_argument(
        "--coco-images-dir",
        default=None,
        help="Optional path to extracted COCO train2014 images.",
    )
    parser.add_argument(
        "--vg-region-descriptions-json",
        default=None,
        help="Optional path to Visual Genome region_descriptions.json.",
    )
    parser.add_argument(
        "--vg-image-data-json",
        default=None,
        help="Optional path to Visual Genome image_data.json.",
    )
    parser.add_argument(
        "--vg-image-dirs",
        nargs="+",
        default=None,
        help="Optional list of extracted Visual Genome image directories.",
    )
    parser.add_argument(
        "--max-coco-images",
        type=int,
        default=80000,
        help="Maximum number of COCO images to include.",
    )
    parser.add_argument(
        "--max-vg-images",
        type=int,
        default=40000,
        help="Maximum number of Visual Genome images to include.",
    )
    parser.add_argument(
        "--max-coco-captions-per-image",
        type=int,
        default=2,
        help="Maximum number of COCO captions kept per image.",
    )
    parser.add_argument(
        "--max-vg-regions-per-image",
        type=int,
        default=2,
        help="Maximum number of Visual Genome region phrases kept per image.",
    )
    parser.add_argument(
        "--max-unique-texts",
        type=int,
        default=250000,
        help="Upper bound for the global text pool to keep hard-negative mining manageable.",
    )
    parser.add_argument("--min-text-chars", type=int, default=6)
    parser.add_argument("--max-text-chars", type=int, default=160)
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output directory.",
    )
    return parser.parse_args()


def normalize_text(text: str, min_chars: int, max_chars: int) -> str | None:
    compact = " ".join(str(text).strip().split())
    if len(compact) < min_chars:
        return None
    if len(compact) > max_chars:
        compact = compact[:max_chars].rstrip()
    return compact or None


def dedupe_preserve_order(texts: Iterable[str]) -> list[str]:
    return list(OrderedDict.fromkeys(texts))


def link_or_copy_image(source_path: Path, output_path: Path, mode: str) -> None:
    if output_path.exists() or output_path.is_symlink():
        output_path.unlink()
    if mode == "symlink":
        output_path.symlink_to(source_path.resolve())
    elif mode == "hardlink":
        output_path.hardlink_to(source_path.resolve())
    else:
        shutil.copy2(source_path, output_path)


def register_texts(
    texts: list[str],
    *,
    text_to_id: dict[str, int],
    text_rows: list[dict[str, object]],
    max_unique_texts: int,
) -> list[int]:
    positive_ids: list[int] = []
    for text in texts:
        text_id = text_to_id.get(text)
        if text_id is None:
            if len(text_rows) >= max_unique_texts:
                continue
            text_id = len(text_rows)
            text_to_id[text] = text_id
            text_rows.append({"text_id": text_id, "text": text})
        positive_ids.append(text_id)
    return positive_ids


class DownloadProgressBar:
    def __init__(self, desc: str):
        self.desc = desc
        self.progress_bar = None

    def __call__(self, block_count: int, block_size: int, total_size: int) -> None:
        if self.progress_bar is None:
            self.progress_bar = tqdm(
                total=total_size if total_size > 0 else None,
                desc=self.desc,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
            )
        downloaded = block_count * block_size
        if total_size > 0:
            downloaded = min(downloaded, total_size)
        self.progress_bar.update(downloaded - self.progress_bar.n)

    def close(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.close()


def download_file(url: str, destination: Path, *, validate_zip: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if validate_zip and not is_valid_zip_archive(destination):
            raise RuntimeError(
                f"Archive exists but is incomplete or corrupted: {destination}. "
                "If another prepare_track1_dataset.py process is still running, wait for it to finish. "
                "Otherwise remove the file and retry."
            )
        return destination

    partial_destination = destination.with_suffix(destination.suffix + ".part")
    if partial_destination.exists():
        raise RuntimeError(
            f"Temporary download already exists: {partial_destination}. "
            "Another dataset download may still be running."
        )

    print(f"Downloading {url} -> {destination}")
    reporter = DownloadProgressBar(desc=f"Downloading {destination.name}")
    try:
        urllib.request.urlretrieve(url, partial_destination, reporthook=reporter)
        if validate_zip and not is_valid_zip_archive(partial_destination):
            raise RuntimeError(
                f"Downloaded archive is invalid: {partial_destination}. Remove it and retry."
            )
        partial_destination.replace(destination)
        return destination
    finally:
        reporter.close()
        if partial_destination.exists():
            partial_destination.unlink()


def extract_zip(zip_path: Path, output_dir: Path) -> Path:
    marker = output_dir / ".extracted"
    if marker.exists():
        return output_dir
    if not is_valid_zip_archive(zip_path):
        raise RuntimeError(
            f"Archive is not ready for extraction: {zip_path}. "
            "The file is likely only partially downloaded or corrupted."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} -> {output_dir}")
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        for member in tqdm(
            members,
            total=len(members),
            desc=f"Extracting {zip_path.name}",
            unit="file",
            dynamic_ncols=True,
        ):
            archive.extract(member, output_dir)
    marker.write_text("ok", encoding="utf-8")
    return output_dir


def resolve_coco_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.coco_captions_json and args.coco_images_dir:
        return Path(args.coco_captions_json), Path(args.coco_images_dir)

    raw_root = Path(args.raw_root) / "coco2014"
    annotations_zip = raw_root / "annotations_trainval2014.zip"
    train_zip = raw_root / "train2014.zip"
    annotations_dir = raw_root / "annotations"
    images_parent = raw_root / "images"
    images_dir = images_parent / "train2014"

    if not args.skip_download:
        download_file(
            "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
            annotations_zip,
            validate_zip=True,
        )
        download_file("http://images.cocodataset.org/zips/train2014.zip", train_zip, validate_zip=True)

    if not annotations_zip.exists() or not train_zip.exists():
        raise FileNotFoundError("COCO archives are missing. Re-run without --skip-download or pass explicit paths.")

    extract_zip(annotations_zip, annotations_dir)
    extract_zip(train_zip, images_parent)
    captions_json = annotations_dir / "annotations" / "captions_train2014.json"
    return captions_json, images_dir


def resolve_vg_paths(args: argparse.Namespace) -> tuple[Path, Path, list[Path]]:
    if args.vg_region_descriptions_json and args.vg_image_data_json and args.vg_image_dirs:
        return (
            Path(args.vg_region_descriptions_json),
            Path(args.vg_image_data_json),
            [Path(path) for path in args.vg_image_dirs],
        )

    raw_root = Path(args.raw_root) / "visual_genome"
    region_zip = raw_root / "region_descriptions.json.zip"
    image_data_zip = raw_root / "image_data.json.zip"
    images1_zip = raw_root / "images.zip"
    images2_zip = raw_root / "images2.zip"
    metadata_dir = raw_root / "metadata"
    images_dir_1 = raw_root / "VG_100K"
    images_dir_2 = raw_root / "VG_100K_2"

    if not args.skip_download:
        download_file(
            "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip",
            region_zip,
            validate_zip=True,
        )
        download_file(
            "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip",
            image_data_zip,
            validate_zip=True,
        )
        download_file("https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip", images1_zip, validate_zip=True)
        download_file("https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip", images2_zip, validate_zip=True)

    required = [region_zip, image_data_zip, images1_zip, images2_zip]
    if any(not path.exists() for path in required):
        raise FileNotFoundError("Visual Genome archives are missing. Re-run without --skip-download or pass explicit paths.")

    extract_zip(region_zip, metadata_dir)
    extract_zip(image_data_zip, metadata_dir)
    extract_zip(images1_zip, images_dir_1)
    extract_zip(images2_zip, images_dir_2)
    return (
        metadata_dir / "region_descriptions.json",
        metadata_dir / "image_data.json",
        [images_dir_1, images_dir_2],
    )


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_vg_image_lookup(image_dirs: list[Path]) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for image_dir in image_dirs:
        image_paths = sorted(image_dir.glob("*.jpg"))
        for image_path in tqdm(
            image_paths,
            total=len(image_paths),
            desc=f"Indexing {image_dir.name}",
            unit="image",
            dynamic_ncols=True,
        ):
            lookup[image_path.name] = image_path
    return lookup


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    txt_csv = output_dir / "txt_list.csv"
    img_csv = output_dir / "img_list.csv"
    summary_json = output_dir / "summary.json"

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already exists and is not empty. Use --overwrite or choose another output-dir."
        )

    images_dir.mkdir(parents=True, exist_ok=True)

    text_to_id: dict[str, int] = {}
    text_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    summary = {
        "sources": {},
        "max_unique_texts": args.max_unique_texts,
        "raw_root": str(Path(args.raw_root).resolve()),
    }

    if not args.skip_coco:
        captions_json, coco_images_dir = resolve_coco_paths(args)
        coco_annotations = load_json(captions_json)
        image_id_to_name = {
            int(image["id"]): str(image["file_name"])
            for image in coco_annotations.get("images", [])
            if "id" in image and "file_name" in image
        }
        captions_by_image: dict[int, list[str]] = defaultdict(list)
        for annotation in tqdm(
            coco_annotations.get("annotations", []),
            desc="Scanning COCO captions",
            unit="caption",
            dynamic_ncols=True,
        ):
            caption = annotation.get("caption")
            image_id = annotation.get("image_id")
            if not isinstance(caption, str) or image_id is None:
                continue
            normalized = normalize_text(caption, args.min_text_chars, args.max_text_chars)
            if normalized is not None:
                captions_by_image[int(image_id)].append(normalized)

        kept_images = 0
        for image_id, captions in tqdm(
            captions_by_image.items(),
            total=min(len(captions_by_image), args.max_coco_images),
            desc="Building COCO samples",
            unit="image",
            dynamic_ncols=True,
        ):
            if kept_images >= args.max_coco_images or len(text_rows) >= args.max_unique_texts:
                break

            texts = dedupe_preserve_order(captions)[:args.max_coco_captions_per_image]
            positive_ids = register_texts(
                texts,
                text_to_id=text_to_id,
                text_rows=text_rows,
                max_unique_texts=args.max_unique_texts,
            )
            if not positive_ids:
                continue

            file_name = image_id_to_name.get(image_id)
            if not file_name:
                continue

            source_path = coco_images_dir / file_name
            if not source_path.exists():
                continue

            image_name = f"coco_{Path(file_name).name}"
            link_or_copy_image(source_path, images_dir / image_name, args.link_mode)
            image_rows.append({"Image_names": image_name, "GT_text_ids": ";".join(map(str, positive_ids))})
            kept_images += 1

        summary["sources"]["coco"] = {
            "images": kept_images,
            "captions_per_image": args.max_coco_captions_per_image,
            "captions_json": str(captions_json),
        }

    if not args.skip_visual_genome and len(text_rows) < args.max_unique_texts:
        region_json, image_data_json, vg_image_dirs = resolve_vg_paths(args)
        region_descriptions = load_json(region_json)
        image_data = load_json(image_data_json)
        image_meta = {
            int(item["image_id"]): item
            for item in image_data
            if "image_id" in item
        }
        vg_lookup = build_vg_image_lookup(vg_image_dirs)
        kept_images = 0
        for sample in tqdm(
            region_descriptions,
            total=min(len(region_descriptions), args.max_vg_images),
            desc="Building Visual Genome samples",
            unit="image",
            dynamic_ncols=True,
        ):
            if kept_images >= args.max_vg_images or len(text_rows) >= args.max_unique_texts:
                break

            texts = []
            for region in sample.get("regions", []):
                phrase = region.get("phrase") if isinstance(region, dict) else None
                if not isinstance(phrase, str):
                    continue
                normalized = normalize_text(phrase, args.min_text_chars, args.max_text_chars)
                if normalized is not None:
                    texts.append(normalized)
            texts = dedupe_preserve_order(texts)[:args.max_vg_regions_per_image]
            positive_ids = register_texts(
                texts,
                text_to_id=text_to_id,
                text_rows=text_rows,
                max_unique_texts=args.max_unique_texts,
            )
            if not positive_ids:
                continue

            image_id = int(sample.get("id") or sample.get("image_id") or 0)
            metadata = image_meta.get(image_id)
            if metadata is None:
                continue

            url = str(metadata.get("url", ""))
            basename = Path(url).name
            source_path = vg_lookup.get(basename)
            if source_path is None or not source_path.exists():
                continue

            image_name = f"vg_{basename}"
            link_or_copy_image(source_path, images_dir / image_name, args.link_mode)
            image_rows.append({"Image_names": image_name, "GT_text_ids": ";".join(map(str, positive_ids))})
            kept_images += 1

        summary["sources"]["visual_genome"] = {
            "images": kept_images,
            "regions_per_image": args.max_vg_regions_per_image,
            "region_descriptions_json": str(region_json),
        }

    if not text_rows or not image_rows:
        raise RuntimeError("No dataset rows were created. Try smaller limits or verify external dataset access.")

    pd.DataFrame(text_rows).to_csv(txt_csv, index=False)
    pd.DataFrame(image_rows).to_csv(img_csv, index=False)

    summary["total_images"] = len(image_rows)
    summary["total_texts"] = len(text_rows)
    summary["image_dir"] = str(images_dir)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved text list to {txt_csv}")
    print(f"Saved image list to {img_csv}")


if __name__ == "__main__":
    main()