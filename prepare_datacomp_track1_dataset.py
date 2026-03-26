from __future__ import annotations

import argparse
import csv
import json
import shutil
import string
import unicodedata
from pathlib import Path

from tqdm.auto import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert img2dataset file output into Track1 dataset format."
    )
    parser.add_argument(
        "--source-roots",
        nargs="+",
        required=True,
        help="One or more roots containing image files with sidecar .txt captions.",
    )
    parser.add_argument("--output-dir", default="dataset")
    parser.add_argument("--max-pairs", type=int, default=30_000_000)
    parser.add_argument("--min-text-chars", type=int, default=6)
    parser.add_argument("--max-text-chars", type=int, default=160)
    parser.add_argument(
        "--english-only",
        action="store_true",
        help="Keep only captions that look like English.",
    )
    parser.add_argument(
        "--english-min-letter-ratio",
        type=float,
        default=0.55,
        help="Minimum ratio of ASCII letters among non-space characters for English filtering.",
    )
    parser.add_argument(
        "--english-max-nonlatin-ratio",
        type=float,
        default=0.12,
        help="Maximum ratio of non-Latin/non-ASCII symbols among non-space characters for English filtering.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
    )
    parser.add_argument("--image-prefix", default="datacomp")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_text(text: str, min_chars: int, max_chars: int) -> str | None:
    compact = " ".join(text.strip().split())
    if len(compact) < min_chars:
        return None
    if len(compact) > max_chars:
        compact = compact[:max_chars].rstrip()
    return compact or None


def looks_english(text: str, *, min_letter_ratio: float, max_nonlatin_ratio: float) -> bool:
    non_space = [ch for ch in text if not ch.isspace()]
    if not non_space:
        return False

    ascii_letters = 0
    non_latin = 0

    for ch in non_space:
        if ch in string.ascii_letters:
            ascii_letters += 1
            continue
        if ch in string.digits or ch in string.punctuation:
            continue

        unicode_name = unicodedata.name(ch, "")
        if "LATIN" in unicode_name:
            continue
        non_latin += 1

    letter_ratio = ascii_letters / len(non_space)
    nonlatin_ratio = non_latin / len(non_space)
    return letter_ratio >= min_letter_ratio and nonlatin_ratio <= max_nonlatin_ratio


def link_or_copy(source_path: Path, target_path: Path, mode: str) -> None:
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()
    if mode == "symlink":
        target_path.symlink_to(source_path.resolve())
    elif mode == "hardlink":
        target_path.hardlink_to(source_path.resolve())
    else:
        shutil.copy2(source_path, target_path)


def iter_image_paths(roots: list[Path]):
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    img_list_path = output_dir / "img_list.csv"
    txt_list_path = output_dir / "txt_list.csv"
    summary_path = output_dir / "summary.json"

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    source_roots = [Path(root) for root in args.source_roots]

    text_to_id: dict[str, int] = {}

    kept_pairs = 0
    skipped_missing_txt = 0
    skipped_bad_text = 0
    skipped_non_english = 0

    with (
        img_list_path.open("w", encoding="utf-8", newline="") as img_file,
        txt_list_path.open("w", encoding="utf-8", newline="") as txt_file,
    ):
        img_writer = csv.writer(
            img_file,
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
            doublequote=True,
            lineterminator="\n",
        )
        txt_writer = csv.writer(
            txt_file,
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
            doublequote=True,
            lineterminator="\n",
        )
        img_writer.writerow(["Image_names", "Text_ids"])
        txt_writer.writerow(["Text_id", "Text"])

        for image_path in tqdm(
            iter_image_paths(source_roots),
            desc="Converting DataComp pairs",
            unit="pair",
            dynamic_ncols=True,
        ):
            if kept_pairs >= args.max_pairs:
                break

            caption_path = image_path.with_suffix(".txt")
            if not caption_path.exists():
                skipped_missing_txt += 1
                continue

            try:
                caption = caption_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                skipped_bad_text += 1
                continue

            caption = normalize_text(caption, args.min_text_chars, args.max_text_chars)
            if caption is None:
                skipped_bad_text += 1
                continue

            if args.english_only and not looks_english(
                caption,
                min_letter_ratio=args.english_min_letter_ratio,
                max_nonlatin_ratio=args.english_max_nonlatin_ratio,
            ):
                skipped_non_english += 1
                continue

            text_id = text_to_id.get(caption)
            if text_id is None:
                text_id = len(text_to_id)
                text_to_id[caption] = text_id
                txt_writer.writerow([text_id, caption])

            image_name = f"{args.image_prefix}_{kept_pairs:09d}{image_path.suffix.lower()}"
            out_image_path = images_dir / image_name
            link_or_copy(image_path, out_image_path, args.link_mode)
            img_writer.writerow([image_name, str(text_id)])

            kept_pairs += 1

    summary = {
        "source_roots": [str(root) for root in source_roots],
        "kept_pairs": kept_pairs,
        "unique_texts": len(text_to_id),
        "skipped_missing_txt": skipped_missing_txt,
        "skipped_bad_text": skipped_bad_text,
        "skipped_non_english": skipped_non_english,
        "max_pairs": args.max_pairs,
        "link_mode": args.link_mode,
        "english_only": bool(args.english_only),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
