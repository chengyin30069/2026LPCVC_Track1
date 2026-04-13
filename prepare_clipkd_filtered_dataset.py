from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter noisy Track1 captions into a cleaner CLIP-KD-style training subset."
    )
    parser.add_argument("--img-list", default="dataset/img_list.csv")
    parser.add_argument("--txt-list", default="dataset/txt_list.csv")
    parser.add_argument("--output-dir", default="dataset_clipkd")
    parser.add_argument("--min-chars", type=int, default=12)
    parser.add_argument("--max-chars", type=int, default=120)
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--max-words", type=int, default=28)
    parser.add_argument("--min-letter-ratio", type=float, default=0.55)
    parser.add_argument("--max-nonlatin-ratio", type=float, default=0.20)
    parser.add_argument("--max-digit-ratio", type=float, default=0.35)
    parser.add_argument("--min-kept-images", type=int, default=100000)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def is_high_quality_caption(
    text: str,
    *,
    min_chars: int,
    max_chars: int,
    min_words: int,
    max_words: int,
    min_letter_ratio: float,
    max_nonlatin_ratio: float,
    max_digit_ratio: float,
) -> bool:
    cleaned = normalize_text(text)
    if len(cleaned) < min_chars or len(cleaned) > max_chars:
        return False

    words = cleaned.split(" ")
    if len(words) < min_words or len(words) > max_words:
        return False

    lowered = cleaned.lower()
    junk_exact = {"listing", "image", "photo", "stock photo", "in stock"}
    if lowered in junk_exact:
        return False
    junk_substrings = (
        "http://",
        "https://",
        "www.",
        ".com",
        "buy now",
        "free shipping",
        "sku",
        "item #",
        "add to cart",
    )
    if any(piece in lowered for piece in junk_substrings):
        return False

    non_space = [ch for ch in cleaned if not ch.isspace()]
    if not non_space:
        return False

    letters = 0
    digits = 0
    non_latin = 0
    for ch in non_space:
        if ch.isdigit():
            digits += 1
            continue
        if ch.isalpha():
            letters += 1
            unicode_name = unicodedata.name(ch, "")
            if "LATIN" not in unicode_name and ord(ch) > 127:
                non_latin += 1
            continue
        unicode_name = unicodedata.name(ch, "")
        if unicode_name and "LATIN" not in unicode_name and ord(ch) > 127:
            non_latin += 1

    letter_ratio = letters / len(non_space)
    digit_ratio = digits / len(non_space)
    nonlatin_ratio = non_latin / len(non_space)
    if letter_ratio < min_letter_ratio:
        return False
    if digit_ratio > max_digit_ratio:
        return False
    if nonlatin_ratio > max_nonlatin_ratio:
        return False

    alpha_words = [word for word in re.findall(r"[A-Za-z]+", cleaned) if len(word) >= 3]
    if len(alpha_words) < 2:
        return False
    return True


def parse_text_ids(raw_text_ids: str) -> list[int]:
    values = [segment.strip() for segment in str(raw_text_ids).split(";")]
    return [int(value) for value in values if value]


def iter_sanitized_lines(path: Path):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as file:
        for line in file:
            if "\x00" in line:
                line = line.replace("\x00", "")
            yield line


def main() -> None:
    args = parse_args()
    img_list_path = Path(args.img_list)
    txt_list_path = Path(args.txt_list)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not img_list_path.exists():
        raise FileNotFoundError(f"img_list not found: {img_list_path}")
    if not txt_list_path.exists():
        raise FileNotFoundError(f"txt_list not found: {txt_list_path}")

    text_rows: dict[int, str] = {}
    reader = csv.reader(iter_sanitized_lines(txt_list_path))
    header = next(reader, None)
    if header is None or len(header) < 2:
        raise ValueError("txt_list.csv has invalid header")
    for row in reader:
        if len(row) < 2:
            continue
        text_id = int(row[0])
        text_rows[text_id] = row[1]

    kept_text_ids: set[int] = set()
    for text_id, text in text_rows.items():
        if is_high_quality_caption(
            text,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            min_words=args.min_words,
            max_words=args.max_words,
            min_letter_ratio=args.min_letter_ratio,
            max_nonlatin_ratio=args.max_nonlatin_ratio,
            max_digit_ratio=args.max_digit_ratio,
        ):
            kept_text_ids.add(text_id)

    output_img_path = output_dir / "img_list.csv"
    output_txt_path = output_dir / "txt_list.csv"

    kept_images = 0
    total_images = 0
    with output_img_path.open("w", encoding="utf-8", newline="") as out_file:
        reader = csv.reader(iter_sanitized_lines(img_list_path))
        writer = csv.writer(out_file)
        header = next(reader, None)
        if header is None or len(header) < 2:
            raise ValueError("img_list.csv has invalid header")
        writer.writerow(header[:2])

        for row in reader:
            if len(row) < 2:
                continue
            total_images += 1
            image_name = row[0]
            text_ids = parse_text_ids(row[1])
            filtered = [text_id for text_id in text_ids if text_id in kept_text_ids]
            if not filtered:
                continue
            kept_images += 1
            writer.writerow([image_name, ";".join(str(text_id) for text_id in filtered)])

    if kept_images < args.min_kept_images:
        raise RuntimeError(
            f"Filtered dataset too small: kept_images={kept_images}, required_min={args.min_kept_images}. "
            "Relax filtering thresholds."
        )

    with output_txt_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Text_id", "Text"])
        for text_id in sorted(kept_text_ids):
            writer.writerow([text_id, text_rows[text_id]])

    summary = {
        "input_img_list": str(img_list_path),
        "input_txt_list": str(txt_list_path),
        "output_dir": str(output_dir),
        "total_images": total_images,
        "kept_images": kept_images,
        "kept_image_ratio": float(kept_images) / float(max(total_images, 1)),
        "total_texts": len(text_rows),
        "kept_texts": len(kept_text_ids),
        "kept_text_ratio": float(len(kept_text_ids)) / float(max(len(text_rows), 1)),
        "thresholds": {
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "min_words": args.min_words,
            "max_words": args.max_words,
            "min_letter_ratio": args.min_letter_ratio,
            "max_nonlatin_ratio": args.max_nonlatin_ratio,
            "max_digit_ratio": args.max_digit_ratio,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
