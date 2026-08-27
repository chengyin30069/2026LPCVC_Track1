import csv
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CocoCaptions


class CocoClipDataset(Dataset):
    def __init__(self, img_dir, cap_json, img_tf, tokenizer, max_samples=None):
        self.ds = CocoCaptions(root=img_dir, annFile=cap_json)
        self.img_tf = img_tf
        self.tokenizer = tokenizer
        self.max_samples = min(max_samples, len(self.ds)) if max_samples is not None else len(self.ds)

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        image, captions = self.ds[idx]
        caption = random.choice(captions)
        image = self.img_tf(image)
        tokens = self.tokenizer([caption])[0]
        return image, tokens


class Track1SampleClipDataset(Dataset):
    def __init__(self, sample_root, img_tf, tokenizer, max_samples=None):
        self.sample_root = Path(sample_root)
        self.img_dir = self.sample_root / "images"
        self.img_tf = img_tf
        self.tokenizer = tokenizer

        img_csv = self.sample_root / "img_list.csv"
        txt_csv = self.sample_root / "txt_list.csv"
        if not img_csv.exists():
            raise FileNotFoundError(f"Image list not found: {img_csv}")
        if not txt_csv.exists():
            raise FileNotFoundError(f"Text list not found: {txt_csv}")

        self.text_by_id = self._load_texts(txt_csv)
        self.samples = self._load_images(img_csv)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise ValueError(f"No samples found in: {self.sample_root}")

    def _load_texts(self, txt_csv):
        text_by_id = {}
        with txt_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                text_id = row.get("Text_nums", "").strip()
                text = row.get("Unique_Texts", "").strip()
                if text_id and text:
                    text_by_id[int(text_id)] = text
        return text_by_id

    def _load_images(self, img_csv):
        samples = []
        with img_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_name = row.get("Image_names", "").strip()
                text_ids = [
                    int(text_id)
                    for text_id in row.get("Text_nums", "").split(";")
                    if text_id.strip()
                ]
                captions = [self.text_by_id[text_id] for text_id in text_ids if text_id in self.text_by_id]
                image_path = self.img_dir / image_name
                if image_path.exists() and captions:
                    samples.append((image_path, captions))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Track1SampleClipDataset requires Pillow to load images. "
                "Install it with `pip install pillow` in the runtime environment."
            ) from exc

        image_path, captions = self.samples[idx]
        with Image.open(image_path) as image:
            image = self.img_tf(image.convert("RGB"))
        caption = random.choice(captions)
        tokens = self.tokenizer([caption])[0]
        return image, tokens


class CalibrationDataset(Dataset):
    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}

    def __init__(self, img_dir, img_tf, max_samples=1000):
        self.img_dir = Path(img_dir)
        self.img_tf = img_tf

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Calibration directory not found: {self.img_dir}")

        self.img_files = sorted(
            path
            for path in self.img_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in self._IMAGE_EXTENSIONS
        )

        if max_samples is not None:
            self.img_files = self.img_files[:max_samples]

        if not self.img_files:
            raise ValueError(f"No calibration images found in: {self.img_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "CalibrationDataset requires Pillow to load images. "
                "Install it with `pip install pillow` in the runtime environment."
            ) from exc

        img_path = self.img_files[idx]
        try:
            with Image.open(img_path) as image:
                image = image.convert("RGB")
                return self.img_tf(image)
        except Exception as exc:
            print(f"Error loading {img_path}: {exc}")
            return torch.randn(3, 224, 224)


class TextCalibrationDataset(Dataset):
    def __init__(
        self,
        text_source,
        tokenizer,
        max_samples=1000,
        text_column=1,
        output_dtype=torch.int32,
        return_raw_text=False,
    ):
        self.text_source = Path(text_source)
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.output_dtype = output_dtype
        self.return_raw_text = return_raw_text

        if not self.text_source.exists():
            raise FileNotFoundError(f"Text calibration source not found: {self.text_source}")

        self.texts = self._load_texts()
        if max_samples is not None:
            self.texts = self.texts[:max_samples]

        if not self.texts:
            raise ValueError(f"No calibration texts found in: {self.text_source}")

    def _load_texts(self):
        if self.text_source.suffix.lower() == ".csv":
            return self._load_texts_from_csv()
        return self._load_texts_from_text_file()

    def _load_texts_from_csv(self):
        texts = []
        with self.text_source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) <= self.text_column:
                    continue
                text = row[self.text_column].strip()
                if text:
                    texts.append(text)
        return texts

    def _load_texts_from_text_file(self):
        texts = []
        with self.text_source.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    texts.append(text)
        return texts

    def _tokenize_text(self, text):
        try:
            tokenized = self.tokenizer([text])
        except TypeError:
            tokenized = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )

        if isinstance(tokenized, dict):
            tokenized = tokenized["input_ids"]

        if not isinstance(tokenized, torch.Tensor):
            tokenized = torch.as_tensor(tokenized)

        if tokenized.ndim > 1:
            tokenized = tokenized.squeeze(0)

        return tokenized.to(dtype=self.output_dtype)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        token_ids = self._tokenize_text(text)
        if self.return_raw_text:
            return token_ids, text
        return token_ids
