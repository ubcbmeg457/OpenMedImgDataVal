"""Dataset download and loading for NIH Chest X-ray 14 (multi-label, 14 classes)."""

import os
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from xray_class import config


def download_dataset():
    """Download full NIH CXR14 via kagglehub. Returns the path to the dataset."""
    os.environ["KAGGLE_CACHE_DIR"] = str(config.DEFAULT_DATA_DIR)
    print(f"KAGGLE_CACHE_DIR = {os.environ['KAGGLE_CACHE_DIR']}")
    print(f"Downloading {config.KAGGLE_DATASET} via kagglehub...")
    path = kagglehub.dataset_download(config.KAGGLE_DATASET)
    print(f"Dataset path: {config.rel(path)}")
    return Path(path)


CSV_FILENAME = "Data_Entry_2017.csv"


def _find_image(root_dir, fn, _cache={}):
    """Find an image file across the dataset directory structure.

    Handles both nested (images_001/images/) and flat layouts.
    Caches the discovered structure on first call for fast subsequent lookups.
    """
    if "dirs" not in _cache:
        # Build a map of filename -> path by scanning the directory tree once
        img_map = {}
        root = Path(root_dir)
        # Nested: images_001/images/*.png, images_002/images/*.png, ...
        for img_dir in sorted(root.glob("images_*/images")):
            for img_file in img_dir.iterdir():
                if img_file.is_file():
                    img_map[img_file.name] = str(img_file)
        # Flat fallback: images/*.png or *.png directly in root
        if not img_map:
            for img_file in root.glob("images/*.png"):
                img_map[img_file.name] = str(img_file)
        if not img_map:
            for img_file in root.glob("*.png"):
                img_map[img_file.name] = str(img_file)
        _cache["dirs"] = img_map
        print(f"Indexed {len(img_map)} images")

    return _cache["dirs"].get(fn)


class ChestXray14(Dataset):
    """
    Multi-label dataset for NIH Chest X-ray 14.

    Each image has a 14-dimensional binary label vector, one per disease.
    "No Finding" maps to all-zeros.

    Loads all images listed in the labels CSV — no split list file required.
    The pipeline handles train/val/test splitting via random_split.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        csv_path = os.path.join(root_dir, CSV_FILENAME)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing {CSV_FILENAME} in {root_dir}")
        df = pd.read_csv(csv_path)
        label_to_idx = {label: i for i, label in enumerate(config.ALL_LABELS)}

        # Clear the image path cache for this dataset instance
        _find_image.__defaults__[0].clear()

        self.filenames = []
        self.paths = []
        self.label_map = {}

        skipped = 0
        for _, row in df.iterrows():
            filename = row["Image Index"]
            findings_raw = str(row["Finding Labels"])
            findings = [x.strip() for x in findings_raw.split("|") if x.strip()]

            y = np.zeros(config.NUM_CLASSES, dtype=np.float32)
            for finding in findings:
                if finding in label_to_idx:
                    y[label_to_idx[finding]] = 1.0
            self.label_map[filename] = y

            p = _find_image(root_dir, filename)
            if p is None:
                skipped += 1
                continue
            self.filenames.append(filename)
            self.paths.append(p)

        print(f"Loaded {len(self.filenames)} images ({skipped} skipped — not found on disk)")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        y = torch.tensor(self.label_map[self.filenames[idx]], dtype=torch.float32)
        return img, y


class TransformWrapper(Dataset):
    """Wrap a Subset to apply different transforms for train/val/test."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    @property
    def indices(self):
        return self.subset.indices

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, y = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, y
