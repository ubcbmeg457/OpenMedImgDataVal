"""Section 1: Dataset classes and data loading."""

import os
from dataclasses import dataclass

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms

from pipeline import config


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------
class ChestXray14Binary(Dataset):
    """Binary labels: y=0 (healthy) iff Finding Labels is exactly "No Finding",
    y=1 (unhealthy) otherwise."""

    def __init__(self, root_dir, split_list_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        csv_path = os.path.join(root_dir, "Data_Entry_2017.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing CSV: {csv_path}")

        df = pd.read_csv(csv_path)

        self.label_map = {}
        for _, row in df.iterrows():
            filename = row["Image Index"]
            findings_raw = str(row["Finding Labels"])
            findings = [x.strip() for x in findings_raw.split("|") if x.strip()]

            if len(findings) == 1 and findings[0] == config.NO_FINDING_LABEL:
                y = 0.0
            else:
                y = 1.0
            self.label_map[filename] = y

        if not os.path.exists(split_list_path):
            raise FileNotFoundError(f"Missing split list: {split_list_path}")

        with open(split_list_path) as f:
            self.filenames = [line.strip() for line in f if line.strip()]

        self.paths = []
        for fn in self.filenames:
            p = self._find(fn)
            if p is None:
                raise FileNotFoundError(fn)
            self.paths.append(p)

    def _find(self, fn):
        for i in range(1, 13):
            p = os.path.join(self.root_dir, f"images_{i:03d}", "images", fn)
            if os.path.exists(p):
                return p
        return None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        y = torch.tensor([self.label_map[self.filenames[idx]]], dtype=torch.float32)
        return img, y


class TransformWrapper(Dataset):
    """Wrap a Subset so train/val/test can use different transforms."""

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


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass
class DataResult:
    """Outputs of the data preparation stage."""

    dataset: ChestXray14Binary
    train_set: Subset
    val_set: Subset
    test_set: Subset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_eval_loader: DataLoader
    n_train: int
    n_val: int
    n_test: int


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, eval_transform)."""
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )

    return train_transform, eval_transform


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def setup_data(args) -> DataResult:
    """Load dataset, split into train/val/test, and create DataLoaders."""
    print("=" * 60)
    print("SECTION 1: DATA LOADING AND PREPARATION")
    print("=" * 60)

    split_list = os.path.join(args.data_root, "train_val_list.txt")
    dataset = ChestXray14Binary(args.data_root, split_list, transform=None)

    n_total = len(dataset)
    n_test = int(args.test_frac * n_total)
    n_val = int(args.val_frac * n_total)
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise ValueError("Split fractions too large; train set would be empty.")

    gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=gen)

    train_transform, eval_transform = get_transforms()

    train_set_t = TransformWrapper(train_set, train_transform)
    val_set_t = TransformWrapper(val_set, eval_transform)
    test_set_t = TransformWrapper(test_set, eval_transform)
    train_set_eval_t = TransformWrapper(train_set, eval_transform)

    train_loader = DataLoader(
        train_set_t, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set_t, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set_t, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    train_eval_loader = DataLoader(
        train_set_eval_t, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    print(f"Total samples: {n_total}")
    print(f"Train/Val/Test sizes: {n_train}/{n_val}/{n_test}")
    print()

    return DataResult(
        dataset=dataset,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_eval_loader=train_eval_loader,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
    )
