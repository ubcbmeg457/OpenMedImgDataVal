"""Section 1: Data download, organization, and exploration."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import kagglehub
import pandas as pd

from pipeline import config


@dataclass
class DataResult:
    """Outputs of the data preparation stage."""

    image_dir: Path
    df_labels: pd.DataFrame


def download_dataset() -> Path:
    """Download the NIH Chest X-rays dataset via kagglehub and return the cache path."""
    print("Downloading NIH Chest X-rays dataset from Kaggle...")
    kaggle_path = kagglehub.dataset_download(config.KAGGLE_DATASET)
    print(f"Downloaded to: {config.rel(kaggle_path)}")
    return Path(kaggle_path)


def copy_to_data_dir(kaggle_path: Path) -> None:
    """Copy the downloaded dataset into the local DATA_DIR."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Copying files to {config.rel(config.DATA_DIR)}...")
    for item in os.listdir(kaggle_path):
        src = os.path.join(kaggle_path, item)
        dst = config.DATA_DIR / item
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    print("Dataset organized successfully")


def load_labels() -> pd.DataFrame:
    """Read the labels CSV and return as a DataFrame."""
    labels_csv = config.DATA_DIR / "sample_labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(f"Expected {config.rel(labels_csv)}")
    df = pd.read_csv(labels_csv)
    print(f"Loaded {len(df)} label rows")
    return df


def explore_data() -> None:
    """Print the directory structure of DATA_DIR."""
    print("Dataset structure:")
    for root, _dirs, files in os.walk(config.DATA_DIR):
        level = root.replace(str(config.DATA_DIR), "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for f in files[:5]:
            print(f"{subindent}{f}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")


def download_and_prepare_data() -> DataResult:
    """Orchestrator: download, copy, explore, and return paths + labels."""
    print("=" * 60)
    print("SECTION 1: DATA DOWNLOAD AND PREPARATION")
    print("=" * 60)

    kaggle_path = download_dataset()
    copy_to_data_dir(kaggle_path)
    explore_data()

    df_labels = load_labels()

    image_dir = config.DATA_DIR / "sample" / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Expected {config.rel(image_dir)}")

    return DataResult(image_dir=image_dir, df_labels=df_labels)
