"""Section 1: Data download, organization, and exploration."""

import os
from dataclasses import dataclass
from pathlib import Path

import kagglehub
import pandas as pd

from pipeline import config

# ---------------------------------------------------------------------------
# Full vs sample dataset layout
# ---------------------------------------------------------------------------
# Sample ("nih-chest-xrays/sample"):
#   sample_labels.csv, sample/images/*.png
#
# Full ("nih-chest-xrays/data"):
#   Data_Entry_2017.csv, images_001/images/*.png … images_012/images/*.png
# ---------------------------------------------------------------------------
_IS_FULL_DATASET = "nih-chest-xrays/data" in config.KAGGLE_DATASET


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


def link_to_data_dir(kaggle_path: Path) -> None:
    """Symlink the downloaded dataset into the local DATA_DIR (avoids duplicating ~45 GB)."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Linking files to {config.rel(config.DATA_DIR)}...")
    for item in os.listdir(kaggle_path):
        src = Path(os.path.join(kaggle_path, item)).resolve()
        dst = config.DATA_DIR / item
        if dst.is_symlink() or dst.exists():
            dst.unlink() if (dst.is_symlink() or dst.is_file()) else None
        if not dst.exists():
            os.symlink(src, dst)
    print("Dataset linked successfully")


def _collect_images(kaggle_path: Path) -> Path:
    """Create a single flat image directory by symlinking images from all sub-folders.

    The full dataset stores images in images_001/images/, images_002/images/, etc.
    This creates DATA_DIR/images/ with symlinks to every individual .png so the rest
    of the pipeline can use a single image_dir.
    """
    image_dir = config.DATA_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for subdir in sorted(kaggle_path.iterdir()):
        inner = subdir / "images"
        if not inner.is_dir():
            continue
        for img in inner.iterdir():
            dst = image_dir / img.name
            if not dst.exists():
                os.symlink(img.resolve(), dst)
                count += 1

    print(f"Linked {count} images into {config.rel(image_dir)}")
    return image_dir


def load_labels() -> pd.DataFrame:
    """Read the labels CSV and return as a DataFrame."""
    if _IS_FULL_DATASET:
        labels_csv = config.DATA_DIR / "Data_Entry_2017.csv"
    else:
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
    """Orchestrator: download, link, explore, and return paths + labels."""
    print("=" * 60)
    print("SECTION 1: DATA DOWNLOAD AND PREPARATION")
    print("=" * 60)

    kaggle_path = download_dataset()
    link_to_data_dir(kaggle_path)

    if _IS_FULL_DATASET:
        image_dir = _collect_images(kaggle_path)
    else:
        image_dir = config.DATA_DIR / "sample" / "images"

    explore_data()

    df_labels = load_labels()

    if not image_dir.exists():
        raise FileNotFoundError(f"Expected {config.rel(image_dir)}")

    return DataResult(image_dir=image_dir, df_labels=df_labels)
