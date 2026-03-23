"""Centralized configuration for X-ray classification pipeline."""

from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_ROOT.parent

DEFAULT_DATA_DIR = _SRC_ROOT / "data"

KAGGLE_DATASET = "nih-chest-xrays/data"

ALL_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]
NUM_CLASSES = len(ALL_LABELS)
NO_FINDING_LABEL = "No Finding"

# Per-channel mean/std of ImageNet training set — must match DenseNet121 pretraining normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def rel(path):
    """Return path relative to project root for display."""
    try:
        return str(Path(path).resolve().relative_to(_PROJECT_ROOT))
    except ValueError:
        try:
            return "~/" + str(Path(path).resolve().relative_to(Path.home()))
        except ValueError:
            return str(path)
