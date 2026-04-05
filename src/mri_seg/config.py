"""Centralized configuration for MRI segmentation pipeline."""

from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_ROOT.parent

DEFAULT_DATA_DIR = _SRC_ROOT / "data"

SYNAPSE_DATASET_ID = "syn64952532"

# BraTS 2023 GLI tumor sub-regions (used as multi-label vector for data valuation)
TUMOR_LABELS = {
    1: "NCR",  # Necrotic tumor core
    2: "ED",  # Peritumoral edematous/invaded tissue
    4: "ET",  # GD-enhancing tumor
}
NUM_TUMOR_CLASSES = len(TUMOR_LABELS)  # 3
ALL_LABELS = ["NCR", "ED", "ET"]

# 2D slice parameters
INPUT_SIZE = (128, 128)
MODALITY = "t2f"  # T2-FLAIR

# Subject directory prefix
SUBJECT_PREFIX = "BraTS-GLI"


def rel(path):
    """Return path relative to project root for display."""
    try:
        return str(Path(path).resolve().relative_to(_PROJECT_ROOT))
    except ValueError:
        try:
            return "~/" + str(Path(path).resolve().relative_to(Path.home()))
        except ValueError:
            return str(path)
