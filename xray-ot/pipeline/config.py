"""Centralized configuration: paths, constants, and helpers."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths – derived from this file's location so the pipeline works regardless
# of the current working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUTS_DIR = _PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_DIR = OUTPUTS_DIR / "results"
OUTPUT_LOG = OUTPUTS_DIR / "output.txt"


def rel(path: Path | str) -> str:
    """Return *path* relative to _PROJECT_ROOT for display (no absolute paths in logs)."""
    try:
        return str(Path(path).resolve().relative_to(_PROJECT_ROOT))
    except ValueError:
        try:
            return "~/" + str(Path(path).resolve().relative_to(Path.home()))
        except ValueError:
            return str(path)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
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
NO_FINDING_LABEL = "No Finding"

# ---------------------------------------------------------------------------
# ImageNet normalization (used by DenseNet121)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
