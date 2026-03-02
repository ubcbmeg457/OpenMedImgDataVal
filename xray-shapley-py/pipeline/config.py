"""Centralized configuration: paths, hyperparameters, and constants."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths – derived from this file's location so the pipeline works regardless
# of the current working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUTS_DIR = _PROJECT_ROOT / "outputs"
DATA_DIR = OUTPUTS_DIR / "data"
MODELS_DIR = OUTPUTS_DIR / "models"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_DIR = OUTPUTS_DIR / "results"
OUTPUT_LOG = OUTPUTS_DIR / "output.txt"


def rel(path: Path | str) -> str:
    """Return *path* relative to _PROJECT_ROOT for display (no absolute paths in logs)."""
    try:
        return str(Path(path).resolve().relative_to(_PROJECT_ROOT))
    except ValueError:
        # Path is outside the project (e.g. kagglehub cache) — show relative to home
        try:
            return "~/" + str(Path(path).resolve().relative_to(Path.home()))
        except ValueError:
            return str(path)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# "nih-chest-xrays/sample" for 5% sample (~2.3 GB, ~5,606 images).
# Change to "nih-chest-xrays/data" for the full 45 GB / 112,120 image dataset.
KAGGLE_DATASET = "nih-chest-xrays/data"

# ---------------------------------------------------------------------------
# ImageNet normalization (used by DenseNet121)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Model / training
# ---------------------------------------------------------------------------
FINETUNE_ALL = False  # True → unfreeze all layers (requires GPU)
BATCH_SIZE = 32
NUM_EPOCHS = 10
PATIENCE = 3  # early-stopping patience
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
SHAP_BACKGROUND_SIZE = 200

# ---------------------------------------------------------------------------
# KNN-Shapley
# ---------------------------------------------------------------------------
KNN_K = 10

# ---------------------------------------------------------------------------
# Data-quality thresholds
# ---------------------------------------------------------------------------
NOISY_THRESHOLD = -0.05
REDUNDANCY_THRESHOLD = 0.01

# ---------------------------------------------------------------------------
# Data-efficiency experiment fractions
# ---------------------------------------------------------------------------
EFFICIENCY_FRACTIONS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# ---------------------------------------------------------------------------
# Retraining experiment (Section 6)
# ---------------------------------------------------------------------------
RETRAIN_FRACTIONS: list[float] = [0.2, 0.5, 0.8, 1.0]
RETRAIN_RANDOM_SEEDS: int = 1  # random-K repeats for error bars
RETRAIN_NUM_EPOCHS: int = 5
RETRAIN_PATIENCE: int = 2
RETRAIN_LR: float = 1e-3
RETRAIN_WEIGHT_DECAY: float = 1e-5
RETRAIN_FINETUNE_ALL: bool = FINETUNE_ALL

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
TSNE_SAMPLE_SIZE = 1000
SALIENCY_SAMPLE_INDICES = [0, 50, 100]
WATERFALL_SAMPLE_INDICES = [0, 50, 100]
