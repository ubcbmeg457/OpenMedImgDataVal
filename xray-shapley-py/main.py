"""
X-Ray Shapley (Python Script) — Entry Point

Runs the full data-valuation pipeline end-to-end.  All print output is tee'd
to outputs/output.txt and all plots are saved as PNGs in outputs/plots/.

Usage:
    cd xray-shapley-py && uv run --project .. python main.py
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must precede any pyplot import

# isort: split
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pipeline import config
from pipeline.data import download_and_prepare_data
from pipeline.model import (
    create_model,
    extract_and_cache_all_embeddings,
    generate_saliency_maps,
    save_model,
    setup_data,
    train_model,
    visualize_pretrained_features,
)
from pipeline.retraining import run_retraining_evaluation
from pipeline.shap_analysis import run_shap_analysis
from pipeline.valuation import run_data_valuation


# ---------------------------------------------------------------------------
# TeeStream: duplicate stdout to console + file
# ---------------------------------------------------------------------------
class TeeStream:
    """Write to both the original stream and a log file.

    The file copy replaces the user's home directory and the project root
    with short placeholders so that no absolute paths leak into the log.
    """

    def __init__(self, stream, log_path: Path):
        self._stream = stream
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "w")  # noqa: SIM115
        # Longer prefix first so project root (which is under home) is matched before home.
        self._replacements = [
            (str(config._PROJECT_ROOT), "."),
            (str(Path.home()), "~"),
        ]

    def write(self, data):
        self._stream.write(data)
        sanitized = data
        for old, new in self._replacements:
            sanitized = sanitized.replace(old, new)
        self._file.write(sanitized)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # --- Output directories ---
    for d in (config.DATA_DIR, config.MODELS_DIR, config.EMBEDDINGS_DIR, config.PLOTS_DIR, config.RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # --- Tee stdout ---
    tee = TeeStream(sys.stdout, config.OUTPUT_LOG)
    sys.stdout = tee

    try:
        _run_pipeline()
    finally:
        sys.stdout = tee._stream
        tee.close()


def _run_pipeline() -> None:
    start = time.time()

    print("=" * 60)
    print("X-RAY SHAPLEY: DATA VALUATION PIPELINE (Python script)")
    print("=" * 60)

    # --- Reproducibility ---
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Seed: {config.SEED}")
    print()

    # --- Section 1: Data ---
    data = download_and_prepare_data()

    # --- Section 2: Model setup + data splits ---
    model = create_model(device)
    split = setup_data(data.image_dir, data.df_labels)
    visualize_pretrained_features(model, split, device)

    # --- Section 3: Training ---
    model = train_model(model, split, device)
    generate_saliency_maps(model, split, device)
    save_model(model)

    # --- Embeddings ---
    train_result = extract_and_cache_all_embeddings(model, split, device)

    # --- Section 4: SHAP ---
    shap_result = run_shap_analysis(train_result, split)  # noqa: F841

    # --- Section 5: Data valuation ---
    valuation_result = run_data_valuation(train_result, split)

    # --- Section 6: Retraining experiments ---
    run_retraining_evaluation(valuation_result, split, device)

    elapsed = time.time() - start
    print(f"\nPipeline completed in {elapsed / 60:.1f} minutes.")
    print(f"Outputs: {config.rel(config.OUTPUTS_DIR)}")


if __name__ == "__main__":
    main()
