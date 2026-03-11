"""
X-Ray OT (Optimal Transport) — Entry Point

Runs the full OT-based data-valuation pipeline end-to-end.  All print output
is tee'd to outputs/output.txt and all plots are saved as PNGs.

Usage:
    cd xray-ot && uv run --project .. python main.py --data_root /path/to/data --out_dir ./outputs
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must precede any pyplot import

# isort: split
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pipeline import config
from pipeline.data import setup_data
from pipeline.model import run_training
from pipeline.ot_valuation import run_ot_valuation
from pipeline.retraining import run_retraining_evaluation


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
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)

    # Early stopping (AUC-based)
    parser.add_argument("--early_stop_patience", type=int, default=5)

    # Offline pretrained weights (HPC)
    parser.add_argument("--pretrained_path", type=str, default=None)

    # Torch cache dir (HPC)
    parser.add_argument("--torch_home", type=str, default=None)

    # Split fractions
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)

    # OT
    parser.add_argument("--ot_reg", type=float, default=0.01)

    # OT-subset experiment fractions (train only)
    parser.add_argument("--subset_fracs", type=str, default="0.95,0.90,0.85,0.80,0.75")

    parser.add_argument("--seed", type=int, default=42)

    # Model head
    parser.add_argument("--dropout", type=float, default=0.4)

    # Backbone freeze/unfreeze
    parser.add_argument(
        "--freeze_backbone", action="store_true", help="Freeze DenseNet features initially; train head only."
    )
    parser.add_argument(
        "--unfreeze_last_block",
        action="store_true",
        help="When freezing backbone, unfreeze denseblock4+norm5 at --unfreeze_epoch.",
    )
    parser.add_argument(
        "--unfreeze_epoch", type=int, default=6, help="1-indexed epoch number when to unfreeze denseblock4+norm5."
    )

    # Learning rates
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay_head", type=float, default=0.0)
    parser.add_argument("--weight_decay_finetune", type=float, default=5e-4)

    # Scheduler
    parser.add_argument("--use_scheduler", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.torch_home is not None:
        os.environ["TORCH_HOME"] = args.torch_home
        os.makedirs(args.torch_home, exist_ok=True)
        print("TORCH_HOME set to:", args.torch_home)

    # --- Tee stdout ---
    tee = TeeStream(sys.stdout, config.OUTPUT_LOG)
    sys.stdout = tee

    try:
        _run_pipeline(args)
    finally:
        sys.stdout = tee._stream
        tee.close()


def _run_pipeline(args) -> None:
    start = time.time()

    print("=" * 60)
    print("X-RAY OT: DATA VALUATION PIPELINE (Optimal Transport)")
    print("=" * 60)

    # --- Reproducibility ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print()

    # --- Section 1: Data ---
    data = setup_data(args)

    # --- Section 2: Full training ---
    train_result = run_training(data, args, device)

    # --- Section 3: OT valuation ---
    ot_result = run_ot_valuation(train_result, data, args)

    # --- Section 4: Subset retraining experiments ---
    run_retraining_evaluation(train_result, ot_result, data, args, device)

    elapsed = time.time() - start
    print(f"\nPipeline completed in {elapsed / 60:.1f} minutes.")
    print(f"Outputs: {config.rel(args.out_dir)}")
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
