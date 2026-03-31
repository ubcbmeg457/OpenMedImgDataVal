"""Main pipeline orchestrator for MRI 2D segmentation with data valuation."""

import copy
import os
import random
import shutil
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from mri_seg import config
from mri_seg.data import (
    BraTSSliceDataset,
    SegAugmentation,
    SegTransformWrapper,
    download_dataset,
    find_training_data,
    slice_volumes,
)
from mri_seg.model import (
    build_model,
    evaluate_segmentation,
    extract_embeddings,
    plot_curve,
    plot_hist,
    plot_retraining_curves,
    train_with_early_stop,
)

matplotlib.use("Agg")


# ──────────────────────────────────────────────────────────────────────
# Pipeline defaults
# ──────────────────────────────────────────────────────────────────────
PIPELINE_DEFAULTS = {
    "epochs": 50,
    "batch_size": 32,
    "num_workers": 4,
    "early_stop_patience": 5,
    "seed": 42,
    "lr": 5e-4,
    "dropout": 0.1,
    "weight_decay": 1e-4,
    "subset_fracs": "0.90,0.80,0.70,0.50,0.30",
    "random_seeds": 1,
    "subset_epochs": 25,
    "ot_reg": 0.01,
    "shapley_mstar": 5000,
    "shapley_batch_val": 32,
    "shapley_k_candidates": "1,3,5,10,20,50",
}

HPO_SEARCH_SPACE = {
    "lr": [1e-4, 3e-4, 5e-4, 1e-3, 2e-3],
    "dropout": [0.05, 0.1, 0.15, 0.2],
    "weight_decay": [1e-5, 1e-4, 5e-4, 1e-3],
}
HPO_N_TRIALS = 3
HPO_EPOCHS = 15


# ──────────────────────────────────────────────────────────────────────
# TeeStream: duplicate stdout to console + log file
# ──────────────────────────────────────────────────────────────────────
class TeeStream:
    def __init__(self, stream, log_path):
        self._stream = stream
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "w")  # noqa: SIM115

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _apply_defaults(args):
    for k, v in PIPELINE_DEFAULTS.items():
        if not hasattr(args, k):
            setattr(args, k, v)


def _train_and_eval_subset(
    keep_indices, train_dataset, val_loader, test_loader, args, device, train_aug, run_name, frac
):
    train_subset = Subset(train_dataset, keep_indices)
    sub_train_loader = DataLoader(
        SegTransformWrapper(train_subset, train_aug),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    sub_args = copy.copy(args)
    sub_args.epochs = args.subset_epochs

    sub_model = build_model(sub_args, device)
    sub_model, _, _, sub_best_epoch, sub_best_val_dice = train_with_early_stop(
        sub_model, sub_train_loader, val_loader, device, sub_args, run_name=run_name
    )

    sub_train_eval_loader = DataLoader(
        SegTransformWrapper(train_subset, None),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    sub_train_loss, sub_train_dice = evaluate_segmentation(sub_model, sub_train_eval_loader, device)
    sub_val_loss, sub_val_dice = evaluate_segmentation(sub_model, val_loader, device)
    sub_test_loss, sub_test_dice = evaluate_segmentation(sub_model, test_loader, device)

    return {
        "frac": frac,
        "n_train": len(keep_indices),
        "train_loss": sub_train_loss,
        "train_dice": sub_train_dice,
        "val_loss": sub_val_loss,
        "val_dice": sub_val_dice,
        "test_loss": sub_test_loss,
        "test_dice": sub_test_dice,
        "best_epoch": sub_best_epoch,
        "best_val_dice": sub_best_val_dice,
    }


def _optimize_hyperparameters(train_loader, val_loader, args, device):
    hpo_dir = os.path.join(args.out_dir, "hpo")
    os.makedirs(hpo_dir, exist_ok=True)

    trials = []
    best_dice = -1.0
    best_config = {}

    print(f"\n{'=' * 60}")
    print(f"Hyperparameter Optimisation ({HPO_N_TRIALS} trials, {HPO_EPOCHS} epochs each)")
    print(f"{'=' * 60}")

    for trial_idx in range(HPO_N_TRIALS):
        trial_config = {k: random.choice(v) for k, v in HPO_SEARCH_SPACE.items()}

        trial_args = copy.copy(args)
        trial_args.lr = trial_config["lr"]
        trial_args.dropout = trial_config["dropout"]
        trial_args.weight_decay = trial_config["weight_decay"]
        trial_args.epochs = HPO_EPOCHS
        trial_args.out_dir = hpo_dir

        model = build_model(trial_args, device)
        _, _, _, _, val_dice = train_with_early_stop(
            model, train_loader, val_loader, device, trial_args, run_name=f"hpo_trial_{trial_idx}"
        )

        trial_config["val_dice"] = val_dice
        trials.append(trial_config)

        marker = " *best*" if val_dice > best_dice else ""
        print(
            f"  Trial {trial_idx + 1}/{HPO_N_TRIALS} | val_dice={val_dice:.4f}{marker} | "
            f"lr={trial_config['lr']:.1e} dropout={trial_config['dropout']} "
            f"wd={trial_config['weight_decay']:.1e}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            best_config = {k: v for k, v in trial_config.items() if k != "val_dice"}

    pd.DataFrame(trials).to_csv(os.path.join(args.out_dir, "hpo_trials.csv"), index=False)
    shutil.rmtree(hpo_dir, ignore_errors=True)

    print(f"\nBest config (val_dice={best_dice:.4f}): {best_config}")
    print(f"{'=' * 60}\n")
    return best_config, trials


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────
def run_pipeline(args, compute_values_fn, method_name, method_params_report):
    """Full data-valuation pipeline for MRI 2D binary segmentation."""
    _apply_defaults(args)
    os.makedirs(args.out_dir, exist_ok=True)
    method_lower = method_name.lower()

    # Tee stdout
    log_path = os.path.join(args.out_dir, "output.txt")
    tee = TeeStream(sys.stdout, log_path)
    sys.stdout = tee

    try:
        _run_pipeline_inner(args, compute_values_fn, method_name, method_lower, method_params_report)
    finally:
        sys.stdout = tee._stream
        tee.close()


def _run_pipeline_inner(args, compute_values_fn, method_name, method_lower, method_params_report):
    start = time.time()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Segmentation: Binary (tumor vs background)")
    print(f"DV labels: {config.NUM_TUMOR_CLASSES} tumor types ({', '.join(config.ALL_LABELS)})")

    train_aug = SegAugmentation()

    # ── Section 1: Data download + slicing ──
    print("\n" + "=" * 60)
    print("SECTION 1: DATA DOWNLOAD & PREPROCESSING")
    print("=" * 60)

    raw_dir = download_dataset()
    training_dir = find_training_data(raw_dir)
    print(f"Training data directory: {config.rel(training_dir)}")

    sliced_dir = os.path.join(config.DEFAULT_DATA_DIR, "brats2023_sliced")
    split_dirs = slice_volumes(training_dir, sliced_dir, seed=args.seed)

    train_dataset = BraTSSliceDataset(split_dirs["train"])
    val_dataset = BraTSSliceDataset(split_dirs["val"])
    test_dataset = BraTSSliceDataset(split_dirs["test"])

    n_train = len(train_dataset)
    n_val = len(val_dataset)
    n_test = len(test_dataset)

    train_loader = DataLoader(
        SegTransformWrapper(train_dataset, train_aug),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        SegTransformWrapper(val_dataset, None),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        SegTransformWrapper(test_dataset, None),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # ── Section 2: HPO ──
    print("\n" + "=" * 60)
    print("SECTION 2: HYPERPARAMETER OPTIMISATION")
    print("=" * 60)
    hpo_best_config, hpo_trials = _optimize_hyperparameters(train_loader, val_loader, args, device)

    args.lr = hpo_best_config["lr"]
    args.dropout = hpo_best_config["dropout"]
    args.weight_decay = hpo_best_config["weight_decay"]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Section 3: Full training ──
    print("\n" + "=" * 60)
    print("SECTION 3: MODEL TRAINING (100% data)")
    print("=" * 60)
    full_model = build_model(args, device)
    full_model, full_hist, full_best_path, full_best_epoch, full_best_val_dice = train_with_early_stop(
        full_model, train_loader, val_loader, device, args, run_name="full_100pct"
    )

    train_eval_loader = DataLoader(
        SegTransformWrapper(train_dataset, None),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    full_train_loss, full_train_dice = evaluate_segmentation(full_model, train_eval_loader, device)
    full_val_loss, full_val_dice = evaluate_segmentation(full_model, val_loader, device)
    full_test_loss, full_test_dice = evaluate_segmentation(full_model, test_loader, device)

    print("\nFull model results:")
    print(f"  Train Dice: {full_train_dice:.4f}")
    print(f"  Val Dice:   {full_val_dice:.4f}")
    print(f"  Test Dice:  {full_test_dice:.4f}")

    # Epoch plots
    epochs_axis = np.arange(1, len(full_hist["train_loss"]) + 1)
    plot_curve(
        epochs_axis,
        full_hist["train_loss"],
        full_hist["val_loss"],
        "Train",
        "Val",
        "Train/Val Loss",
        "Epoch",
        "Dice+BCE Loss",
        os.path.join(args.out_dir, "train_val_loss.png"),
    )
    plot_curve(
        epochs_axis,
        full_hist["val_dice"],
        full_hist["val_dice"],
        "Val Dice",
        "",
        "Val Dice vs Epoch",
        "Epoch",
        "Dice Coefficient",
        os.path.join(args.out_dir, "val_dice.png"),
    )

    # Save model
    model_path = os.path.join(args.out_dir, "unet_binary_seg.pt")
    torch.save(full_model.state_dict(), model_path)

    # ── Section 4: Data valuation ──
    print("\n" + "=" * 60)
    print(f"SECTION 4: {method_name.upper()} DATA VALUATION")
    print("=" * 60)

    print("Extracting embeddings...")
    train_feats, train_y = extract_embeddings(full_model, train_dataset, args, device)
    val_feats, val_y = extract_embeddings(full_model, val_dataset, args, device)

    np.savez(
        os.path.join(args.out_dir, "embeddings_train.npz"),
        features=train_feats.cpu().numpy(),
        labels=train_y.cpu().numpy(),
    )
    np.savez(
        os.path.join(args.out_dir, "embeddings_val.npz"),
        features=val_feats.cpu().numpy(),
        labels=val_y.cpu().numpy(),
    )

    print(f"Computing {method_name} values...")
    v = compute_values_fn(train_feats, train_y, val_feats, val_y, args)
    v = np.asarray(v, dtype=np.float64)

    np.save(os.path.join(args.out_dir, f"{method_lower}_values.npy"), v)
    plot_hist(v, os.path.join(args.out_dir, f"{method_lower}_histogram.png"), method_name)

    train_filenames = train_dataset.filenames

    df_val = pd.DataFrame({"PatientID": train_filenames, f"{method_name}_value": v})
    df_val.to_csv(os.path.join(args.out_dir, f"{method_lower}_values_with_ids.csv"), index=False)

    order_asc = np.argsort(v)
    bottom_idx = order_asc[:50]
    top_idx = order_asc[-50:][::-1]

    print(f"\n{method_name} values: min={v.min():.6f}, max={v.max():.6f}, mean={v.mean():.6f}, std={v.std():.6f}")
    print(f"\nTop 10 {method_name} samples:")
    for rank, idx in enumerate(top_idx[:10], 1):
        print(f"  {rank}. {train_filenames[idx]}: {v[idx]:.6f}")
    print(f"\nBottom 10 {method_name} samples:")
    for rank, idx in enumerate(bottom_idx[:10], 1):
        print(f"  {rank}. {train_filenames[idx]}: {v[idx]:.6f}")

    # ── Section 5: Subset retraining experiments ──
    print("\n" + "=" * 60)
    print("SECTION 5: SUBSET RETRAINING EXPERIMENTS")
    print("=" * 60)

    subset_fracs = [float(s.strip()) for s in args.subset_fracs.split(",") if s.strip()]
    n_random_seeds = args.random_seeds
    order_desc = np.argsort(v)[::-1]
    order_asc_full = np.argsort(v)
    n_train_total = len(train_dataset)

    print(f"Fractions: {[int(f * 100) for f in subset_fracs]}%")
    print(f"Strategies: Top-N, Bottom-N, Random-N (x{n_random_seeds} seeds)")

    baseline = {
        "direction": "baseline",
        "frac": 1.00,
        "seed": args.seed,
        "n_train": n_train_total,
        "train_loss": full_train_loss,
        "train_dice": full_train_dice,
        "val_loss": full_val_loss,
        "val_dice": full_val_dice,
        "test_loss": full_test_loss,
        "test_dice": full_test_dice,
        "best_epoch": full_best_epoch,
        "best_val_dice": full_best_val_dice,
    }

    top_results = [dict(baseline, direction="top")]
    bottom_results = [dict(baseline, direction="bottom")]
    random_results = [dict(baseline, direction="random")]

    random_by_frac = {}

    for frac in subset_fracs:
        if frac <= 0.0 or frac > 1.0:
            continue
        n_keep = max(1, int(round(frac * n_train_total)))
        pct = int(frac * 100)

        # Top-N
        print(f"\n--- Top-{pct}% ({n_keep} samples) ---")
        top_res = _train_and_eval_subset(
            order_desc[:n_keep].tolist(),
            train_dataset,
            val_loader,
            test_loader,
            args,
            device,
            train_aug,
            f"top_{pct}pct",
            frac,
        )
        top_res["direction"] = "top"
        top_res["seed"] = args.seed
        top_results.append(top_res)

        # Bottom-N
        print(f"\n--- Bottom-{pct}% ({n_keep} samples) ---")
        bottom_res = _train_and_eval_subset(
            order_asc_full[:n_keep].tolist(),
            train_dataset,
            val_loader,
            test_loader,
            args,
            device,
            train_aug,
            f"bottom_{pct}pct",
            frac,
        )
        bottom_res["direction"] = "bottom"
        bottom_res["seed"] = args.seed
        bottom_results.append(bottom_res)

        # Random-N
        random_by_frac[frac] = []
        for seed_i in range(n_random_seeds):
            rng_seed = args.seed + seed_i + 1
            rng = np.random.RandomState(rng_seed)
            random_indices = rng.choice(n_train_total, size=n_keep, replace=False).tolist()

            torch.manual_seed(rng_seed)
            np.random.seed(rng_seed)

            print(f"\n--- Random-{pct}% seed={rng_seed} ({n_keep} samples) ---")
            rand_res = _train_and_eval_subset(
                random_indices,
                train_dataset,
                val_loader,
                test_loader,
                args,
                device,
                train_aug,
                f"random_{pct}pct_s{rng_seed}",
                frac,
            )
            rand_res["direction"] = "random"
            rand_res["seed"] = rng_seed
            random_results.append(rand_res)
            random_by_frac[frac].append(rand_res)

    # Reset seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Sort for plotting
    top_results = sorted(top_results, key=lambda d: d["frac"], reverse=True)
    bottom_results = sorted(bottom_results, key=lambda d: d["frac"], reverse=True)

    fracs_for_plot = sorted(set([1.0] + subset_fracs), reverse=True)
    fracs_pct = [int(round(f * 100)) for f in fracs_for_plot]

    metric_keys = ["train_dice", "train_loss", "val_dice", "val_loss", "test_dice", "test_loss"]
    top_by_frac = {r["frac"]: r for r in top_results}
    bottom_by_frac = {r["frac"]: r for r in bottom_results}

    top_metrics = {k: [float(top_by_frac[f][k]) for f in fracs_for_plot] for k in metric_keys}
    bottom_metrics = {k: [float(bottom_by_frac[f][k]) for f in fracs_for_plot] for k in metric_keys}

    random_mean_metrics = {}
    random_std_metrics = {}
    for k in metric_keys:
        means, stds = [], []
        for f in fracs_for_plot:
            if f >= 1.0:
                means.append(float(baseline[k]))
                stds.append(0.0)
            else:
                vals = [float(r[k]) for r in random_by_frac.get(f, [])]
                means.append(float(np.mean(vals)) if vals else 0.0)
                stds.append(float(np.std(vals)) if vals else 0.0)
        random_mean_metrics[k] = means
        random_std_metrics[k] = stds

    # Plots
    plot_retraining_curves(
        fracs_pct,
        top_metrics,
        bottom_metrics,
        random_mean_metrics,
        random_std_metrics,
        method_name,
        args.out_dir,
    )

    # CSV with all results
    all_results = (
        top_results + [r for r in bottom_results if r["frac"] < 1.0] + [r for r in random_results if r["frac"] < 1.0]
    )
    all_results = sorted(all_results, key=lambda d: (-d["frac"], d["direction"], d.get("seed", 0)))
    pd.DataFrame(all_results).to_csv(os.path.join(args.out_dir, "subset_results.csv"), index=False)

    # ── Final report ──
    report_path = os.path.join(args.out_dir, "final_report.txt")
    with open(report_path, "w") as f:
        f.write("=== RUN SUMMARY ===\n")
        f.write(f"Device: {device}\n")
        f.write("Task: Binary segmentation (tumor vs background)\n")
        f.write(f"DV labels: {config.NUM_TUMOR_CLASSES} tumor types ({', '.join(config.ALL_LABELS)})\n")
        f.write(f"Total samples: {n_train + n_val + n_test}\n")
        f.write(f"Train/Val/Test: {n_train}/{n_val}/{n_test}\n\n")

        f.write("=== HYPERPARAMETERS (after HPO) ===\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"early_stop_patience: {args.early_stop_patience}\n")
        f.write(f"lr: {args.lr}\n")
        f.write(f"dropout: {args.dropout}\n")
        f.write(f"weight_decay: {args.weight_decay}\n")
        f.write(f"seed: {args.seed}\n\n")

        f.write(f"=== HPO TRIALS ({len(hpo_trials)} trials, {HPO_EPOCHS} epochs each) ===\n")
        f.write("Trial\tlr\t\tdropout\tweight_decay\tval_dice\n")
        for i, t in enumerate(hpo_trials):
            f.write(f"{i + 1}\t{t['lr']:.1e}\t{t['dropout']}\t{t['weight_decay']:.1e}\t\t{t['val_dice']:.6f}\n")
        f.write(f"Selected: {hpo_best_config}\n\n")

        f.write(f"=== {method_name.upper()} PARAMETERS ===\n")
        f.write(f"{method_params_report}\n\n")

        f.write("=== FULL MODEL PERFORMANCE ===\n")
        f.write(f"Best epoch: {full_best_epoch}\n")
        f.write(f"Train: loss={full_train_loss:.6f}, dice={full_train_dice:.6f}\n")
        f.write(f"Val:   loss={full_val_loss:.6f}, dice={full_val_dice:.6f}\n")
        f.write(f"Test:  loss={full_test_loss:.6f}, dice={full_test_dice:.6f}\n\n")

        f.write(f"=== {method_name.upper()} VALUES SUMMARY ===\n")
        f.write(f"shape: {v.shape}\n")
        f.write(f"min: {v.min():.10f}\n")
        f.write(f"max: {v.max():.10f}\n")
        f.write(f"mean: {v.mean():.10f}\n")
        f.write(f"std: {v.std():.10f}\n\n")

        f.write(f"=== TOP 50 {method_name.upper()} SAMPLES ===\n")
        f.write(f"Rank\t{method_name}_value\tPatientID\n")
        for rank, idx in enumerate(top_idx, start=1):
            f.write(f"{rank}\t{v[idx]:.10f}\t{train_filenames[idx]}\n")

        f.write(f"\n=== BOTTOM 50 {method_name.upper()} SAMPLES ===\n")
        f.write(f"Rank\t{method_name}_value\tPatientID\n")
        for rank, idx in enumerate(bottom_idx, start=1):
            f.write(f"{rank}\t{v[idx]:.10f}\t{train_filenames[idx]}\n")

        f.write(f"\n=== TOP {method_name.upper()} SUBSET EXPERIMENTS ===\n")
        f.write("Kept%\tN_train\tTrainLoss\tTrainDice\tValLoss\tValDice\tTestLoss\tTestDice\tBestEpoch\tBestValDice\n")
        for r in top_results:
            kept = int(round(r["frac"] * 100))
            f.write(
                f"{kept}\t{r['n_train']}\t"
                f"{float(r['train_loss']):.6f}\t{float(r['train_dice']):.6f}\t"
                f"{float(r['val_loss']):.6f}\t{float(r['val_dice']):.6f}\t"
                f"{float(r['test_loss']):.6f}\t{float(r['test_dice']):.6f}\t"
                f"{int(r['best_epoch'])}\t{float(r['best_val_dice']):.6f}\n"
            )

        f.write(f"\n=== BOTTOM {method_name.upper()} SUBSET EXPERIMENTS ===\n")
        f.write("Kept%\tN_train\tTrainLoss\tTrainDice\tValLoss\tValDice\tTestLoss\tTestDice\tBestEpoch\tBestValDice\n")
        for r in bottom_results:
            if r["frac"] >= 1.0:
                continue
            kept = int(round(r["frac"] * 100))
            f.write(
                f"{kept}\t{r['n_train']}\t"
                f"{float(r['train_loss']):.6f}\t{float(r['train_dice']):.6f}\t"
                f"{float(r['val_loss']):.6f}\t{float(r['val_dice']):.6f}\t"
                f"{float(r['test_loss']):.6f}\t{float(r['test_dice']):.6f}\t"
                f"{int(r['best_epoch'])}\t{float(r['best_val_dice']):.6f}\n"
            )

        f.write(f"\n=== RANDOM SUBSET EXPERIMENTS (mean +/- std over {n_random_seeds} seeds) ===\n")
        f.write("Kept%\tTestDice_mean\tTestDice_std\tTestLoss_mean\tTestLoss_std\n")
        for fi, frac in enumerate(fracs_for_plot):
            if frac >= 1.0:
                continue
            pct = int(round(frac * 100))
            f.write(
                f"{pct}\t"
                f"{random_mean_metrics['test_dice'][fi]:.6f}\t{random_std_metrics['test_dice'][fi]:.6f}\t"
                f"{random_mean_metrics['test_loss'][fi]:.6f}\t{random_std_metrics['test_loss'][fi]:.6f}\n"
            )

        f.write("\n=== OUTPUT FILES ===\n")
        f.write("HPO trials: hpo_trials.csv\n")
        f.write(f"Model: {os.path.basename(model_path)}\n")
        f.write("Embeddings: embeddings_train.npz, embeddings_val.npz\n")
        f.write(f"{method_name} values: {method_lower}_values.npy\n")
        f.write(f"{method_name} values + IDs: {method_lower}_values_with_ids.csv\n")
        f.write("Subset results: subset_results.csv\n")
        f.write("Report: final_report.txt\n")

    elapsed = time.time() - start
    print(f"\nPipeline completed in {elapsed / 60:.1f} minutes.")
    print(f"Outputs saved to: {args.out_dir}")
    print(f"Report: {report_path}")
