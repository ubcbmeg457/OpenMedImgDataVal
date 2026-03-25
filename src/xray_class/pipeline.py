"""Main pipeline orchestrator for X-ray multi-label classification with data valuation."""

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
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms

from xray_class import config
from xray_class.data import ChestXray14, TransformWrapper, download_dataset
from xray_class.model import (
    build_model,
    compute_multilabel_auc,
    evaluate_multilabel,
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
    "epochs": 20,
    "batch_size": 32,
    "num_workers": 4,
    "early_stop_patience": 3,
    "pretrained_path": None,
    "torch_home": None,
    "seed": 42,
    "val_frac": 0.1,
    "test_frac": 0.1,
    "subset_fracs": "0.90,0.80,0.70,0.50,0.30",
    "random_seeds": 1,
    "subset_epochs": 8,
    "dropout": 0.4,
    "freeze_backbone": True,
    "unfreeze_last_block": True,
    "unfreeze_epoch": 6,
    "head_lr": 1e-3,
    "finetune_lr": 1e-4,
    "weight_decay_head": 0.0,
    "weight_decay_finetune": 5e-4,
    "use_scheduler": True,
    "ot_reg": 0.01,
    "shapley_mstar": 5000,
    "shapley_batch_val": 32,
    "shapley_k_candidates": "1,3,5,10,20,50",
}

HPO_SEARCH_SPACE = {
    "head_lr": [3e-4, 5e-4, 1e-3, 2e-3, 3e-3],
    "finetune_lr": [3e-5, 5e-5, 1e-4, 2e-4],
    "dropout": [0.2, 0.3, 0.4, 0.5],
    "weight_decay_finetune": [1e-4, 3e-4, 5e-4, 1e-3],
    "unfreeze_epoch": [3, 4, 5, 6, 8],
}
HPO_N_TRIALS = 3
HPO_EPOCHS = 10


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
    keep_indices, train_set, val_loader, test_loader, args, device, train_transform, eval_transform, run_name, frac
):
    train_subset = Subset(train_set, keep_indices)
    sub_train_loader = DataLoader(
        TransformWrapper(train_subset, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    sub_args = copy.copy(args)
    sub_args.epochs = args.subset_epochs
    sub_args.unfreeze_epoch = min(args.unfreeze_epoch, max(1, args.subset_epochs // 3))

    sub_model = build_model(sub_args, device)
    sub_model, _, _, sub_best_epoch, sub_best_val_auc = train_with_early_stop(
        sub_model, sub_train_loader, val_loader, device, sub_args, run_name=run_name
    )

    sub_train_eval_loader = DataLoader(
        TransformWrapper(train_subset, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    sub_train_loss, sub_train_auc, _, _ = evaluate_multilabel(sub_model, sub_train_eval_loader, device)
    sub_val_loss, sub_val_auc, _, _ = evaluate_multilabel(sub_model, val_loader, device)
    sub_test_loss, sub_test_auc, _, _ = evaluate_multilabel(sub_model, test_loader, device)

    return {
        "frac": frac,
        "n_train": len(keep_indices),
        "train_loss": sub_train_loss,
        "train_auc": sub_train_auc,
        "val_loss": sub_val_loss,
        "val_auc": sub_val_auc,
        "test_loss": sub_test_loss,
        "test_auc": sub_test_auc,
        "best_epoch": sub_best_epoch,
        "best_val_auc": sub_best_val_auc,
    }


def _optimize_hyperparameters(train_loader, val_loader, args, device):
    hpo_dir = os.path.join(args.out_dir, "hpo")
    os.makedirs(hpo_dir, exist_ok=True)

    trials = []
    best_auc = -1.0
    best_config = {}

    print(f"\n{'=' * 60}")
    print(f"Hyperparameter Optimisation ({HPO_N_TRIALS} trials, {HPO_EPOCHS} epochs each)")
    print(f"{'=' * 60}")

    for trial_idx in range(HPO_N_TRIALS):
        trial_config = {k: random.choice(v) for k, v in HPO_SEARCH_SPACE.items()}

        trial_args = copy.copy(args)
        trial_args.head_lr = trial_config["head_lr"]
        trial_args.finetune_lr = trial_config["finetune_lr"]
        trial_args.dropout = trial_config["dropout"]
        trial_args.weight_decay_finetune = trial_config["weight_decay_finetune"]
        trial_args.unfreeze_epoch = trial_config["unfreeze_epoch"]
        trial_args.epochs = HPO_EPOCHS
        trial_args.out_dir = hpo_dir

        model = build_model(trial_args, device)
        _, _, _, _, val_auc = train_with_early_stop(
            model, train_loader, val_loader, device, trial_args, run_name=f"hpo_trial_{trial_idx}"
        )

        trial_config["val_auc"] = val_auc
        trials.append(trial_config)

        marker = " *best*" if val_auc > best_auc else ""
        print(
            f"  Trial {trial_idx + 1}/{HPO_N_TRIALS} | val_auc={val_auc:.4f}{marker} | "
            f"head_lr={trial_config['head_lr']:.1e} finetune_lr={trial_config['finetune_lr']:.1e} "
            f"dropout={trial_config['dropout']} wd_ft={trial_config['weight_decay_finetune']:.1e} "
            f"unfreeze_ep={trial_config['unfreeze_epoch']}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_config = {k: v for k, v in trial_config.items() if k != "val_auc"}

    pd.DataFrame(trials).to_csv(os.path.join(args.out_dir, "hpo_trials.csv"), index=False)
    shutil.rmtree(hpo_dir, ignore_errors=True)

    print(f"\nBest config (val_auc={best_auc:.4f}): {best_config}")
    print(f"{'=' * 60}\n")
    return best_config, trials


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────
def run_pipeline(args, compute_values_fn, method_name, method_params_report):
    """Full data-valuation pipeline for X-ray multi-label classification."""
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

    if args.torch_home is not None:
        os.environ["TORCH_HOME"] = args.torch_home

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classification: Multi-label ({config.NUM_CLASSES} classes)")

    # ── Transforms ──
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

    # ── Section 1: Data download ──
    print("\n" + "=" * 60)
    print("SECTION 1: DATA DOWNLOAD")
    print("=" * 60)
    data_path = download_dataset()

    dataset = ChestXray14(str(data_path), transform=None)

    n_total = len(dataset)
    n_test = int(args.test_frac * n_total)
    n_val = int(args.val_frac * n_total)
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise ValueError("Split fractions too large; train set would be empty.")

    gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=gen)

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

    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
    print(f"Classes: {config.NUM_CLASSES} ({', '.join(config.ALL_LABELS)})")

    # ── Section 2: HPO ──
    print("\n" + "=" * 60)
    print("SECTION 2: HYPERPARAMETER OPTIMISATION")
    print("=" * 60)
    hpo_best_config, hpo_trials = _optimize_hyperparameters(train_loader, val_loader, args, device)

    args.head_lr = hpo_best_config["head_lr"]
    args.finetune_lr = hpo_best_config["finetune_lr"]
    args.dropout = hpo_best_config["dropout"]
    args.weight_decay_finetune = hpo_best_config["weight_decay_finetune"]
    args.unfreeze_epoch = hpo_best_config["unfreeze_epoch"]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Section 3: Full training ──
    print("\n" + "=" * 60)
    print("SECTION 3: MODEL TRAINING (100% data)")
    print("=" * 60)
    full_model = build_model(args, device)
    full_model, full_hist, full_best_path, full_best_epoch, full_best_val_auc = train_with_early_stop(
        full_model, train_loader, val_loader, device, args, run_name="full_100pct"
    )

    train_eval_loader = DataLoader(
        train_set_eval_t, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    full_train_loss, full_train_auc, _, _ = evaluate_multilabel(full_model, train_eval_loader, device)
    full_val_loss, full_val_auc, _, _ = evaluate_multilabel(full_model, val_loader, device)
    full_test_loss, full_test_auc, y_test_true, y_test_prob = evaluate_multilabel(full_model, test_loader, device)

    _, per_class_aucs = compute_multilabel_auc(y_test_true, y_test_prob)

    print("\nFull model results:")
    print(f"  Train AUC: {full_train_auc:.4f}")
    print(f"  Val AUC:   {full_val_auc:.4f}")
    print(f"  Test AUC:  {full_test_auc:.4f}")
    print("  Per-class Test AUC:")
    for label, auc_val in per_class_aucs.items():
        print(f"    {label}: {auc_val:.4f}")

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
        "Loss",
        os.path.join(args.out_dir, "train_val_loss.png"),
    )
    plot_curve(
        epochs_axis,
        full_hist["val_auc"],
        full_hist["val_auc"],
        "Val AUC",
        "",
        "Val Mean AUC vs Epoch",
        "Epoch",
        "Mean AUC-ROC",
        os.path.join(args.out_dir, "val_auc.png"),
    )

    # Save model
    model_path = os.path.join(args.out_dir, "densenet121_multilabel.pt")
    torch.save(full_model.state_dict(), model_path)

    # ── Section 4: Data valuation ──
    print("\n" + "=" * 60)
    print(f"SECTION 4: {method_name.upper()} DATA VALUATION")
    print("=" * 60)

    print("Extracting embeddings...")
    train_feats, train_y = extract_embeddings(full_model, train_set_eval_t, args, device)
    val_feats, val_y = extract_embeddings(full_model, val_set_t, args, device)

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

    train_indices = train_set.indices
    train_filenames = [dataset.filenames[i] for i in train_indices]

    df_val = pd.DataFrame({"ImageID": train_filenames, f"{method_name}_value": v})
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

    # ── Section 5: Subset retraining experiments (Top-N, Bottom-N, Random-N) ──
    print("\n" + "=" * 60)
    print("SECTION 5: SUBSET RETRAINING EXPERIMENTS")
    print("=" * 60)

    subset_fracs = [float(s.strip()) for s in args.subset_fracs.split(",") if s.strip()]
    n_random_seeds = args.random_seeds
    order_desc = np.argsort(v)[::-1]
    order_asc_full = np.argsort(v)
    n_train_total = len(train_set)

    print(f"Fractions: {[int(f * 100) for f in subset_fracs]}%")
    print(f"Strategies: Top-N, Bottom-N, Random-N (x{n_random_seeds} seeds)")

    baseline = {
        "direction": "baseline",
        "frac": 1.00,
        "seed": args.seed,
        "n_train": n_train_total,
        "train_loss": full_train_loss,
        "train_auc": full_train_auc,
        "val_loss": full_val_loss,
        "val_auc": full_val_auc,
        "test_loss": full_test_loss,
        "test_auc": full_test_auc,
        "best_epoch": full_best_epoch,
        "best_val_auc": full_best_val_auc,
    }

    top_results = [dict(baseline, direction="top")]
    bottom_results = [dict(baseline, direction="bottom")]
    random_results = [dict(baseline, direction="random")]

    # Collect per-frac random results for computing mean/std
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
            train_set,
            val_loader,
            test_loader,
            args,
            device,
            train_transform,
            eval_transform,
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
            train_set,
            val_loader,
            test_loader,
            args,
            device,
            train_transform,
            eval_transform,
            f"bottom_{pct}pct",
            frac,
        )
        bottom_res["direction"] = "bottom"
        bottom_res["seed"] = args.seed
        bottom_results.append(bottom_res)

        # Random-N (multiple seeds for error bars)
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
                train_set,
                val_loader,
                test_loader,
                args,
                device,
                train_transform,
                eval_transform,
                f"random_{pct}pct_s{rng_seed}",
                frac,
            )
            rand_res["direction"] = "random"
            rand_res["seed"] = rng_seed
            random_results.append(rand_res)
            random_by_frac[frac].append(rand_res)

    # Reset seeds after subset experiments
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Sort for plotting (descending frac so 100% is first)
    top_results = sorted(top_results, key=lambda d: d["frac"], reverse=True)
    bottom_results = sorted(bottom_results, key=lambda d: d["frac"], reverse=True)

    # Build aligned arrays for plotting: 100% baseline + each frac
    fracs_for_plot = sorted(set([1.0] + subset_fracs), reverse=True)
    fracs_pct = [int(round(f * 100)) for f in fracs_for_plot]

    metric_keys = ["train_auc", "train_loss", "val_auc", "val_loss", "test_auc", "test_loss"]
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
        f.write(f"Classification: Multi-label ({config.NUM_CLASSES} classes)\n")
        f.write(f"Classes: {', '.join(config.ALL_LABELS)}\n")
        f.write(f"Total samples: {n_total}\n")
        f.write(f"Train/Val/Test: {n_train}/{n_val}/{n_test}\n\n")

        f.write("=== HYPERPARAMETERS (after HPO) ===\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"early_stop_patience: {args.early_stop_patience}\n")
        f.write(f"dropout: {args.dropout}\n")
        f.write(f"head_lr: {args.head_lr}\n")
        f.write(f"finetune_lr: {args.finetune_lr}\n")
        f.write(f"weight_decay_head: {args.weight_decay_head}\n")
        f.write(f"weight_decay_finetune: {args.weight_decay_finetune}\n")
        f.write(f"freeze_backbone: {args.freeze_backbone}\n")
        f.write(f"unfreeze_last_block: {args.unfreeze_last_block}\n")
        f.write(f"unfreeze_epoch: {args.unfreeze_epoch}\n")
        f.write(f"seed: {args.seed}\n\n")

        f.write(f"=== HPO TRIALS ({len(hpo_trials)} trials, {HPO_EPOCHS} epochs each) ===\n")
        f.write("Trial\thead_lr\t\tfinetune_lr\tdropout\twd_finetune\tunfreeze_ep\tval_auc\n")
        for i, t in enumerate(hpo_trials):
            f.write(
                f"{i + 1}\t{t['head_lr']:.1e}\t{t['finetune_lr']:.1e}\t\t{t['dropout']}\t"
                f"{t['weight_decay_finetune']:.1e}\t\t{t['unfreeze_epoch']}\t\t{t['val_auc']:.6f}\n"
            )
        f.write(f"Selected: {hpo_best_config}\n\n")

        f.write(f"=== {method_name.upper()} PARAMETERS ===\n")
        f.write(f"{method_params_report}\n\n")

        f.write("=== FULL MODEL PERFORMANCE ===\n")
        f.write(f"Best epoch: {full_best_epoch}\n")
        f.write(f"Train: loss={full_train_loss:.6f}, mean_auc={full_train_auc:.6f}\n")
        f.write(f"Val:   loss={full_val_loss:.6f}, mean_auc={full_val_auc:.6f}\n")
        f.write(f"Test:  loss={full_test_loss:.6f}, mean_auc={full_test_auc:.6f}\n\n")

        f.write("=== PER-CLASS TEST AUC ===\n")
        for label, auc_val in per_class_aucs.items():
            f.write(f"{label}: {auc_val:.6f}\n")
        f.write(f"Mean: {full_test_auc:.6f}\n\n")

        f.write(f"=== {method_name.upper()} VALUES SUMMARY ===\n")
        f.write(f"shape: {v.shape}\n")
        f.write(f"min: {v.min():.10f}\n")
        f.write(f"max: {v.max():.10f}\n")
        f.write(f"mean: {v.mean():.10f}\n")
        f.write(f"std: {v.std():.10f}\n\n")

        f.write(f"=== TOP 50 {method_name.upper()} SAMPLES ===\n")
        f.write(f"Rank\t{method_name}_value\tImageID\n")
        for rank, idx in enumerate(top_idx, start=1):
            f.write(f"{rank}\t{v[idx]:.10f}\t{train_filenames[idx]}\n")

        f.write(f"\n=== BOTTOM 50 {method_name.upper()} SAMPLES ===\n")
        f.write(f"Rank\t{method_name}_value\tImageID\n")
        for rank, idx in enumerate(bottom_idx, start=1):
            f.write(f"{rank}\t{v[idx]:.10f}\t{train_filenames[idx]}\n")

        f.write(f"\n=== TOP {method_name.upper()} SUBSET EXPERIMENTS ===\n")
        f.write("Kept%\tN_train\tTrainLoss\tTrainAUC\tValLoss\tValAUC\tTestLoss\tTestAUC\tBestEpoch\tBestValAUC\n")
        for r in top_results:
            kept = int(round(r["frac"] * 100))
            f.write(
                f"{kept}\t{r['n_train']}\t"
                f"{float(r['train_loss']):.6f}\t{float(r['train_auc']):.6f}\t"
                f"{float(r['val_loss']):.6f}\t{float(r['val_auc']):.6f}\t"
                f"{float(r['test_loss']):.6f}\t{float(r['test_auc']):.6f}\t"
                f"{int(r['best_epoch'])}\t{float(r['best_val_auc']):.6f}\n"
            )

        f.write(f"\n=== BOTTOM {method_name.upper()} SUBSET EXPERIMENTS ===\n")
        f.write("Kept%\tN_train\tTrainLoss\tTrainAUC\tValLoss\tValAUC\tTestLoss\tTestAUC\tBestEpoch\tBestValAUC\n")
        for r in bottom_results:
            if r["frac"] >= 1.0:
                continue
            kept = int(round(r["frac"] * 100))
            f.write(
                f"{kept}\t{r['n_train']}\t"
                f"{float(r['train_loss']):.6f}\t{float(r['train_auc']):.6f}\t"
                f"{float(r['val_loss']):.6f}\t{float(r['val_auc']):.6f}\t"
                f"{float(r['test_loss']):.6f}\t{float(r['test_auc']):.6f}\t"
                f"{int(r['best_epoch'])}\t{float(r['best_val_auc']):.6f}\n"
            )

        f.write(f"\n=== RANDOM SUBSET EXPERIMENTS (mean +/- std over {n_random_seeds} seeds) ===\n")
        f.write("Kept%\tTestAUC_mean\tTestAUC_std\tTestLoss_mean\tTestLoss_std\n")
        for fi, frac in enumerate(fracs_for_plot):
            if frac >= 1.0:
                continue
            pct = int(round(frac * 100))
            f.write(
                f"{pct}\t"
                f"{random_mean_metrics['test_auc'][fi]:.6f}\t{random_std_metrics['test_auc'][fi]:.6f}\t"
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
