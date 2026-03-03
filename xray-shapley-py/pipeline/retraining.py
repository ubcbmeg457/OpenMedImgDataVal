"""Section 6: DenseNet121 retraining experiments on Shapley-ranked subsets.

This module validates the KNN-Shapley data rankings produced in Section 5 by
retraining DenseNet121 **from scratch** (ImageNet weights) on subsets of the
training data at various retention rates, then evaluating on the held-out test
set.

**Experimental design:**
For each retention fraction in ``RETRAIN_FRACTIONS``:
  - **Top-K**: keep the *k* highest-Shapley samples.
  - **Bottom-K**: keep the *k* lowest-Shapley samples.
  - **Random-K** (x ``RETRAIN_RANDOM_SEEDS``): keep *k* random samples (provides
    error bars / standard deviation bands).

**Fair-comparison guarantees:**
  - Every run starts from *fresh ImageNet weights* — not from the Section 3
    checkpoint — so performance differences are attributable to data quality
    alone.
  - The test set is **never** used during Shapley computation or training; it
    is reserved exclusively for final evaluation.
  - The validation set is reused for early stopping (acknowledged limitation,
    shared with Section 5's Shapley computation).

**References:**
  - Ghorbani & Zou (2019). "Data Shapley." ICML.
  - Jia et al. (2019). "Efficient Task-Specific Data Valuation." VLDB.
"""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pipeline import config
from pipeline.model import (
    MetricsResult,
    SplitData,
    TransformSubset,
    compute_metrics,
    create_fresh_model,
    get_transforms,
    predict_dataset,
    train_model_on_loaders,
)
from pipeline.valuation import ValuationResult


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass
class RetrainingResult:
    """Aggregated results of all retraining runs."""

    fractions: list[float]
    strategies: list[str]
    metrics_table: pd.DataFrame
    baseline_metrics: MetricsResult = field(default_factory=lambda: None)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Subset loaders
# ---------------------------------------------------------------------------
def _create_subset_loaders(
    split: SplitData,
    subset_indices: np.ndarray,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val loaders for a subset of the training data.

    *subset_indices* are indices into ``split.y_train`` / ``split.train_actual``
    (aligned with ``data_shapley``).  The absolute dataset indices are obtained
    via ``split.train_actual[subset_indices]``.

    The **validation loader always uses the full validation set** so that all
    runs are compared fairly.
    """
    train_transform, eval_transform = get_transforms()
    abs_train_indices = split.train_actual[subset_indices]
    train_ds = TransformSubset(split.dataset, abs_train_indices, train_transform)
    val_ds = TransformSubset(split.dataset, split.val_actual, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Single retrain + evaluate
# ---------------------------------------------------------------------------
def _retrain_and_evaluate(
    split: SplitData,
    subset_indices: np.ndarray,
    device: torch.device,
) -> MetricsResult:
    """Retrain DenseNet121 from ImageNet weights on *subset_indices* and evaluate on the test set."""
    model = create_fresh_model(device, finetune_all=config.RETRAIN_FINETUNE_ALL, quiet=True)

    train_loader, val_loader = _create_subset_loaders(split, subset_indices)
    y_train_sub = split.y_train[subset_indices]

    model = train_model_on_loaders(
        model,
        train_loader,
        val_loader,
        y_train_sub,
        split.y_val,
        device,
        num_epochs=config.RETRAIN_NUM_EPOCHS,
        patience=config.RETRAIN_PATIENCE,
        lr=config.RETRAIN_LR,
        weight_decay=config.RETRAIN_WEIGHT_DECAY,
        quiet=True,
    )

    y_pred, y_pred_proba, _y_true = predict_dataset(model, split.test_loader, device)
    metrics = compute_metrics(split.y_test, y_pred, y_pred_proba)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return metrics


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
def run_retraining_experiments(
    data_shapley: np.ndarray,
    split: SplitData,
    device: torch.device,
) -> RetrainingResult:
    """Run Top-K / Bottom-K / Random-K retraining experiments across all fractions."""
    fractions = config.RETRAIN_FRACTIONS
    n_train = len(data_shapley)
    sorted_indices = np.argsort(data_shapley)

    rows: list[dict] = []
    total_runs = len(fractions) * (2 + config.RETRAIN_RANDOM_SEEDS)
    run_idx = 0

    for frac in fractions:
        k = max(1, int(n_train * frac))

        # At 100% all strategies select the same samples — train once and
        # reuse the result so the metrics are guaranteed identical.
        if frac == 1.0:
            all_indices = np.arange(n_train)
            metrics = _retrain_and_evaluate(split, all_indices, device)
            for strategy in ["top_k", "bottom_k"]:
                run_idx += 1
                print(
                    f"  [{run_idx:2d}/{total_runs}] {strategy:<9s} {frac:.0%} ({k:4d} samples)"
                    f"  AUC={metrics.auc_roc:.4f}  F1={metrics.f1:.4f}"
                )
                row = {"fraction": frac, "k": k, "strategy": strategy, "seed": None}
                row.update(metrics.to_dict())
                rows.append(row)
            for seed_offset in range(config.RETRAIN_RANDOM_SEEDS):
                run_idx += 1
                seed_val = config.SEED + seed_offset
                print(
                    f"  [{run_idx:2d}/{total_runs}] random    {frac:.0%} ({k:4d} samples, seed={seed_val})"
                    f"  AUC={metrics.auc_roc:.4f}  F1={metrics.f1:.4f}"
                )
                row = {
                    "fraction": frac,
                    "k": k,
                    "strategy": "random",
                    "seed": seed_val,
                }
                row.update(metrics.to_dict())
                rows.append(row)
            continue

        # --- Top-K ---
        run_idx += 1
        top_k_indices = sorted_indices[-k:]
        metrics = _retrain_and_evaluate(split, top_k_indices, device)
        print(
            f"  [{run_idx:2d}/{total_runs}] top_k     {frac:.0%} ({k:4d} samples)"
            f"  AUC={metrics.auc_roc:.4f}  F1={metrics.f1:.4f}"
        )
        row = {"fraction": frac, "k": k, "strategy": "top_k", "seed": None}
        row.update(metrics.to_dict())
        rows.append(row)

        # --- Bottom-K ---
        run_idx += 1
        bottom_k_indices = sorted_indices[:k]
        metrics = _retrain_and_evaluate(split, bottom_k_indices, device)
        print(
            f"  [{run_idx:2d}/{total_runs}] bottom_k  {frac:.0%} ({k:4d} samples)"
            f"  AUC={metrics.auc_roc:.4f}  F1={metrics.f1:.4f}"
        )
        row = {"fraction": frac, "k": k, "strategy": "bottom_k", "seed": None}
        row.update(metrics.to_dict())
        rows.append(row)

        # --- Random-K (multiple seeds) ---
        for seed_offset in range(config.RETRAIN_RANDOM_SEEDS):
            run_idx += 1
            rng = np.random.RandomState(config.SEED + seed_offset)
            random_indices = rng.choice(n_train, k, replace=False)
            metrics = _retrain_and_evaluate(split, random_indices, device)
            seed_val = config.SEED + seed_offset
            print(
                f"  [{run_idx:2d}/{total_runs}] random    {frac:.0%} ({k:4d} samples, seed={seed_val})"
                f"  AUC={metrics.auc_roc:.4f}  F1={metrics.f1:.4f}"
            )
            row = {
                "fraction": frac,
                "k": k,
                "strategy": "random",
                "seed": seed_val,
            }
            row.update(metrics.to_dict())
            rows.append(row)

    df = pd.DataFrame(rows)
    strategies = ["top_k", "bottom_k", "random"]

    # Baseline = the fraction=1.0 top_k run (uses all training data)
    baseline_row = df[(df["fraction"] == 1.0) & (df["strategy"] == "top_k")].iloc[0]
    baseline_metrics = MetricsResult(
        accuracy=baseline_row["accuracy"],
        precision=baseline_row["precision"],
        recall=baseline_row["recall"],
        specificity=baseline_row["specificity"],
        f1=baseline_row["f1"],
        auc_roc=baseline_row["auc_roc"],
    )

    return RetrainingResult(
        fractions=fractions,
        strategies=strategies,
        metrics_table=df,
        baseline_metrics=baseline_metrics,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_retraining_curves(result: RetrainingResult) -> None:
    """2-panel figure: AUC-ROC and F1 vs data fraction."""
    df = result.metrics_table
    fracs = sorted(df["fraction"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metric_names = ["auc_roc", "f1"]
    metric_labels = ["AUC-ROC", "F1 Score"]

    for ax, metric, label in zip(axes, metric_names, metric_labels):
        # Top-K
        top = df[df["strategy"] == "top_k"].sort_values("fraction")
        ax.plot(top["fraction"], top[metric], "o-", label="Top-K (Shapley)", linewidth=2)

        # Bottom-K
        bot = df[df["strategy"] == "bottom_k"].sort_values("fraction")
        ax.plot(bot["fraction"], bot[metric], "^:", label="Bottom-K (Worst)", linewidth=2)

        # Random-K (mean +/- std)
        rand_df = df[df["strategy"] == "random"]
        rand_mean = rand_df.groupby("fraction")[metric].mean().reindex(fracs)
        rand_std = rand_df.groupby("fraction")[metric].std().reindex(fracs).fillna(0)
        ax.plot(fracs, rand_mean, "s--", label="Random-K (Baseline)", linewidth=2)
        ax.fill_between(
            fracs,
            rand_mean - rand_std,
            rand_mean + rand_std,
            alpha=0.2,
        )

        # Baseline reference
        baseline_val = getattr(result.baseline_metrics, metric)
        ax.axhline(baseline_val, color="gray", linestyle="--", alpha=0.6, label="Full-dataset baseline")

        ax.set_xlabel("Data Fraction", fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f"{label} vs Data Fraction", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = config.PLOTS_DIR / "retraining_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {config.rel(out_path)}")


def plot_retraining_comprehensive_metrics(result: RetrainingResult) -> None:
    """2x3 grid of all six metrics vs data fraction."""
    df = result.metrics_table
    fracs = sorted(df["fraction"].unique())

    metric_names = ["auc_roc", "f1", "precision", "recall", "specificity", "accuracy"]
    metric_labels = ["AUC-ROC", "F1 Score", "Precision", "Recall (Sensitivity)", "Specificity", "Accuracy"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for ax, metric, label in zip(axes_flat, metric_names, metric_labels):
        # Top-K
        top = df[df["strategy"] == "top_k"].sort_values("fraction")
        ax.plot(top["fraction"], top[metric], "o-", label="Top-K", linewidth=2)

        # Bottom-K
        bot = df[df["strategy"] == "bottom_k"].sort_values("fraction")
        ax.plot(bot["fraction"], bot[metric], "^:", label="Bottom-K", linewidth=2)

        # Random-K
        rand_df = df[df["strategy"] == "random"]
        rand_mean = rand_df.groupby("fraction")[metric].mean().reindex(fracs)
        rand_std = rand_df.groupby("fraction")[metric].std().reindex(fracs).fillna(0)
        ax.plot(fracs, rand_mean, "s--", label="Random-K", linewidth=2)
        ax.fill_between(fracs, rand_mean - rand_std, rand_mean + rand_std, alpha=0.2)

        # Baseline
        baseline_val = getattr(result.baseline_metrics, metric)
        ax.axhline(baseline_val, color="gray", linestyle="--", alpha=0.6, label="Baseline")

        ax.set_xlabel("Data Fraction")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Retraining Metrics vs Data Fraction (DenseNet121 from ImageNet)", fontsize=14)
    plt.tight_layout()
    out_path = config.PLOTS_DIR / "retraining_all_metrics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {config.rel(out_path)}")


# ---------------------------------------------------------------------------
# Save / print
# ---------------------------------------------------------------------------
def save_retraining_results(result: RetrainingResult) -> None:
    """Save the full metrics table to CSV."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "retraining_results.csv"
    result.metrics_table.to_csv(out_path, index=False)
    print(f"Saved: {config.rel(out_path)}")


def print_retraining_summary(result: RetrainingResult) -> None:
    """Print a formatted summary table with all metrics."""
    df = result.metrics_table
    metric_cols = ["auc_roc", "f1", "precision", "recall", "specificity", "accuracy"]
    header = f"{'Frac':>6} {'Strategy':>10}" + "".join(f" {m:>11}" for m in metric_cols)

    print("\n" + "=" * len(header))
    print("RETRAINING RESULTS SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for frac in sorted(df["fraction"].unique()):
        frac_df = df[df["fraction"] == frac]

        # Top-K
        top_row = frac_df[frac_df["strategy"] == "top_k"].iloc[0]
        vals = "".join(f" {top_row[m]:>11.4f}" for m in metric_cols)
        print(f"{frac:>6.0%} {'top_k':>10}{vals}")

        # Bottom-K
        bot_row = frac_df[frac_df["strategy"] == "bottom_k"].iloc[0]
        vals = "".join(f" {bot_row[m]:>11.4f}" for m in metric_cols)
        print(f"{'':>6} {'bottom_k':>10}{vals}")

        # Random-K (mean +/- std)
        rand_rows = frac_df[frac_df["strategy"] == "random"]
        if len(rand_rows) > 0:
            means = rand_rows[metric_cols].mean()
            stds = rand_rows[metric_cols].std().fillna(0)
            vals = "".join(f" {means[m]:>.4f}±{stds[m]:.4f}" for m in metric_cols)
            print(f"{'':>6} {'random':>10}{vals}")

    print("-" * len(header))
    bl = result.baseline_metrics
    vals = "".join(f" {getattr(bl, m):>11.4f}" for m in metric_cols)
    print(f"{'100%':>6} {'baseline':>10}{vals}")
    print(
        "\nNote: 'baseline' is the full-dataset retrain. 'random' shows mean±std"
        f" across {config.RETRAIN_RANDOM_SEEDS} seeds."
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_retraining_evaluation(
    valuation_result: ValuationResult,
    split: SplitData,
    device: torch.device,
) -> RetrainingResult:
    """Section 6 orchestrator: retrain DenseNet121 on Shapley-ranked subsets."""
    print("\n" + "=" * 60)
    print("SECTION 6: RETRAINING EXPERIMENTS")
    print("=" * 60)

    n_train = len(valuation_result.data_shapley)
    total_runs = len(config.RETRAIN_FRACTIONS) * (2 + config.RETRAIN_RANDOM_SEEDS)

    print(f"Training samples: {n_train}")
    print(f"Fractions: {config.RETRAIN_FRACTIONS}")
    print(f"Strategies: top_k, bottom_k, random (x{config.RETRAIN_RANDOM_SEEDS} seeds)")
    print(f"Total retraining runs: {total_runs}")
    print(f"Epochs per run: {config.RETRAIN_NUM_EPOCHS} (patience={config.RETRAIN_PATIENCE})")
    print(f"LR: {config.RETRAIN_LR}, Weight decay: {config.RETRAIN_WEIGHT_DECAY}")
    print(f"Finetune all layers: {config.RETRAIN_FINETUNE_ALL}")
    print()

    result = run_retraining_experiments(valuation_result.data_shapley, split, device)

    print_retraining_summary(result)
    plot_retraining_curves(result)
    plot_retraining_comprehensive_metrics(result)
    save_retraining_results(result)

    return result
