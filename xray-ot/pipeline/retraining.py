"""Section 4: OT-subset retraining experiments."""

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

from pipeline import config
from pipeline.data import DataResult, TransformWrapper, get_transforms
from pipeline.model import (
    TrainResult,
    build_model,
    compute_roc_auc_and_best_threshold,
    evaluate_binary,
    train_with_freeze_unfreeze_and_early_stop,
)
from pipeline.ot_valuation import OTResult


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass
class RetrainingResult:
    """Aggregated results of all OT-subset retraining runs."""

    results: list[dict]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_acc_vs_size(xs_pct, ys, title, out_path):
    plt.figure()
    plt.plot(xs_pct, ys, marker="o")
    plt.title(title)
    plt.xlabel("Training data kept (%) (top OT)")
    plt.ylabel("Accuracy")
    plt.xticks(xs_pct)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_subset_results(results, out_dir):
    """Plot accuracy vs training data size for train/val/test."""
    xs_pct = [int(round(r["frac"] * 100)) for r in results]
    train_accs = [float(r["train_acc"]) for r in results]
    val_accs = [float(r["val_acc"]) for r in results]
    test_accs = [float(r["test_acc"]) for r in results]

    _plot_acc_vs_size(
        xs_pct,
        train_accs,
        "Train Accuracy vs Training Data Kept (Top OT)",
        os.path.join(out_dir, "subset_train_accuracy_vs_size.png"),
    )
    _plot_acc_vs_size(
        xs_pct,
        val_accs,
        "Validation Accuracy vs Training Data Kept (Top OT)",
        os.path.join(out_dir, "subset_val_accuracy_vs_size.png"),
    )
    _plot_acc_vs_size(
        xs_pct,
        test_accs,
        "Test Accuracy vs Training Data Kept (Top OT)",
        os.path.join(out_dir, "subset_test_accuracy_vs_size.png"),
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_final_report(
    train_result: TrainResult, ot_result: OTResult, retrain_result: RetrainingResult, data: DataResult, args
):
    """Write a comprehensive final report to disk."""
    v = ot_result.ot_values
    results = retrain_result.results

    k50 = 50
    order_asc = np.argsort(v)
    bottom_idx = order_asc[:k50]
    top_idx = order_asc[-k50:][::-1]

    device = next(train_result.model.parameters()).device
    report_path = os.path.join(args.out_dir, "final_report.txt")

    with open(report_path, "w") as f:
        f.write("=== RUN SUMMARY ===\n")
        f.write(f"Device: {device}\n")
        f.write(f"Total samples: {data.n_train + data.n_val + data.n_test}\n")
        f.write(f"Train/Val/Test sizes: {data.n_train}/{data.n_val}/{data.n_test}\n\n")

        f.write("=== HYPERPARAMETERS ===\n")
        f.write(f"epochs_requested: {args.epochs}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"early_stop_patience (val_auc no-improve): {args.early_stop_patience}\n")
        f.write(f"ot_reg: {args.ot_reg}\n")
        f.write(f"pretrained_path: {args.pretrained_path}\n")
        f.write(f"seed: {args.seed}\n\n")

        f.write("=== IMPROVEMENTS ENABLED ===\n")
        f.write(f"dropout: {args.dropout}\n")
        f.write(f"freeze_backbone: {args.freeze_backbone}\n")
        f.write(f"unfreeze_last_block: {args.unfreeze_last_block}\n")
        f.write(f"unfreeze_epoch (1-indexed): {args.unfreeze_epoch}\n")
        f.write(f"head_lr: {args.head_lr}\n")
        f.write(f"finetune_lr: {args.finetune_lr}\n")
        f.write(f"weight_decay_head: {args.weight_decay_head}\n")
        f.write(f"weight_decay_finetune: {args.weight_decay_finetune}\n")
        f.write(f"scheduler: {'ReduceLROnPlateau(mode=max,val_auc)' if args.use_scheduler else 'None'}\n\n")

        f.write("=== FULL DATA (100%) BEST MODEL (by VAL AUC) ===\n")
        f.write(f"best_epoch: {train_result.best_epoch}\n")
        f.write(f"best_val_auc: {train_result.best_val_auc:.6f}\n")
        f.write(f"Train: loss={train_result.train_loss:.6f}, acc@0.5={train_result.train_acc:.6f}\n")
        f.write(
            f"Val:   loss={train_result.val_loss:.6f}, acc@0.5={train_result.val_acc:.6f}, "
            f"auc={train_result.val_auc:.6f}, best_thr={train_result.val_best_thr:.6f}\n"
        )
        f.write(
            f"Test:  loss={train_result.test_loss:.6f}, acc@0.5={train_result.test_acc:.6f}, "
            f"auc={train_result.test_auc:.6f}\n\n"
        )

        f.write("=== OT VALUES SUMMARY (train samples) ===\n")
        f.write(f"OT shape: {v.shape}\n")
        f.write(f"min: {float(v.min()):.10f}\n")
        f.write(f"max: {float(v.max()):.10f}\n")
        f.write(f"mean: {float(v.mean()):.10f}\n")
        f.write(f"std: {float(v.std()):.10f}\n\n")

        f.write("=== TOP 50 OT SAMPLES (TRAIN SET) ===\n")
        f.write("Rank\tOT_value\tImageID\n")
        for rank, idx in enumerate(top_idx, start=1):
            f.write(f"{rank}\t{float(v[idx]):.10f}\t{ot_result.train_filenames[idx]}\n")

        f.write("\n=== BOTTOM 50 OT SAMPLES (TRAIN SET) ===\n")
        f.write("Rank\tOT_value\tImageID\n")
        for rank, idx in enumerate(bottom_idx, start=1):
            f.write(f"{rank}\t{float(v[idx]):.10f}\t{ot_result.train_filenames[idx]}\n")

        f.write("\n=== OT SUBSET EXPERIMENTS (TRAIN ON TOP OT%) ===\n")
        f.write(
            "Kept%\tN_train\tTrainLoss\tTrainAcc\tValLoss\tValAcc\tValAUC\tValBestThr\t"
            "TestLoss\tTestAcc\tTestAUC\tBestEpoch\tBestValAUC\n"
        )
        for r in results:
            kept = int(round(r["frac"] * 100))
            f.write(
                f"{kept}\t{r['n_train']}\t"
                f"{float(r['train_loss']):.6f}\t{float(r['train_acc']):.6f}\t"
                f"{float(r['val_loss']):.6f}\t{float(r['val_acc']):.6f}\t"
                f"{float(r.get('val_auc', np.nan)):.6f}\t{float(r.get('val_best_thr', np.nan)):.6f}\t"
                f"{float(r['test_loss']):.6f}\t{float(r['test_acc']):.6f}\t"
                f"{float(r['test_auc']):.6f}\t{int(r['best_epoch'])}\t"
                f"{float(r.get('best_val_auc', np.nan)):.6f}\n"
            )

        f.write("\n=== OUTPUT FILES ===\n")
        f.write(f"Model (full best): {train_result.best_path}\n")
        f.write(f"OT values (npy): {os.path.join(args.out_dir, 'ot_values.npy')}\n")
        f.write(f"OT values + IDs (csv): {os.path.join(args.out_dir, 'ot_values_with_ids.csv')}\n")
        f.write(f"Subset results (csv): {os.path.join(args.out_dir, 'subset_results.csv')}\n")
        f.write(f"Report: {report_path}\n")

    print(f"Saved: {config.rel(report_path)}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_retraining_evaluation(
    train_result: TrainResult,
    ot_result: OTResult,
    data: DataResult,
    args,
    device,
) -> RetrainingResult:
    """Section 4 orchestrator: retrain on OT-ranked subsets and evaluate."""
    print("\n" + "=" * 60)
    print("SECTION 4: OT SUBSET RETRAINING EXPERIMENTS")
    print("=" * 60)

    subset_fracs = []
    for s in args.subset_fracs.split(","):
        s = s.strip()
        if s:
            subset_fracs.append(float(s))

    order_desc = ot_result.order_desc
    train_transform, eval_transform = get_transforms()

    # Include 100% baseline first
    results = [
        {
            "frac": 1.00,
            "n_train": len(data.train_set),
            "train_loss": train_result.train_loss,
            "train_acc": train_result.train_acc,
            "val_loss": train_result.val_loss,
            "val_acc": train_result.val_acc,
            "val_auc": train_result.val_auc,
            "val_best_thr": train_result.val_best_thr,
            "test_loss": train_result.test_loss,
            "test_acc": train_result.test_acc,
            "test_auc": train_result.test_auc,
            "best_epoch": train_result.best_epoch,
            "best_val_auc": train_result.best_val_auc,
            "best_path": train_result.best_path,
        }
    ]

    for frac in subset_fracs:
        frac = float(frac)
        if frac <= 0.0 or frac > 1.0:
            continue

        n_keep = max(1, int(round(frac * len(data.train_set))))
        keep_indices = order_desc[:n_keep].tolist()

        train_subset = Subset(data.train_set, keep_indices)
        train_subset_t = TransformWrapper(train_subset, train_transform)
        train_subset_eval_t = TransformWrapper(train_subset, eval_transform)

        sub_train_loader = DataLoader(
            train_subset_t, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
        )
        sub_train_eval_loader = DataLoader(
            train_subset_eval_t,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        run_name = f"subset_top_{int(frac * 100)}pct"
        sub_model = build_model(args, device)

        sub_model, sub_hist, sub_best_path, sub_best_epoch, sub_best_val_auc = (
            train_with_freeze_unfreeze_and_early_stop(
                sub_model,
                sub_train_loader,
                data.val_loader,
                device,
                args,
                run_name=run_name,
            )
        )

        sub_train_loss, sub_train_acc, _, _ = evaluate_binary(sub_model, sub_train_eval_loader, device, threshold=0.5)
        sub_val_loss, sub_val_acc, y_vt, y_vp = evaluate_binary(sub_model, data.val_loader, device, threshold=0.5)
        _, _, sub_val_auc, sub_val_best_thr = compute_roc_auc_and_best_threshold(y_vt, y_vp)

        sub_test_loss, sub_test_acc, y_t, y_p = evaluate_binary(sub_model, data.test_loader, device, threshold=0.5)
        _, _, sub_test_auc, _ = compute_roc_auc_and_best_threshold(y_t, y_p)

        results.append(
            {
                "frac": frac,
                "n_train": n_keep,
                "train_loss": sub_train_loss,
                "train_acc": sub_train_acc,
                "val_loss": sub_val_loss,
                "val_acc": sub_val_acc,
                "val_auc": sub_val_auc,
                "val_best_thr": sub_val_best_thr,
                "test_loss": sub_test_loss,
                "test_acc": sub_test_acc,
                "test_auc": sub_test_auc,
                "best_epoch": sub_best_epoch,
                "best_val_auc": sub_best_val_auc,
                "best_path": sub_best_path,
            }
        )

    # Sort by frac descending
    results = sorted(results, key=lambda d: d["frac"], reverse=True)

    # Plots
    plot_subset_results(results, args.out_dir)

    # Save CSV
    subset_csv_path = os.path.join(args.out_dir, "subset_results.csv")
    pd.DataFrame(results).to_csv(subset_csv_path, index=False)
    print(f"Saved: {config.rel(subset_csv_path)}")

    retrain_result = RetrainingResult(results=results)

    # Final report
    write_final_report(train_result, ot_result, retrain_result, data, args)

    return retrain_result
