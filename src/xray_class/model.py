"""DenseNet121 model, training, evaluation, and embedding extraction for multi-label classification."""

import os
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")

# isort: split
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models

from xray_class import config


# ──────────────────────────────────────────────────────────────────────
# Model utilities
# ──────────────────────────────────────────────────────────────────────
def _fix_densenet_state_dict_keys(state_dict):
    fixed = OrderedDict()
    for k, v in state_dict.items():
        nk = k.replace("norm.1.", "norm1.").replace("norm.2.", "norm2.")
        nk = nk.replace("conv.1.", "conv1.").replace("conv.2.", "conv2.")
        fixed[nk] = v
    return fixed


def load_pretrained_densenet121(model, local_weight_path):
    if local_weight_path is None:
        print("No pretrained_path -> using torchvision ImageNet weights.")
        return model

    if not os.path.exists(local_weight_path):
        raise FileNotFoundError(f"Pretrained weights not found: {local_weight_path}")

    state_dict = torch.load(local_weight_path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    state_dict = _fix_densenet_state_dict_keys(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    missing_nc = [m for m in missing if not m.startswith("classifier.")]
    unexpected_nc = [u for u in unexpected if not u.startswith("classifier.")]
    if missing_nc or unexpected_nc:
        print("WARNING: Weight load mismatches detected.")
    print(f"Loaded pretrained weights from: {local_weight_path}")
    return model


def build_model(args, device):
    if args.pretrained_path is not None:
        m = models.densenet121(weights=None)
        m = load_pretrained_densenet121(m, args.pretrained_path)
    else:
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    in_features = m.classifier.in_features
    m.classifier = nn.Sequential(nn.Dropout(args.dropout), nn.Linear(in_features, config.NUM_CLASSES))
    return m.to(device)


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad = flag


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────
def compute_multilabel_auc(y_true, y_score):
    """Compute per-class AUC-ROC and mean AUC. Returns (mean_auc, per_class_aucs dict)."""
    per_class = {}
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) < 2:
            continue
        try:
            per_class[config.ALL_LABELS[c]] = roc_auc_score(y_true[:, c], y_score[:, c])
        except ValueError:
            continue
    mean_auc = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return mean_auc, per_class


@torch.no_grad()
def evaluate_multilabel(model, loader, device):
    """Evaluate multi-label model. Returns (avg_loss, mean_auc, y_true, y_prob)."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_true, all_prob = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        all_true.append(y.detach().cpu())
        all_prob.append(torch.sigmoid(logits).detach().cpu())

    avg_loss = total_loss / max(1, len(loader))
    y_true = torch.cat(all_true, dim=0).numpy()
    y_prob = torch.cat(all_prob, dim=0).numpy()
    mean_auc, _ = compute_multilabel_auc(y_true, y_prob)
    return avg_loss, mean_auc, y_true, y_prob


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def train_with_early_stop(model, train_loader, val_loader, device, args, run_name):
    """Train with optional backbone freeze/unfreeze and early stopping on val mean AUC."""
    criterion = nn.BCEWithLogitsLoss()

    if args.freeze_backbone:
        set_requires_grad(model.features, False)
        set_requires_grad(model.classifier, True)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.head_lr,
        weight_decay=args.weight_decay_head,
    )

    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_auc = -1.0
    best_epoch = -1
    best_path = os.path.join(args.out_dir, f"{run_name}_best.pt")
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "val_auc": [], "lr": []}

    for epoch in range(args.epochs):
        epoch_num = epoch + 1

        if args.freeze_backbone and args.unfreeze_last_block and epoch_num == args.unfreeze_epoch:
            print(f"[{run_name}] Unfreezing denseblock4 + norm5 at epoch {epoch_num}")
            set_requires_grad(model.features.denseblock4, True)
            set_requires_grad(model.features.norm5, True)
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.finetune_lr,
                weight_decay=args.weight_decay_finetune,
            )
            if args.use_scheduler:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))
        val_loss, val_auc, _, _ = evaluate_multilabel(model, val_loader, device)

        if scheduler is not None:
            scheduler.step(val_auc)

        lr_now = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["lr"].append(lr_now)

        print(
            f"[{run_name}] Epoch {epoch_num}/{args.epochs} | lr={lr_now:.2e} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} val_auc={val_auc:.4f}"
        )

        if val_auc > best_val_auc + 1e-6:
            best_val_auc = val_auc
            best_epoch = epoch_num
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"[{run_name}] Early stopping at epoch {epoch_num}.")
                break

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return model, history, best_path, best_epoch, best_val_auc


# ──────────────────────────────────────────────────────────────────────
# Embedding extraction
# ──────────────────────────────────────────────────────────────────────
def extract_embeddings(model, dataset, args, device):
    """Extract 1024-dim DenseNet121 feature embeddings."""
    feature_model = models.densenet121(weights=None)
    feature_model.features = model.features
    feature_model.classifier = nn.Identity()
    feature_model = feature_model.to(device)
    feature_model.eval()

    @torch.no_grad()
    def forward_features(x):
        f = feature_model.features(x)
        f = nn.functional.relu(f, inplace=False)
        f = nn.functional.adaptive_avg_pool2d(f, (1, 1)).flatten(1)
        return f

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    feats, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        feats.append(forward_features(x))
        ys.append(y)

    return torch.cat(feats, dim=0), torch.cat(ys, dim=0)


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────
def plot_curve(x, y1, y2, label1, label2, title, xlabel, ylabel, out_path):
    plt.figure()
    plt.plot(x, y1, label=label1)
    plt.plot(x, y2, label=label2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_hist(values, out_path, method_name, bins=50):
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title(f"Histogram of {method_name} Values (Train Samples)")
    plt.xlabel(f"{method_name} value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_auc_vs_size(xs_pct, ys_top, ys_bottom, title, method_name, out_path):
    plt.figure()
    plt.plot(xs_pct, ys_top, marker="o", label=f"Top {method_name}")
    plt.plot(xs_pct, ys_bottom, marker="s", label=f"Bottom {method_name}")
    plt.title(title)
    plt.xlabel("Training data kept (%)")
    plt.ylabel("Mean AUC-ROC")
    plt.xticks(xs_pct)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_retraining_curves(fracs_pct, top_aucs, bottom_aucs, random_mean, random_std, method_name, out_dir):
    """
    Two-panel retraining plot: AUC-ROC and Loss.
    Top-N, Bottom-N as solid lines; Random-N as line with shaded +/-1 std band.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Test AUC
    ax = axes[0]
    ax.plot(fracs_pct, top_aucs["test_auc"], "o-", label=f"Top-{method_name}", color="tab:green")
    ax.plot(fracs_pct, bottom_aucs["test_auc"], "s-", label=f"Bottom-{method_name}", color="tab:red")
    rm = np.array(random_mean["test_auc"])
    rs = np.array(random_std["test_auc"])
    ax.plot(fracs_pct, rm, "^--", label="Random (mean)", color="tab:blue")
    ax.fill_between(fracs_pct, rm - rs, rm + rs, alpha=0.2, color="tab:blue")
    ax.set_title("Test Mean AUC-ROC vs Data Retained")
    ax.set_xlabel("Training data kept (%)")
    ax.set_ylabel("Mean AUC-ROC")
    ax.set_ylim(0.0, 1.0)
    ax.legend()

    # Panel 2: Test Loss
    ax = axes[1]
    ax.plot(fracs_pct, top_aucs["test_loss"], "o-", label=f"Top-{method_name}", color="tab:green")
    ax.plot(fracs_pct, bottom_aucs["test_loss"], "s-", label=f"Bottom-{method_name}", color="tab:red")
    rm = np.array(random_mean["test_loss"])
    rs = np.array(random_std["test_loss"])
    ax.plot(fracs_pct, rm, "^--", label="Random (mean)", color="tab:blue")
    ax.fill_between(fracs_pct, rm - rs, rm + rs, alpha=0.2, color="tab:blue")
    ax.set_title("Test Loss vs Data Retained")
    ax.set_xlabel("Training data kept (%)")
    ax.set_ylabel("BCEWithLogitsLoss")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "retraining_curves.png"), dpi=200)
    plt.close(fig)

    # Comprehensive 2x3 grid: train/val/test × AUC/Loss
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    splits = ["train", "val", "test"]
    metrics = ["auc", "loss"]
    ylabels = ["Mean AUC-ROC", "BCEWithLogitsLoss"]

    for row, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        for col, split in enumerate(splits):
            ax = axes[row, col]
            key = f"{split}_{metric}"
            ax.plot(fracs_pct, top_aucs[key], "o-", label=f"Top-{method_name}", color="tab:green")
            ax.plot(fracs_pct, bottom_aucs[key], "s-", label=f"Bottom-{method_name}", color="tab:red")
            rm = np.array(random_mean[key])
            rs = np.array(random_std[key])
            ax.plot(fracs_pct, rm, "^--", label="Random (mean)", color="tab:blue")
            ax.fill_between(fracs_pct, rm - rs, rm + rs, alpha=0.2, color="tab:blue")
            ax.set_title(f"{split.capitalize()} {ylabel}")
            ax.set_xlabel("Training data kept (%)")
            ax.set_ylabel(ylabel)
            if metric == "auc":
                ax.set_ylim(0.0, 1.0)
            ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "retraining_all_metrics.png"), dpi=200)
    plt.close(fig)
