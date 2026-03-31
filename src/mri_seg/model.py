"""U-Net model, training, evaluation, and embedding extraction for binary MRI segmentation."""

import os

import matplotlib

matplotlib.use("Agg")

# isort: split
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# ──────────────────────────────────────────────────────────────────────
# U-Net architecture
# ──────────────────────────────────────────────────────────────────────
class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """U-Net for binary segmentation with extractable bottleneck features."""

    def __init__(self, in_channels=1, out_channels=1, features=None, dropout=0.1):
        super().__init__()
        if features is None:
            features = [32, 64, 128]

        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoder_blocks.append(_ConvBlock(ch, f, dropout))
            self.pools.append(nn.MaxPool2d(2))
            ch = f

        self.bottleneck = _ConvBlock(ch, features[-1] * 2, dropout)
        self.bottleneck_dim = features[-1] * 2

        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        ch = features[-1] * 2
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(ch, f, 2, stride=2))
            self.decoder_blocks.append(_ConvBlock(f * 2, f, dropout))
            ch = f

        self.head = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoder_blocks, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, dec, skip in zip(self.upconvs, self.decoder_blocks, reversed(skips)):
            x = upconv(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.head(x)  # logits

    @torch.no_grad()
    def extract_bottleneck(self, x):
        """Extract bottleneck features with global average pooling -> [B, D]."""
        for enc, pool in zip(self.encoder_blocks, self.pools):
            x = enc(x)
            x = pool(x)
        x = self.bottleneck(x)
        return x.mean(dim=[2, 3])  # GAP


# ──────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────
class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for binary segmentation."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return bce + (1.0 - dice)


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────
def dice_score(logits, targets, threshold=0.5, smooth=1.0):
    """Compute Dice coefficient from logits."""
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum()
    return ((2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)).item()


# ──────────────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────────────
def build_model(args, device):
    m = UNet(in_channels=1, out_channels=1, dropout=args.dropout)
    return m.to(device)


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_segmentation(model, loader, device):
    """Evaluate segmentation model. Returns (avg_loss, avg_dice)."""
    model.eval()
    criterion = DiceBCELoss()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    for batch in loader:
        img, mask = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        logits = model(img)
        loss = criterion(logits, mask)
        total_loss += loss.item()
        total_dice += dice_score(logits, mask)
        n_batches += 1

    n = max(1, n_batches)
    return total_loss / n, total_dice / n


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def train_with_early_stop(model, train_loader, val_loader, device, args, run_name):
    """Train U-Net with early stopping on validation Dice."""
    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_dice = -1.0
    best_epoch = -1
    best_path = os.path.join(args.out_dir, f"{run_name}_best.pt")
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "val_dice": [], "lr": []}

    for epoch in range(args.epochs):
        epoch_num = epoch + 1

        model.train()
        total_loss = 0.0
        for batch in train_loader:
            img, mask = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(img)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))
        val_loss, val_dice = evaluate_segmentation(model, val_loader, device)

        scheduler.step(val_dice)

        lr_now = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["lr"].append(lr_now)

        print(
            f"[{run_name}] Epoch {epoch_num}/{args.epochs} | lr={lr_now:.2e} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} val_dice={val_dice:.4f}"
        )

        if val_dice > best_val_dice + 1e-6:
            best_val_dice = val_dice
            best_epoch = epoch_num
            patience_counter = 0
            os.makedirs(os.path.dirname(best_path), exist_ok=True)
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"[{run_name}] Early stopping at epoch {epoch_num}.")
                break

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return model, history, best_path, best_epoch, best_val_dice


# ──────────────────────────────────────────────────────────────────────
# Embedding extraction
# ──────────────────────────────────────────────────────────────────────
def extract_embeddings(model, dataset, args, device):
    """Extract bottleneck embeddings and tumor-type labels for data valuation.

    Returns:
        features: [N, D] float32 tensor (bottleneck features after GAP)
        labels:   [N, 3] float32 tensor (NCR, ED, ET presence)
    """
    model.eval()
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    feats, ys = [], []
    for batch in loader:
        img, _mask, label = batch
        img = img.to(device, non_blocking=True)
        feats.append(model.extract_bottleneck(img).cpu())
        ys.append(label)

    return torch.cat(feats, dim=0), torch.cat(ys, dim=0)


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────
def plot_curve(x, y1, y2, label1, label2, title, xlabel, ylabel, out_path):
    plt.figure()
    plt.plot(x, y1, label=label1)
    if label2:
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


def plot_retraining_curves(fracs_pct, top_metrics, bottom_metrics, random_mean, random_std, method_name, out_dir):
    """Two-panel retraining plot: Dice and Loss."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Test Dice
    ax = axes[0]
    ax.plot(fracs_pct, top_metrics["test_dice"], "o-", label=f"Top-{method_name}", color="tab:green")
    ax.plot(fracs_pct, bottom_metrics["test_dice"], "s-", label=f"Bottom-{method_name}", color="tab:red")
    rm = np.array(random_mean["test_dice"])
    rs = np.array(random_std["test_dice"])
    ax.plot(fracs_pct, rm, "^--", label="Random (mean)", color="tab:blue")
    ax.fill_between(fracs_pct, rm - rs, rm + rs, alpha=0.2, color="tab:blue")
    ax.set_title("Test Dice vs Data Retained")
    ax.set_xlabel("Training data kept (%)")
    ax.set_ylabel("Dice Coefficient")
    ax.set_ylim(0.0, 1.0)
    ax.legend()

    # Panel 2: Test Loss
    ax = axes[1]
    ax.plot(fracs_pct, top_metrics["test_loss"], "o-", label=f"Top-{method_name}", color="tab:green")
    ax.plot(fracs_pct, bottom_metrics["test_loss"], "s-", label=f"Bottom-{method_name}", color="tab:red")
    rm = np.array(random_mean["test_loss"])
    rs = np.array(random_std["test_loss"])
    ax.plot(fracs_pct, rm, "^--", label="Random (mean)", color="tab:blue")
    ax.fill_between(fracs_pct, rm - rs, rm + rs, alpha=0.2, color="tab:blue")
    ax.set_title("Test Loss vs Data Retained")
    ax.set_xlabel("Training data kept (%)")
    ax.set_ylabel("Dice+BCE Loss")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "retraining_curves.png"), dpi=200)
    plt.close(fig)

    # Comprehensive 2x3 grid: train/val/test x Dice/Loss
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    splits = ["train", "val", "test"]
    metrics = ["dice", "loss"]
    ylabels = ["Dice Coefficient", "Dice+BCE Loss"]

    for row, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        for col, split in enumerate(splits):
            ax = axes[row, col]
            key = f"{split}_{metric}"
            ax.plot(fracs_pct, top_metrics[key], "o-", label=f"Top-{method_name}", color="tab:green")
            ax.plot(fracs_pct, bottom_metrics[key], "s-", label=f"Bottom-{method_name}", color="tab:red")
            rm = np.array(random_mean[key])
            rs = np.array(random_std[key])
            ax.plot(fracs_pct, rm, "^--", label="Random (mean)", color="tab:blue")
            ax.fill_between(fracs_pct, rm - rs, rm + rs, alpha=0.2, color="tab:blue")
            ax.set_title(f"{split.capitalize()} {ylabel}")
            ax.set_xlabel("Training data kept (%)")
            ax.set_ylabel(ylabel)
            if metric == "dice":
                ax.set_ylim(0.0, 1.0)
            ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "retraining_all_metrics.png"), dpi=200)
    plt.close(fig)
