"""Sections 2–3: Model building, training, evaluation, and embedding extraction."""

import os
from collections import OrderedDict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from pipeline import config
from pipeline.data import DataResult


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class TrainResult:
    """Outputs of the training stage."""

    model: nn.Module
    history: dict
    best_path: str
    best_epoch: int
    best_val_auc: float
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_auc: float
    val_best_thr: float
    test_loss: float
    test_acc: float
    test_auc: float
    test_fpr: np.ndarray
    test_tpr: np.ndarray


# ---------------------------------------------------------------------------
# Pretrained weight loading
# ---------------------------------------------------------------------------
def _fix_densenet_state_dict_keys(state_dict):
    fixed = OrderedDict()
    for k, v in state_dict.items():
        nk = k
        nk = nk.replace("norm.1.", "norm1.")
        nk = nk.replace("norm.2.", "norm2.")
        nk = nk.replace("conv.1.", "conv1.")
        nk = nk.replace("conv.2.", "conv2.")
        fixed[nk] = v
    return fixed


def load_pretrained_densenet121_from_local(model, local_weight_path):
    if local_weight_path is None:
        print("No pretrained_path provided -> training from scratch (weights=None).")
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
    if len(missing_nc) > 0 or len(unexpected_nc) > 0:
        print("WARNING: Pretrained weight load had mismatches.")
        if len(missing_nc) > 0:
            print("  Missing (non-classifier) example:", missing_nc[:10])
        if len(unexpected_nc) > 0:
            print("  Unexpected (non-classifier) example:", unexpected_nc[:10])

    print(f"Loaded pretrained weights from: {local_weight_path}")
    return model


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_model(args, device):
    """Build a DenseNet121 with a dropout + linear head."""
    m = models.densenet121(weights=None)
    m = load_pretrained_densenet121_from_local(m, args.pretrained_path)

    in_features = m.classifier.in_features
    m.classifier = nn.Sequential(
        nn.Dropout(args.dropout),
        nn.Linear(in_features, 1),
    )
    return m.to(device)


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def compute_roc_auc_and_best_threshold(y_true, y_score):
    """Return (fpr, tpr, auc_value, best_threshold) where best_threshold maximises Youden J."""
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)

    try:
        from sklearn.metrics import auc, roc_curve

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc_value = float(auc(fpr, tpr))
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        best_thr = float(thresholds[best_idx])
        return fpr, tpr, auc_value, best_thr
    except Exception:
        order = np.argsort(-y_score)
        y_true_sorted = y_true[order]
        y_score_sorted = y_score[order]

        P = max(1, int((y_true_sorted == 1).sum()))
        N = max(1, int((y_true_sorted == 0).sum()))

        tpr_list, fpr_list, thr_list = [], [], []
        tp, fp = 0, 0
        prev_score = None

        for i in range(len(y_true_sorted)):
            s = y_score_sorted[i]
            if prev_score is None or s != prev_score:
                tpr_list.append(tp / P)
                fpr_list.append(fp / N)
                thr_list.append(s)
                prev_score = s
            if y_true_sorted[i] == 1:
                tp += 1
            else:
                fp += 1

        tpr_list.append(tp / P)
        fpr_list.append(fp / N)
        thr_list.append(y_score_sorted[-1] - 1e-12)

        fpr = np.array(fpr_list, dtype=np.float64)
        tpr = np.array(tpr_list, dtype=np.float64)
        thresholds = np.array(thr_list, dtype=np.float64)

        auc_value = float(np.trapz(tpr, fpr))
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        best_thr = float(thresholds[best_idx])
        return fpr, tpr, auc_value, best_thr


@torch.no_grad()
def evaluate_binary(model, loader, device, threshold=0.5):
    """Return (avg_loss, acc_at_threshold, y_true, y_prob)."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    correct = 0
    total = 0
    all_true, all_prob = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        correct += (preds == y).sum().item()
        total += y.numel()

        all_true.append(y.detach().cpu())
        all_prob.append(probs.detach().cpu())

    avg_loss = total_loss / max(1, len(loader))
    acc = correct / max(1, total)

    y_true = torch.cat(all_true, dim=0).numpy().reshape(-1)
    y_prob = torch.cat(all_prob, dim=0).numpy().reshape(-1)
    return avg_loss, acc, y_true, y_prob


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_embeddings(model, loader, device):
    """Extract pooled DenseNet121 feature embeddings."""
    feature_model = models.densenet121(weights=None)
    feature_model.features = model.features
    feature_model.classifier = nn.Identity()
    feature_model = feature_model.to(device)
    feature_model.eval()

    def forward_features(x):
        f = feature_model.features(x)
        f = nn.functional.relu(f, inplace=False)
        f = nn.functional.adaptive_avg_pool2d(f, (1, 1)).flatten(1)
        return f

    feats, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        feats.append(forward_features(x))
        ys.append(y)
    return torch.cat(feats, dim=0), torch.cat(ys, dim=0)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_with_freeze_unfreeze_and_early_stop(model, train_loader, val_loader, device, args, run_name):
    """Train with optional backbone freeze/unfreeze and AUC-based early stopping."""
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
            verbose=True,
        )

    best_val_auc = -1.0
    best_epoch = -1
    best_path = os.path.join(args.out_dir, f"{run_name}_best.pt")
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc@0.5": [],
        "val_auc": [],
        "val_best_thr": [],
        "val_acc@best_thr": [],
        "lr": [],
    }

    unfreeze_epoch = args.unfreeze_epoch

    for epoch in range(args.epochs):
        epoch_num = epoch + 1

        if args.freeze_backbone and args.unfreeze_last_block and epoch_num == unfreeze_epoch:
            print(f"[{run_name}] Unfreezing denseblock4 + norm5 at epoch {epoch_num}")
            set_requires_grad(model.features.denseblock4, True)
            set_requires_grad(model.features.norm5, True)

            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.finetune_lr,
                weight_decay=args.weight_decay_finetune,
            )

            if args.use_scheduler:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=0.5,
                    patience=2,
                    verbose=True,
                )

        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct += (preds == y).sum().item()
                total += y.numel()

        train_loss = total_loss / max(1, len(train_loader))
        train_acc = correct / max(1, total)

        val_loss, val_acc_05, y_val_true, y_val_prob = evaluate_binary(model, val_loader, device, threshold=0.5)
        fpr, tpr, val_auc, best_thr = compute_roc_auc_and_best_threshold(y_val_true, y_val_prob)
        _, val_acc_best, _, _ = evaluate_binary(model, val_loader, device, threshold=best_thr)

        if scheduler is not None:
            scheduler.step(val_auc)

        lr_now = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc@0.5"].append(val_acc_05)
        history["val_auc"].append(val_auc)
        history["val_best_thr"].append(best_thr)
        history["val_acc@best_thr"].append(val_acc_best)
        history["lr"].append(lr_now)

        print(
            f"[{run_name}] Epoch {epoch_num}/{args.epochs} | lr={lr_now:.2e} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc@0.5={val_acc_05:.4f} | "
            f"val_auc={val_auc:.4f} best_thr={best_thr:.3f} val_acc@thr={val_acc_best:.4f}"
        )

        if val_auc > best_val_auc + 1e-6:
            best_val_auc = val_auc
            best_epoch = epoch_num
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"[{run_name}] Early stopping: val_auc did not improve for {args.early_stop_patience} epochs.")
                break

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return model, history, best_path, best_epoch, best_val_auc


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_curve(x, y1, y2, label1, label2, title, xlabel, ylabel, out_path):
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


def _plot_roc(fpr, tpr, auc_value, out_path):
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve (Test Set)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_training_curves(history, out_dir):
    """Plot train/val loss and accuracy curves."""
    epochs_axis = np.arange(1, len(history["train_loss"]) + 1)

    _plot_curve(
        epochs_axis,
        history["train_loss"],
        history["val_loss"],
        "Train Loss",
        "Val Loss",
        "Train/Val Loss vs Epoch",
        "Epoch",
        "Loss",
        os.path.join(out_dir, "train_val_loss.png"),
    )
    _plot_curve(
        epochs_axis,
        history["train_acc"],
        history["val_acc@0.5"],
        "Train Acc",
        "Val Acc@0.5",
        "Train/Val Accuracy vs Epoch",
        "Epoch",
        "Accuracy",
        os.path.join(out_dir, "train_val_accuracy.png"),
    )


def plot_roc_curve(test_fpr, test_tpr, test_auc, out_dir):
    """Plot ROC curve for the test set."""
    _plot_roc(test_fpr, test_tpr, test_auc, os.path.join(out_dir, "roc_curve_test.png"))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_training(data: DataResult, args, device) -> TrainResult:
    """Section 2–3 orchestrator: build model, train on full data, evaluate, and extract embeddings."""
    print("\n" + "=" * 60)
    print("SECTION 2: FULL TRAINING (100%)")
    print("=" * 60)

    model = build_model(args, device)
    model, history, best_path, best_epoch, best_val_auc = train_with_freeze_unfreeze_and_early_stop(
        model,
        data.train_loader,
        data.val_loader,
        device,
        args,
        run_name="full_100pct",
    )

    # Final metrics (best model)
    train_loss, train_acc, _, _ = evaluate_binary(model, data.train_eval_loader, device, threshold=0.5)
    val_loss, val_acc, y_val_true, y_val_prob = evaluate_binary(model, data.val_loader, device, threshold=0.5)
    _, _, val_auc, val_best_thr = compute_roc_auc_and_best_threshold(y_val_true, y_val_prob)

    test_loss, test_acc, y_test_true, y_test_prob = evaluate_binary(model, data.test_loader, device, threshold=0.5)
    test_fpr, test_tpr, test_auc, _ = compute_roc_auc_and_best_threshold(y_test_true, y_test_prob)

    # Plots
    plot_training_curves(history, args.out_dir)
    plot_roc_curve(test_fpr, test_tpr, test_auc, args.out_dir)

    # Save model
    model_path = os.path.join(args.out_dir, "densenet121_binary.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model: {config.rel(model_path)}")

    return TrainResult(
        model=model,
        history=history,
        best_path=best_path,
        best_epoch=best_epoch,
        best_val_auc=best_val_auc,
        train_loss=train_loss,
        train_acc=train_acc,
        val_loss=val_loss,
        val_acc=val_acc,
        val_auc=val_auc,
        val_best_thr=val_best_thr,
        test_loss=test_loss,
        test_acc=test_acc,
        test_auc=test_auc,
        test_fpr=test_fpr,
        test_tpr=test_tpr,
    )
