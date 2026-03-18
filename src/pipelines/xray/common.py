"""
Shared utilities for X-ray DenseNet121 data valuation pipelines.

Contains: dataset classes, model building, training, evaluation,
plotting, embedding extraction, subset experiments, and report writing.
"""

import copy
import os
import random
from collections import OrderedDict

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import models, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
ALL_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]
NO_FINDING_LABEL = "No Finding"


# ──────────────────────────────────────────────────────────────────────
# Dataset classes
# ──────────────────────────────────────────────────────────────────────
class ChestXray14Binary(Dataset):
    """
    Binary labels:
      y = 0 (healthy)    iff Finding Labels is exactly "No Finding"
      y = 1 (unhealthy)  otherwise (anything else, including unknown labels)
    """

    def __init__(self, root_dir, split_list_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        csv_path = os.path.join(root_dir, "Data_Entry_2017.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing CSV: {csv_path}")

        df = pd.read_csv(csv_path)

        self.label_map = {}
        for _, row in df.iterrows():
            filename = row["Image Index"]
            findings_raw = str(row["Finding Labels"])
            findings = [x.strip() for x in findings_raw.split("|") if x.strip()]

            if len(findings) == 1 and findings[0] == NO_FINDING_LABEL:
                y = 0.0
            else:
                y = 1.0
            self.label_map[filename] = y

        if not os.path.exists(split_list_path):
            raise FileNotFoundError(f"Missing split list: {split_list_path}")

        with open(split_list_path) as f:
            self.filenames = [line.strip() for line in f if line.strip()]

        self.paths = []
        for fn in self.filenames:
            p = self._find(fn)
            if p is None:
                raise FileNotFoundError(fn)
            self.paths.append(p)

    def _find(self, fn):
        for i in range(1, 13):
            p = os.path.join(self.root_dir, f"images_{i:03d}", "images", fn)
            if os.path.exists(p):
                return p
        return None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        y = torch.tensor([self.label_map[self.filenames[idx]]], dtype=torch.float32)
        return img, y


class TransformWrapper(Dataset):
    """
    Wrap a Subset so train/val/test can use DIFFERENT transforms.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    @property
    def indices(self):
        return self.subset.indices

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, y = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, y


# ──────────────────────────────────────────────────────────────────────
# Model utilities
# ──────────────────────────────────────────────────────────────────────
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


def build_model(args, device):
    m = models.densenet121(weights=None)
    m = load_pretrained_densenet121_from_local(m, args.pretrained_path)

    in_features = m.classifier.in_features
    m.classifier = nn.Sequential(nn.Dropout(args.dropout), nn.Linear(in_features, 1))
    return m.to(device)


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────
def compute_roc_auc_and_best_threshold(y_true, y_score):
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
        tp, fp, prev_score = 0, 0, None

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
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    correct = 0
    total = 0
    all_true = []
    all_prob = []

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


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def train_with_freeze_unfreeze_and_early_stop(model, train_loader, val_loader, device, args, run_name):
    criterion = nn.BCEWithLogitsLoss()

    if args.freeze_backbone:
        set_requires_grad(model.features, False)
        set_requires_grad(model.classifier, True)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.head_lr, weight_decay=args.weight_decay_head
    )

    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

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
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

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


def plot_roc(fpr, tpr, auc_value, out_path):
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


def plot_hist(values, out_path, method_name, bins=50):
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title(f"Histogram of {method_name} Values (Train Samples)")
    plt.xlabel(f"{method_name} value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_acc_vs_size(xs_pct, ys_top, ys_bottom, title, method_name, out_path):
    plt.figure()
    plt.plot(xs_pct, ys_top, marker="o", label=f"Top {method_name}")
    plt.plot(xs_pct, ys_bottom, marker="s", label=f"Bottom {method_name}")
    plt.title(title)
    plt.xlabel("Training data kept (%)")
    plt.ylabel("Accuracy")
    plt.xticks(xs_pct)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# Embedding extraction
# ──────────────────────────────────────────────────────────────────────
def extract_embeddings(model, train_eval_dataset, val_dataset, args, device):
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

    @torch.no_grad()
    def get_embeddings(loader):
        feats, ys = [], []
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            feats.append(forward_features(x))
            ys.append(y)
        return torch.cat(feats, dim=0), torch.cat(ys, dim=0)

    train_loader = DataLoader(
        train_eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    train_feats, train_y = get_embeddings(train_loader)
    val_feats, val_y = get_embeddings(val_loader)

    return train_feats, train_y, val_feats, val_y


# ──────────────────────────────────────────────────────────────────────
# Subset experiment helper
# ──────────────────────────────────────────────────────────────────────
def _train_and_eval_subset(
    keep_indices, train_set, val_loader, test_loader, args, device, train_transform, eval_transform, run_name, frac
):
    train_subset = Subset(train_set, keep_indices)

    train_subset_t = TransformWrapper(train_subset, train_transform)
    train_subset_eval_t = TransformWrapper(train_subset, eval_transform)

    sub_train_loader = DataLoader(
        train_subset_t, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    sub_train_eval_loader = DataLoader(
        train_subset_eval_t, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    sub_model = build_model(args, device)
    sub_model, sub_hist, sub_best_path, sub_best_epoch, sub_best_val_auc = train_with_freeze_unfreeze_and_early_stop(
        sub_model, sub_train_loader, val_loader, device, args, run_name=run_name
    )

    sub_train_loss, sub_train_acc, _, _ = evaluate_binary(sub_model, sub_train_eval_loader, device)
    sub_val_loss, sub_val_acc, y_vt, y_vp = evaluate_binary(sub_model, val_loader, device)
    _, _, sub_val_auc, sub_val_best_thr = compute_roc_auc_and_best_threshold(y_vt, y_vp)

    sub_test_loss, sub_test_acc, y_t, y_p = evaluate_binary(sub_model, test_loader, device)
    _, _, sub_test_auc, _ = compute_roc_auc_and_best_threshold(y_t, y_p)

    return {
        "frac": frac,
        "n_train": len(keep_indices),
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
    }


# ──────────────────────────────────────────────────────────────────────
# Hyperparameter optimisation
# ──────────────────────────────────────────────────────────────────────
HPO_SEARCH_SPACE = {
    "head_lr": [3e-4, 5e-4, 1e-3, 2e-3, 3e-3],
    "finetune_lr": [3e-5, 5e-5, 1e-4, 2e-4],
    "dropout": [0.2, 0.3, 0.4, 0.5],
    "weight_decay_finetune": [1e-4, 3e-4, 5e-4, 1e-3],
    "unfreeze_epoch": [3, 4, 5, 6, 8],
}
HPO_N_TRIALS = 10
HPO_EPOCHS = 15


def optimize_hyperparameters(train_loader, val_loader, args, device):
    """
    Random search over training hyperparameters.
    Trains each configuration for a reduced number of epochs and picks
    the one with the highest validation AUC.
    Returns (best_config dict, all_trials list[dict]).
    """
    hpo_dir = os.path.join(args.out_dir, "hpo")
    os.makedirs(hpo_dir, exist_ok=True)

    trials = []
    best_auc = -1.0
    best_config = {}

    print(f"\n{'=' * 60}")
    print(f"Hyperparameter Optimisation ({HPO_N_TRIALS} trials, {HPO_EPOCHS} epochs each)")
    print(f"{'=' * 60}")

    for trial_idx in range(HPO_N_TRIALS):
        config = {k: random.choice(v) for k, v in HPO_SEARCH_SPACE.items()}

        trial_args = copy.copy(args)
        trial_args.head_lr = config["head_lr"]
        trial_args.finetune_lr = config["finetune_lr"]
        trial_args.dropout = config["dropout"]
        trial_args.weight_decay_finetune = config["weight_decay_finetune"]
        trial_args.unfreeze_epoch = config["unfreeze_epoch"]
        trial_args.epochs = HPO_EPOCHS
        trial_args.out_dir = hpo_dir

        model = build_model(trial_args, device)
        _, _, _, _, val_auc = train_with_freeze_unfreeze_and_early_stop(
            model, train_loader, val_loader, device, trial_args, run_name=f"hpo_trial_{trial_idx}"
        )

        config["val_auc"] = val_auc
        trials.append(config)

        marker = " *best*" if val_auc > best_auc else ""
        print(
            f"  Trial {trial_idx + 1}/{HPO_N_TRIALS} | val_auc={val_auc:.4f}{marker} | "
            f"head_lr={config['head_lr']:.1e} finetune_lr={config['finetune_lr']:.1e} "
            f"dropout={config['dropout']} wd_ft={config['weight_decay_finetune']:.1e} "
            f"unfreeze_ep={config['unfreeze_epoch']}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_config = {k: v for k, v in config.items() if k != "val_auc"}

    # Save trials CSV
    pd.DataFrame(trials).to_csv(os.path.join(args.out_dir, "hpo_trials.csv"), index=False)

    # Clean up HPO checkpoint files
    import shutil

    shutil.rmtree(hpo_dir, ignore_errors=True)

    print(f"\nBest config (val_auc={best_auc:.4f}): {best_config}")
    print(f"{'=' * 60}\n")

    return best_config, trials


def _apply_hpo_config(args, config):
    """Apply optimised hyperparameters to args in-place."""
    args.head_lr = config["head_lr"]
    args.finetune_lr = config["finetune_lr"]
    args.dropout = config["dropout"]
    args.weight_decay_finetune = config["weight_decay_finetune"]
    args.unfreeze_epoch = config["unfreeze_epoch"]


# ──────────────────────────────────────────────────────────────────────
# Pipeline defaults
# ──────────────────────────────────────────────────────────────────────
PIPELINE_DEFAULTS = {
    # Training
    "epochs": 50,
    "batch_size": 32,
    "num_workers": 8,
    "early_stop_patience": 5,
    "pretrained_path": None,
    "torch_home": None,
    "seed": 42,
    # Data split
    "val_frac": 0.1,
    "test_frac": 0.1,
    "subset_fracs": "0.95,0.90,0.85,0.80,0.75",
    # Model / backbone
    "dropout": 0.4,
    "freeze_backbone": True,
    "unfreeze_last_block": True,
    "unfreeze_epoch": 6,
    "head_lr": 1e-3,
    "finetune_lr": 1e-4,
    "weight_decay_head": 0.0,
    "weight_decay_finetune": 5e-4,
    "use_scheduler": True,
    # OT
    "ot_reg": 0.01,
    # Shapley
    "shapley_k": 10,
    "shapley_mstar": 5000,
    "shapley_batch_val": 32,
    "k_candidates": "1,3,5,10,20,50",
}


def _apply_defaults(args):
    """Set any missing attributes on args to their pipeline defaults."""
    for k, v in PIPELINE_DEFAULTS.items():
        if not hasattr(args, k):
            setattr(args, k, v)


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────
def run_valuation_pipeline(args, compute_values_fn, method_name, method_params_report):
    """
    Full data-valuation pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain at minimum: data_root, out_dir.
        All other training / method parameters use built-in defaults
        if not already set on args.
    compute_values_fn : callable
        (train_feats, train_y, val_feats, val_y, args) -> np.ndarray [n_train]
    method_name : str
        e.g. "OT", "Shapley"
    method_params_report : str or object with __str__
        Method-specific hyperparameters for the report.
    """
    _apply_defaults(args)
    os.makedirs(args.out_dir, exist_ok=True)
    method_lower = method_name.lower()

    if args.torch_home is not None:
        os.environ["TORCH_HOME"] = args.torch_home
        os.makedirs(args.torch_home, exist_ok=True)
        print("TORCH_HOME set to:", args.torch_home)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ── Transforms ──
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # ── Data loading ──
    split_list = os.path.join(args.data_root, "train_val_list.txt")
    dataset = ChestXray14Binary(args.data_root, split_list, transform=None)

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
    train_eval_loader = DataLoader(
        train_set_eval_t, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # ── 0) Hyperparameter optimisation ──
    hpo_best_config, hpo_trials = optimize_hyperparameters(train_loader, val_loader, args, device)
    _apply_hpo_config(args, hpo_best_config)

    # Reset seeds so full training is reproducible after HPO
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── 1) Full training (100%) with optimised hyperparameters ──
    full_model = build_model(args, device)
    full_model, full_hist, full_best_path, full_best_epoch, full_best_val_auc = (
        train_with_freeze_unfreeze_and_early_stop(
            full_model, train_loader, val_loader, device, args, run_name="full_100pct"
        )
    )

    full_train_loss, full_train_acc, _, _ = evaluate_binary(full_model, train_eval_loader, device)
    full_val_loss, full_val_acc, y_val_true, y_val_prob = evaluate_binary(full_model, val_loader, device)
    _, _, full_val_auc, full_val_best_thr = compute_roc_auc_and_best_threshold(y_val_true, y_val_prob)

    full_test_loss, full_test_acc, y_test_true, y_test_prob = evaluate_binary(full_model, test_loader, device)
    full_fpr, full_tpr, full_test_auc, _ = compute_roc_auc_and_best_threshold(y_test_true, y_test_prob)

    # Epoch-curve plots
    epochs_axis = np.arange(1, len(full_hist["train_loss"]) + 1)
    loss_plot_path = os.path.join(args.out_dir, "train_val_loss.png")
    acc_plot_path = os.path.join(args.out_dir, "train_val_accuracy.png")
    plot_curve(
        epochs_axis,
        full_hist["train_loss"],
        full_hist["val_loss"],
        "Train Loss",
        "Val Loss",
        "Train/Val Loss vs Epoch",
        "Epoch",
        "Loss",
        loss_plot_path,
    )
    plot_curve(
        epochs_axis,
        full_hist["train_acc"],
        full_hist["val_acc@0.5"],
        "Train Acc",
        "Val Acc@0.5",
        "Train/Val Accuracy vs Epoch",
        "Epoch",
        "Accuracy",
        acc_plot_path,
    )

    roc_plot_path = os.path.join(args.out_dir, "roc_curve_test.png")
    plot_roc(full_fpr, full_tpr, full_test_auc, roc_plot_path)

    # ── 2) Compute data values ──
    print(f"\nExtracting embeddings for {method_name} computation...")
    train_feats, train_y, val_feats, val_y = extract_embeddings(full_model, train_set_eval_t, val_set_t, args, device)

    print(f"Computing {method_name} values...")
    v = compute_values_fn(train_feats, train_y, val_feats, val_y, args)
    v = np.asarray(v, dtype=np.float64)

    val_npy_path = os.path.join(args.out_dir, f"{method_lower}_values.npy")
    np.save(val_npy_path, v)

    hist_path = os.path.join(args.out_dir, f"{method_lower}_histogram.png")
    plot_hist(v, hist_path, method_name)

    train_indices = train_set.indices
    train_filenames = [dataset.filenames[i] for i in train_indices]

    df_val = pd.DataFrame({"ImageID": train_filenames, f"{method_name}_value": v})
    val_csv_path = os.path.join(args.out_dir, f"{method_lower}_values_with_ids.csv")
    df_val.to_csv(val_csv_path, index=False)

    k50 = 50
    order_asc = np.argsort(v)
    bottom_idx = order_asc[:k50]
    top_idx = order_asc[-k50:][::-1]

    # ── 3) Subset experiments (top-k AND bottom-k) ──
    subset_fracs = [float(s.strip()) for s in args.subset_fracs.split(",") if s.strip()]
    order_desc = np.argsort(v)[::-1]
    order_asc_full = np.argsort(v)

    baseline = {
        "direction": "baseline",
        "frac": 1.00,
        "n_train": len(train_set),
        "train_loss": full_train_loss,
        "train_acc": full_train_acc,
        "val_loss": full_val_loss,
        "val_acc": full_val_acc,
        "val_auc": full_val_auc,
        "val_best_thr": full_val_best_thr,
        "test_loss": full_test_loss,
        "test_acc": full_test_acc,
        "test_auc": full_test_auc,
        "best_epoch": full_best_epoch,
        "best_val_auc": full_best_val_auc,
    }

    top_results = [baseline.copy()]
    top_results[0]["direction"] = "top"
    bottom_results = [baseline.copy()]
    bottom_results[0]["direction"] = "bottom"

    for frac in subset_fracs:
        if frac <= 0.0 or frac > 1.0:
            continue
        n_keep = max(1, int(round(frac * len(train_set))))

        keep_top = order_desc[:n_keep].tolist()
        top_res = _train_and_eval_subset(
            keep_top,
            train_set,
            val_loader,
            test_loader,
            args,
            device,
            train_transform,
            eval_transform,
            f"subset_top_{int(frac * 100)}pct",
            frac,
        )
        top_res["direction"] = "top"
        top_results.append(top_res)

        keep_bottom = order_asc_full[:n_keep].tolist()
        bottom_res = _train_and_eval_subset(
            keep_bottom,
            train_set,
            val_loader,
            test_loader,
            args,
            device,
            train_transform,
            eval_transform,
            f"subset_bottom_{int(frac * 100)}pct",
            frac,
        )
        bottom_res["direction"] = "bottom"
        bottom_results.append(bottom_res)

    top_results = sorted(top_results, key=lambda d: d["frac"], reverse=True)
    bottom_results = sorted(bottom_results, key=lambda d: d["frac"], reverse=True)

    xs_pct = [int(round(r["frac"] * 100)) for r in top_results]

    for metric, metric_label in [("train_acc", "Train"), ("val_acc", "Validation"), ("test_acc", "Test")]:
        ys_top = [float(r[metric]) for r in top_results]
        ys_bottom = [float(r[metric]) for r in bottom_results]
        out_path = os.path.join(args.out_dir, f"subset_{metric}_vs_size.png")
        plot_acc_vs_size(
            xs_pct, ys_top, ys_bottom, f"{metric_label} Accuracy vs Training Data Kept", method_name, out_path
        )

    all_results = top_results + [r for r in bottom_results if r["frac"] < 1.0]
    all_results = sorted(all_results, key=lambda d: (-d["frac"], d["direction"]))
    subset_csv_path = os.path.join(args.out_dir, "subset_results.csv")
    pd.DataFrame(all_results).to_csv(subset_csv_path, index=False)

    model_path = os.path.join(args.out_dir, "densenet121_binary.pt")
    torch.save(full_model.state_dict(), model_path)

    # ── 4) Final report ──
    report_path = os.path.join(args.out_dir, "final_report.txt")
    with open(report_path, "w") as f:
        f.write("=== RUN SUMMARY ===\n")
        f.write(f"Device: {device}\n")
        f.write(f"Total samples: {n_total}\n")
        f.write(f"Train/Val/Test sizes: {n_train}/{n_val}/{n_test}\n\n")

        f.write("=== HYPERPARAMETERS ===\n")
        f.write(f"epochs_requested: {args.epochs}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"early_stop_patience (val_auc no-improve): {args.early_stop_patience}\n")
        f.write(f"pretrained_path: {args.pretrained_path}\n")
        f.write(f"seed: {args.seed}\n\n")

        f.write("=== TRAINING CONFIG (after HPO) ===\n")
        f.write(f"dropout: {args.dropout}\n")
        f.write(f"freeze_backbone: {args.freeze_backbone}\n")
        f.write(f"unfreeze_last_block: {args.unfreeze_last_block}\n")
        f.write(f"unfreeze_epoch (1-indexed): {args.unfreeze_epoch}\n")
        f.write(f"head_lr: {args.head_lr}\n")
        f.write(f"finetune_lr: {args.finetune_lr}\n")
        f.write(f"weight_decay_head: {args.weight_decay_head}\n")
        f.write(f"weight_decay_finetune: {args.weight_decay_finetune}\n")
        f.write(f"scheduler: {'ReduceLROnPlateau(mode=max,val_auc)' if args.use_scheduler else 'None'}\n\n")

        f.write(f"=== HYPERPARAMETER OPTIMISATION ({len(hpo_trials)} trials, {HPO_EPOCHS} epochs each) ===\n")
        f.write("Trial\thead_lr\t\tfinetune_lr\tdropout\twd_finetune\tunfreeze_ep\tval_auc\n")
        for i, t in enumerate(hpo_trials):
            f.write(
                f"{i + 1}\t{t['head_lr']:.1e}\t{t['finetune_lr']:.1e}\t\t{t['dropout']}\t"
                f"{t['weight_decay_finetune']:.1e}\t\t{t['unfreeze_epoch']}\t\t{t['val_auc']:.6f}\n"
            )
        f.write(f"Selected: {hpo_best_config}\n\n")

        f.write(f"=== {method_name.upper()} PARAMETERS ===\n")
        f.write(f"{method_params_report}\n\n")

        f.write("=== FULL DATA (100%) BEST MODEL (by VAL AUC) ===\n")
        f.write(f"best_epoch: {full_best_epoch}\n")
        f.write(f"best_val_auc: {full_best_val_auc:.6f}\n")
        f.write(f"Train: loss={full_train_loss:.6f}, acc@0.5={full_train_acc:.6f}\n")
        f.write(
            f"Val:   loss={full_val_loss:.6f}, acc@0.5={full_val_acc:.6f}, auc={full_val_auc:.6f}, best_thr={full_val_best_thr:.6f}\n"
        )
        f.write(f"Test:  loss={full_test_loss:.6f}, acc@0.5={full_test_acc:.6f}, auc={full_test_auc:.6f}\n\n")

        f.write(f"=== {method_name.upper()} VALUES SUMMARY (train samples) ===\n")
        f.write(f"shape: {v.shape}\n")
        f.write(f"min: {float(v.min()):.10f}\n")
        f.write(f"max: {float(v.max()):.10f}\n")
        f.write(f"mean: {float(v.mean()):.10f}\n")
        f.write(f"std: {float(v.std()):.10f}\n\n")

        f.write(f"=== TOP 50 {method_name.upper()} SAMPLES (TRAIN SET) ===\n")
        f.write(f"Rank\t{method_name}_value\tImageID\n")
        for rank, idx in enumerate(top_idx, start=1):
            f.write(f"{rank}\t{float(v[idx]):.10f}\t{train_filenames[idx]}\n")

        f.write(f"\n=== BOTTOM 50 {method_name.upper()} SAMPLES (TRAIN SET) ===\n")
        f.write(f"Rank\t{method_name}_value\tImageID\n")
        for rank, idx in enumerate(bottom_idx, start=1):
            f.write(f"{rank}\t{float(v[idx]):.10f}\t{train_filenames[idx]}\n")

        f.write(f"\n=== TOP {method_name.upper()} SUBSET EXPERIMENTS ===\n")
        f.write(
            "Kept%\tN_train\tTrainLoss\tTrainAcc\tValLoss\tValAcc\tValAUC\tValBestThr\tTestLoss\tTestAcc\tTestAUC\tBestEpoch\tBestValAUC\n"
        )
        for r in top_results:
            kept = int(round(r["frac"] * 100))
            f.write(
                f"{kept}\t{r['n_train']}\t"
                f"{float(r['train_loss']):.6f}\t{float(r['train_acc']):.6f}\t"
                f"{float(r['val_loss']):.6f}\t{float(r['val_acc']):.6f}\t"
                f"{float(r.get('val_auc', np.nan)):.6f}\t{float(r.get('val_best_thr', np.nan)):.6f}\t"
                f"{float(r['test_loss']):.6f}\t{float(r['test_acc']):.6f}\t"
                f"{float(r['test_auc']):.6f}\t{int(r['best_epoch'])}\t{float(r.get('best_val_auc', np.nan)):.6f}\n"
            )

        f.write(f"\n=== BOTTOM {method_name.upper()} SUBSET EXPERIMENTS ===\n")
        f.write(
            "Kept%\tN_train\tTrainLoss\tTrainAcc\tValLoss\tValAcc\tValAUC\tValBestThr\tTestLoss\tTestAcc\tTestAUC\tBestEpoch\tBestValAUC\n"
        )
        for r in bottom_results:
            if r["frac"] >= 1.0:
                continue
            kept = int(round(r["frac"] * 100))
            f.write(
                f"{kept}\t{r['n_train']}\t"
                f"{float(r['train_loss']):.6f}\t{float(r['train_acc']):.6f}\t"
                f"{float(r['val_loss']):.6f}\t{float(r['val_acc']):.6f}\t"
                f"{float(r.get('val_auc', np.nan)):.6f}\t{float(r.get('val_best_thr', np.nan)):.6f}\t"
                f"{float(r['test_loss']):.6f}\t{float(r['test_acc']):.6f}\t"
                f"{float(r['test_auc']):.6f}\t{int(r['best_epoch'])}\t{float(r.get('best_val_auc', np.nan)):.6f}\n"
            )

        f.write("\n=== OUTPUT FILES ===\n")
        f.write(f"HPO trials (csv): {os.path.join(args.out_dir, 'hpo_trials.csv')}\n")
        f.write(f"Model (full best): {full_best_path}\n")
        f.write(f"Model (full final state): {model_path}\n")
        f.write(f"{method_name} values (npy): {val_npy_path}\n")
        f.write(f"{method_name} values + IDs (csv): {val_csv_path}\n")
        f.write(f"{method_name} histogram: {hist_path}\n")
        f.write(f"Subset results (csv): {subset_csv_path}\n")
        f.write(f"Loss plot (epoch, full): {loss_plot_path}\n")
        f.write(f"Accuracy plot (epoch, full): {acc_plot_path}\n")
        f.write(f"ROC plot (test, full): {roc_plot_path}\n")
        f.write(f"Report: {report_path}\n")

    print("\n=== DONE ===")
    print("Saved outputs to:", args.out_dir)
    print("Report:", report_path)
