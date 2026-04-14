import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, models

import ot

# matplotlib for plots (HPC-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ALL_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]
NO_FINDING_LABEL = "No Finding"


class ChestXray14Binary(Dataset):
    """
    Binary labels:
      y = 0 (healthy)    iff Finding Labels is exactly "No Finding"
      y = 1 (unhealthy)  otherwise (anything else, including unknown labels)

    IMPORTANT:
    - Only exact "No Finding" => healthy.
    - Do NOT treat "none of the 14 labels" as healthy.
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
        y = torch.tensor([self.label_map[self.filenames[idx]]], dtype=torch.float32)  # [1]
        return img, y


class TransformWrapper(Dataset):
    """
    Wrap a Subset(random_split output) so train/val/test can use DIFFERENT transforms.
    random_split returns Subset objects that share the same underlying dataset instance,
    so you cannot safely assign dataset.transform differently per split.
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    @property
    def indices(self):
        # convenience passthrough (used later in OT mapping)
        return self.subset.indices

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, y = self.subset[idx]  # img is PIL if base dataset.transform is None
        if self.transform is not None:
            img = self.transform(img)
        return img, y


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


def compute_roc_auc_and_best_threshold(y_true, y_score):
    """
    Returns (fpr, tpr, auc_value, best_threshold) where best_threshold maximizes Youden J = tpr - fpr.
    Uses sklearn if available, else falls back to numpy implementations.
    """
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)

    try:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc_value = float(auc(fpr, tpr))
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        best_thr = float(thresholds[best_idx])
        return fpr, tpr, auc_value, best_thr
    except Exception:
        # fallback ROC curve calculation
        order = np.argsort(-y_score)
        y_true_sorted = y_true[order]
        y_score_sorted = y_score[order]

        P = max(1, int((y_true_sorted == 1).sum()))
        N = max(1, int((y_true_sorted == 0).sum()))

        tpr_list = []
        fpr_list = []
        thr_list = []

        tp = 0
        fp = 0
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
    """
    Returns:
      avg_loss, acc_at_threshold, y_true, y_prob
    (AUC + best threshold are computed separately via compute_roc_auc_and_best_threshold)
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    all_true = []
    all_prob = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)  # [B,1]

        logits = model(x)  # [B,1]
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


def ot_binary_row_normalized(train_feats, train_y, val_feats, val_y, reg=0.01, eps=1e-12):
    train_feats = nn.functional.normalize(train_feats, dim=1)
    val_feats = nn.functional.normalize(val_feats, dim=1)

    C = 1.0 - (train_feats @ val_feats.T)  # [n_train, n_val]
    R = (train_y.round() == val_y.T.round()).float()  # [n_train, n_val]

    n_train = train_feats.size(0)
    n_val = val_feats.size(0)
    a = np.ones(n_train, dtype=np.float64) / n_train
    b = np.ones(n_val, dtype=np.float64) / n_val

    C_np = C.detach().cpu().numpy().astype(np.float64)
    P = ot.sinkhorn(a, b, C_np, reg)
    P = torch.tensor(P, dtype=torch.float32)  # CPU

    row_sums = P.sum(dim=1, keepdim=True).clamp_min(eps)
    P_row = P / row_sums
    scores = (P_row * R.detach().cpu()).sum(dim=1)  # [n_train]
    return scores


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


def plot_hist(values, out_path, bins=50):
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title("Histogram of OT Values (Train Samples)")
    plt.xlabel("OT value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_acc_vs_size(xs_pct, ys, title, out_path):
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


def build_model(args, device):
    """
    Uses your existing local-pretrained loader if provided.
    Otherwise weights=None (safe for offline/HPC).
    Adds dropout head (improved code logic).
    """
    m = models.densenet121(weights=None)
    m = load_pretrained_densenet121_from_local(m, args.pretrained_path)

    in_features = m.classifier.in_features
    m.classifier = nn.Sequential(
        nn.Dropout(args.dropout),
        nn.Linear(in_features, 1)
    )
    return m.to(device)


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def train_with_freeze_unfreeze_and_early_stop(model, train_loader, val_loader, device, args, run_name):
    """
    Improvements integrated:
    - Freeze backbone initially; train classifier head with head_lr.
    - After 'unfreeze_epoch' (1-indexed epoch number), unfreeze denseblock4+norm5 and continue
      with finetune_lr + weight_decay.
    - Early stop on VAL AUC (patience), save best checkpoint by VAL AUC.
    Also still tracks val_acc at 0.5 and at best-threshold (reported).
    """
    criterion = nn.BCEWithLogitsLoss()

    # Phase 1: freeze backbone
    if args.freeze_backbone:
        set_requires_grad(model.features, False)
        set_requires_grad(model.classifier, True)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.head_lr,
        weight_decay=args.weight_decay_head
    )

    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, verbose=True
        )

    best_val_auc = -1.0
    best_epoch = -1
    best_path = os.path.join(args.out_dir, f"{run_name}_best.pt")

    patience_counter = 0

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc@0.5": [], "val_auc": [], "val_best_thr": [],
        "val_acc@best_thr": [],
        "lr": []
    }

    # (1-indexed like your printouts)
    unfreeze_epoch = args.unfreeze_epoch

    for epoch in range(args.epochs):
        epoch_num = epoch + 1

        # Phase 2: unfreeze last block at requested epoch
        if args.freeze_backbone and args.unfreeze_last_block and epoch_num == unfreeze_epoch:
            print(f"[{run_name}] Unfreezing denseblock4 + norm5 at epoch {epoch_num}")
            set_requires_grad(model.features.denseblock4, True)
            set_requires_grad(model.features.norm5, True)

            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.finetune_lr,
                weight_decay=args.weight_decay_finetune
            )

            if args.use_scheduler:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5, patience=2, verbose=True
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

        # Val metrics
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

        # Best model by val AUC
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

    # load best
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return model, history, best_path, best_epoch, best_val_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)

    # Early stopping (AUC-based, like your improved code)
    parser.add_argument("--early_stop_patience", type=int, default=5)

    # Offline pretrained weights (your original behavior)
    parser.add_argument("--pretrained_path", type=str, default=None)

    # Torch cache dir (HPC)
    parser.add_argument("--torch_home", type=str, default=None)

    # Split fractions
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)

    # OT
    parser.add_argument("--ot_reg", type=float, default=0.01)

    # OT-subset experiment fractions (train only)
    parser.add_argument("--subset_fracs", type=str, default="0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50,0.45,0.40,0.35,0.30,0.25,0.20,0.15,0.10,0.05,0.02,0.01")

    parser.add_argument("--seed", type=int, default=42)

    # =========================
    # NEW (from improved code)
    # =========================
    parser.add_argument("--dropout", type=float, default=0.4)

    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze DenseNet features initially; train head only.")
    parser.add_argument("--unfreeze_last_block", action="store_true",
                        help="When freezing backbone, unfreeze denseblock4+norm5 at --unfreeze_epoch.")
    parser.add_argument("--unfreeze_epoch", type=int, default=6,
                        help="1-indexed epoch number when to unfreeze denseblock4+norm5 (default: 6, i.e., after 5 epochs).")

    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay_head", type=float, default=0.0)
    parser.add_argument("--weight_decay_finetune", type=float, default=5e-4)

    # Scheduler (now driven by val_auc)
    parser.add_argument("--use_scheduler", action="store_true")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.torch_home is not None:
        os.environ["TORCH_HOME"] = args.torch_home
        os.makedirs(args.torch_home, exist_ok=True)
        print("TORCH_HOME set to:", args.torch_home)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =========================
    # NEW: separate transforms
    # - Train: augmentation
    # - Val/Test/Embeddings: deterministic
    # =========================
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    split_list = os.path.join(args.data_root, "train_val_list.txt")

    # IMPORTANT: base dataset transform=None so wrappers can apply different transforms per split
    dataset = ChestXray14Binary(args.data_root, split_list, transform=None)

    # --- Train/Val/Test split ---
    n_total = len(dataset)
    n_test = int(args.test_frac * n_total)
    n_val = int(args.val_frac * n_total)
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise ValueError("Split fractions too large; train set would be empty.")

    gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=gen)

    # Apply different transforms safely
    train_set_t = TransformWrapper(train_set, train_transform)
    val_set_t = TransformWrapper(val_set, eval_transform)
    test_set_t = TransformWrapper(test_set, eval_transform)

    # For train-eval metrics (no augmentation), use eval_transform
    train_set_eval_t = TransformWrapper(train_set, eval_transform)

    train_loader = DataLoader(train_set_t, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set_t, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set_t, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    train_eval_loader = DataLoader(train_set_eval_t, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, pin_memory=True)

    # =========================
    # 1) FULL TRAINING (100%)
    # =========================
    full_model = build_model(args, device)
    full_model, full_hist, full_best_path, full_best_epoch, full_best_val_auc = train_with_freeze_unfreeze_and_early_stop(
        full_model, train_loader, val_loader, device, args, run_name="full_100pct"
    )

    # Final metrics (best model)
    full_train_loss, full_train_acc, _, _ = evaluate_binary(full_model, train_eval_loader, device, threshold=0.5)
    full_val_loss, full_val_acc, y_val_true, y_val_prob = evaluate_binary(full_model, val_loader, device, threshold=0.5)
    _, _, full_val_auc, full_val_best_thr = compute_roc_auc_and_best_threshold(y_val_true, y_val_prob)

    full_test_loss, full_test_acc, y_test_true, y_test_prob = evaluate_binary(full_model, test_loader, device, threshold=0.5)
    full_fpr, full_tpr, full_test_auc, _ = compute_roc_auc_and_best_threshold(y_test_true, y_test_prob)

    # Plots for epoch curves (full run)
    epochs_axis = np.arange(1, len(full_hist["train_loss"]) + 1)
    loss_plot_path = os.path.join(args.out_dir, "train_val_loss.png")
    acc_plot_path = os.path.join(args.out_dir, "train_val_accuracy.png")
    plot_curve(epochs_axis, full_hist["train_loss"], full_hist["val_loss"],
               "Train Loss", "Val Loss", "Train/Val Loss vs Epoch", "Epoch", "Loss", loss_plot_path)

    # keep your original accuracy plot, but use val_acc@0.5 series
    plot_curve(epochs_axis, full_hist["train_acc"], full_hist["val_acc@0.5"],
               "Train Acc", "Val Acc@0.5", "Train/Val Accuracy vs Epoch", "Epoch", "Accuracy", acc_plot_path)

    roc_plot_path = os.path.join(args.out_dir, "roc_curve_test.png")
    plot_roc(full_fpr, full_tpr, full_test_auc, roc_plot_path)

    # =========================
    # 2) OT COMPUTATION (on train_set vs val_set)
    # IMPORTANT: use deterministic transforms for embeddings
    # =========================
    feature_model = models.densenet121(weights=None)
    feature_model.features = full_model.features
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
    def get_embeddings_pooled(loader):
        feats, ys = [], []
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            feats.append(forward_features(x))
            ys.append(y)
        return torch.cat(feats, dim=0), torch.cat(ys, dim=0)

    # deterministic embedding loaders
    train_embed_loader = DataLoader(train_set_eval_t, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)
    val_embed_loader = DataLoader(val_set_t, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

    train_feats, train_y = get_embeddings_pooled(train_embed_loader)
    val_feats, val_y = get_embeddings_pooled(val_embed_loader)

    ot_values = ot_binary_row_normalized(train_feats, train_y, val_feats, val_y, reg=args.ot_reg)
    v = ot_values.numpy()

    ot_path = os.path.join(args.out_dir, "ot_values.npy")
    np.save(ot_path, v)

    ot_hist_path = os.path.join(args.out_dir, "ot_histogram.png")
    plot_hist(v, ot_hist_path, bins=50)

    # Map train order -> filenames
    train_indices = train_set.indices
    train_filenames = [dataset.filenames[i] for i in train_indices]

    df_ot = pd.DataFrame({"ImageID": train_filenames, "OT_value": v.astype(np.float64)})
    ot_csv_path = os.path.join(args.out_dir, "ot_values_with_ids.csv")
    df_ot.to_csv(ot_csv_path, index=False)

    # Top/bottom 50 listing (for full train-set OT scores)
    k50 = 50
    order_asc = np.argsort(v)
    bottom_idx = order_asc[:k50]
    top_idx = order_asc[-k50:][::-1]

    # =========================
    # 3) OT SUBSET EXPERIMENTS
    #    Train on top-X% OT of TRAIN SET only.
    #    Val/Test stay fixed.
    # =========================
    subset_fracs = []
    for s in args.subset_fracs.split(","):
        s = s.strip()
        if not s:
            continue
        subset_fracs.append(float(s))
    
    # Sort OT descending indices for train_set order
    order_desc = np.argsort(v)[::-1]  # highest -> lowest, indices into train_set

    results_top = []
    results_bottom = []

    # Include 100% baseline first in both tables
    baseline_row = {
        "subset_type": "full",
        "frac": 1.00,
        "n_train": len(train_set),
        "train_loss": full_train_loss, "train_acc": full_train_acc,
        "val_loss": full_val_loss, "val_acc": full_val_acc,
        "val_auc": full_val_auc,
        "val_best_thr": full_val_best_thr,
        "test_loss": full_test_loss, "test_acc": full_test_acc,
        "test_auc": full_test_auc,
        "best_epoch": full_best_epoch,
        "best_val_auc": full_best_val_auc,
        "best_path": full_best_path
    }
    results_top.append(dict(baseline_row))
    results_bottom.append(dict(baseline_row))

    for frac in subset_fracs:
        frac = float(frac)
        if frac <= 0.0 or frac > 1.0:
            continue

        n_keep = max(1, int(round(frac * len(train_set))))

        # -------------------------
        # TOP-X% OT experiment
        # -------------------------
        keep_top_indices = order_desc[:n_keep].tolist()
        train_subset_top = Subset(train_set, keep_top_indices)

        train_subset_top_t = TransformWrapper(train_subset_top, train_transform)
        train_subset_top_eval_t = TransformWrapper(train_subset_top, eval_transform)

        top_train_loader = DataLoader(train_subset_top_t, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True)
        top_train_eval_loader = DataLoader(train_subset_top_eval_t, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, pin_memory=True)

        run_name_top = f"subset_top_{int(frac*100)}pct"
        top_model = build_model(args, device)

        top_model, top_hist, top_best_path, top_best_epoch, top_best_val_auc = train_with_freeze_unfreeze_and_early_stop(
            top_model, top_train_loader, val_loader, device, args, run_name=run_name_top
        )

        top_train_loss, top_train_acc, _, _ = evaluate_binary(top_model, top_train_eval_loader, device, threshold=0.5)
        top_val_loss, top_val_acc, y_vt, y_vp = evaluate_binary(top_model, val_loader, device, threshold=0.5)
        _, _, top_val_auc, top_val_best_thr = compute_roc_auc_and_best_threshold(y_vt, y_vp)

        top_test_loss, top_test_acc, y_t, y_p = evaluate_binary(top_model, test_loader, device, threshold=0.5)
        _, _, top_test_auc, _ = compute_roc_auc_and_best_threshold(y_t, y_p)

        results_top.append({
            "subset_type": "top",
            "frac": frac,
            "n_train": n_keep,
            "train_loss": top_train_loss, "train_acc": top_train_acc,
            "val_loss": top_val_loss, "val_acc": top_val_acc,
            "val_auc": top_val_auc,
            "val_best_thr": top_val_best_thr,
            "test_loss": top_test_loss, "test_acc": top_test_acc,
            "test_auc": top_test_auc,
            "best_epoch": top_best_epoch,
            "best_val_auc": top_best_val_auc,
            "best_path": top_best_path
        })

        # -------------------------
        # BOTTOM-X% OT experiment
        # -------------------------
        keep_bottom_indices = order_desc[-n_keep:].tolist()
        train_subset_bottom = Subset(train_set, keep_bottom_indices)

        train_subset_bottom_t = TransformWrapper(train_subset_bottom, train_transform)
        train_subset_bottom_eval_t = TransformWrapper(train_subset_bottom, eval_transform)

        bottom_train_loader = DataLoader(train_subset_bottom_t, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=True)
        bottom_train_eval_loader = DataLoader(train_subset_bottom_eval_t, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers, pin_memory=True)

        run_name_bottom = f"subset_bottom_{int(frac*100)}pct"
        bottom_model = build_model(args, device)

        bottom_model, bottom_hist, bottom_best_path, bottom_best_epoch, bottom_best_val_auc = train_with_freeze_unfreeze_and_early_stop(
            bottom_model, bottom_train_loader, val_loader, device, args, run_name=run_name_bottom
        )

        bottom_train_loss, bottom_train_acc, _, _ = evaluate_binary(bottom_model, bottom_train_eval_loader, device, threshold=0.5)
        bottom_val_loss, bottom_val_acc, y_vt, y_vp = evaluate_binary(bottom_model, val_loader, device, threshold=0.5)
        _, _, bottom_val_auc, bottom_val_best_thr = compute_roc_auc_and_best_threshold(y_vt, y_vp)

        bottom_test_loss, bottom_test_acc, y_t, y_p = evaluate_binary(bottom_model, test_loader, device, threshold=0.5)
        _, _, bottom_test_auc, _ = compute_roc_auc_and_best_threshold(y_t, y_p)

        results_bottom.append({
            "subset_type": "bottom",
            "frac": frac,
            "n_train": n_keep,
            "train_loss": bottom_train_loss, "train_acc": bottom_train_acc,
            "val_loss": bottom_val_loss, "val_acc": bottom_val_acc,
            "val_auc": bottom_val_auc,
            "val_best_thr": bottom_val_best_thr,
            "test_loss": bottom_test_loss, "test_acc": bottom_test_acc,
            "test_auc": bottom_test_auc,
            "best_epoch": bottom_best_epoch,
            "best_val_auc": bottom_best_val_auc,
            "best_path": bottom_best_path
        })

    # Sort results by frac descending
    results_top = sorted(results_top, key=lambda d: d["frac"], reverse=True)
    results_bottom = sorted(results_bottom, key=lambda d: d["frac"], reverse=True)

    

    # TOP plots
    xs_pct_top = [int(round(r["frac"] * 100)) for r in results_top]
    top_train_accs = [float(r["train_acc"]) for r in results_top]
    top_val_accs = [float(r["val_acc"]) for r in results_top]
    top_test_accs = [float(r["test_acc"]) for r in results_top]

    top_train_acc_plot = os.path.join(args.out_dir, "subset_top_train_accuracy_vs_size.png")
    top_val_acc_plot = os.path.join(args.out_dir, "subset_top_val_accuracy_vs_size.png")
    top_test_acc_plot = os.path.join(args.out_dir, "subset_top_test_accuracy_vs_size.png")

    plot_acc_vs_size(xs_pct_top, top_train_accs, "Train Accuracy vs Training Data Kept (Top OT)", top_train_acc_plot)
    plot_acc_vs_size(xs_pct_top, top_val_accs, "Validation Accuracy vs Training Data Kept (Top OT)", top_val_acc_plot)
    plot_acc_vs_size(xs_pct_top, top_test_accs, "Test Accuracy vs Training Data Kept (Top OT)", top_test_acc_plot)

    # BOTTOM plots
    xs_pct_bottom = [int(round(r["frac"] * 100)) for r in results_bottom]
    bottom_train_accs = [float(r["train_acc"]) for r in results_bottom]
    bottom_val_accs = [float(r["val_acc"]) for r in results_bottom]
    bottom_test_accs = [float(r["test_acc"]) for r in results_bottom]

    bottom_train_acc_plot = os.path.join(args.out_dir, "subset_bottom_train_accuracy_vs_size.png")
    bottom_val_acc_plot = os.path.join(args.out_dir, "subset_bottom_val_accuracy_vs_size.png")
    bottom_test_acc_plot = os.path.join(args.out_dir, "subset_bottom_test_accuracy_vs_size.png")

    plot_acc_vs_size(xs_pct_bottom, bottom_train_accs, "Train Accuracy vs Training Data Kept (Bottom OT)", bottom_train_acc_plot)
    plot_acc_vs_size(xs_pct_bottom, bottom_val_accs, "Validation Accuracy vs Training Data Kept (Bottom OT)", bottom_val_acc_plot)
    plot_acc_vs_size(xs_pct_bottom, bottom_test_accs, "Test Accuracy vs Training Data Kept (Bottom OT)", bottom_test_acc_plot)

    # Save subset results CSVs
    subset_top_csv_path = os.path.join(args.out_dir, "subset_results_top.csv")
    subset_bottom_csv_path = os.path.join(args.out_dir, "subset_results_bottom.csv")

    pd.DataFrame(results_top).to_csv(subset_top_csv_path, index=False)
    pd.DataFrame(results_bottom).to_csv(subset_bottom_csv_path, index=False)

    # Save model (full best already saved; also save final full model state)
    model_path = os.path.join(args.out_dir, "densenet121_binary.pt")
    torch.save(full_model.state_dict(), model_path)

    # =========================
    # 4) FINAL REPORT
    # =========================
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
        f.write(f"best_epoch: {full_best_epoch}\n")
        f.write(f"best_val_auc: {full_best_val_auc:.6f}\n")
        f.write(f"Train: loss={full_train_loss:.6f}, acc@0.5={full_train_acc:.6f}\n")
        f.write(f"Val:   loss={full_val_loss:.6f}, acc@0.5={full_val_acc:.6f}, auc={full_val_auc:.6f}, best_thr={full_val_best_thr:.6f}\n")
        f.write(f"Test:  loss={full_test_loss:.6f}, acc@0.5={full_test_acc:.6f}, auc={full_test_auc:.6f}\n\n")

        f.write("=== OT VALUES SUMMARY (train samples) ===\n")
        f.write(f"OT shape: {v.shape}\n")
        f.write(f"min: {float(v.min()):.10f}\n")
        f.write(f"max: {float(v.max()):.10f}\n")
        f.write(f"mean: {float(v.mean()):.10f}\n")
        f.write(f"std: {float(v.std()):.10f}\n\n")

        f.write("=== TOP 50 OT SAMPLES (TRAIN SET) ===\n")
        f.write("Rank\tOT_value\tImageID\n")
        for rank, idx in enumerate(top_idx, start=1):
            f.write(f"{rank}\t{float(v[idx]):.10f}\t{train_filenames[idx]}\n")

        f.write("\n=== BOTTOM 50 OT SAMPLES (TRAIN SET) ===\n")
        f.write("Rank\tOT_value\tImageID\n")
        for rank, idx in enumerate(bottom_idx, start=1):
            f.write(f"{rank}\t{float(v[idx]):.10f}\t{train_filenames[idx]}\n")

        f.write("\n=== OT SUBSET EXPERIMENTS: TOP OT% ===\n")
        f.write("Kept%\tN_train\tTrainLoss\tTrainAcc\tValLoss\tValAcc\tValAUC\tValBestThr\tTestLoss\tTestAcc\tTestAUC\tBestEpoch\tBestValAUC\n")
        for r in results_top:
            kept = int(round(r["frac"] * 100))
            f.write(
                f"{kept}\t{r['n_train']}\t"
                f"{float(r['train_loss']):.6f}\t{float(r['train_acc']):.6f}\t"
                f"{float(r['val_loss']):.6f}\t{float(r['val_acc']):.6f}\t"
                f"{float(r.get('val_auc', np.nan)):.6f}\t{float(r.get('val_best_thr', np.nan)):.6f}\t"
                f"{float(r['test_loss']):.6f}\t{float(r['test_acc']):.6f}\t"
                f"{float(r['test_auc']):.6f}\t{int(r['best_epoch'])}\t{float(r.get('best_val_auc', np.nan)):.6f}\n"
            )

        f.write("\n=== OT SUBSET EXPERIMENTS: BOTTOM OT% ===\n")
        f.write("Kept%\tN_train\tTrainLoss\tTrainAcc\tValLoss\tValAcc\tValAUC\tValBestThr\tTestLoss\tTestAcc\tTestAUC\tBestEpoch\tBestValAUC\n")
        for r in results_bottom:
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
        f.write(f"Model (full best): {full_best_path}\n")
        f.write(f"Model (full final state): {model_path}\n")
        f.write(f"OT values (npy): {ot_path}\n")
        f.write(f"OT values + IDs (csv): {ot_csv_path}\n")
        f.write(f"Subset results TOP (csv): {subset_top_csv_path}\n")
        f.write(f"Subset results BOTTOM (csv): {subset_bottom_csv_path}\n")
        f.write(f"Top train acc vs size: {top_train_acc_plot}\n")
        f.write(f"Top val acc vs size: {top_val_acc_plot}\n")
        f.write(f"Top test acc vs size: {top_test_acc_plot}\n")
        f.write(f"Bottom train acc vs size: {bottom_train_acc_plot}\n")
        f.write(f"Bottom val acc vs size: {bottom_val_acc_plot}\n")
        f.write(f"Bottom test acc vs size: {bottom_test_acc_plot}\n")
        f.write(f"Loss plot (epoch, full): {loss_plot_path}\n")
        f.write(f"Accuracy plot (epoch, full): {acc_plot_path}\n")
        f.write(f"ROC plot (test, full): {roc_plot_path}\n")
        f.write(f"OT histogram: {ot_hist_path}\n")
        f.write(f"Report: {report_path}\n")

    print("\n=== DONE ===")
    print("Saved outputs to:", args.out_dir)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
