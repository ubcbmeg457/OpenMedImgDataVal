"""Sections 2 & 3: DenseNet121 setup, dataset, training, evaluation, embeddings."""

import copy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import DenseNet121_Weights
from tqdm import tqdm

from pipeline import config


# ---------------------------------------------------------------------------
# Dataclasses for stage results
# ---------------------------------------------------------------------------
@dataclass
class SplitData:
    """Everything needed to describe the train/val/test splits."""

    dataset: "XRayDataset"
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_actual: np.ndarray
    val_actual: np.ndarray
    y_train_full: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


@dataclass
class TrainResult:
    """Embeddings and predictions produced after training."""

    model: nn.Module
    X_train_full: np.ndarray
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_val_pred: np.ndarray
    y_val_pred_proba: np.ndarray
    y_test_pred: np.ndarray
    y_test_pred_proba: np.ndarray


@dataclass
class MetricsResult:
    """Standard classification metrics bundle."""

    accuracy: float
    precision: float
    recall: float  # = sensitivity
    specificity: float
    f1: float
    auc_roc: float

    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "f1": self.f1,
            "auc_roc": self.auc_roc,
        }


# ---------------------------------------------------------------------------
# Custom datasets
# ---------------------------------------------------------------------------
class XRayDataset(Dataset):
    """Custom dataset for loading X-ray images with labels."""

    def __init__(self, image_dir: Path, labels_df: pd.DataFrame, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform

        self.valid_images: list[tuple[Path, int]] = []
        for _, row in labels_df.iterrows():
            img_name = str(row["Image Index"]).strip()
            findings = str(row["Finding Labels"]).strip()
            label = 0 if findings == "No Finding" else 1
            img_path = self.image_dir / img_name
            if img_path.exists():
                self.valid_images.append((img_path, label))

        n_no = sum(1 for _, lbl in self.valid_images if lbl == 0)
        n_has = sum(1 for _, lbl in self.valid_images if lbl == 1)
        print(f"Found {len(self.valid_images)} images (No Finding: {n_no}, Has Finding: {n_has})")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_path, label = self.valid_images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class TransformSubset(Dataset):
    """Wraps a base XRayDataset with a different transform for a subset of indices."""

    def __init__(self, base_dataset: XRayDataset, indices: np.ndarray, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path, label = self.base_dataset.valid_images[self.indices[idx]]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, eval_transform)."""
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
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
    return train_transform, eval_transform


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------
def create_fresh_model(device: torch.device, finetune_all: bool | None = None, quiet: bool = False) -> nn.Module:
    """Load pretrained DenseNet121 with fresh ImageNet weights, replace head for binary classification.

    This is the reusable core shared by ``create_model`` (Section 2) and the
    retraining experiments (Section 6).  *quiet=True* suppresses print output
    so that retraining loops stay concise.
    """
    if finetune_all is None:
        finetune_all = config.FINETUNE_ALL

    densenet = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    num_features = densenet.classifier.in_features  # 1024
    densenet.classifier = nn.Linear(num_features, 1)

    if not finetune_all:
        for param in densenet.features.parameters():
            param.requires_grad = False
        if not quiet:
            print("Frozen conv layers (head-only training)")
    else:
        if not quiet:
            print("All layers unfrozen (full fine-tuning)")

    densenet.to(device)
    if not quiet:
        print(f"DenseNet121 loaded on {device}")
        print(f"Classifier head: Linear({num_features}, 1)")
        trainable = sum(p.numel() for p in densenet.parameters() if p.requires_grad)
        total = sum(p.numel() for p in densenet.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,}")
    return densenet


def create_model(device: torch.device) -> nn.Module:
    """Load pretrained DenseNet121, replace head for binary classification."""
    print("\n" + "=" * 60)
    print("SECTION 2: MODEL SETUP AND DATA PREPARATION")
    print("=" * 60)
    return create_fresh_model(device, quiet=False)


# ---------------------------------------------------------------------------
# Data setup
# ---------------------------------------------------------------------------
def setup_data(image_dir: Path, df_labels: pd.DataFrame) -> SplitData:
    """Create dataset, stratified splits, and dataloaders."""
    _, eval_transform = get_transforms()
    train_transform, _ = get_transforms()

    dataset = XRayDataset(image_dir, df_labels, transform=eval_transform)

    all_labels = np.array([dataset[i][1] for i in range(len(dataset))])
    indices = np.arange(len(dataset))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, stratify=all_labels, random_state=config.SEED
    )

    y_train_full = all_labels[train_indices]
    y_test = all_labels[test_indices]

    print(f"\nTrain: {len(train_indices)}, Test: {len(test_indices)}")
    print(f"Train class distribution: {np.bincount(y_train_full)}")
    print(f"Test class distribution: {np.bincount(y_test)}")

    # Further split training into train/val for early stopping
    train_sub_indices, val_sub_indices = train_test_split(
        np.arange(len(train_indices)),
        test_size=0.2,
        stratify=y_train_full,
        random_state=config.SEED,
    )
    train_actual = train_indices[train_sub_indices]
    val_actual = train_indices[val_sub_indices]

    train_ds = TransformSubset(dataset, train_actual, train_transform)
    val_ds = TransformSubset(dataset, val_actual, eval_transform)
    test_ds = TransformSubset(dataset, test_indices, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    y_train = y_train_full[train_sub_indices]
    y_val = y_train_full[val_sub_indices]

    print(f"Train: {len(train_ds)} samples ({config.BATCH_SIZE}-batch, {len(train_loader)} batches)")
    print(f"Val:   {len(val_ds)} samples")
    print(f"Test:  {len(test_ds)} samples")
    print(f"\nTrain class dist: {np.bincount(y_train)}")
    print(f"Val class dist:   {np.bincount(y_val)}")
    print(f"Test class dist:  {np.bincount(y_test)}")

    return SplitData(
        dataset=dataset,
        train_indices=train_indices,
        test_indices=test_indices,
        train_actual=train_actual,
        val_actual=val_actual,
        y_train_full=y_train_full,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )


# ---------------------------------------------------------------------------
# Pre-training visualisation
# ---------------------------------------------------------------------------
def visualize_pretrained_features(model: nn.Module, split: SplitData, device: torch.device) -> None:
    """t-SNE of pretrained features (before fine-tuning)."""
    _, eval_transform = get_transforms()
    model.eval()
    sample_loader = DataLoader(
        TransformSubset(split.dataset, split.train_indices[: config.TSNE_SAMPLE_SIZE], eval_transform),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    pretrained_features: list[np.ndarray] = []
    pretrained_labels: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in tqdm(sample_loader, desc="Extracting pretrained features"):
            images = images.to(device)
            feats = model.features(images)
            feats = torch.nn.functional.adaptive_avg_pool2d(torch.relu(feats), 1)
            feats = feats.view(feats.size(0), -1)
            pretrained_features.append(feats.cpu().numpy())
            pretrained_labels.append(labels.numpy())

    features_sample = np.vstack(pretrained_features)
    labels_sample = np.concatenate(pretrained_labels)

    tsne = TSNE(n_components=2, random_state=config.SEED, perplexity=30)
    features_2d = tsne.fit_transform(features_sample)

    plt.figure(figsize=(10, 8))
    for label, name, color in [(0, "No Finding", "#4477AA"), (1, "Has Finding", "#EE6677")]:
        mask = labels_sample == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], c=color, label=name, alpha=0.5, s=20, edgecolors="none")
    plt.legend(fontsize=12)
    plt.title("Pretrained DenseNet121 Feature Space (t-SNE, before fine-tuning)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "pretrained_tsne.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(
        f"Samples plotted: {len(labels_sample)} "
        f"(No Finding: {np.sum(labels_sample == 0)}, Has Finding: {np.sum(labels_sample == 1)})"
    )
    print(f"Saved: {config.rel(config.PLOTS_DIR / 'pretrained_tsne.png')}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model_on_loaders(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_train: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    *,
    num_epochs: int = config.NUM_EPOCHS,
    patience: int = config.PATIENCE,
    lr: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    quiet: bool = False,
) -> nn.Module:
    """Train a model with BCEWithLogitsLoss, Adam, ReduceLROnPlateau, and early stopping.

    This is the reusable core shared by ``train_model`` (Section 3) and the
    retraining experiments (Section 6).  *quiet=True* suppresses per-epoch
    prints and tqdm progress bars.
    """
    neg_count = int(np.sum(y_train == 0))
    pos_count = int(np.sum(y_train == 1))
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(device)
    if not quiet:
        print(f"Class balance — Negative: {neg_count}, Positive: {pos_count}, pos_weight: {pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    if not quiet:
        print(f"Optimizer: Adam (lr={lr}, wd={weight_decay})")
        print("Scheduler: ReduceLROnPlateau (patience=2)")
        print(f"Epochs: {num_epochs}, Early stopping patience: {patience}")

    best_val_auc = 0.0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        loader_iter = (
            train_loader if quiet else tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [train]", leave=False)
        )
        for images, labels in loader_iter:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_logits_list, val_labels_list = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = model(images).cpu()
                val_logits_list.append(logits)
                val_labels_list.append(labels)

        val_logits = torch.cat(val_logits_list).squeeze(1)
        val_labels = torch.cat(val_labels_list)
        val_probs = torch.sigmoid(val_logits).numpy()
        val_auc = roc_auc_score(val_labels.numpy(), val_probs)

        scheduler.step(val_auc)
        if not quiet:
            print(f"Epoch {epoch + 1}/{num_epochs} — Loss: {train_loss:.4f} — Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if not quiet:
                    print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                break

    model.load_state_dict(best_model_state)
    model.eval()
    if not quiet:
        print(f"\nBest validation AUC: {best_val_auc:.4f}")
    return model


def train_model(model: nn.Module, split: SplitData, device: torch.device) -> nn.Module:
    """Fine-tune the DenseNet121 classifier head with early stopping."""
    print("\n" + "=" * 60)
    print("SECTION 3: MODEL TRAINING")
    print("=" * 60)
    return train_model_on_loaders(
        model,
        split.train_loader,
        split.val_loader,
        split.y_train,
        split.y_val,
        device,
        quiet=False,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> MetricsResult:
    """Compute standard binary classification metrics from predictions."""
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return MetricsResult(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        specificity=specificity,
        f1=f1_score(y_true, y_pred, zero_division=0),
        auc_roc=roc_auc_score(y_true, y_pred_proba),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def predict_dataset(model: nn.Module, loader: DataLoader, device: torch.device):
    """Run inference and return (preds, probs, labels)."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images).cpu().squeeze(1)
            all_logits.append(logits)
            all_labels.append(labels)
    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    preds = (probs >= 0.5).astype(int)
    return preds, probs, labels_np


def evaluate_model(
    model: nn.Module, split: SplitData, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Print metrics and plot confusion matrices. Return val/test preds+probs."""
    y_val_pred, y_val_pred_proba, _ = predict_dataset(model, split.val_loader, device)
    y_test_pred, y_test_pred_proba, _ = predict_dataset(model, split.test_loader, device)

    print("=" * 50)
    print("VALIDATION SET METRICS")
    print("=" * 50)
    print(f"Accuracy:  {accuracy_score(split.y_val, y_val_pred):.4f}")
    print(f"Precision: {precision_score(split.y_val, y_val_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(split.y_val, y_val_pred, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(split.y_val, y_val_pred, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(split.y_val, y_val_pred_proba):.4f}")

    print("\n" + "=" * 50)
    print("TEST SET METRICS (FINAL EVALUATION)")
    print("=" * 50)
    print(f"Accuracy:  {accuracy_score(split.y_test, y_test_pred):.4f}")
    print(f"Precision: {precision_score(split.y_test, y_test_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(split.y_test, y_test_pred, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(split.y_test, y_test_pred, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(split.y_test, y_test_pred_proba):.4f}")

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT (TEST SET)")
    print("=" * 50)
    print(
        classification_report(
            split.y_test, y_test_pred, labels=[0, 1], target_names=["NO FINDING", "HAS FINDING"], zero_division=0
        )
    )

    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm_val = confusion_matrix(split.y_val, y_val_pred, labels=[0, 1])
    sns.heatmap(cm_val, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title("Validation Set Confusion Matrix")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_xticklabels(["NO FINDING", "HAS FINDING"])
    axes[0].set_yticklabels(["NO FINDING", "HAS FINDING"])

    cm_test = confusion_matrix(split.y_test, y_test_pred, labels=[0, 1])
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Greens", ax=axes[1], cbar=False)
    axes[1].set_title("Test Set Confusion Matrix")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_xticklabels(["NO FINDING", "HAS FINDING"])
    axes[1].set_yticklabels(["NO FINDING", "HAS FINDING"])

    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {config.rel(config.PLOTS_DIR / 'confusion_matrices.png')}")

    return y_val_pred, y_val_pred_proba, y_test_pred, y_test_pred_proba


# ---------------------------------------------------------------------------
# Saliency maps
# ---------------------------------------------------------------------------
def generate_saliency_maps(model: nn.Module, split: SplitData, device: torch.device) -> None:
    """Gradient-based saliency maps for a few test images."""
    _, eval_transform = get_transforms()
    test_ds = TransformSubset(split.dataset, split.test_indices, eval_transform)

    fig, axes = plt.subplots(
        len(config.SALIENCY_SAMPLE_INDICES), 2, figsize=(8, 4 * len(config.SALIENCY_SAMPLE_INDICES))
    )
    model.eval()

    for row, idx in enumerate(config.SALIENCY_SAMPLE_INDICES):
        image, label = test_ds[idx]
        input_tensor = image.unsqueeze(0).to(device).requires_grad_(True)

        logit = model(input_tensor)
        logit.backward()

        saliency = input_tensor.grad.data.abs().squeeze().cpu()
        saliency = saliency.max(dim=0).values

        img_np = image.permute(1, 2, 0).numpy()
        img_np = img_np * np.array(config.IMAGENET_STD) + np.array(config.IMAGENET_MEAN)
        img_np = np.clip(img_np, 0, 1)

        pred_label = "HAS FINDING" if torch.sigmoid(logit).item() >= 0.5 else "NO FINDING"
        true_label = "HAS FINDING" if label == 1 else "NO FINDING"

        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f"True: {true_label}\nPred: {pred_label}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(saliency.numpy(), cmap="hot")
        axes[row, 1].set_title("Saliency Map")
        axes[row, 1].axis("off")

    plt.suptitle("Gradient-Based Saliency Maps", fontsize=14)
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "saliency_maps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {config.rel(config.PLOTS_DIR / 'saliency_maps.png')}")


# ---------------------------------------------------------------------------
# Save model
# ---------------------------------------------------------------------------
def save_model(model: nn.Module) -> None:
    """Save the model state dict."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / "densenet121_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {config.rel(model_path)}")
    print(f"File size: {model_path.stat().st_size / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
def extract_embeddings(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Extract 1024-dim embeddings from the penultimate layer via a forward hook."""
    model.eval()
    labels_list: list[torch.Tensor] = []
    hook_outputs: list[torch.Tensor] = []

    def hook_fn(module, input, output):
        pooled = torch.nn.functional.adaptive_avg_pool2d(torch.relu(output), 1)
        hook_outputs.append(pooled.squeeze(-1).squeeze(-1).detach().cpu())

    handle = model.features.register_forward_hook(hook_fn)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting embeddings"):
            images = images.to(device)
            _ = model(images)
            labels_list.append(labels)

    handle.remove()

    embeddings = torch.cat(hook_outputs).numpy()
    labels = torch.cat(labels_list).numpy()
    return embeddings, labels


def extract_and_cache_all_embeddings(model: nn.Module, split: SplitData, device: torch.device) -> TrainResult:
    """Extract embeddings for all splits, cache to disk, and return a TrainResult."""
    _, eval_transform = get_transforms()

    # Full training set (for KNN-Shapley)
    full_train_loader = DataLoader(
        TransformSubset(split.dataset, split.train_indices, eval_transform),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    print("Extracting training embeddings...")
    X_train_full, _ = extract_embeddings(model, full_train_loader, device)
    print(f"Training embeddings: {X_train_full.shape}")

    print("Extracting test embeddings...")
    X_test, _ = extract_embeddings(model, split.test_loader, device)
    print(f"Test embeddings: {X_test.shape}")

    print("Extracting train-subset embeddings...")
    train_sub_loader = DataLoader(
        TransformSubset(split.dataset, split.train_actual, eval_transform),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    X_train, _ = extract_embeddings(model, train_sub_loader, device)
    print(f"Train-subset embeddings: {X_train.shape}")

    print("Extracting val-subset embeddings...")
    val_sub_loader = DataLoader(
        TransformSubset(split.dataset, split.val_actual, eval_transform),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    X_val, _ = extract_embeddings(model, val_sub_loader, device)
    print(f"Val-subset embeddings: {X_val.shape}")

    # Cache
    config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(config.EMBEDDINGS_DIR / "train_features.npz", features=X_train_full, labels=split.y_train_full)
    np.savez(config.EMBEDDINGS_DIR / "test_features.npz", features=X_test, labels=split.y_test)
    print(f"\nEmbeddings cached to {config.rel(config.EMBEDDINGS_DIR)}/")

    # Predictions for downstream use
    y_val_pred, y_val_pred_proba, y_test_pred, y_test_pred_proba = evaluate_model(model, split, device)

    return TrainResult(
        model=model,
        X_train_full=X_train_full,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_val_pred=y_val_pred,
        y_val_pred_proba=y_val_pred_proba,
        y_test_pred=y_test_pred,
        y_test_pred_proba=y_test_pred_proba,
    )
