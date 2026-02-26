import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

import ot


ALL_LABELS = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
    "Nodule","Pneumonia","Pneumothorax","Consolidation","Edema",
    "Emphysema","Fibrosis","Pleural_Thickening","Hernia"
]


class ChestXray14MultiLabel(Dataset):
    """
    Expected archive structure:
      archive/
        Data_Entry_2017.csv
        train_val_list.txt
        images_001/images/*.png
        images_002/images/*.png
        ...
        images_012/images/*.png
    """

    def __init__(self, root_dir, split_list_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        csv_path = os.path.join(root_dir, "Data_Entry_2017.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing CSV: {csv_path}")

        df = pd.read_csv(csv_path)

        # Map filename -> 14-dim multi-label vector
        self.label_map = {}
        for _, row in df.iterrows():
            filename = row["Image Index"]
            findings = str(row["Finding Labels"]).split("|")

            vec = np.zeros(len(ALL_LABELS), dtype=np.float32)
            for f in findings:
                if f in ALL_LABELS:
                    vec[ALL_LABELS.index(f)] = 1.0

            self.label_map[filename] = vec

        if not os.path.exists(split_list_path):
            raise FileNotFoundError(f"Missing split list: {split_list_path}")

        with open(split_list_path) as f:
            self.filenames = [line.strip() for line in f if line.strip()]

        # Build full paths
        self.paths = []
        for fn in self.filenames:
            p = self._find(fn)
            if p is None:
                raise FileNotFoundError(fn)
            self.paths.append(p)

    def _find(self, fn):
        # Folder has extra 'images' directory
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
        label = torch.tensor(self.label_map[self.filenames[idx]], dtype=torch.float32)
        return img, label


def _fix_densenet_state_dict_keys(state_dict):
    """
    Some DenseNet checkpoints (older/other repos) use keys like:
      norm.1, norm.2, conv.1, conv.2
    while torchvision expects:
      norm1, norm2, conv1, conv2
    """
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
    """
    Load DenseNet121 ImageNet weights from a local .pth file,
    fixing key naming if needed.
    """
    if local_weight_path is None:
        return model

    if not os.path.exists(local_weight_path):
        raise FileNotFoundError(f"Pretrained weights not found: {local_weight_path}")

    state_dict = torch.load(local_weight_path, map_location="cpu", weights_only=True)

    # Some checkpoints wrap weights under a key
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Fix key naming differences
    state_dict = _fix_densenet_state_dict_keys(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # We EXPECT classifier mismatch because we'll replace classifier anyway.
    # But if you see tons of missing/unexpected besides classifier, something is wrong.
    if missing:
        # Filter out classifier-related missing keys
        missing_non_classifier = [m for m in missing if not m.startswith("classifier.")]
        if len(missing_non_classifier) > 0:
            print("WARNING: Missing non-classifier keys:", missing_non_classifier[:20], "...")
    if unexpected:
        unexpected_non_classifier = [u for u in unexpected if not u.startswith("classifier.")]
        if len(unexpected_non_classifier) > 0:
            print("WARNING: Unexpected non-classifier keys:", unexpected_non_classifier[:20], "...")

    print(f"Loaded pretrained weights from: {local_weight_path}")
    return model


def ot_multilabel(train_feats, train_labels, val_feats, val_labels, reg=0.05):
    # Normalize features and compute cosine distance cost matrix
    train_feats = nn.functional.normalize(train_feats, dim=1)
    val_feats = nn.functional.normalize(val_feats, dim=1)
    C = 1.0 - (train_feats @ val_feats.T)  # [n_train, n_val]

    # Label similarity: cosine between multi-hot vectors
    train_labels = nn.functional.normalize(train_labels, dim=1)
    val_labels = nn.functional.normalize(val_labels, dim=1)
    R = train_labels @ val_labels.T  # [n_train, n_val]

    # Uniform weights
    a = torch.ones(train_feats.size(0), dtype=torch.float32) / train_feats.size(0)
    b = torch.ones(val_feats.size(0), dtype=torch.float32) / val_feats.size(0)

    # POT expects numpy arrays on CPU
    P = ot.sinkhorn(a.numpy(), b.numpy(), C.detach().cpu().numpy(), reg)
    P = torch.tensor(P, dtype=torch.float32)

    values = (P * R.detach().cpu()).sum(dim=1)  # [n_train]
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)

    # NEW: local pretrained weight path (no internet)
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Path to local DenseNet121 ImageNet weights (.pth)."
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_list = os.path.join(args.data_root, "train_val_list.txt")
    dataset = ChestXray14MultiLabel(args.data_root, train_list, transform)

    # simple random split
    n_val = int(0.1 * len(dataset))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # DenseNet121 base (NO online weights)
    model = models.densenet121(weights=None)

    # load local ImageNet weights into base (optional)
    model = load_pretrained_densenet121_from_local(model, args.pretrained_path)

    # Replace classifier for 14 multi-label
    model.classifier = nn.Linear(model.classifier.in_features, len(ALL_LABELS))
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(args.epochs):
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

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1}/{args.epochs} - train loss: {avg_loss:.4f}")

    # Feature extractor: reuse trained features
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
    def get_embeddings_pooled(loader):
        feats, labels = [], []
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            feats.append(forward_features(x))
            labels.append(y)
        return torch.cat(feats, dim=0), torch.cat(labels, dim=0)

    train_embed_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_embed_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    train_feats, train_labels = get_embeddings_pooled(train_embed_loader)
    val_feats, val_labels = get_embeddings_pooled(val_embed_loader)

    values = ot_multilabel(train_feats, train_labels, val_feats, val_labels, reg=0.05)

    np.save(os.path.join(args.out_dir, "ot_values.npy"), values.numpy())
    torch.save(model.state_dict(), os.path.join(args.out_dir, "densenet121_multilabel.pt"))

    print("Saved:")
    print(" - ot_values.npy")
    print(" - densenet121_multilabel.pt")


if __name__ == "__main__":
    main()
