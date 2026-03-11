"""Section 3: Optimal Transport data valuation."""

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import torch
import torch.nn as nn

from pipeline import config
from pipeline.data import DataResult
from pipeline.model import TrainResult, extract_embeddings


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass
class OTResult:
    """Outputs of the OT data valuation stage."""

    ot_values: np.ndarray
    train_filenames: list[str]
    order_desc: np.ndarray


# ---------------------------------------------------------------------------
# OT computation
# ---------------------------------------------------------------------------
def ot_binary_row_normalized(train_feats, train_y, val_feats, val_y, reg=0.01, eps=1e-12):
    """Compute per-sample OT data values using Sinkhorn transport."""
    train_feats = nn.functional.normalize(train_feats, dim=1)
    val_feats = nn.functional.normalize(val_feats, dim=1)

    C = 1.0 - (train_feats @ val_feats.T)
    R = (train_y.round() == val_y.T.round()).float()

    n_train = train_feats.size(0)
    n_val = val_feats.size(0)
    a = np.ones(n_train, dtype=np.float64) / n_train
    b = np.ones(n_val, dtype=np.float64) / n_val

    C_np = C.detach().cpu().numpy().astype(np.float64)
    P = ot.sinkhorn(a, b, C_np, reg)
    P = torch.tensor(P, dtype=torch.float32)

    row_sums = P.sum(dim=1, keepdim=True).clamp_min(eps)
    P_row = P / row_sums
    scores = (P_row * R.detach().cpu()).sum(dim=1)
    return scores


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_ot_histogram(values, out_path, bins=50):
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title("Histogram of OT Values (Train Samples)")
    plt.xlabel("OT value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_ot_valuation(train_result: TrainResult, data: DataResult, args) -> OTResult:
    """Section 3 orchestrator: compute OT data values on train vs val embeddings."""
    print("\n" + "=" * 60)
    print("SECTION 3: OPTIMAL TRANSPORT DATA VALUATION")
    print("=" * 60)

    device = next(train_result.model.parameters()).device

    # Extract embeddings using deterministic (eval) transforms
    from torch.utils.data import DataLoader

    from pipeline.data import TransformWrapper, get_transforms

    _, eval_transform = get_transforms()

    train_embed_set = TransformWrapper(data.train_set, eval_transform)
    val_embed_set = TransformWrapper(data.val_set, eval_transform)

    train_embed_loader = DataLoader(
        train_embed_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    val_embed_loader = DataLoader(
        val_embed_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    train_feats, train_y = extract_embeddings(train_result.model, train_embed_loader, device)
    val_feats, val_y = extract_embeddings(train_result.model, val_embed_loader, device)

    ot_values = ot_binary_row_normalized(train_feats, train_y, val_feats, val_y, reg=args.ot_reg)
    v = ot_values.numpy()

    # Save OT values
    ot_path = os.path.join(args.out_dir, "ot_values.npy")
    np.save(ot_path, v)

    plot_ot_histogram(v, os.path.join(args.out_dir, "ot_histogram.png"))

    # Map train order -> filenames
    train_indices = data.train_set.indices
    train_filenames = [data.dataset.filenames[i] for i in train_indices]

    df_ot = pd.DataFrame({"ImageID": train_filenames, "OT_value": v.astype(np.float64)})
    ot_csv_path = os.path.join(args.out_dir, "ot_values_with_ids.csv")
    df_ot.to_csv(ot_csv_path, index=False)

    print(
        f"\nOT values — shape: {v.shape}, min: {v.min():.10f}, max: {v.max():.10f}, "
        f"mean: {v.mean():.10f}, std: {v.std():.10f}"
    )
    print(f"Saved: {config.rel(ot_path)}")
    print(f"Saved: {config.rel(ot_csv_path)}")

    order_desc = np.argsort(v)[::-1]

    return OTResult(
        ot_values=v,
        train_filenames=train_filenames,
        order_desc=order_desc,
    )
