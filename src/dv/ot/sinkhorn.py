"""
Optimal Transport (OT) data valuation for multi-label classification.

Uses Sinkhorn OT with cosine distance cost and multi-label agreement reward.

This module is modality-agnostic: it operates on embeddings + label tensors
and can be reused for any classification pipeline (xray, mri, etc.).
"""

import numpy as np
import ot as pot
import torch
import torch.nn as nn


def ot_multilabel(train_feats, train_y, val_feats, val_y, reg=0.01, eps=1e-12):
    """
    Compute per-training-sample OT scores using Sinkhorn transport.

    Cost: cosine distance between embeddings.
    Reward: mean per-class label agreement (fraction of matching classes).
    """
    train_feats = nn.functional.normalize(train_feats, dim=1)
    val_feats = nn.functional.normalize(val_feats, dim=1)

    C = 1.0 - (train_feats @ val_feats.T)

    train_labels = train_y.round()
    val_labels = val_y.round()
    R = (train_labels.unsqueeze(1) == val_labels.unsqueeze(0)).float().mean(dim=2)

    n_train = train_feats.size(0)
    n_val = val_feats.size(0)
    a = np.ones(n_train, dtype=np.float64) / n_train
    b = np.ones(n_val, dtype=np.float64) / n_val

    C_np = C.detach().cpu().numpy().astype(np.float64)
    P = pot.sinkhorn(a, b, C_np, reg)
    P = torch.tensor(P, dtype=torch.float32)

    row_sums = P.sum(dim=1, keepdim=True).clamp_min(eps)
    P_row = P / row_sums
    scores = (P_row * R.detach().cpu()).sum(dim=1)
    return scores


def compute_ot_values(train_feats, train_y, val_feats, val_y, args):
    """compute_values_fn adapter for the pipeline."""
    scores = ot_multilabel(train_feats, train_y, val_feats, val_y, reg=args.ot_reg)
    return scores.numpy()
