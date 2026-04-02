"""
KNN-Shapley data valuation for multi-label classification.

Implements the exact KNN-Shapley recursion from Jia et al. 2019,
extended for multi-label settings using per-class agreement scores.

This module is modality-agnostic: it operates on embeddings + label tensors
and can be reused for any classification pipeline (xray, mri, etc.).
"""

import os

import matplotlib

matplotlib.use("Agg")

# isort: split
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def _inverse_frequency_weights(labels, eps=1e-8):
    """Compute inverse-frequency class weights from binary label matrix.

    Returns a 1-D tensor of shape [n_classes] where rare classes receive higher
    weight, preventing the agreement metric from being dominated by the
    majority class (e.g. "No Finding" in CXR-14).

    Uses sqrt(1/freq) to moderate the correction — pure 1/freq can over-weight
    ultra-rare classes (e.g. Hernia at ~0.2%) making valuation noisy.
    """
    freq = labels.float().mean(dim=0).clamp(min=eps)  # per-class positive rate
    w = 1.0 / freq.sqrt()
    w = w / w.sum() * float(w.numel())  # normalise so weights sum to n_classes
    return w


@torch.no_grad()
def knn_accuracy(train_feats, train_y, val_feats, val_y, k):
    """Multi-label KNN accuracy: per-class majority vote, macro averaged."""
    tf = nn.functional.normalize(train_feats, dim=1)
    vf = nn.functional.normalize(val_feats, dim=1)

    sim = vf @ tf.T
    k_eff = min(k, sim.size(1))
    _, topk_idx = torch.topk(sim, k=k_eff, dim=1)

    train_labels = train_y.round().long()
    topk_labels = train_labels[topk_idx.cpu()]  # [n_val, k, n_classes]
    preds = (topk_labels.float().mean(dim=1) >= 0.5).long()  # [n_val, n_classes]
    val_labels = val_y.round().long()

    per_class_acc = (preds == val_labels).float().mean(dim=0)
    return per_class_acc.mean().item()


def optimize_knn_k(train_feats, train_y, val_feats, val_y, candidates, out_dir=None):
    """Try each candidate k, return (best_k, {k: accuracy})."""
    results = {}
    for k in candidates:
        acc = knn_accuracy(train_feats, train_y, val_feats, val_y, k)
        results[k] = acc
        print(f"  KNN k={k:>4d}  val_acc={acc:.6f}")

    best_k = max(results, key=results.get)
    print(f"  -> Best k={best_k} (val_acc={results[best_k]:.6f})")

    if out_dir is not None:
        ks = sorted(results.keys())
        accs = [results[k] for k in ks]
        plt.figure()
        plt.plot(ks, accs, marker="o")
        plt.axvline(best_k, color="r", linestyle="--", label=f"best k={best_k}")
        plt.title("KNN Validation Accuracy vs k")
        plt.xlabel("k")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "knn_k_optimization.png"), dpi=200)
        plt.close()

    return best_k, results


def knn_shapley_values_embeddings(train_feats, train_y, val_feats, val_y, k=10, m_star=5000, val_batch=32):
    """
    Multi-label KNN-Shapley computation.

    Agreement between training and validation labels uses inverse-frequency
    weighted per-class matching, so that rare pathology classes contribute
    proportionally more than the dominant majority class.
    """
    train_feats = nn.functional.normalize(train_feats, dim=1)
    val_feats = nn.functional.normalize(val_feats, dim=1)

    train_labels = train_y.round().to(torch.int64)
    val_labels = val_y.round().to(torch.int64)

    # Compute class weights from the combined label distribution
    all_labels = torch.cat([train_labels, val_labels], dim=0)
    class_weights = _inverse_frequency_weights(all_labels)  # [n_classes]

    n_train = train_feats.size(0)
    n_val = val_feats.size(0)

    shapley = torch.zeros(n_train, dtype=torch.float64, device="cpu")

    if m_star is None:
        m_star = 5000
    use_exact = (m_star == -1) or (m_star >= n_train)
    M = n_train if use_exact else int(m_star)
    K = int(k)

    for start in range(0, n_val, val_batch):
        end = min(n_val, start + val_batch)
        vb = val_feats[start:end]
        yb = val_labels[start:end]

        sim = (train_feats @ vb.T).detach()

        if use_exact:
            order = torch.argsort(sim, dim=0, descending=True)
        else:
            _, order = torch.topk(sim, k=M, dim=0, largest=True, sorted=True)

        order_cpu = order.detach().cpu()
        sim = None

        for j in range(order_cpu.size(1)):
            idx_sorted = order_cpu[:, j].numpy()
            y_val_j = yb[j]

            lab = train_labels[idx_sorted]
            Np = len(idx_sorted)
            K_eff = min(K, Np)

            # Weighted per-class agreement: rare classes count more
            a_lab = lab
            b_lab = y_val_j.unsqueeze(0)
            match = (a_lab == b_lab).float()  # [Np, n_classes]
            agreement = (match * class_weights.unsqueeze(0)).mean(dim=1).numpy()

            s_next = agreement[Np - 1] / float(Np)
            shapley[idx_sorted[Np - 1]] += s_next

            for pos in range(Np - 2, -1, -1):
                i_1idx = pos + 1
                coeff = (min(K_eff, i_1idx) / float(i_1idx)) / float(K_eff)
                s_i = s_next + (agreement[pos] - agreement[pos + 1]) * coeff
                shapley[idx_sorted[pos]] += s_i
                s_next = s_i

    shapley /= float(n_val)
    return shapley.to(torch.float32).numpy()


def compute_shapley_values(train_feats, train_y, val_feats, val_y, args):
    """compute_values_fn adapter for the pipeline."""
    candidates = [int(c) for c in args.shapley_k_candidates.split(",") if c.strip()]
    print("Optimising KNN k value...")
    best_k, k_results = optimize_knn_k(train_feats, train_y, val_feats, val_y, candidates, out_dir=args.out_dir)
    args._optimized_k = best_k
    args._k_results = k_results

    m_star = args.shapley_mstar
    if m_star == -1:
        print("Shapley: exact mode (all train points per val) - may be very slow.")
    else:
        print(f"Shapley: approximate mode using top M*={m_star} nearest train points per val.")

    v = knn_shapley_values_embeddings(
        train_feats, train_y, val_feats, val_y, k=best_k, m_star=args.shapley_mstar, val_batch=args.shapley_batch_val
    )
    return v.astype(np.float64)
