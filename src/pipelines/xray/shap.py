"""
KNN-Shapley data valuation for X-ray classification.

Implements the exact KNN-Shapley recursion from Jia et al. 2019,
with optional M* approximation and automatic k optimisation.
"""

import os

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────
# KNN accuracy (used for k optimisation)
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def knn_accuracy(train_feats, train_y, val_feats, val_y, k):
    tf = nn.functional.normalize(train_feats, dim=1)
    vf = nn.functional.normalize(val_feats, dim=1)

    sim = vf @ tf.T
    k_eff = min(k, sim.size(1))
    _, topk_idx = torch.topk(sim, k=k_eff, dim=1)

    topk_labels = train_y.view(-1).round().long()[topk_idx]
    preds = (topk_labels.float().mean(dim=1) >= 0.5).long()
    correct = (preds == val_y.view(-1).round().long()).sum().item()
    return correct / len(val_y)


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


# ──────────────────────────────────────────────────────────────────────
# KNN-Shapley computation
# ──────────────────────────────────────────────────────────────────────
def knn_shapley_values_embeddings(train_feats, train_y, val_feats, val_y, k=10, m_star=5000, val_batch=32):
    train_feats = nn.functional.normalize(train_feats, dim=1)
    val_feats = nn.functional.normalize(val_feats, dim=1)

    train_labels = train_y.view(-1).round().to(torch.int64)
    val_labels = val_y.view(-1).round().to(torch.int64)

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
            y_val = int(yb[j].item())

            lab = train_labels[idx_sorted].detach().cpu().numpy()
            Np = len(idx_sorted)
            K_eff = min(K, Np)

            s_next = (1.0 if lab[Np - 1] == y_val else 0.0) / float(Np)
            shapley[idx_sorted[Np - 1]] += s_next

            for pos in range(Np - 2, -1, -1):
                i_1idx = pos + 1
                I_i = 1.0 if lab[pos] == y_val else 0.0
                I_ip1 = 1.0 if lab[pos + 1] == y_val else 0.0

                coeff = (min(K_eff, i_1idx) / float(i_1idx)) / float(K_eff)
                s_i = s_next + (I_i - I_ip1) * coeff

                shapley[idx_sorted[pos]] += s_i
                s_next = s_i

    shapley /= float(n_val)
    return shapley.to(torch.float32).numpy()


# ──────────────────────────────────────────────────────────────────────
# Adapter for the shared pipeline
# ──────────────────────────────────────────────────────────────────────
def compute_shapley_values(train_feats, train_y, val_feats, val_y, args):
    """compute_values_fn adapter for run_valuation_pipeline."""
    candidates = [int(c) for c in args.k_candidates.split(",") if c.strip()]
    print("Optimising KNN k value...")
    best_k, k_results = optimize_knn_k(train_feats, train_y, val_feats, val_y, candidates, out_dir=args.out_dir)
    k = best_k
    args._optimized_k = best_k
    args._k_results = k_results

    m_star = args.shapley_mstar
    if m_star == -1:
        print("Shapley: exact mode (uses all train points per val) - may be very slow.")
    else:
        print(f"Shapley: approximate mode using top M*={m_star} nearest train points per val.")

    v = knn_shapley_values_embeddings(
        train_feats, train_y, val_feats, val_y, k=k, m_star=args.shapley_mstar, val_batch=args.shapley_batch_val
    )
    return v.astype(np.float64)
