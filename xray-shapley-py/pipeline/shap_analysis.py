"""Section 4: SHAP feature-level analysis on classifier head embeddings."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn as nn

from pipeline import config
from pipeline.model import SplitData, TrainResult


@dataclass
class ShapResult:
    """Outputs of the SHAP analysis stage."""

    shap_values: np.ndarray
    expected_value: float


# ---------------------------------------------------------------------------
# Explainer initialisation
# ---------------------------------------------------------------------------
def _model_device(module: nn.Module) -> torch.device:
    """Return the device of the first parameter in *module*."""
    return next(module.parameters()).device


def init_explainer(classifier_head: nn.Module, X_train: np.ndarray) -> tuple[shap.Explainer, torch.Tensor]:
    """Create a GradientExplainer (with DeepExplainer fallback) on the classifier head."""
    device = _model_device(classifier_head)
    background_size = min(config.SHAP_BACKGROUND_SIZE, len(X_train))
    bg_indices = np.random.choice(len(X_train), background_size, replace=False)
    background = torch.tensor(X_train[bg_indices], dtype=torch.float32).to(device)

    print("Initializing SHAP GradientExplainer on classifier head...")
    print(f"Background samples: {background_size}")

    try:
        explainer = shap.GradientExplainer(classifier_head, background)
        print("GradientExplainer initialized successfully")
    except Exception as e:
        print(f"GradientExplainer failed ({e}), falling back to DeepExplainer")
        explainer = shap.DeepExplainer(classifier_head, background)

    return explainer, background


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------
def compute_shap_values(
    explainer: shap.Explainer,
    X_test: np.ndarray,
    classifier_head: nn.Module,
    background: torch.Tensor,
) -> tuple[np.ndarray, float]:
    """Compute SHAP values for test embeddings and return (shap_values, expected_value)."""
    print("Computing SHAP values for test set...")
    device = _model_device(classifier_head)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    shap_values = explainer.shap_values(X_test_tensor)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.array(shap_values)

    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values.squeeze(-1)

    print(f"SHAP values shape: {shap_values.shape}")
    print(f"Sample SHAP values (first instance): {shap_values[0, :5]}")

    with torch.no_grad():
        base_logits = classifier_head(background)
    expected_value = base_logits.mean().item()
    print(f"Expected prediction (base value): {expected_value:.4f}")

    return shap_values, expected_value


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_shap_summary(shap_values: np.ndarray, X_test: np.ndarray) -> None:
    """Bar chart (mean |SHAP|) and beeswarm plot."""
    # Bar chart
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("Feature Importance (Mean |SHAP| values)")
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {config.rel(config.PLOTS_DIR / 'shap_bar.png')}")

    # Beeswarm
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.title("SHAP Summary Plot — Feature Impact on Predictions")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {config.rel(config.PLOTS_DIR / 'shap_beeswarm.png')}")


def analyze_top_features(shap_values: np.ndarray) -> np.ndarray:
    """Print the top-10 features by mean |SHAP| and return their indices."""
    feature_importance = np.abs(shap_values).mean(axis=0).flatten()
    top_k = 10
    top_indices = np.argsort(feature_importance)[-top_k:][::-1]

    print(f"Top {top_k} most important features (by SHAP):")
    for rank, idx in enumerate(top_indices, 1):
        print(f"{rank:2d}. Feature {idx}: {feature_importance[idx]:.4f}")

    return top_indices


def plot_shap_dependence(shap_values: np.ndarray, X_test: np.ndarray, top_indices: np.ndarray) -> None:
    """Dependence plots for the top 3 features."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, idx in enumerate(top_indices[:3]):
        shap.dependence_plot(idx, shap_values, X_test, ax=axes[i], show=False, title=f"Feature {idx} Dependence Plot")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "shap_dependence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {config.rel(config.PLOTS_DIR / 'shap_dependence.png')}")


def plot_shap_waterfall(
    shap_values: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    expected_value: float,
    classifier_head: nn.Module,
) -> None:
    """Waterfall plots for selected individual predictions."""
    print("Waterfall plots for individual predictions:\n")
    for idx in config.WATERFALL_SAMPLE_INDICES:
        if idx >= len(X_test):
            continue
        print(f"\n--- Sample {idx} ---")
        print(f"True label: {y_test[idx]} ({'NO FINDING' if y_test[idx] == 0 else 'HAS FINDING'})")

        with torch.no_grad():
            device = _model_device(classifier_head)
            logit = classifier_head(torch.tensor(X_test[idx : idx + 1], dtype=torch.float32).to(device)).item()
        prob = 1.0 / (1.0 + np.exp(-logit))
        prediction = 1 if prob >= 0.5 else 0
        print(f"Predicted label: {prediction} ({'NO FINDING' if prediction == 0 else 'HAS FINDING'})")
        print(f"Prediction probability: {prob:.4f}")

        plt.figure(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(values=shap_values[idx], base_values=expected_value, data=X_test[idx]),
            show=False,
            max_display=15,
        )
        plt.title(f"Sample {idx} — Prediction Explanation")
        plt.tight_layout()
        plt.savefig(config.PLOTS_DIR / f"shap_waterfall_{idx}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {config.rel(config.PLOTS_DIR / f'shap_waterfall_{idx}.png')}")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_shap_values(shap_values: np.ndarray, expected_value: float) -> None:
    """Persist SHAP values and expected value to RESULTS_DIR."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.RESULTS_DIR / "shap_values_test.npy", shap_values)
    np.save(config.RESULTS_DIR / "expected_value.npy", np.array([expected_value]))
    print(f"SHAP values saved to {config.rel(config.RESULTS_DIR)}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_shap_analysis(train_result: TrainResult, split: SplitData) -> ShapResult:
    """Run the full SHAP analysis pipeline (Section 4)."""
    print("\n" + "=" * 60)
    print("SECTION 4: SHAP FEATURE-LEVEL ANALYSIS")
    print("=" * 60)

    classifier_head = train_result.model.classifier
    classifier_head.eval()

    explainer, background = init_explainer(classifier_head, train_result.X_train)
    shap_values, expected_value = compute_shap_values(explainer, train_result.X_test, classifier_head, background)

    plot_shap_summary(shap_values, train_result.X_test)
    top_indices = analyze_top_features(shap_values)
    plot_shap_dependence(shap_values, train_result.X_test, top_indices)
    plot_shap_waterfall(shap_values, train_result.X_test, split.y_test, expected_value, classifier_head)
    save_shap_values(shap_values, expected_value)

    return ShapResult(shap_values=shap_values, expected_value=expected_value)
