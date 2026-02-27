# X-Ray Shapley: Data Valuation Pipeline

## Overview

This pipeline answers the question: **which training samples actually help a chest X-ray classifier, and which ones hurt it?**

It uses Data Shapley values to assign a numerical "contribution score" to every training image. A high score means the image helps the model make correct predictions. A negative score means the image actively misleads the model (e.g., it may be mislabeled or an unusual outlier). This is called **data valuation**, improving ML models by understanding the training data itself rather than tuning the model architecture.

> **Data valuation**: Instead of asking "how do we build a better model?", we ask "which data points actually matter?" This is a data-centric approach to ML.

The pipeline runs end-to-end in a single notebook (`xray_shapley.ipynb`) and covers:

1. Downloading the NIH Chest X-rays dataset from Kaggle
2. Setting up a DenseNet121 classifier (pretrained on ImageNet)
3. Fine-tuning the model, extracting embeddings
4. Computing feature-level explanations (SHAP)
5. Computing sample-level data valuations (KNN-Shapley)

### Dataset

We use the [NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data) **5% sample** (~5,606 images, ~2.3 GB). The full dataset is 112,120 images / 45 GB. You can switch to it by changing one line in the download cell.

The classification task is **binary**: "No Finding" (healthy) vs. "Has Finding" (any disease). The 5% sample has roughly a **54% / 46%** split, which is balanced enough for meaningful model training and data valuation.

## Pipeline Walkthrough and Results

### Section 1: Data Download and Preparation

**What it does**: Downloads the dataset from Kaggle using `kagglehub`, copies it into a local `data/` directory, and prints the file structure.

**Why it matters**: Reproducibility. Anyone with Kaggle credentials can re-run this and get the exact same dataset. The 5% sample is used for development speed.

**Output**: `data/sample/images/` containing 5,606 PNG chest X-ray images, plus `sample_labels.csv` with metadata and disease labels for each image.

---

### Section 2: Model Setup and Data Preparation

**What it does**: Loads a pretrained DenseNet121 (the backbone of CheXNet), replaces the classifier head with a single linear layer for binary classification, and creates stratified train/val/test splits with data augmentation for training.

> **DenseNet121**: A convolutional neural network where each layer receives inputs from all preceding layers. This dense connectivity pattern encourages feature reuse and requires fewer parameters than ResNet. DenseNet121 is the backbone of CheXNet (Rajpurkar et al. 2017), a model specifically designed for chest X-ray diagnosis.

> **Transfer learning**: DenseNet121 was already trained on millions of everyday photos (ImageNet). We reuse the visual knowledge it learned (edges, textures, shapes) and fine-tune it on chest X-rays. By default, we freeze the convolutional layers and only train the classifier head, which is CPU-feasible. Set `FINETUNE_ALL = True` to unfreeze all layers for GPU training.

**The t-SNE visualization** projects the pretrained 1024-dimensional embeddings down to 2D for plotting (before fine-tuning).

> **t-SNE**: An algorithm that squishes high-dimensional data into 2D so you can plot it. Points close together in 2D were similar in the original space. But distances between faraway points and cluster sizes are distorted, so only local neighborhoods are trustworthy.

**Output**:
- Dataset split: ~4,484 train / ~1,122 test (80/20 stratified)
- Train further split into ~3,587 train / ~897 val for early stopping
- t-SNE scatter plot of the pretrained feature space

---

### Section 3: Model Training (DenseNet121)

**What it does**: Fine-tunes the DenseNet121 classifier head on the training images using a PyTorch training loop with early stopping on validation AUC.

> **BCEWithLogitsLoss**: Binary cross-entropy loss that combines a sigmoid activation with binary cross-entropy in a single numerically stable operation. The `pos_weight` parameter upweights the minority class to handle class imbalance (replaces XGBoost's `scale_pos_weight`).

> **Early stopping**: Monitors validation AUC each epoch and stops training if it doesn't improve for 3 consecutive epochs. The best model checkpoint is restored, preventing overfitting.

**Why DenseNet121?** It is the backbone of CheXNet, a model purpose-built for chest X-ray classification. The 1024-dimensional penultimate layer provides compact, medically relevant embeddings. After training, these embeddings are extracted and cached for Sections 4 and 5.

**Gradient-based saliency maps** replace XGBoost feature importance. They show which image regions most influence the model's predictions, which is more interpretable for a CNN than abstract feature indices.

**Output**:
- Trained model saved to `models/densenet121_model.pt`
- Confusion matrices and saliency map visualizations
- Cached 1024-dim embeddings in `embeddings/` for downstream analysis

---

### Section 4: SHAP Feature-Level Analysis

**What it does**: Uses SHAP's GradientExplainer to explain _why_ the model made each prediction. For every test image, SHAP computes how much each of the 1024 embedding dimensions pushed the prediction toward "Has Finding" or "No Finding".

> **SHAP values**: Each feature gets a score for each prediction. Positive = pushed toward "Has Finding". Negative = pushed toward "No Finding". Add them all up plus the base value and you get the model's final output. It's a principled way to decompose any prediction into per-feature contributions.

> **GradientExplainer**: Uses gradient information to compute SHAP values for neural network models. Applied to the classifier head operating on 1024-dim embeddings, producing a (1122, 1024) SHAP matrix.

> **Shapley values** (the math behind SHAP): A concept from game theory that fairly distributes a "payout" among "players" based on their marginal contribution. Here the "payout" is the model's prediction and the "players" are features (Section 4) or training samples (Section 5).

**How to read the plots**:

- **Bar chart (mean |SHAP|)**: Which embedding dimensions matter most on average. Higher bar = more influence.

- **Beeswarm plot**: Each dot is one test image. X-axis = SHAP value for that feature. Color = the feature's actual value (red = high, blue = low). This shows both importance AND direction.

- **Dependence plots**: How one feature's value (x-axis) relates to its SHAP contribution (y-axis). Reveals non-linear relationships.

- **Waterfall plots**: Explain a single prediction step by step. Starting from the base value, each feature adds or subtracts, arriving at the final score.

**Output**: `shap_values/shap_values_test.npy` and `shap_values/expected_value.npy`

---

### Section 5: Data Valuation with Shapley Values (Core Analysis)

This is the central section of the pipeline. It shifts from explaining _features_ to valuing _training samples_.

#### 5.1-5.3: KNN-Shapley Computation

**What it does**: For each validation image, finds the 10 nearest training images (by distance in 1024-dim embedding space) and scores them:
- Neighbor has the **same label** as the validation image's true label: **+1/k** contribution (helpful)
- Neighbor has a **different label**: **-1/k** contribution (harmful)

Each training sample accumulates contributions across all validation images it appears as a neighbor for. The final score is its **Data Shapley value**.

> **KNN-Shapley**: A fast approximation of Data Shapley. Instead of retraining the model thousands of times (the exact method), it uses neighbor relationships to estimate each sample's contribution. Runs in seconds instead of hours.

**Intuition**: A training image is valuable if it tends to be near validation images _of the same class_. It is harmful if it tends to be near validation images _of the wrong class_, suggesting it may be mislabeled, an outlier, or in a confusing region of feature space.

#### 5.4: Distribution Analysis

**The histogram** shows the Shapley value distribution. A roughly symmetric distribution centered near zero is typical. Long tails on either side indicate a few very influential (positive or negative) samples.

**The cumulative distribution plot** shows what fraction of total data value comes from what fraction of samples. A steep curve means a small number of samples contribute disproportionately.

**The sorted values plot** visualizes the full spectrum from most harmful to most helpful training samples.

#### 5.5: High-Value and Low-Value Samples

**Top samples**: The most "helpful" training images. These are likely very clean, prototypical images that clearly anchor their class cluster in embedding space.

**Bottom samples**: The most "harmful" training images. They include both classes, suggesting they live in ambiguous regions where healthy and diseased X-rays overlap. These are prime candidates for manual review.

#### 5.6: Data Quality Detection

The pipeline categorizes training samples into four groups:

| Category | Criterion | Interpretation |
|----------|-----------|----------------|
| **Noisy labels** | Shapley < -0.05 | May be mislabeled or in ambiguous regions |
| **Outliers** | Negative Shapley AND flagged by LOF | Unusual images far from any cluster |
| **Redundant** | \|Shapley\| < 0.01 | Near-zero contribution, interchangeable with neighbors |
| **High-value** | Top 10th percentile | Most helpful for model quality |

> **Local Outlier Factor (LOF)**: An algorithm that flags points unusually far from their neighbors. Combined with negative Shapley, this identifies images that are both isolated AND harmful.

#### 5.7-5.9: Data Efficiency Experiments

**What it does**: Trains separate `nn.Linear(1024, 1)` classifier heads on different fractions of the training embeddings (50%-100%), selected by three strategies:
- **Top-K**: Use the highest-Shapley samples (should be best)
- **Random**: Random sample (baseline)
- **Bottom-K**: Use the lowest-Shapley samples (should be worst)

Evaluates each on the validation set using **F1 score**. The 18 runs are fast (~seconds each) since they only train a linear layer on pre-extracted embeddings.

> **Data efficiency**: How well a model performs when trained on less data. If you can use 70% of the data and get the same performance, the other 30% was redundant or harmful.

**The data efficiency curve plot** visualizes these three lines. An ideal result shows Top-K above Random above Bottom-K, with Top-K maintaining performance even at low fractions.

#### 5.10-5.11: Results and Recommendations

Saves all results to `data_valuation/`:
- `data_shapley.npy`: Raw Shapley values for all training samples
- `data_valuation_results.csv`: Every training sample with its Shapley value, label, and quality flags
- `problematic_indices.npy`: Indices of samples flagged as noisy

**Actionable recommendations** from the analysis:
1. **Data pruning**: Remove redundant samples to reduce storage with minimal performance loss
2. **Label review**: Manually review the most negative-Shapley samples
3. **Training optimization**: Use only the top 70-80% of samples by Shapley value
4. **Active learning**: Use high-value samples as seeds for further annotation campaigns

## Architecture

```
DenseNet121 (pretrained ImageNet, fine-tuned on chest X-rays)
  -> Frozen conv layers (head-only by default) or full fine-tuning (GPU)
  -> Classifier head: Linear(1024, 1) with BCEWithLogitsLoss
  -> Early stopping on validation AUC

After training:
  -> Extract 1024-dim embeddings from penultimate layer
  -> SHAP GradientExplainer on classifier head for feature-level explanations
  -> KNN-Shapley on embeddings for sample-level data valuation
  -> Data efficiency experiments with nn.Linear heads (~seconds each)
```

### Why this architecture?

- **DenseNet121 is the backbone of CheXNet**, designed specifically for chest X-ray classification
- **1024-dim embeddings** are compact and medically relevant (vs. 2048 for ResNet50)
- **Head-only training is CPU-feasible** (~minutes), with full fine-tuning available for GPU
- **GradientExplainer** provides gradient-based SHAP values for the neural classifier head
- **KNN-Shapley is O(n log n)**, much faster than exact Shapley (exponential) or Monte Carlo approximations
- **Linear head retraining** for data efficiency experiments runs in seconds on pre-extracted embeddings

## Directory Structure

```
xray-shapley-nb/
├── xray_shapley.ipynb      # Complete pipeline (run this)
├── pyproject.toml           # Dependencies
├── README.md                # This file
├── data/                    # Downloaded dataset (git-ignored)
│   └── sample/images/       # 5,606 chest X-ray PNGs
├── embeddings/              # Cached DenseNet121 embeddings
│   ├── train_features.npz   # (4484, 1024) + labels
│   └── test_features.npz    # (1122, 1024) + labels
├── models/                  # Trained DenseNet121
│   └── densenet121_model.pt
├── shap_values/             # Feature-level SHAP
│   ├── shap_values_test.npy
│   └── expected_value.npy
└── data_valuation/          # Sample-level valuation
    ├── data_shapley.npy
    ├── problematic_indices.npy
    └── data_valuation_results.csv
```

## Installation

### Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- [Kaggle API credentials](https://github.com/Kaggle/kaggle-api#api-credentials) (for dataset download in Section 1)
- CUDA optional (for full fine-tuning; head-only training runs on CPU)

### Local Setup

From the repository root:

```bash
make setup                        # Install all dependencies
make notebook MODULE=xray-shapley-nb # Launch Jupyter Lab in this directory
```

### JupyterHub / HPC (Sockeye, Fir, etc.)

```bash
make setup                            # Install dependencies
make kernel MODULE=xray-shapley-nb       # Register the Jupyter kernel
```

Then open JupyterHub, navigate to `xray-shapley-nb/xray_shapley.ipynb`, and select the **"Python (xray-shapley-nb)"** kernel.

## Key Takeaways

1. **Not all training data is equal.** Some samples actively hurt model performance. Removing the worst samples and keeping the best 70-80% can improve F1 over using all data.

2. **Data Shapley values provide a principled ranking.** The Top-K strategy consistently outperforms random selection and Bottom-K, validating that the Shapley values capture real signal about data quality.

3. **Redundancy is significant.** A large fraction of training samples contribute near-zero value. They are interchangeable with their neighbors in feature space, which has implications for data storage and annotation budgets.

4. **"Noisy" samples are not necessarily mislabeled.** Many negative-Shapley samples may be genuinely ambiguous cases where the disease boundary is unclear, rather than labeling errors.

5. **DenseNet121 provides medically relevant embeddings.** As the backbone of CheXNet, its features are more suited to chest X-ray analysis than general-purpose ImageNet models, improving both classification and valuation quality.

## References

- Lundberg & Lee (2017). [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874) (SHAP)
- Ghorbani & Zou (2019). [Data Shapley: Equitable Valuation of Data for Machine Learning](https://arxiv.org/abs/1904.02868) (Data Shapley)
- Jia et al. (2019). [Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms](https://arxiv.org/abs/1908.08619) (KNN-Shapley)
- Rajpurkar et al. (2017). [CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning](https://arxiv.org/abs/1711.05225) (CheXNet / DenseNet121)
- Huang et al. (2017). [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (DenseNet)
- Wang et al. (2017). [ChestX-ray8: Hospital-scale Chest X-ray Database](https://arxiv.org/abs/1705.02315) (NIH dataset)
