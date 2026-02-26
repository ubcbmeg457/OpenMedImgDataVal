# X-Ray Shapley: Data Valuation Pipeline

## Overview

This pipeline answers the question: **which training samples actually help a chest X-ray classifier, and which ones hurt it?**

It uses Data Shapley values to assign a numerical "contribution score" to every training image. A high score means the image helps the model make correct predictions. A negative score means the image actively misleads the model (e.g., it may be mislabeled or an unusual outlier). This is called **data valuation**, improving ML models by understanding the training data itself rather than tuning the model architecture.

> **Data valuation**: Instead of asking "how do we build a better model?", we ask "which data points actually matter?" This is a data-centric approach to ML.

The pipeline runs end-to-end in a single notebook (`xray_shapley.ipynb`) and covers:

1. Downloading the NIH Chest X-rays dataset from Kaggle
2. Extracting image embeddings with a pretrained CNN
3. Training a classifier on those embeddings
4. Computing feature-level explanations (SHAP)
5. Computing sample-level data valuations (KNN-Shapley)

### Dataset

We use the [NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data) **5% sample** (~5,606 images, ~2.3 GB). The full dataset is 112,120 images / 45 GB. You can switch to it by changing one line in the download cell.

The classification task is **binary**: "No Finding" (healthy) vs. "Has Finding" (any disease). The 5% sample has roughly a **54% / 46%** split, which is balanced enough for meaningful model training and data valuation.

## Pipeline Walkthrough and Results

### Section 1: Data Download and Preparation

**What it does**: Downloads the dataset from Kaggle using `kagglehub`, copies it into a local `data/` directory, and prints the file structure.

**Why it matters**: Reproducibility. Anyone with Kaggle credentials can re-run this and get the exact same dataset. The 5% sample is used for development speed, since feature extraction takes ~10 minutes on CPU vs. hours on the full dataset.

**Output**: `data/sample/images/` containing 5,606 PNG chest X-ray images, plus `sample_labels.csv` with metadata and disease labels for each image.

---

### Section 2: Feature Extraction with Pretrained CNN

**What it does**: Loads a pretrained ResNet50, removes the final classification layer, and passes every X-ray through it to produce a 2048-dimensional vector per image. These vectors are the "features" used for all downstream tasks.

> **Embedding**: A list of numbers that summarizes an image, like a fingerprint. Similar images produce similar numbers. Here, each image becomes 2048 numbers.

> **Pretrained model / Transfer learning**: ResNet50 was already trained on millions of everyday photos (ImageNet). We reuse the visual knowledge it learned (edges, textures, shapes) instead of training from scratch. This works surprisingly well even for medical images.

**Why not train a CNN directly?** Data valuation requires training many models on different subsets of data. Training a CNN each time would take hours. Instead, we extract features once (~10 min) and train fast XGBoost models on those features (~seconds each). This two-stage approach is standard in data valuation research.

**The t-SNE visualization** projects these 2048-dimensional embeddings down to 2D for plotting.

> **t-SNE**: An algorithm that squishes high-dimensional data into 2D so you can plot it. Points close together in 2D were similar in the original space. But distances between faraway points and cluster sizes are distorted, so only local neighborhoods are trustworthy.

If you see two distinct clusters colored by class, the features separate the classes well. If the classes are heavily intermixed, the features don't cleanly distinguish healthy from diseased X-rays, which tells you the classification task is hard.

**Output**:
- `embeddings/train_features.npz`: training features (4,484 images x 2,048 features)
- `embeddings/test_features.npz`: test features (1,122 images x 2,048 features)
- t-SNE scatter plot of the training feature space

**Our results**:
- Dataset split: 4,484 train / 1,122 test (80/20 stratified)
- Train distribution: 2,435 No Finding / 2,049 Has Finding
- Test distribution: 609 No Finding / 513 Has Finding

> **Stratified split**: Splitting data so each subset keeps the same class ratio as the original (~54/46 everywhere). Without this, one subset could accidentally end up with mostly one class.

---

### Section 3: Model Training (XGBoost)

**What it does**: Trains an XGBoost classifier on the cached embeddings. The training set is further split 80/20 into train/validation for monitoring during training.

> **XGBoost**: A fast ML algorithm that makes predictions by chaining many simple decision trees together. Each tree corrects the mistakes of the previous ones.

> **Train/validation/test split**: Training data teaches the model. Validation data monitors it during training (catches overfitting). Test data gives the final, unbiased score. They must never overlap.

**Why XGBoost?** It trains in seconds (vs. minutes/hours for neural networks), which is critical for Section 5 where we train ~18 separate models. SHAP's TreeExplainer also provides exact Shapley values for tree models, making the feature-level analysis in Section 4 theoretically sound.

**Handling class imbalance**: Even with the ~54/46 split, we set `scale_pos_weight` (~1.19) to slightly upweight the minority class.

> **Class imbalance**: When one class has more samples than the other. A model can "cheat" by always guessing the majority class and still looking accurate. `scale_pos_weight` tells XGBoost to care more about getting the smaller class right.

**How to read the metrics**:

| Metric | What it measures | Our result (test) |
|--------|-----------------|-------------------|
| **Accuracy** | % of all predictions that are correct | 0.647 |
| **Precision** | Of images predicted "Has Finding", what % actually have findings | 0.628 |
| **Recall** | Of images that actually have findings, what % did the model catch | 0.561 |
| **F1 Score** | Balances precision and recall into one number | 0.593 |
| **ROC-AUC** | How well the model ranks positive cases above negative ones | 0.692 |

> **Precision vs. Recall tradeoff**: Precision asks "when the model says yes, is it right?" Recall asks "did the model catch all the actual positives?" You usually can't max out both, since improving one hurts the other. F1 is a single number that balances them (1.0 = perfect, 0.0 = useless).

> **ROC-AUC**: Measures ranking quality rather than a single threshold. 0.5 means the model is no better than a coin flip. 1.0 means it perfectly separates the classes. Our 0.692 means the model is better than random but far from perfect.

> **Accuracy can be misleading**: If 95% of images are healthy, a model that always predicts "healthy" gets 95% accuracy while being completely useless. That's why we use F1 and AUC instead.

**Interpreting our results**: The model performs modestly (AUC ~0.69, F1 ~0.59). This is expected. We're using a general-purpose ImageNet model (not fine-tuned for medical images) with a simple XGBoost head. For data valuation purposes, this is fine. We don't need a perfect model, just one that is sensitive enough to the training data so that the Shapley values are meaningful.

**The confusion matrix** shows the breakdown of all predictions in a 2x2 table:

> **Confusion matrix**: Rows = what the image actually is, columns = what the model predicted. The diagonal (top-left, bottom-right) = correct predictions. Off-diagonal = errors. "False negatives" (missed diseases) are usually worse than "false positives" (false alarms) in medical settings.

**Output**:
- Trained model saved to `models/xgb_model.pkl` and `models/xgb_model.json`
- Confusion matrices and feature importance plots
- Validation AUC tracked during training (~0.685)

---

### Section 4: SHAP Feature-Level Analysis

**What it does**: Uses SHAP to explain _why_ the model made each prediction. For every test image, SHAP computes how much each of the 2048 features pushed the prediction toward "Has Finding" or "No Finding".

> **SHAP values**: Each feature gets a score for each prediction. Positive = pushed toward "Has Finding". Negative = pushed toward "No Finding". Add them all up plus the base value and you get the model's final output. It's a principled way to decompose any prediction into per-feature contributions.

> **Shapley values** (the math behind SHAP): A concept from game theory that fairly distributes a "payout" among "players" based on their marginal contribution. Here the "payout" is the model's prediction and the "players" are features (Section 4) or training samples (Section 5).

**The base value** is the model's average prediction before looking at any features. For our model it's ~0.0 (log-odds space), meaning it starts at roughly 50/50 before features push it one way or the other.

**How to read the plots**:

- **Bar chart (mean |SHAP|)**: Which features matter most on average. Higher bar = more influence. Our top features were Feature 861, 812, and 1666. These are abstract CNN features (not human-interpretable names), but the ranking shows which parts of the ResNet embedding the model relies on.

- **Beeswarm plot**: Each dot is one test image. X-axis = SHAP value for that feature. Color = the feature's actual value (red = high, blue = low). This shows both importance AND direction. For example, if red dots cluster on the right for a feature, high values of that feature push toward "Has Finding".

- **Dependence plots**: How one feature's value (x-axis) relates to its SHAP contribution (y-axis). Reveals non-linear relationships. A feature might only matter above a certain threshold.

- **Waterfall plots**: Explain a single prediction step by step. Starting from the base value, each feature adds or subtracts, arriving at the final score. Answers "why did the model predict this for this specific image?"

**Our results**: All SHAP values are relatively small (top feature mean |SHAP| ~0.09), consistent with the model's modest performance. No single feature dominates. The model relies on a diffuse combination of many features.

**Output**: `shap_values/shap_values_test.npy` and `shap_values/expected_value.npy`

---

### Section 5: Data Valuation with Shapley Values (Core Analysis)

This is the central section of the pipeline. It shifts from explaining _features_ to valuing _training samples_.

#### 5.1-5.3: KNN-Shapley Computation

**What it does**: For each validation image, finds the 10 nearest training images (by distance in embedding space) and scores them:
- Neighbor has the **same label** as the validation image's true label: **+1/k** contribution (helpful)
- Neighbor has a **different label**: **-1/k** contribution (harmful)

Each training sample accumulates contributions across all validation images it appears as a neighbor for. The final score is its **Data Shapley value**.

> **K-nearest neighbors (KNN)**: Find the K most similar training images to a given image by measuring distance between their embeddings. "Similar" here means close in 2048-dimensional space.

> **KNN-Shapley**: A fast approximation of Data Shapley. Instead of retraining the model thousands of times (the exact method), it uses neighbor relationships to estimate each sample's contribution. Runs in seconds instead of hours.

**Intuition**: A training image is valuable if it tends to be near validation images _of the same class_. It is harmful if it tends to be near validation images _of the wrong class_, suggesting it may be mislabeled, an outlier, or in a confusing region of feature space.

**Our results**:
- Shapley values range from **-1.0 to +1.3**
- Mean: **0.03**, Median: **0.0**
- Most samples have near-zero values (they don't strongly help or hurt)
- A small number of samples have high positive values (very helpful) or negative values (actively harmful)

The median of 0.0 means many training samples are never selected as neighbors for any validation sample. They sit in dense regions where other nearby samples already provide the same information.

#### 5.4: Distribution Analysis

**The histogram** shows the Shapley value distribution. A roughly symmetric distribution centered near zero is typical. Long tails on either side indicate a few very influential (positive or negative) samples.

**The cumulative distribution plot** shows what fraction of total data value comes from what fraction of samples. A steep curve means a small number of samples contribute disproportionately. This is the basis for data pruning (removing low-value samples without hurting performance).

**The sorted values plot** visualizes the full spectrum from most harmful to most helpful training samples.

#### 5.5: High-Value and Low-Value Samples

**Top samples** (Shapley > 0.8): The most "helpful" training images. In our results, most were "No Finding" images, likely very clean, prototypical healthy X-rays that clearly anchor the healthy cluster in feature space.

**Bottom samples** (Shapley < -0.5): The most "harmful" training images. They include both classes, suggesting they live in ambiguous regions of feature space where healthy and diseased X-rays overlap. These are prime candidates for manual review. They may be mislabeled, have unusual image quality, or represent genuinely ambiguous cases.

#### 5.6: Data Quality Detection

The pipeline categorizes training samples into four groups:

| Category | Criterion | Our Count | Our % | Interpretation |
|----------|-----------|-----------|-------|----------------|
| **Noisy labels** | Shapley < -0.05 | 844 | 23.5% | May be mislabeled or in ambiguous regions |
| **Outliers** | Negative Shapley AND flagged by LOF | 0 | 0.0% | Unusual images far from any cluster |
| **Redundant** | \|Shapley\| < 0.01 | 1,474 | 41.1% | Near-zero contribution, interchangeable with neighbors |
| **High-value** | Top 10th percentile | 345 | 9.6% | Most helpful for model quality |

> **Local Outlier Factor (LOF)**: An algorithm that flags points unusually far from their neighbors. Combined with negative Shapley, this would identify images that are both isolated AND harmful, i.e. true outliers. We found none, meaning the harmful samples aren't isolated weirdos but rather images stuck in confusing overlap zones between classes.

**Interpreting the high noisy label count (23.5%)**: This does NOT mean 23.5% of labels are actually wrong. The NIH labels are known to be noisy (derived from NLP on radiology reports, not expert annotation). Additionally, with a modest model (AUC ~0.69), many images near the decision boundary will have negative Shapley values simply because the model can't cleanly separate them. They may be genuinely hard cases rather than mislabeled.

**Interpreting high redundancy (41.1%)**: Many training images have near-zero Shapley values because they sit in dense clusters where neighboring images provide the same information. Removing these would reduce storage with minimal performance impact.

#### 5.7-5.9: Data Efficiency Experiments

**What it does**: Trains separate XGBoost models on different fractions of the training data (50%-100%), selected by three strategies:
- **Top-K**: Use the highest-Shapley samples (should be best)
- **Random**: Random sample (baseline)
- **Bottom-K**: Use the lowest-Shapley samples (should be worst)

Evaluates each on the validation set using **F1 score** (not accuracy, which is misleading with class imbalance).

> **Data efficiency**: How well a model performs when trained on less data. If you can use 70% of the data and get the same performance, the other 30% was redundant or harmful.

**Our results**:

| Fraction | Top-K F1 | Random F1 | Bottom-K F1 |
|----------|----------|-----------|-------------|
| 50% | 0.621 | 0.596 | 0.531 |
| 60% | 0.625 | 0.558 | 0.534 |
| 70% | 0.634 | 0.619 | 0.538 |
| 80% | 0.641 | 0.622 | 0.566 |
| 90% | 0.612 | 0.607 | 0.582 |
| 100% | 0.574 | 0.574 | 0.574 |

**How to read this**:
- At 100%, all three strategies are identical (they all use the full training set).
- **Top-K consistently outperforms Random and Bottom-K**, validating that Shapley values correctly identify more useful training samples.
- **Bottom-K is consistently the worst**, confirming that low-Shapley samples hurt model performance.
- **Top-K at 80% (F1=0.641) outperforms 100% of data (F1=0.574)**. Using all data actually hurts because noisy/harmful samples degrade the model. This is the key finding: _less data can be better if you select the right data_.
- The gap between Top-K and Bottom-K is ~0.08-0.10 F1 at lower fractions, which is substantial and shows the Shapley values carry real signal.

**The data efficiency curve plot** visualizes these three lines. An ideal result shows Top-K above Random above Bottom-K, with Top-K maintaining performance even at low fractions.

#### 5.10-5.11: Results and Recommendations

Saves all results to `data_valuation/`:
- `data_shapley.npy`: Raw Shapley values for all training samples
- `data_valuation_results.csv`: Every training sample with its Shapley value, label, and quality flags
- `problematic_indices.npy`: Indices of samples flagged as noisy

**Actionable recommendations** from the analysis:
1. **Data pruning**: Remove the ~41% redundant samples to reduce storage with minimal performance loss
2. **Label review**: Manually review the 844 negative-Shapley samples, especially the most negative ones
3. **Training optimization**: Use only the top 70-80% of samples by Shapley value for better F1 than using all data
4. **Active learning**: Use high-value samples as seeds for further annotation campaigns

## Architecture

```
Stage 1: Feature Extraction (run once, ~10 min on CPU)
  ResNet50 (pretrained on ImageNet)
    -> Remove classification head
    -> Extract 2048-dim embeddings per image
    -> Cache as .npz files

Stage 2: Classification + Valuation (fast, seconds per model)
  XGBoost on cached embeddings
    -> SHAP TreeExplainer for feature-level explanations
    -> KNN-Shapley for sample-level data valuation
    -> Data efficiency experiments (18 model retrainings)
```

### Why this architecture?

- **XGBoost trains in seconds**, which is critical for data valuation that requires many retrainings
- **SHAP TreeExplainer is exact** for tree models, so no sampling approximation is needed
- **Two-stage separation** means you extract features once and experiment freely with models
- **KNN-Shapley is O(n log n)**, much faster than exact Shapley (exponential) or Monte Carlo approximations

## Directory Structure

```
xray-shapley/
├── xray_shapley.ipynb      # Complete pipeline (run this)
├── pyproject.toml           # Dependencies
├── README.md                # This file
├── data/                    # Downloaded dataset (git-ignored)
│   └── sample/images/       # 5,606 chest X-ray PNGs
├── embeddings/              # Cached ResNet50 features
│   ├── train_features.npz   # (4484, 2048) + labels
│   └── test_features.npz    # (1122, 2048) + labels
├── models/                  # Trained XGBoost
│   ├── xgb_model.pkl
│   └── xgb_model.json
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
- CUDA optional (for faster feature extraction)

### Local Setup

From the repository root:

```bash
make setup                        # Install all dependencies
make notebook MODULE=xray-shapley # Launch Jupyter Lab in this directory
```

### JupyterHub / HPC (Sockeye, Fir, etc.)

```bash
make setup                            # Install dependencies
make kernel MODULE=xray-shapley       # Register the Jupyter kernel
```

Then open JupyterHub, navigate to `xray-shapley/xray_shapley.ipynb`, and select the **"Python (xray-shapley)"** kernel.

## Key Takeaways

1. **Not all training data is equal.** Some samples actively hurt model performance. Removing the worst 20% and keeping the best 80% improved F1 from 0.574 to 0.641.

2. **Data Shapley values provide a principled ranking.** The Top-K strategy consistently outperformed random selection and Bottom-K, validating that the Shapley values capture real signal about data quality.

3. **Redundancy is significant.** 41% of training samples contribute near-zero value. They are interchangeable with their neighbors in feature space, which has implications for data storage and annotation budgets.

4. **"Noisy" samples are not necessarily mislabeled.** Many negative-Shapley samples may be genuinely ambiguous cases where the disease boundary is unclear, rather than labeling errors.

5. **Model quality matters for valuation quality.** With a modest AUC of 0.69, the data valuation has more noise than it would with a stronger model. Fine-tuning the CNN on medical images or using a medical-specific pretrained model (e.g., CheXNet) would likely improve both classification and valuation quality.

## References

- Lundberg & Lee (2017). [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874) (SHAP)
- Ghorbani & Zou (2019). [Data Shapley: Equitable Valuation of Data for Machine Learning](https://arxiv.org/abs/1904.02868) (Data Shapley)
- Jia et al. (2019). [Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms](https://arxiv.org/abs/1908.08619) (KNN-Shapley)
- Chen & Guestrin (2016). [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754) (XGBoost)
- He et al. (2016). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (ResNet)
- Wang et al. (2017). [ChestX-ray8: Hospital-scale Chest X-ray Database](https://arxiv.org/abs/1705.02315) (NIH dataset)
