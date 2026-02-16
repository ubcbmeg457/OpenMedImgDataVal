# SHAP-Based X-Ray Data Valuation Pipeline

## Overview

This pipeline provides a principled, SHAP-based approach to **data-centric interpretability** and **data valuation** for chest X-ray classification. It addresses the critical research question: **Which training samples contribute most to model quality?**

Unlike the existing `prototype/` which uses custom KNN-Shapley approximations with deep learning models, this pipeline combines:

1. **Interpretable Models**: XGBoost trained on CNN-extracted features (fast, data valuation-friendly)
2. **SHAP Library**: Principled Shapley value computation for both features and samples
3. **Data-Centric Analysis**: Detect noisy labels, outliers, redundant samples, and high-value data
4. **Actionable Insights**: Identify which samples to relabel, remove, or prioritize

## Architecture

### Two-Stage Pipeline

```
Stage 1: Feature Extraction
├─ Load pretrained ResNet50 (transfer learning)
├─ Remove classification layer → 2048-dim embeddings
└─ Cache features as .npz for fast reuse

Stage 2: Data Valuation
├─ Train XGBoost on cached features
├─ Compute SHAP values (features + samples)
├─ Identify valuable/noisy/outlier samples
└─ Validate with data efficiency experiments
```

### Why This Architecture?

- **XGBoost trains 100x faster than CNNs** - critical for data valuation requiring many retrainings
- **SHAP TreeExplainer provides exact, fast Shapley values** for tree models
- **Two-stage approach** combines deep learning's feature extraction with interpretable classification
- **Pattern validated** in existing `prototype/prototype.ipynb` cells 13-16

## Directory Structure

```
xray-shapley/
├── xray_shapley.ipynb            # Complete unified pipeline (data → valuation)
├── pyproject.toml
├── README.md
├── data/                          # Downloaded Kaggle dataset (git-ignored)
├── embeddings/                    # Cached CNN embeddings
├── models/                        # Trained XGBoost models
├── shap_values/                   # Computed SHAP values
└── data_valuation/                # Data Shapley results and analysis
```

## Installation

### Prerequisites

- Python 3.11+
- UV package manager
- CUDA (optional, for faster feature extraction)

### Setup

From the repository root:

```bash
# Install all dependencies including xray-shapley
uv sync

# Activate the environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

## Usage

### Single Unified Notebook

Run the complete pipeline in one notebook:

```bash
cd xray-shapley
jupyter notebook xray_shapley.ipynb
```

The notebook contains 5 integrated sections:

**Section 1: Data Download**
- Downloads NIH Chest X-rays 5% sample (~2.3 GB, ~5,606 images)
- Change to full dataset (45 GB) in the download cell if needed
- **Output**: Dataset in `data/`

**Section 2: Feature Extraction**
- Loads pretrained ResNet50 and extracts 2048-dim embeddings
- Caches features for fast reuse
- **Output**: Cached features in `embeddings/`
  - `train_features.npz` (~5-20 MB)
  - `test_features.npz` (~1-5 MB)

**Section 3: Model Training**
- Trains XGBoost on cached CNN features
- Evaluates on test set
- **Output**: Trained model in `models/`
  - XGBoost baseline accuracy: >85% on test set

**Section 4: SHAP Feature Analysis**
- Computes feature-level SHAP values
- Generates feature importance visualizations
- **Output**: Feature importance plots and SHAP values
  - Summary plots (bar chart and beeswarm)
  - Dependence plots for top features
  - Individual prediction explanations

**Section 5: Data Valuation (Core Analysis)**
- Computes data-level Shapley values using KNN-Shapley
- Detects noisy labels, outliers, and redundant samples
- Validates with data efficiency experiments
- **Output**: Data quality analysis in `data_valuation/`
  - `data_shapley.npy` - Shapley values for all training samples
  - `data_valuation_results.csv` - Ranked samples with quality flags
  - Visualizations showing data efficiency curves

**Key Results**:
- Identifies high-value samples (maintain 95%+ accuracy with 70% of data)
- Detects potentially noisy labels
- Finds outliers and redundant samples
- Validates with random baseline comparison

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **XGBoost over CNNs** | 100x faster training, enables many retrainings for data valuation |
| **TreeExplainer** | Exact, fast Shapley values (vs. KernelExplainer which is slower) |
| **Two-stage pipeline** | Combines CNN feature extraction with interpretable classification |
| **KNN-SHAP approximation** | O(n log n) complexity, good accuracy-speed tradeoff |
| **Cached features** | Extract once, train many times - critical for experimentation |

## Dataset Size

The notebook is configured to use the **5% sample dataset** (~2.3 GB):
- ~5,606 X-ray images
- Fast download and processing
- Suitable for development and testing

To use the **full dataset** (45 GB, 112,120 images), change this line in the notebook:

```python
kaggle_path = kagglehub.dataset_download("nih-chest-xrays/sample")
# to:
kaggle_path = kagglehub.dataset_download("nih-chest-xrays/data")
```

## Expected Performance

### Baseline Metrics (XGBoost on CNN Features)
- **Test Accuracy**: >85%
- **ROC-AUC**: >0.90
- **Training time**: <5 minutes (vs. hours for CNN)

### Data Efficiency (from Notebook 5)
- **Top 70% of samples**: maintains >95% of full accuracy
- **Top 50% of samples**: maintains >85% of full accuracy
- **Random baseline**: lower efficiency (validates Shapley values)

### Data Quality Detection
- **Noisy labels detected**: ~5-10% of training data
- **Outliers**: ~1-3% (hard cases or mislabeled)
- **Redundant samples**: ~10-15% (near-zero Shapley value)

## Validation Plan

After running all notebooks, verify:

1. ✓ **Directory structure**: `xray-shapley/` created with proper subdirectories
2. ✓ **Workspace integration**: `uv sync` successfully installs all dependencies
3. ✓ **Data download**: Kaggle dataset in `data/raw/`, git-ignored
4. ✓ **Feature extraction**: Cached features exist, reasonable size (~20-50 MB)
5. ✓ **Model training**: XGBoost achieves >85% test accuracy
6. ✓ **SHAP analysis**: Summary plots render, feature importance makes sense
7. ✓ **Data valuation**:
   - Data Shapley values computed for all training samples
   - High-value samples appear cleaner (higher quality images)
   - Low-value samples show ambiguity or potential label errors
8. ✓ **Data efficiency**: Top 70% maintains >95% of full accuracy
9. ✓ **Results saved**: CSVs and visualizations in `data_valuation/`

## Comparison with Prototype

| Aspect | Prototype | XRay-Shapley |
|--------|-----------|--------------|
| **Model** | Custom CNN + KNN-Shapley | XGBoost + SHAP TreeExplainer |
| **Data Valuation** | Custom LOO approximation | KNN-Shapley + optional exact LOO |
| **Speed** | Slower (CNN training) | Fast (XGBoost) |
| **Interpretability** | Deep features | Shallow, interpretable |
| **Best For** | Feature-level analysis | Data-centric valuation |

## Results & Outputs

### Notebook 5 Outputs (in `data_valuation/`)

```
data_valuation/
├── data_shapley.npy                    # Shapley values (n_train,)
├── data_valuation_results.csv          # Ranked samples with flags
├── plots/
│   ├── shapley_distribution.png        # Histogram, box plot, cumulative
│   └── data_efficiency_curves.png      # Top-K vs Random vs Bottom-K
└── recommendations.txt                 # Actionable data curation suggestions
```

### Key CSV Columns (data_valuation_results.csv)

| Column | Meaning |
|--------|---------|
| `sample_idx` | Training sample index |
| `shapley_value` | Data Shapley value (-∞ to +∞) |
| `label` | Original label (0=NORMAL, 1=PNEUMONIA) |
| `is_noisy` | Detected as potentially mislabeled |
| `is_outlier` | Detected as outlier/hard case |
| `is_redundant` | Low information, high similarity to others |
| `is_high_value` | Top 10% most valuable samples |

## Next Steps

After completing the pipeline:

1. **Data Curation**: Review and relabel noisy samples
2. **Data Removal**: Remove redundant samples to reduce storage
3. **Active Learning**: Use Shapley values for smart sampling of unlabeled data
4. **Retraining**: Iterate - recompute Shapley after cleaning
5. **Comparison**: Compare with other data valuation methods (LOO, influence functions)

## Dependencies

See `pyproject.toml` for full list. Key packages:

- **ML/DL**: torch, torchvision, xgboost, scikit-learn
- **Interpretation**: shap
- **Data**: kagglehub, numpy, pandas
- **Viz**: matplotlib, seaborn
- **Notebook**: jupyter, ipykernel

## Citations

- SHAP: Lundberg & Lee (2017) - "A Unified Approach to Interpreting Model Predictions"
- Data Valuation: Ghorbani et al. (2019) - "Data Shapley: Equitable Valuation of Data for Machine Learning"
- XGBoost: Chen & Guestrin (2016) - "XGBoost: A Scalable Tree Boosting System"

## References

- [SHAP Documentation](https://shap.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Kaggle NIH Chest X-rays Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data)

## License

This project is part of OpenMedImgDataVal. See parent repository LICENSE.

## Contact

Questions or issues? Open a GitHub issue in the parent repository.
