# Archive

Archive of experimental results, legacy scripts, and project documents. This directory contains outputs from completed experiment runs and supporting documentation. The current, maintained pipeline code lives in [`../src/`](../src/).

## Contents

```
archive/
├── docs/                                                # Project documentation/presentations
│
├── mri-ot/                                              # MRI segmentation + Optimal Transport
│   ├── results/                                         # Experiment outputs
│   │   ├── final_report.txt                             # Run summary, hyperparameters, and metrics
│   │   ├── ot_values.csv                                # Per-sample OT data values
│   │   ├── ot_values_with_ids.csv                       # OT values mapped to BraTS image IDs
│   │   ├── ot_histogram.png                             # Distribution of OT values
│   │   ├── training_curves.png                          # Train/val loss and Dice over epochs
│   │   ├── subset_retraining_results.csv                # Subset retraining metrics
│   │   ├── subset_comparison_ot_vs_remove.png           # Performance: top-X% vs bottom-X%
│   │   ├── subset_comparison_emissions_ot_vs_remove.png # Emissions: top-X% vs bottom-X%
│   │   ├── top_*.png / bottom_*.png                     # Highest/lowest valued sample predictions
│   │   ├── top50_filenames.csv / bottom50_filenames.csv # Top/bottom 50 sample IDs
│   │   └── sample_*.png                                 # Example segmentation overlays
│   └── scripts/                                         # Legacy training scripts
│       ├── train.sh                                     # SLURM job script
│       └── trainMRIsegBottom.py                         # Standalone training script (pre-refactor)
│
├── xray-ot/                                             # X-ray classification + Optimal Transport
│   ├── results/                                         # Experiment outputs
│   │   ├── shapley_values_with_ids.csv                  # OT values mapped to CXR-14 image IDs
│   │   ├── shapley_histogram.png                        # Distribution of OT values
│   │   ├── subset_results.csv                           # Subset retraining (all splits)
│   │   ├── subset_results_bottom_only.csv               # Bottom-OT subset retraining
│   │   ├── X-Ray Shap.png                               # Summary visualization
│   │   └── slurm-9786475.{out,err}                      # SLURM job logs
│   └── scripts/                                         # Legacy training scripts
│       ├── run_gpu.sh                                   # SLURM job script
│       └── xray_densenet121_shap.py                     # Standalone training script (pre-refactor)
│
└── xray-shap/                                           # X-ray classification + KNN-Shapley
    ├── results/                                         # Experiment outputs
    │   ├── shapley_values_with_ids.csv                  # Shapley values mapped to CXR-14 image IDs
    │   ├── shapley_histogram.png                        # Distribution of Shapley values
    │   ├── subset_results.csv                           # Subset retraining (all splits)
    │   ├── subset_results_bottom_only.csv               # Bottom-Shapley subset retraining
    │   ├── X-Ray Shap.png                               # Summary visualization
    │   └── slurm-9786475.{out,err}                      # SLURM job logs
    └── scripts/                                         # Legacy training scripts
        ├── run_gpu.sh                                   # SLURM job script
        └── xray_densenet121_shap.py                     # Standalone training script (pre-refactor)
```

## Experiment Summary

| Experiment  | Modality | Task           | Data Valuation Method | Dataset                      | Model       |
| ----------- | -------- | -------------- | --------------------- | ---------------------------- | ----------- |
| `mri-ot`    | MRI      | Segmentation   | Sinkhorn OT           | BraTS 2023 GLI (1251 slices) | 2D U-Net    |
| `xray-ot`   | X-ray    | Classification | Sinkhorn OT           | NIH CXR-14 (86,524 images)   | DenseNet121 |
| `xray-shap` | X-ray    | Classification | KNN-Shapley           | NIH CXR-14 (86,524 images)   | DenseNet121 |

Each experiment followed the same protocol:

1. Train a model on the full dataset and compute per-sample data values
2. Rank samples by their computed value (high to low)
3. Retrain on subsets (top-X% and bottom-X%) to validate that high-value samples contribute more to model performance
