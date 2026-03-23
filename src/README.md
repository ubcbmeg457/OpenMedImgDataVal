# Pipeline Architecture

```mermaid
graph TD
    CLI["main.py<br/>--modality xray --task class --dv {shap,ot}"]

    CLI --> Download["Section 1: Data Download<br/>kagglehub → NIH CXR-14<br/>(14-class multi-label)"]
    Download --> HPO["Section 2: HPO<br/>Random search (10 trials)"]
    HPO --> Train["Section 3: Full Training<br/>DenseNet121 + early stopping"]
    Train --> Embed["Extract Embeddings<br/>1024-dim features"]

    Embed --> SHAP["Section 4a: KNN-Shapley<br/>dv/shap/"]
    Embed --> OT["Section 4b: Optimal Transport<br/>dv/ot/ (POT Sinkhorn)"]

    SHAP --> Retrain["Section 5: Subset Retraining<br/>Top-N / Bottom-N / Random-N<br/>20%–90% retention"]
    OT --> Retrain

    Retrain --> Out["Outputs<br/>report, plots, CSVs, model"]

    style CLI fill:#e0e0e0,stroke:#333
    style SHAP fill:#d4edda,stroke:#28a745
    style OT fill:#d4edda,stroke:#28a745
    style Out fill:#fff3cd,stroke:#856404
```

## Directory Structure

```
src/
├── main.py                  # CLI entry point
├── xray_class/              # X-ray classification pipeline
│   ├── config.py            # Labels, paths, constants
│   ├── data.py              # kagglehub download + ChestXray14 dataset
│   ├── model.py             # DenseNet121 training, eval, embeddings
│   └── pipeline.py          # Orchestrator (sections 1–5)
└── dv/                      # Data valuation methods (reusable)
    ├── shap/knn_shapley.py  # KNN-Shapley (Jia et al. 2019)
    └── ot/sinkhorn.py       # Sinkhorn OT (POT library)
```
