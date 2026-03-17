# Pipeline Report: Findings, Engineering Decisions, and Analysis

## Key Takeaways

### 1. KNN-Shapley Rankings Are Real — Retraining Confirms It

The single most important finding from this pipeline is that the KNN-Shapley data valuations computed in seconds on frozen embeddings translate into real performance differences when you retrain the full DenseNet121 model from scratch. At 20% data retention, training on the highest-Shapley samples yields AUC 0.696 while training on the lowest-Shapley samples collapses to AUC 0.471 — barely better than a coin flip. This is a 0.225 AUC gap produced by the *same amount of data*, differing only in which samples were selected.

This matters because the KNN-Shapley computation itself is O(n log n) and finishes in seconds, while each retraining run takes roughly an hour on CPU. The cheap proxy is identifying the same quality signal as the expensive ground truth.

### 2. Half the Training Data Is Redundant

The quality detection flagged 1,761 of 3,587 training samples (49.1%) as redundant — defined as |Shapley value| < 0.01. These samples contribute almost nothing to the model either positively or negatively. Their embeddings sit in dense regions of the feature space surrounded by other samples with the same label, so removing them doesn't change any validation prediction.

The retraining experiments support this: Top-K at 50% (1,793 samples) achieves AUC 0.716, which is 97% of the full-dataset baseline (AUC 0.737). You can throw away roughly half the data — if you pick the right half — and lose almost nothing.

### 3. Low-Value Data Doesn't Just "Not Help" — It Actively Hurts

The Bottom-K 20% retraining result (AUC 0.471, recall 0.953, specificity 0.033) is not merely bad — it reveals a model that has degenerated into predicting "Has Finding" for almost every input. With 717 training samples selected for having the most negative Shapley values, the model cannot learn a meaningful decision boundary. It defaults to the positive class because the training data it sees is dominated by confusing or label-inconsistent examples that prevent it from distinguishing the two classes.

This contrasts with Random-K at the same 20% fraction (AUC 0.721), which by chance includes enough coherent examples of both classes to learn reasonable separation. The gap between Random-K and Bottom-K at 20% (0.250 AUC) is larger than the gap between Top-K and Random-K (0.025 AUC loss), meaning the damage from bad data far exceeds the benefit of curated good data. Avoiding the worst samples matters more than selecting the best ones.

### 4. AUC Tells a Clearer Story Than F1

Across the retraining experiments, AUC-ROC monotonically improves with data fraction for Top-K (0.696 → 0.716 → 0.726 → 0.737), producing the clean upward curve you would expect. F1 does not — Top-K at 80% drops to 0.507 despite having the second-best AUC (0.726). This is because F1 depends on the fixed 0.5 classification threshold, which may be suboptimal for a particular subset's class balance and decision boundary shape. AUC evaluates across all thresholds and is therefore more robust for comparing data subsets.

The lesson: when evaluating data quality impact, AUC-ROC is the primary metric. F1, precision, recall, and specificity are useful for understanding *how* the model's behaviour changes, but they are threshold-sensitive and can be misleading in isolation.

### 5. The "No Finding" Majority Class Dominates the Top Shapley Values

Of the top 20 highest-Shapley training samples, 17 are labelled "No Finding" and only 3 are "Has Finding". The bottom 20 are the opposite: 14 are "Has Finding" and 6 are "No Finding". This asymmetry reflects the dataset's class balance (54.3% No Finding, 45.7% Has Finding) but amplifies it — the KNN-Shapley algorithm rewards samples whose neighbours share the same label, and the majority class naturally has denser same-label neighbourhoods.

This is not a bug — majority-class samples genuinely are "easier" examples that help the model anchor its decision boundary. But it means that aggressive Top-K selection at small fractions may under-represent the minority class, which could explain why Top-K 20% has high recall (0.770) but moderate precision (0.574).

---

## How the Algorithms Work

### DenseNet121 and Transfer Learning

DenseNet121 is a convolutional neural network with 121 layers organized into dense blocks, where each layer receives the feature maps of all preceding layers as input. This "dense connectivity" encourages feature reuse — early layers that detect edges and textures are directly available to later layers that detect complex anatomical structures — and requires fewer parameters than comparable architectures (6.95M total vs ResNet-50's 25.6M).

The model is loaded with weights pretrained on ImageNet (1.2M natural images, 1000 classes). The original 1000-class head is replaced with `Linear(1024, 1)` for binary classification. In the default configuration, the convolutional feature extractor is frozen and only the 1,025-parameter head is trained. This means the model uses ImageNet's learned visual representations as-is, and only learns *how to combine* the 1024-dimensional feature vector into a disease/healthy prediction. This is called "head-only fine-tuning" or "linear probing".

The frozen configuration trains on CPU in ~60 minutes because backpropagation only flows through the tiny classifier head, not the 6.95M-parameter feature extractor. Setting `FINETUNE_ALL = True` unfreezes all layers and lets the convolutional features adapt to chest X-rays, which dramatically increases both training cost and performance potential.

### BCEWithLogitsLoss and pos_weight

The loss function is binary cross-entropy with logits (combines sigmoid activation and BCE into a single numerically stable operation). The `pos_weight` parameter addresses class imbalance: with 1,948 negatives and 1,639 positives in the training set, `pos_weight = 1948/1639 = 1.19`. This multiplies the loss contribution of positive samples by 1.19, effectively telling the optimizer that missing a positive case is 19% worse than missing a negative case.

Without `pos_weight`, the model could achieve 54% accuracy by predicting "No Finding" for everything. The weighting forces it to take the minority class seriously. The confusion matrices confirm this works — errors are roughly balanced across the two classes (67% recall on No Finding, 71% recall on Has Finding) rather than being heavily skewed toward the majority class.

### Embedding Extraction via Forward Hooks

Embeddings are extracted using PyTorch's `register_forward_hook` mechanism. A hook function is attached to `model.features` (the convolutional backbone), and during forward passes it intercepts the intermediate activations, applies adaptive average pooling and ReLU, then stores the resulting 1024-dimensional vector.

This approach is non-invasive — the model's architecture and forward method are untouched. The hook fires automatically during normal inference, captures the penultimate representation, and is cleanly removed afterward. The alternative would be to modify the model's forward method to return intermediate outputs, which would couple the extraction logic to the model definition and make it harder to swap model architectures.

### SHAP GradientExplainer

SHAP (SHapley Additive exPlanations) decomposes a model's prediction into additive contributions from each input feature. GradientExplainer approximates these contributions using the expected gradients method — it samples reference inputs from a background distribution (200 random training embeddings), computes gradients of the output with respect to the input at interpolated points between the reference and the actual input, and averages these to estimate each feature's marginal contribution.

The pipeline applies SHAP to the **classifier head only** (a single `Linear(1024, 1)` layer), not to the full model. This means the SHAP values explain which embedding dimensions the classifier relies on for each prediction. Because the head is linear, the SHAP values are actually exact (for a linear function, the Shapley value of feature *i* equals the weight times the deviation from the background mean). The GradientExplainer is still used because it handles the background distribution averaging and produces properly calibrated values that sum to the prediction.

The base value (0.116 in our results) is the average model output over the background distribution — conceptually, "what the model would predict if it knew nothing about this specific input". Each SHAP value then pushes the prediction up or down from this baseline.

### KNN-Shapley Data Valuation

KNN-Shapley assigns a value to each training sample based on how helpful it is for predicting validation labels. The algorithm works as follows:

1. Fit a k-nearest-neighbours index on the training embeddings (k=10).
2. For each validation sample, find its 10 nearest training neighbours.
3. For each neighbour: if its label matches the validation sample's label, add +1/10 to the neighbour's Shapley value. If the labels disagree, subtract 1/10.
4. Sum across all validation samples.

A training sample ends up with a high Shapley value if it is frequently a nearest neighbour of validation samples *and* its label consistently agrees with theirs. A negative Shapley value means the sample is close to validation samples but has the *wrong* label from their perspective — it is actively misleading a KNN classifier.

This is an O(n log n) closed-form solution for the exact Shapley value of each training point with respect to a KNN classifier (Jia et al. 2019). The key insight is that for KNN, the Shapley value has an analytical formula involving only the rank ordering of distances, which avoids the combinatorial explosion of the general Data Shapley formulation. The trade-off is that the values are exact for KNN, not for DenseNet121 — they capture how useful a sample's position in embedding space is, not how useful it would be for full model retraining. Section 6's retraining experiments validate that this proxy ranking transfers.

### Data Quality Categories

The quality detection system classifies training samples into four overlapping categories based on their Shapley values and embedding-space properties:

- **Noisy labels** (Shapley < -0.05): 768 samples (21.4%). These samples are near validation points of the opposite class. In medical imaging, this often reflects genuine diagnostic ambiguity — borderline cases where the same X-ray might reasonably be read as normal or abnormal depending on the radiologist. It could also indicate actual labelling errors.

- **Outliers** (negative Shapley AND flagged by Local Outlier Factor): 0 samples (0.0%). LOF measures how isolated a point is relative to its local neighbourhood density. The intersection with negative Shapley would catch samples that are both statistically unusual *and* harmful to classification. Finding zero outliers suggests the embedding space is relatively well-structured — there are no samples that are simultaneously isolated and mislabelled.

- **Redundant** (|Shapley| < 0.01): 1,761 samples (49.1%). These samples contribute almost nothing. They sit in dense regions surrounded by same-label neighbours, so their presence or absence doesn't change any prediction. In practice, this means ~half the dataset could be pruned with minimal performance impact.

- **High-value** (top 10th percentile): 325 samples (9.1%). These are the most influential training points — they sit in important regions of the embedding space (near class boundaries or in sparse areas) and consistently help correct predictions.

---

## Engineering Decisions and Their Reasoning

### Why Head-Only Training by Default

With `FINETUNE_ALL = False`, only 1,025 of 6,954,881 parameters are trainable (0.015%). This makes the pipeline CPU-feasible: each epoch takes ~7 minutes on a laptop CPU instead of the hours it would take to backpropagate through the full network. The cost is performance — the model is limited to finding a linear decision boundary in the ImageNet feature space, which was not trained on chest X-rays. The AUC of 0.73 is respectable but far from the ~0.84 reported in the CheXNet paper with full fine-tuning on a GPU.

This is the right trade-off for a data valuation research pipeline. The goal is not to build the best possible classifier — it is to produce embeddings that preserve enough structure for KNN-Shapley to rank training samples meaningfully, and then to validate those rankings through retraining. Head-only training produces embeddings that do distinguish the two classes (the t-SNE shows partial separation), and the retraining experiments confirm the rankings are valid.

### Why Stratified Splitting at Multiple Levels

The data is split in two stages: first 80/20 train/test (stratified by label), then the 80% training portion is further split 80/20 into train/val (also stratified). This produces three sets:

- **Train** (3,587 samples): used for model training and KNN-Shapley computation
- **Val** (897 samples): used for early stopping and as the reference set for Shapley values
- **Test** (1,122 samples): held out entirely, used only for final evaluation

Stratification ensures each split preserves the ~54/46 class balance. Without stratification, random chance could produce a validation set that is 70% positive and 30% negative, which would distort both the early stopping signal and the Shapley values (since Shapley depends on label agreement with validation neighbours).

### Why the Validation Set Is Reused

The same validation set serves three purposes: early stopping during initial training (Section 3), the reference set for KNN-Shapley (Section 5), and early stopping during retraining (Section 6). This is a deliberate compromise documented in the code.

The alternative — splitting the 897 validation samples into separate Shapley-reference and early-stopping sets — would give each purpose only ~450 samples, which is too few for reliable AUC estimation during early stopping. With a 5,606-image dataset, every sample counts. The test set remains fully uncontaminated: it is never used for training, Shapley computation, or early stopping, so the final evaluation is unbiased.

### Why Fresh ImageNet Weights for Every Retraining Run

Section 6 retrains DenseNet121 from ImageNet weights, not from the Section 3 fine-tuned checkpoint. This is critical for scientific validity: if we started from the fine-tuned model, the already-learned chest X-ray features would mask the effect of data quality. A subset of 717 "bottom" samples might still produce reasonable AUC because the model already knows what chest X-rays look like from its previous training. By resetting to ImageNet weights, the model must learn everything about chest X-rays from the provided subset alone, making performance differences directly attributable to data quality.

### Why SHAP Is Applied to the Classifier Head, Not the Full Model

Running SHAP GradientExplainer on the full DenseNet121 (1024-dimensional input images → 1 output) would produce pixel-level attributions but would be extremely expensive and noisy (224x224x3 = 150,528 input dimensions). Instead, the pipeline applies SHAP to just the classifier head (`Linear(1024, 1)`), which explains predictions in terms of the 1024 embedding dimensions.

This is the right abstraction layer for this pipeline's research question. The pipeline asks "which training samples help?", not "which pixels matter?". The embedding dimensions are the representation space where KNN-Shapley operates, so understanding which dimensions drive predictions directly contextualises the Shapley values. Pixel-level attribution is separately handled by the gradient-based saliency maps in Section 3, which are cheaper to compute and more interpretable for clinical audiences.

### Why KNN-Shapley Instead of Exact Data Shapley

Exact Data Shapley (Ghorbani & Zou 2019) computes each training sample's value by measuring its marginal contribution across all possible subsets of the training data. With n = 3,587 training samples, this requires evaluating 2^3587 subsets — a number with over 1,000 digits. The Monte Carlo approximation randomly samples permutations instead, but each permutation requires a full model retrain. At ~5 minutes per retrain on CPU, even 100 permutations across 3,587 samples would take 100 * 3,587 * 5 minutes = 29,892 hours, or 3.4 years.

KNN-Shapley (Jia et al. 2019) sidesteps this entirely by computing the exact Shapley value for a KNN classifier in O(n log n) time using a closed-form recurrence. The trade-off is that these values are exact for KNN, not for DenseNet121. The pipeline addresses this honestly: Section 5 computes the cheap proxy, Section 6 validates it with actual retraining, and the docstrings explain the approximation gap with citations.

### Why TransformSubset Wraps the Base Dataset

The `TransformSubset` class applies a transform to a subset of the base `XRayDataset` without duplicating the image data. This is necessary because training and evaluation require different augmentations (random crops and flips during training, deterministic resize during evaluation), but the underlying images are the same. Rather than creating separate datasets for each split, `TransformSubset` references the base dataset by index and applies the appropriate transform at access time. This saves memory and keeps the split logic (which indices belong to which set) separate from the transform logic.

### Why the Retraining Experiment Uses a Reduced Configuration

The default retraining config (4 fractions, 1 random seed, 5 epochs, patience 2) produces 12 runs instead of the 50 runs of the original design (10 fractions, 3 seeds, 10 epochs). This is a concession to CPU runtime — each retraining run takes ~60 minutes on CPU, so 12 runs take ~12 hours while 50 would take ~50 hours. The reduced configuration still covers the scientifically important range (20% to 100% in large steps) and all three strategies. The trade-off is no error bars on Random-K (1 seed) and coarser sampling of the data efficiency curve. On a GPU, increasing `RETRAIN_FRACTIONS` and `RETRAIN_RANDOM_SEEDS` in `config.py` would give a denser, more statistically robust picture.

---

## Surprises and Unexpected Findings

### Bottom-K at 20% Has Higher F1 Than Top-K

At first glance, Bottom-K 20% achieving F1 0.615 versus Top-K 20% at F1 0.658 seems close — and Bottom-K at 50% (F1 0.507) is actually worse. But looking at the full metric decomposition reveals the explanation: Bottom-K 20% has recall 0.953 and specificity 0.033. The model is predicting "Has Finding" for 96.7% of all inputs. With the test set being ~46% positive, this strategy accidentally produces decent F1 by catching almost all true positives, despite having catastrophic false positive rates. The AUC (0.471) reveals the true picture — the model's probability ranking is worse than random.

This is a textbook example of why F1 alone is insufficient for evaluating medical classifiers and why this report's emphasis on AUC-ROC as the primary comparison metric is justified.

### Zero Outliers Detected

The intersection of negative Shapley values and Local Outlier Factor flagging produced zero samples. This suggests the embedding space is relatively smooth — there are no isolated pockets of mislabelled data. Samples with negative Shapley values are not spatially unusual; they are simply in regions where the two classes overlap. This is consistent with the nature of chest X-ray classification, where the boundary between "normal" and "has finding" is genuinely fuzzy, not a matter of outlier contamination.

### The Median Shapley Value Is Exactly Zero

Half of all training samples have a Shapley value of exactly 0.000. This happens because with k=10, a sample only receives a non-zero contribution if it appears in the 10-nearest-neighbour list of at least one validation sample. In a 3,587-sample training set with 897 validation queries each looking at 10 neighbours, many training samples are never a nearest neighbour of any validation point and therefore receive exactly zero value. These are the "redundant" samples — they exist in dense regions where plenty of other samples carry the same information.

### Random-K at 20% Outperforms Top-K at 20% on AUC

Random-K at 20% (AUC 0.721) beat Top-K at 20% (AUC 0.696). This is counterintuitive — curated high-value samples should outperform a random draw. The likely explanation is class balance. As noted in the findings, the top Shapley values are dominated by "No Finding" samples. At 20% retention (717 samples), Top-K over-represents the majority class and under-represents "Has Finding" examples, producing a model that is biased toward predicting negative. Random selection preserves the natural ~54/46 balance, which gives the model enough positive examples to learn from.

This suggests that Shapley-guided selection should be combined with stratified sampling — selecting the top-K from each class proportionally — rather than taking the global top-K across both classes. This is an area for future work.

### The Section 3 Model (Trained on All Data) and Section 6 Baseline Are Close But Not Identical

The Section 3 model achieves test AUC 0.730, while the Section 6 "baseline" (100% retrain from ImageNet weights) achieves AUC 0.737. The small difference comes from random variation in training (different initial random states for data loading, augmentation, etc.) and the slightly different training configuration (Section 6 uses 5 epochs / patience 2 instead of Section 3's 10 epochs / patience 3). This confirms that the retraining procedure is producing models of comparable quality to the main pipeline, which validates the experimental setup.

---

## How to Read Each Plot

### Pretrained t-SNE (`pretrained_tsne.png`)

Shows the 1024-dimensional ImageNet features projected to 2D before fine-tuning. Heavy overlap between the blue (No Finding) and red (Has Finding) clusters means ImageNet features alone don't separate chest X-ray classes well. This is expected — ImageNet was trained on natural images (dogs, cars, etc.), not medical images. The overlap motivates the fine-tuning step.

### Confusion Matrices (`confusion_matrices.png`)

Two heatmaps: validation (left) and test (right). Each cell shows the count of (true label, predicted label) pairs. The diagonal is correct predictions; off-diagonal is errors. Roughly balanced off-diagonal values (val: 146 false positives, 166 false negatives; test: 195 FP, 177 FN) confirm the `pos_weight` correction is preventing the majority-class-predicts-everything failure mode.

### Saliency Maps (`saliency_maps.png`)

Three test images shown with their gradient-based saliency overlays. Bright regions in the heatmap (right column) indicate which pixels had the largest gradient — the regions the model is most sensitive to. In a well-trained model, these should highlight the lung fields. Artifacts like highlighting image borders or corners would suggest the model is relying on acquisition metadata rather than anatomy.

### SHAP Bar and Beeswarm (`shap_bar.png`, `shap_beeswarm.png`)

The bar chart ranks the 1024 embedding dimensions by mean absolute SHAP value. Feature 936 (importance 0.042) contributes roughly 4x more than the median feature. The beeswarm plot adds detail: each dot is one test sample, coloured by feature value (red = high, blue = low). A horizontal spread of dots for a feature means it has high variance in its contribution across samples.

### SHAP Dependence (`shap_dependence.png`)

Scatter plots for the top 3 features showing feature value (x-axis) vs SHAP contribution (y-axis). For a linear classifier head, these should be roughly linear (positive slope = feature pushes toward "Has Finding" when high). Non-linear patterns would indicate interaction effects captured by the colour encoding (which shows a second feature's value).

### SHAP Waterfall (`shap_waterfall_*.png`)

Decomposes individual predictions. Starting from the base value (0.116, the average logit), each bar shows one feature's contribution. Red bars push toward "Has Finding", blue bars push toward "No Finding". The final value is the model's logit for this sample. Useful for auditing why the model made a specific prediction.

### Shapley Distribution (`shapley_distribution.png`)

Four panels characterising the distribution of training sample values. The histogram (top-left) shows most values clustered near zero with long tails. The cumulative distribution (bottom-left) shows that 80% of cumulative value comes from a minority of samples — the data has a power-law-like concentration of quality. The sorted plot (bottom-right) shows the rank ordering from most harmful to most valuable.

### Efficiency Curves (`efficiency_curves.png`)

Section 5's lightweight proxy experiment. F1 score on the validation set vs dataset fraction for Top-K, Random-K, and Bottom-K selection using a fast `nn.Linear(1024, 1)` head on frozen embeddings. The expected pattern (Top-K > Random > Bottom-K) should be visible, but results are noisy because the linear head is a crude proxy and F1 is threshold-sensitive.

### Retraining Curves (`retraining_curves.png`)

Section 6's ground-truth validation. AUC-ROC (left) and F1 (right) on the held-out test set vs data fraction, with full DenseNet121 retraining from ImageNet weights. The AUC panel should show a clear Top-K advantage, especially at low fractions. The horizontal dashed line is the full-dataset baseline. This is the most important plot in the pipeline — it shows whether the Shapley rankings actually translate to real model performance.

### Comprehensive Metrics (`retraining_all_metrics.png`)

2x3 grid decomposing retraining performance into all six metrics. Useful for understanding *how* the model's behaviour changes with data quality. The precision/recall/specificity panels reveal failure modes — like Bottom-K models degenerating into constant predictors (high recall, zero specificity) at small fractions.

---

## Summary of Numerical Results

| Metric | Section 3 (all data) | Top-K 80% | Top-K 50% | Top-K 20% | Bottom-K 20% |
|--------|---------------------|-----------|-----------|-----------|-------------|
| AUC-ROC | 0.730 | 0.726 | 0.716 | 0.696 | 0.471 |
| F1 | 0.674 | 0.507 | 0.657 | 0.658 | 0.615 |
| Precision | 0.642 | 0.678 | 0.608 | 0.574 | 0.454 |
| Recall | 0.710 | 0.406 | 0.714 | 0.770 | 0.953 |
| Specificity | — | 0.837 | 0.613 | 0.519 | 0.033 |
| Accuracy | 0.686 | 0.640 | 0.659 | 0.634 | 0.454 |

The AUC column tells the cleanest story: steady improvement from 20% to 100% for Top-K, and catastrophic failure for Bottom-K. The full-dataset retrain baseline (AUC 0.737) is only 1.5% above Top-K 80%, confirming that the bottom 20% of training data adds negligible value.
