# Core imports
import os
import argparse
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import random
import glob
import math
import torch
import torch.nn as nn
import ot
import warnings
import pandas as pd
from codecarbon import OfflineEmissionsTracker
warnings.filterwarnings('ignore')

# Global variables
BATCH_SIZE = 32
INPUT_SIZE = (128, 128)
EMP_RATIO = 0.0  # Keep no empty slices for training

# Parse command-line arguments to select task, modality, and DV method
parser = argparse.ArgumentParser(description="BRATS pipeline: segmentation or classification")

parser.add_argument("--data_path", type=str, required=True, help="Path to sliced dataset directory")
parser.add_argument("--task", type=str, required=True, choices=["segmentation", "classification"], help="Task to run: segmentation or classification")
parser.add_argument("--modality", type=str, required=True, choices=["MRI", "X-ray"], help="Modality of data: MRI or X-ray")
parser.add_argument("--DV", type=str, required=True, choices=["Shap", "OT"], help="DV method to run: Shapely or Optimal Transport")
parser.add_argument("--out_dir", type=str, default="otmri_results", help="Directory to save OT outputs and subset results")

# Training
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for TensorFlow pipelines and subset retraining")
parser.add_argument("--num_workers", type=int, default=8, help="Reserved for data loading parallelism")
parser.add_argument("--early_stop_patience", type=int, default=5, help="Early stopping patience based on validation Dice")

parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--ot_reg", type=float, default=0.01, help="Sinkhorn regularization used by OT")
parser.add_argument("--pretrained_path", type=str, default="", help="Optional path to a pretrained checkpoint")

args = parser.parse_args()

def reset_output_dir(path):
    """Remove and recreate a directory so each run starts with clean outputs."""
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

data_path = args.data_path
task = args.task
modality = args.modality
DV = args.DV
out_dir = args.out_dir

os.makedirs(out_dir, exist_ok=True)
plots_dir = os.path.join(out_dir, "plots")
ot_dir = os.path.join(out_dir, "OT")
emissions_dir = os.path.join(out_dir, "emissions")
best_model_path = os.path.join(out_dir, "best_binary_unet.h5")

reset_output_dir(plots_dir)
reset_output_dir(ot_dir)
reset_output_dir(emissions_dir)

print(f"Data:   {data_path}")
print(f"Task selected:   {task}")
print(f"Modality selected:   {modality}")
print(f"DV method selected:   {DV}")

# Set random seeds for reproducibility
BATCH_SIZE = args.batch_size
INPUT_SIZE = (128, 128)
SEED = args.seed
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"

# =========================
# 1) HELPER FUNCTIONS
# =========================

def count_total_slices(folder_path, tumor_only=True):
    """Count one slice per .npz file to match 'one slice per sample' strategy"""
    return len(glob.glob(os.path.join(folder_path, "*.npz")))

def patient_slice_generator(folder_path, tumor_only=True, empty_ratio=0.0):
    patient_files = sorted(glob.glob(os.path.join(folder_path, "*.npz")))
    while True:
        random.shuffle(patient_files)
        for f in patient_files:
            try:
                with np.load(f) as data:
                    images = data['images']
                    masks = data['masks']

                # Strategy: Select the slice with the largest tumor area
                mask_sums = np.sum(masks, axis=(1, 2))
                if tumor_only and np.max(mask_sums) > 0:
                    idx = np.argmax(mask_sums)
                else:
                    idx = len(images) // 2  # Fallback to middle slice

                img_slice = np.squeeze(images[idx]).astype(np.float32)
                # Min-Max Normalization: [0, 1]
                img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice) + 1e-7)

                binary_mask = (np.squeeze(masks[idx]) > 0).astype(np.float32)

                yield np.expand_dims(img_slice, axis=-1), np.expand_dims(binary_mask, axis=-1)
            except Exception:
                continue

def build_unet(input_shape=(128, 128, 1)):
    def conv_block(x, filters, dropout=0.1):
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        return x

    inputs = layers.Input(shape=input_shape)
    # Encoder
    c1 = conv_block(inputs, 16); p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = conv_block(p1, 32); p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = conv_block(p2, 64); p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = conv_block(p3, 128)
    c4 = layers.Identity(name="bottleneck")(c4)

    # Decoder
    u5 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c3]); c5 = conv_block(u5, 64)
    u6 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2]); c6 = conv_block(u6, 32)
    u7 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1]); c7 = conv_block(u7, 16)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)
    model = models.Model(inputs=inputs, outputs=outputs)
    bottleneck_model = models.Model(inputs=inputs, outputs=c4)
    return model, bottleneck_model

def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def build_emissions_tracker(run_name):
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in run_name)
    return OfflineEmissionsTracker(
        country_iso_code="CAN",
        region="British Columbia",
        output_dir=emissions_dir,
        output_file=f"emissions_{safe_name}.csv"
    )

# =========================
# 2) DATASET PREPARATION
# =========================

train_path = os.path.join(data_path, "train")
val_path = os.path.join(data_path, "validation")
test_path = os.path.join(data_path, "test")

total_train = count_total_slices(train_path)
total_val = count_total_slices(val_path)

train_dataset = tf.data.Dataset.from_generator(
    lambda: patient_slice_generator(train_path),
    output_signature=(tf.TensorSpec(shape=(128,128,1), dtype=tf.float32),
                      tf.TensorSpec(shape=(128,128,1), dtype=tf.float32))
).repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: patient_slice_generator(val_path),
    output_signature=(tf.TensorSpec(shape=(128,128,1), dtype=tf.float32),
                      tf.TensorSpec(shape=(128,128,1), dtype=tf.float32))
).repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def load_full_dataset(folder_path):
    X_list, Y_list = [], []
    files = sorted(glob.glob(os.path.join(folder_path, "*.npz")))

    for f in files:
        with np.load(f) as data:
            images = data['images']
            masks = data['masks']

        mask_sums = np.sum(masks, axis=(1, 2))
        idx = np.argmax(mask_sums) if np.max(mask_sums) > 0 else len(images)//2

        img = np.squeeze(images[idx]).astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)
        mask = (np.squeeze(masks[idx]) > 0).astype(np.float32)

        X_list.append(np.expand_dims(img, axis=-1))
        Y_list.append(np.expand_dims(mask, axis=-1))

    filenames = [os.path.basename(f) for f in files]
    return np.array(X_list), np.array(Y_list), filenames

X_train_full, Y_train_full, train_filenames = load_full_dataset(train_path)
X_val_full, Y_val_full, _ = load_full_dataset(val_path)

# =========================
# 3) FULL MODEL TRAINING + EVALUATION
# =========================

model, feature_model = build_unet()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss='binary_crossentropy',
    metrics=[dice_coef, 'accuracy']
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    best_model_path,
    monitor="val_dice_coef",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_dice_coef',
    patience=args.early_stop_patience,
    min_delta=0.001,
    mode="max",
    restore_best_weights=True,
    verbose=1
)

lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_dice_coef',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    mode='max',
    verbose=1
)

full_train_tracker = build_emissions_tracker("full_training")
full_train_tracker.start()

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=args.epochs,
    steps_per_epoch=math.ceil(total_train / BATCH_SIZE),
    validation_steps=math.ceil(total_val / BATCH_SIZE),
    callbacks=[checkpoint, early_stop, lr_reducer],
)

full_training_emissions = full_train_tracker.stop()
print(f"[EMISSIONS] full_training: {full_training_emissions:.6f} kg CO2eq")

def plot_training_history(history, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title('Loss'); ax1.legend()

    ax2.plot(history.history['dice_coef'], label='Train Dice')
    ax2.plot(history.history['val_dice_coef'], label='Val Dice')
    ax2.set_title('Dice Coefficient'); ax2.legend()

    plt.savefig(os.path.join(out_dir, "training_curves.png"))
    plt.close()

plot_training_history(history, out_dir=plots_dir)

model.load_weights(best_model_path)
print("Loaded best model for final evaluation.")

# --- TEST EVALUATION (FULL MODEL) ---
total_test_slices = count_total_slices(test_path)

test_dataset = tf.data.Dataset.from_generator(
    lambda: patient_slice_generator(test_path),
    output_signature=(
        tf.TensorSpec(shape=(128,128,1), dtype=tf.float32),
        tf.TensorSpec(shape=(128,128,1), dtype=tf.float32)
    )
).batch(BATCH_SIZE)

results = model.evaluate(
    test_dataset,
    steps=math.ceil(total_test_slices/BATCH_SIZE)
)

print(f"\nFull Model → Test Loss: {results[0]:.4f}, Test Dice: {results[1]:.4f}")

def save_binary_predictions(model, dataset, out_dir="plots", n=5):
    os.makedirs(out_dir, exist_ok=True)

    for X, Y in dataset.take(1):
        preds = model.predict(X[:n])

        for i in range(n):
            plt.figure(figsize=(12,4))

            plt.subplot(1,3,1)
            plt.imshow(X[i,:,:,0], cmap='gray')
            plt.title("Input")

            plt.subplot(1,3,2)
            plt.imshow(Y[i,:,:,0], cmap='viridis')
            plt.title("Ground Truth")

            plt.subplot(1,3,3)
            plt.imshow(preds[i,:,:,0], cmap='viridis')
            plt.title("Prediction")

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"sample_{i}.png"))
            plt.close()

save_binary_predictions(model, val_dataset, out_dir=plots_dir)

# =========================
# 4) OT SCORING + VISUALIZATION
# =========================

def plot_hist(values, out_path, bins=50):
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title("Histogram of OT Values (Train Samples)")
    plt.xlabel("OT value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# def get_embeddings_np(X, Y, embedding_model, batch_size=16):
#     feats = embedding_model.predict(X, batch_size=batch_size, verbose=0)
#     feats = feats.reshape(feats.shape[0], -1)
#     tumor_present = (Y.sum(axis=(1, 2, 3)) > 0).astype(np.float32)
#     return feats, tumor_present

def get_embeddings_np(X, Y, embedding_model, batch_size=16):
    # Get bottleneck feature maps from the trained U-Net bottleneck model
    feats = embedding_model.predict(X, batch_size=batch_size, verbose=0)
    # feats shape: (N, H, W, C)

    # Global average pooling across spatial dimensions H and W
    feats = feats.mean(axis=(1, 2))
    # pooled feats shape: (N, C)

    tumor_present = (Y.sum(axis=(1, 2, 3)) > 0).astype(np.float32)
    return feats, tumor_present


def ot_binary_row_normalized(train_feats, train_y, val_feats, val_y, reg=0.01, eps=1e-12):
    train_feats = nn.functional.normalize(train_feats, dim=1)
    val_feats = nn.functional.normalize(val_feats, dim=1)

    C = 1.0 - (train_feats @ val_feats.T)
    R = (train_y.round().unsqueeze(1) == val_y.round().unsqueeze(0)).float()

    n_train = train_feats.size(0)
    n_val = val_feats.size(0)
    a = np.ones(n_train, dtype=np.float64) / n_train
    b = np.ones(n_val, dtype=np.float64) / n_val

    C_np = C.detach().cpu().numpy().astype(np.float64)
    P = ot.sinkhorn(a, b, C_np, reg)
    P = torch.tensor(P, dtype=torch.float32)

    row_sums = P.sum(dim=1, keepdim=True).clamp_min(eps)
    P_row = P / row_sums
    row_cost = (P_row * C.detach().cpu()).sum(dim=1)

    train_unique = int(torch.unique(train_y.round()).numel())
    val_unique = int(torch.unique(val_y.round()).numel())

    if train_unique > 1 and val_unique > 1:
        scores = (P_row * R.detach().cpu()).sum(dim=1)
    else:
        cmin = row_cost.min()
        cmax = row_cost.max()
        scores = 1.0 - (row_cost - cmin) / (cmax - cmin + eps)

    return scores, row_cost

print("[INFO] Extracting bottleneck embeddings for OT...")
train_feats_np, train_y_np = get_embeddings_np(X_train_full, Y_train_full, feature_model, batch_size=16)
val_feats_np, val_y_np = get_embeddings_np(X_val_full, Y_val_full, feature_model, batch_size=16)

train_feats = torch.tensor(train_feats_np, dtype=torch.float32)
val_feats = torch.tensor(val_feats_np, dtype=torch.float32)
train_y = torch.tensor(train_y_np, dtype=torch.float32)
val_y = torch.tensor(val_y_np, dtype=torch.float32)

print("[INFO] Computing OT scores...")
ot_values, ot_row_cost = ot_binary_row_normalized(
    train_feats,
    train_y,
    val_feats,
    val_y,
    reg=args.ot_reg,
)
v = ot_values.numpy()
row_cost_np = ot_row_cost.numpy()
sorted_idx = np.argsort(v)[::-1]

np.save(os.path.join(ot_dir, "ot_values.npy"), v)
plot_hist(v, os.path.join(plots_dir, "ot_histogram.png"))

pd.DataFrame({"OT_value": v.astype(np.float64)}).to_csv(os.path.join(ot_dir, "ot_values.csv"), index=False)

df_ot = pd.DataFrame({
    "ImageID": train_filenames,
    "OT_value": v.astype(np.float64),
    "OT_similarity": v.astype(np.float64),
    "OT_row_cost": row_cost_np.astype(np.float64),
})
df_ot.to_csv(os.path.join(ot_dir, "ot_values_with_ids.csv"), index=False)

k50 = min(50, len(v))
order_asc = np.argsort(v)
bottom_idx = order_asc[:k50]
top_idx = order_asc[-k50:][::-1]

np.save(os.path.join(ot_dir, "bottom50_indices.npy"), bottom_idx)
np.save(os.path.join(ot_dir, "top50_indices.npy"), top_idx)
pd.DataFrame({"Bottom50": [train_filenames[i] for i in bottom_idx]}).to_csv(os.path.join(ot_dir, "bottom50_filenames.csv"), index=False)
pd.DataFrame({"Top50": [train_filenames[i] for i in top_idx]}).to_csv(os.path.join(ot_dir, "top50_filenames.csv"), index=False)

def plot_ot_extremes(X, Y, ot_vals, sorted_indices, out_dir="plots", top_k=5):
    os.makedirs(out_dir, exist_ok=True)
    ot_norm = (ot_vals - ot_vals.min()) / (ot_vals.max() - ot_vals.min() + 1e-8)
    top_indices = sorted_indices[:top_k]
    bottom_indices = sorted_indices[-top_k:]

    def plot_group(indices, tag):
        for rank, idx in enumerate(indices):
            plt.figure(figsize=(8,4))
            plt.subplot(1,2,1)
            plt.imshow(X[idx,:,:,0], cmap="gray")
            plt.title("Input")
            plt.subplot(1,2,2)
            plt.imshow(Y[idx,:,:,0], cmap="viridis")
            plt.title("Ground Truth")
            plt.suptitle(f"{tag} Sample #{rank+1}\nOT Value: {ot_norm[idx]:.4f}", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{tag.lower()}_{rank+1}_val_{ot_norm[idx]:.4f}.png"))
            plt.close()

    plot_group(top_indices, "Top")
    plot_group(bottom_indices, "Bottom")

plot_ot_extremes(X_train_full, Y_train_full, v, sorted_idx, out_dir=plots_dir)

# =========================
# 5) OT SUBSET EXPERIMENTS
# =========================

# Percentages of data to use for Top-K / Bottom-K OT subset retraining
percentages = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
n = len(sorted_idx)

def train_and_evaluate(X_train, Y_train, tag="subset"):
    sub_model, _ = build_unet()
    steps_per_epoch = math.ceil(len(X_train) / BATCH_SIZE)
    validation_steps = math.ceil(total_val / BATCH_SIZE)

    sub_model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss='binary_crossentropy',
        metrics=[dice_coef]
    )

    early_stop_sub = tf.keras.callbacks.EarlyStopping(
        monitor='val_dice_coef',
        patience=args.early_stop_patience,
        min_delta=0.001,
        mode='max',
        restore_best_weights=True,
        verbose=0
    )

    lr_reducer_sub = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_dice_coef',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        mode='max',
        verbose=1
    )

    subset_best_path = os.path.join(out_dir, f"best_binary_unet_{tag}.h5")
    checkpoint_sub = tf.keras.callbacks.ModelCheckpoint(
        subset_best_path,
        monitor="val_dice_coef",
        mode="max",
        save_best_only=True,
        verbose=1
    )

    subset_train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    run_tracker = build_emissions_tracker(tag)
    run_tracker.start()

    sub_model.fit(
        subset_train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        steps_per_epoch=math.ceil(len(X_train) / BATCH_SIZE),
        validation_steps=math.ceil(total_val / BATCH_SIZE),
        callbacks=[checkpoint_sub, early_stop_sub, lr_reducer_sub],
    )

    sub_model.load_weights(subset_best_path)

    run_emissions = run_tracker.stop()
    print(f"[EMISSIONS] {tag}: {run_emissions:.6f} kg CO2eq")

    res = sub_model.evaluate(
        test_dataset,
        steps=math.ceil(total_test_slices/BATCH_SIZE),
        verbose=0
    )

    return res[1], run_emissions

# Initialize result containers
results_dict = {}          # keyed by fraction kept for TOP-p%
results_bottom_dict = {}   # keyed by fraction kept for BOTTOM-p%
ot_emissions_runtime = {}
bottom_emissions_runtime = {}

for p in percentages:
    k_samples = int(n * p)

    if p == 1.0:
        print(f"\nUsing PRE-TRAINED FULL MODEL results for 100% data")
        full_test_dice = results[1]
        results_dict[p] = full_test_dice
        results_bottom_dict[p] = full_test_dice
        ot_emissions_runtime[p] = full_training_emissions
        bottom_emissions_runtime[p] = full_training_emissions
        continue

    shap_idx = sorted_idx[:k_samples]
    print(f"\nTraining with TOP {int(p*100)}% Shapley data ({len(shap_idx)} samples)")
    dice_shap, emissions_shap = train_and_evaluate(
        X_train_full[shap_idx],
        Y_train_full[shap_idx],
        tag=f"shap_top_{int(p*100)}"
    )
    results_dict[p] = dice_shap
    ot_emissions_runtime[p] = emissions_shap

    shap_idx_bottom = sorted_idx[-k_samples:]
    print(f"\nTraining with BOTTOM {int(p*100)}% Shapley data ({len(shap_idx_bottom)} samples)")
    dice_shap_bottom, emissions_shap_bottom = train_and_evaluate(
        X_train_full[shap_idx_bottom],
        Y_train_full[shap_idx_bottom],
        tag=f"shap_bottom_{int(p*100)}"
    )
    results_bottom_dict[p] = dice_shap_bottom
    bottom_emissions_runtime[p] = emissions_shap_bottom

def read_emissions_from_csv(tag, fallback=np.nan):
    csv_path = os.path.join(emissions_dir, f"emissions_{tag}.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] Missing emissions file: {csv_path}. Using fallback value.")
        return float(fallback)
    df_e = pd.read_csv(csv_path)
    if "emissions" not in df_e.columns or len(df_e) == 0:
        print(f"[WARN] Invalid emissions format in: {csv_path}. Using fallback value.")
        return float(fallback)
    return float(df_e["emissions"].iloc[-1])

# --- RESULTS FROM SUBSET RETRAINING ---
print("\n===== OT Data Valuation Results =====")
for p in percentages:
    print(f"Top {int(p*100)}% → Test Dice: {results_dict[p]:.4f}")
for p in percentages:
    print(f"Bottom {int(p*100)}% → Test Dice: {results_bottom_dict[p]:.4f}")

# Prepare reporting arrays
keep_percents = [int(p*100) for p in percentages]

# Read emissions for top and bottom subset experiments
emissions_ot = []
emissions_bottom = []

for pct in keep_percents:
    tag = "full_training" if pct == 100 else f"shap_top_{pct}"
    key = pct / 100.0
    emissions_ot.append(read_emissions_from_csv(tag, fallback=ot_emissions_runtime.get(key, np.nan)))

for pct in keep_percents:
    tag = "full_training" if pct == 100 else f"shap_bottom_{pct}"
    key = pct / 100.0
    emissions_bottom.append(read_emissions_from_csv(tag, fallback=bottom_emissions_runtime.get(key, np.nan)))

# Build dataframe for CSV output
df_subset = pd.DataFrame({
    "KeepPercent": keep_percents,
    "Top_Dice": [results_dict[p/100.0] for p in keep_percents],
    "Top_Emissions_kgCO2eq": emissions_ot,
    "Bottom_Dice": [results_bottom_dict[p/100.0] for p in keep_percents],
    "Bottom_Emissions_kgCO2eq": emissions_bottom,
})
df_subset.to_csv(os.path.join(out_dir, "subset_retraining_results.csv"), index=False)

# --- PLOTTING COMPARISONS ---
# Performance comparison
plt.figure(figsize=(8,6))
plt.plot(keep_percents, [results_dict[p/100.0] for p in keep_percents], marker='o', label='Top-X%')
plt.plot(keep_percents, [results_bottom_dict[p/100.0] for p in keep_percents], marker='s', label='Bottom-X%')
plt.xlabel("Percent of Data Kept")
plt.ylabel("Test Dice")
plt.title("Top-X% vs Bottom-X% Subset Performance")
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(plots_dir, "subset_comparison_ot_vs_remove.png"))
plt.close()

# Emissions comparison
plt.figure(figsize=(8,6))
plt.plot(keep_percents, emissions_ot, marker='o', label='Top-X%')
plt.plot(keep_percents, emissions_bottom, marker='s', label='Bottom-X%')
plt.axhline(y=full_training_emissions, linestyle='--', label='Full Training (100%)')
plt.xlabel("Percent of Data Kept")
plt.ylabel("CO2 Emissions (kg CO2eq)")
plt.title("Emissions Comparison: Top-X% vs Bottom-X%")
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(plots_dir, "subset_comparison_emissions_ot_vs_remove.png"))
plt.close()

# =========================
# 6) FINAL REPORT
# =========================
report_path = os.path.join(args.out_dir, "final_report.txt")
with open(report_path, "w") as f:
    f.write("=== RUN SUMMARY ===\n")
    f.write(f"Device: {device}\n")
    f.write(f"Total test slices: {total_test_slices + total_train + total_val}\n")
    f.write(f"Train/Val/Test sizes: {total_train}/{total_val}/{total_test_slices}\n\n")

    f.write("=== HYPERPARAMETERS ===\n")
    f.write(f"epochs_requested: {args.epochs}\n")
    f.write(f"batch_size: {args.batch_size}\n")
    f.write(f"early_stop_patience (val_dice no-improve): {args.early_stop_patience}\n")
    f.write(f"ot_reg: {args.ot_reg}\n")
    f.write(f"pretrained_path: {args.pretrained_path}\n")
    f.write(f"seed: {args.seed}\n\n")

    best_idx = int(np.argmax(history.history['val_dice_coef']))
    best_epoch = best_idx + 1
    best_train_loss = history.history['loss'][best_idx]
    best_train_dice = history.history['dice_coef'][best_idx]
    best_val_loss = history.history['val_loss'][best_idx]
    best_val_dice = history.history['val_dice_coef'][best_idx]
    full_test_loss = results[0]
    full_test_dice = results[1]

    f.write("=== FULL DATA (100%) BEST MODEL (by VAL DICE) ===\n")
    f.write(f"best_epoch: {best_epoch}\n")
    f.write(f"best_val_dice: {best_val_dice:.6f}\n")
    f.write(f"Train: loss={best_train_loss:.6f}, dice={best_train_dice:.6f}\n")
    f.write(f"Val:   loss={best_val_loss:.6f}, dice={best_val_dice:.6f}\n")
    f.write(f"Test:  loss={full_test_loss:.6f}, dice={full_test_dice:.6f}\n\n")

    f.write("=== OT VALUES SUMMARY (train samples) ===\n")
    f.write(f"OT shape: {v.shape}\n")
    f.write(f"min: {float(v.min()):.10f}\n")
    f.write(f"max: {float(v.max()):.10f}\n")
    f.write(f"mean: {float(v.mean()):.10f}\n")
    f.write(f"std: {float(v.std()):.10f}\n\n")

    f.write("=== TOP 50 OT SAMPLES (TRAIN SET) ===\n")
    f.write("Rank\tOT_value\tImageID\n")
    for rank, idx in enumerate(top_idx, start=1):
        f.write(f"{rank}\t{float(v[idx]):.10f}\t{train_filenames[idx]}\n")

    f.write("\n=== BOTTOM 50 OT SAMPLES (TRAIN SET) ===\n")
    f.write("Rank\tOT_value\tImageID\n")
    for rank, idx in enumerate(bottom_idx, start=1):
        f.write(f"{rank}\t{float(v[idx]):.10f}\t{train_filenames[idx]}\n")

    f.write("\n=== SUBSET RETRAINING RESULTS (Top-X% vs Bottom-X%) ===\n")
    f.write("KeepPercent\tTop_Dice\tTop_Emissions_kgCO2eq\tBottom_Dice\tBottom_Emissions_kgCO2eq\n")
    for i in range(len(keep_percents)):
        kp = keep_percents[i]
        f.write(
            f"{kp}\t{results_dict[kp/100.0]:.6f}\t{emissions_ot[i]:.6f}\t"
            f"{results_bottom_dict[kp/100.0]:.6f}\t{emissions_bottom[i]:.6f}\n"
        )

    f.write("\n=== OUTPUT FILES ===\n")
    f.write(f"Best model checkpoint: {best_model_path}\n")
    f.write(f"Training curves: {os.path.join(plots_dir, 'training_curves.png')}\n")
    f.write(f"Sample predictions + OT extremes dir: {plots_dir}\n")
    f.write(f"OT directory: {ot_dir}\n")
    f.write(f"OT values (csv): {os.path.join(ot_dir, 'ot_values.csv')}\n")
    f.write(f"OT values (npy): {os.path.join(ot_dir, 'ot_values.npy')}\n")
    f.write(f"OT values + IDs (csv): {os.path.join(ot_dir, 'ot_values_with_ids.csv')}\n")
    f.write(f"OT histogram: {os.path.join(plots_dir, 'ot_histogram.png')}\n")
    f.write(f"Bottom-50 indices (npy): {os.path.join(ot_dir, 'bottom50_indices.npy')}\n")
    f.write(f"Top-50 indices (npy): {os.path.join(ot_dir, 'top50_indices.npy')}\n")
    f.write(f"Bottom-50 filenames (csv): {os.path.join(ot_dir, 'bottom50_filenames.csv')}\n")
    f.write(f"Top-50 filenames (csv): {os.path.join(ot_dir, 'top50_filenames.csv')}\n")
    f.write(f"Subset retraining results (csv): {os.path.join(out_dir, 'subset_retraining_results.csv')}\n")
    f.write(f"Subset comparison plot (performance): {os.path.join(plots_dir, 'subset_comparison_ot_vs_remove.png')}\n")
    f.write(f"Subset comparison emissions plot: {os.path.join(plots_dir, 'subset_comparison_emissions_ot_vs_remove.png')}\n")
    f.write(f"Emissions directory: {emissions_dir}\n")
    f.write(f"Full training emissions (csv): {os.path.join(emissions_dir, 'emissions_full_training.csv')}\n")
    f.write(f"Report: {report_path}\n")

print("\n=== DONE ===")
print("Saved outputs to:", args.out_dir)
print("Emissions CSVs saved in:", emissions_dir)
print("Plots saved in:", plots_dir)
print("OT values saved in:", ot_dir)
print("Report:", report_path)
