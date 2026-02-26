import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import nibabel as nib
import random
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm
import math

#Global variables
BATCH_SIZE = 15
INPUT_SIZE = (128, 128)

# Parse command-line arguments to select task, modality, and DV method
parser = argparse.ArgumentParser(
    description="BRATS pipeline: segmentation or classification"
)

parser.add_argument(
    "--data_path",
    type=str,
    required=True,
    help="Path to sliced dataset directory"
)


parser.add_argument(
    "--task",
    type=str,
    required=True,
    choices=["segmentation", "classification"],
    help="Task to run: segmentation or classification"
)

parser.add_argument(
    "--modality",
    type=str,
    required=True,
    choices=["MRI", "X-ray"],
    help="Modality of data: MRI or X-ray"
)

parser.add_argument(
    "--DV",
    type=str,
    required=True,
    choices=["Shap", "OT"],
    help="DV method to run: Shapely or Optimal Transport"
)


args = parser.parse_args()

data_path = args.data_path
task = args.task
modality = args.modality
DV = args.DV

for p, name in [(data_path, "data_path")]:
    if not os.path.isdir(p):
        raise ValueError(f"{name} does not exist or is not a directory: {p}")

print(f"Data:   {data_path}")
print(f"Task selected:   {task}")
print(f"Modality selected:   {modality}")
print(f"DV method selected:   {DV}")


# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Load train and validation cases
def patient_slice_generator(folder_path):
    """
    Loads one patient at a time and yields their slices individually.
    Fixes the missing channel dimension in the mask.
    """
    patient_files = glob.glob(os.path.join(folder_path, "*.npz"))
    
    while True:
        random.shuffle(patient_files)
        
        for f in patient_files:
            try:
                with np.load(f) as data:
                    images = data['images']  # Shape: (146, 128, 128, 1)
                    masks = data['masks']    # Shape: (146, 128, 128)
                
                # Shuffle slices within the patient
                indices = np.arange(len(images))
                np.random.shuffle(indices)
                
                for idx in indices:
                    # Input is already (128, 128, 1)
                    img_slice = images[idx]
                    
                    # Fix: Turn (128, 128) into (128, 128, 1)
                    mask_slice = np.expand_dims(masks[idx], axis=-1)
                    
                    yield img_slice, mask_slice
                    
            except Exception as e:
                print(f"Error loading {f}: {e}")
                continue



# Count total slices to determine steps
def count_total_slices(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.npz"))
    total = 0
    for f in files:
        # We only need to load the metadata to be fast
        with np.load(f) as data:
            total += data['images'].shape[0]
    return total

# Calculate steps
total_train_slices = count_total_slices(os.path.join(data_path, "train"))
total_val_slices = count_total_slices(os.path.join(data_path, "validation"))

steps_per_epoch = math.ceil(total_train_slices / BATCH_SIZE)

validation_steps = math.ceil(total_val_slices / BATCH_SIZE)

print(f"Steps per epoch: {steps_per_epoch}")

# Create Training Dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: patient_slice_generator(os.path.join(data_path, "train")),
    output_signature=(
        tf.TensorSpec(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 1), dtype=tf.float32),
        tf.TensorSpec(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 1), dtype=tf.int32)
    )
)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Create Validation Dataset
val_dataset = tf.data.Dataset.from_generator(
    lambda: patient_slice_generator(os.path.join(data_path, "validation")),
    output_signature=(
        tf.TensorSpec(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 1), dtype=tf.float32),
        tf.TensorSpec(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 1), dtype=tf.int32)
    )
)
val_dataset = val_dataset.repeat()
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# Build U-Net model for segmentation
# ---------------------------
# Build U-Net (2D) model
# ---------------------------
def build_unet(input_shape=(128, 128, 1), num_classes=4):
    """Create a U-Net model for multi-class segmentation"""
    
    def conv_block(x, filters):
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        return x
    
    inputs = layers.Input(shape=input_shape)
    
    # Encoder path
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    c5 = conv_block(p4, 1024)
    
    # Decoder path
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = conv_block(u6, 512)
    
    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = conv_block(u7, 256)
    
    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = conv_block(u8, 128)
    
    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = conv_block(u9, 64)
    
    # Output layer for 4 classes with softmax
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c9)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


# Dice Loss
def dice_coefficient_multi(y_true, y_pred, num_classes=4, smooth=1e-6):
    y_true = tf.squeeze(y_true, axis=-1)  # remove channel
    
    y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
    
    y_true_flat = tf.reshape(y_true_onehot, [-1, num_classes])
    y_pred_flat = tf.reshape(y_pred, [-1, num_classes])
    
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=0)
    union = tf.reduce_sum(y_true_flat, axis=0) + tf.reduce_sum(y_pred_flat, axis=0)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return tf.reduce_mean(dice)


# Sparse categorical crossentropy loss 
def sparse_categorical_focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.squeeze(y_true, axis=-1)
        y_true = tf.cast(y_true, tf.int32)
        
        y_true_onehot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        cross_entropy = -y_true_onehot * tf.math.log(y_pred)
        weight = alpha * tf.pow(1. - y_pred, gamma)
        
        fl = weight * cross_entropy
        
        return tf.reduce_mean(tf.reduce_sum(fl, axis=-1))
    
    return loss


# Build model
model = build_unet(
    input_shape=(128, 128, 1),
    num_classes=4
)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=sparse_categorical_focal_loss(),
    metrics=[
        "accuracy",
        dice_coefficient_multi
    ]
)

# Save best model
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_unet_model_multi.h5",
    monitor="val_dice_coefficient_multi",
    mode="max",
    save_best_only=True,
    verbose=1
)

# Train
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[checkpoint]
)

model.save("unet_2d_brats_multi_class.h5")

# Save sample slices with class visualization
def save_sample_slices_with_classes(X, Y, out_dir="samples", n=5):
    """Save sample slices showing all 4 classes with different colors"""
    os.makedirs(out_dir, exist_ok=True)
    
    for i in range(min(n, len(X))):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Input image
        axes[0].imshow(X[i, :, :, 0], cmap="gray")
        axes[0].set_title("Input (FLAIR)")
        axes[0].axis("off")
        
        # Ground truth mask with colors
        mask_rgb = np.zeros((*INPUT_SIZE, 3))
        # Squeeze the mask to (128, 128)
        y_plot = np.squeeze(Y[i]) 
        
        for class_idx in range(1, 4):  # Skip background (0)
            mask_class = (y_plot == class_idx) # Use the squeezed array here
            if class_idx == 1:  # Red
                mask_rgb[mask_class, 0] = 1
            elif class_idx == 2:  # Green
                mask_rgb[mask_class, 1] = 1
            elif class_idx == 3:  # Blue
                mask_rgb[mask_class, 2] = 1
        
        axes[1].imshow(mask_rgb)
        axes[1].set_title("Ground Truth (Colored)")
        axes[1].axis("off")
        
        # Class distribution
        unique_classes, class_counts = np.unique(y_plot, return_counts=True)
        class_text = "\n".join([f"Class {cls}: {count} pixels" 
                               for cls, count in zip(unique_classes, class_counts)])
        
        axes[2].text(0.1, 0.5, class_text, fontsize=10, 
                    verticalalignment='center', transform=axes[2].transAxes)
        axes[2].set_title("Class Distribution")
        axes[2].axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"sample_{i}.png"), dpi=150, bbox_inches='tight')
        plt.close()

print("Saving sample slices with class visualization...")
# Get a batch of data
for X_sample, Y_sample in val_dataset.take(1):
    save_sample_slices_with_classes(X_sample.numpy(), Y_sample.numpy(), n=5)
    break

# Plot training curves
def plot_training(history, out_dir="plots"):
    """Plot training history for multi-class segmentation"""
    os.makedirs(out_dir, exist_ok=True)
    
    # Dice coefficient
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["dice_coefficient_multi"], label="Train Dice", linewidth=2)
    plt.plot(history.history["val_dice_coefficient_multi"], label="Val Dice", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Dice Coefficient", fontsize=12)
    plt.title("Dice Coefficient (Multi-Class) vs Epoch", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "dice_curve_multi.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
    plt.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Focal Loss", fontsize=12)
    plt.title("Loss (Focal) vs Epoch", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "loss_curve_multi.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    plt.plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Accuracy vs Epoch", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "accuracy_curve_multi.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # All metrics in one figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(history.history["loss"], label="Train", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Focal Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history["dice_coefficient_multi"], label="Train", linewidth=2)
    axes[1].plot(history.history["val_dice_coefficient_multi"], label="Val", linewidth=2)
    axes[1].set_title("Dice Coefficient")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(history.history["accuracy"], label="Train", linewidth=2)
    axes[2].plot(history.history["val_accuracy"], label="Val", linewidth=2)
    axes[2].set_title("Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "all_metrics_multi.png"), dpi=150, bbox_inches='tight')
    plt.close()

print("Saving training plots...")
plot_training(history)

# Save predictions with class visualization
def save_predictions_with_classes(model, X, Y, out_dir="predictions", n=5):
    """Save predictions with colored class visualization"""
    os.makedirs(out_dir, exist_ok=True)
    
    preds = model.predict(X[:n])
    pred_classes = np.argmax(preds, axis=-1) 
    
    for i in range(n):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # Squeeze arrays for boolean indexing
        gt_plot = np.squeeze(Y[i])
        pred_plot = np.squeeze(pred_classes[i])
        
        axes[0].imshow(X[i, :, :, 0], cmap="gray")
        axes[0].set_title("Input (FLAIR)")
        axes[0].axis("off")
        
        # Ground truth colors
        gt_rgb = np.zeros((*INPUT_SIZE, 3))
        for class_idx in range(1, 4):
            mask_class = (gt_plot == class_idx) # Use squeezed version
            if class_idx == 1: gt_rgb[mask_class, 0] = 1
            elif class_idx == 2: gt_rgb[mask_class, 1] = 1
            elif class_idx == 3: gt_rgb[mask_class, 2] = 1
        
        axes[1].imshow(gt_rgb)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        
        # Prediction colors
        pred_rgb = np.zeros((*INPUT_SIZE, 3))
        for class_idx in range(1, 4):
            mask_class = (pred_plot == class_idx) # Use squeezed version
            if class_idx == 1: pred_rgb[mask_class, 0] = 1
            elif class_idx == 2: pred_rgb[mask_class, 1] = 1
            elif class_idx == 3: pred_rgb[mask_class, 2] = 1
        
        axes[2].imshow(pred_rgb)
        axes[2].set_title("Prediction")
        axes[2].axis("off")
        
        # Uncertainty
        uncertainty = 1 - np.max(preds[i], axis=-1)
        im = axes[3].imshow(uncertainty, cmap="hot", vmin=0, vmax=1)
        axes[3].set_title("Uncertainty")
        axes[3].axis("off")
        plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"pred_sample_{i}.png"), dpi=150, bbox_inches='tight')
        plt.close()

print("Saving predictions with class visualization...")
for X_sample, Y_sample in val_dataset.take(1):
    save_predictions_with_classes(model, X_sample.numpy(), Y_sample.numpy(), n=5)
    break

print("\nTraining completed successfully!")
print(f"Model saved as: unet_2d_brats_multi_class.h5")
print(f"Best model saved as: best_unet_model_multi.h5")
print(f"Samples saved in: samples/")
print(f"Plots saved in: plots/")
print(f"Predictions saved in: predictions/")




# Create the test dataset (Using the correct NPZ generator)
test_dataset = tf.data.Dataset.from_generator(
    lambda: patient_slice_generator(os.path.join(data_path, "test")),
    output_signature=(
        tf.TensorSpec(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 1), dtype=tf.float32),
        tf.TensorSpec(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 1), dtype=tf.int32)
    )
)


# Batch it and evaluate
test_dataset = test_dataset.batch(BATCH_SIZE)

# Calculate test steps based on total slices in the test folder
total_test_slices = count_total_slices(os.path.join(data_path, "test"))
test_steps = math.ceil(total_test_slices / BATCH_SIZE)

# Evaluate on Internal Test Set
results = model.evaluate(test_dataset, steps=test_steps)
print(f"Final Test Loss: {results[0]:.4f}")
print(f"Final Test Accuracy: {results[1]:.4f}")
print(f"Final Test Dice: {results[2]:.4f}")



