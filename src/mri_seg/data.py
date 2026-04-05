"""Dataset download, slicing, and loading for BraTS 2023 MRI segmentation."""

import glob
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from mri_seg import config


# ──────────────────────────────────────────────────────────────────────
# Download
# ──────────────────────────────────────────────────────────────────────
def download_dataset(data_dir=None):
    """Download BraTS 2023 dataset from Synapse. Returns path to raw data directory."""
    if data_dir is None:
        data_dir = config.DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    raw_dir = data_dir / "brats2023"
    raw_dir.mkdir(parents=True, exist_ok=True)

    import synapseclient
    import synapseutils
    from dotenv import load_dotenv

    load_dotenv()
    auth_token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if not auth_token:
        raise RuntimeError("SYNAPSE_AUTH_TOKEN not set. Add it to your .env file.")

    syn = synapseclient.Synapse()
    syn.login(authToken=auth_token)

    print(f"Syncing BraTS 2023 (Synapse: {config.SYNAPSE_DATASET_ID})...")
    synapseutils.syncFromSynapse(syn, config.SYNAPSE_DATASET_ID, path=str(raw_dir))
    print(f"Sync complete: {config.rel(raw_dir)}")
    return raw_dir


def find_training_data(raw_dir):
    """Locate the directory containing BraTS training case folders."""
    raw_dir = Path(raw_dir)

    for pattern in [
        "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
        "*TrainingData*",
        "*Training*",
    ]:
        for m in sorted(raw_dir.rglob(pattern)):
            if m.is_dir() and any(m.glob(f"{config.SUBJECT_PREFIX}*")):
                return m

    # Fallback: find case directories directly
    case_dirs = sorted(raw_dir.rglob(f"{config.SUBJECT_PREFIX}*"))
    if case_dirs:
        return case_dirs[0].parent

    raise FileNotFoundError(f"Could not find BraTS training data in {raw_dir}")


# ──────────────────────────────────────────────────────────────────────
# Volume slicing
# ──────────────────────────────────────────────────────────────────────
def _load_volume(file_path):
    """Load a NIfTI file and return a 3D numpy array."""
    return nib.load(file_path).get_fdata()


def _get_modality_file(case_dir, modality):
    """Find a NIfTI file matching the given modality keyword in a case directory."""
    nii_files = glob.glob(os.path.join(case_dir, "*.nii*"))
    for f in nii_files:
        if modality.lower() in f.lower():
            return f
    return None


def _resize_2d(arr, target_size):
    """Resize a 2D array using PyTorch interpolation (bilinear)."""
    t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(t, size=target_size, mode="bilinear", align_corners=False)
    return resized.squeeze().numpy()


def _resize_2d_nearest(arr, target_size):
    """Resize a 2D array using nearest-neighbor (for masks)."""
    t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(t, size=target_size, mode="nearest")
    return resized.squeeze().numpy().astype(np.int32)


def slice_volumes(raw_dir, output_dir, prefix=None, modality=None, input_size=None, seed=42):
    """Slice 3D NIfTI volumes into 2D .npz files (one per patient).

    Preprocessing logic adapted from mri-seg/slice.py:
    - Loads the specified modality volume and segmentation mask
    - Extracts all non-empty axial slices
    - Normalizes each slice to [0, 1]
    - Resizes to input_size
    - Saves per-patient .npz with images and masks arrays

    Also splits cases into train/val/test (75%/15%/10%).
    Returns dict mapping split name to directory path.
    """
    if prefix is None:
        prefix = config.SUBJECT_PREFIX
    if modality is None:
        modality = config.MODALITY
    if input_size is None:
        input_size = config.INPUT_SIZE

    raw_dir = str(raw_dir)
    output_dir = str(output_dir)

    # Find all subject directories
    case_dirs = sorted(glob.glob(os.path.join(raw_dir, f"{prefix}*")))
    if not case_dirs:
        case_dirs = sorted(glob.glob(os.path.join(raw_dir, "**", f"{prefix}*"), recursive=True))
        # Keep only directories
        case_dirs = [d for d in case_dirs if os.path.isdir(d)]

    if not case_dirs:
        raise FileNotFoundError(f"No cases found with prefix '{prefix}' in {raw_dir}")

    print(f"Found {len(case_dirs)} cases")

    # Split into train/val/test (75%/15%/10%)
    train_cases, test_cases = train_test_split(case_dirs, test_size=0.10, random_state=seed)
    train_cases, val_cases = train_test_split(train_cases, test_size=0.15 / 0.90, random_state=seed)

    print(f"Train: {len(train_cases)}, Val: {len(val_cases)}, Test: {len(test_cases)}")

    splits = {"train": train_cases, "val": val_cases, "test": test_cases}
    split_dirs = {}

    for split_name, cases in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        split_dirs[split_name] = split_dir

        # Skip if already processed
        existing = glob.glob(os.path.join(split_dir, "*.npz"))
        if len(existing) >= len(cases):
            print(f"{split_name}: {len(existing)} files already exist, skipping")
            continue

        print(f"\nProcessing {split_name} split ({len(cases)} cases)...")
        for case_dir in tqdm(cases, desc=split_name):
            case_id = os.path.basename(case_dir)
            img_file = _get_modality_file(case_dir, modality)
            mask_file = _get_modality_file(case_dir, "seg")

            if img_file is None or mask_file is None:
                continue

            img_vol = _load_volume(img_file)
            mask_vol = _load_volume(mask_file)

            case_images = []
            case_masks = []

            for z in range(img_vol.shape[2]):
                img_slice = img_vol[:, :, z].astype(np.float32)
                mask_slice = mask_vol[:, :, z].astype(np.int32)

                # Skip empty slices
                if np.max(img_slice) == 0:
                    continue

                # Normalize to [0, 1]
                img_slice /= img_slice.max()

                # Resize
                img_res = _resize_2d(img_slice, input_size)
                mask_res = _resize_2d_nearest(mask_slice, input_size)

                case_images.append(np.expand_dims(img_res, -1))
                case_masks.append(mask_res)

            if case_images:
                save_path = os.path.join(split_dir, f"{case_id}.npz")
                np.savez_compressed(
                    save_path,
                    images=np.array(case_images),
                    masks=np.array(case_masks),
                )

    return split_dirs


# ──────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────
class BraTSSliceDataset(Dataset):
    """PyTorch Dataset for BraTS 2D slices stored as .npz files.

    Each .npz file contains all slices for one patient.
    Selects the slice with the largest tumor area per patient
    (matching the reference mri-seg/shap.py strategy).

    Returns:
        image: [1, H, W] float32 tensor (normalized to [0, 1])
        mask:  [1, H, W] float32 tensor (binary: tumor > 0)
        label: [3] float32 tensor (multi-label: NCR, ED, ET presence for DV)
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        self.filenames = [os.path.basename(f) for f in self.files]

        # Pre-compute best slice index and labels for each patient
        self._slice_idx = {}
        self._labels = {}
        for i, f in enumerate(self.files):
            with np.load(f) as data:
                masks = data["masks"]

            # Select slice with largest tumor area
            mask_sums = np.sum(masks > 0, axis=(1, 2))
            if np.max(mask_sums) > 0:
                idx = int(np.argmax(mask_sums))
            else:
                idx = len(masks) // 2
            self._slice_idx[i] = idx

            # Multi-label: presence of each tumor sub-region in the selected slice
            raw_mask = masks[idx]
            label = np.zeros(config.NUM_TUMOR_CLASSES, dtype=np.float32)
            for li, (val, _) in enumerate(config.TUMOR_LABELS.items()):
                if np.any(raw_mask == val):
                    label[li] = 1.0
            self._labels[i] = label

        print(f"Loaded {len(self.files)} patients from {config.rel(data_dir)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx]) as data:
            images = data["images"]
            masks = data["masks"]

        slice_idx = self._slice_idx[idx]

        # Image: normalize to [0, 1]
        img = np.squeeze(images[slice_idx]).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-7)
        img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]

        # Binary mask: tumor > 0
        mask = (np.squeeze(masks[slice_idx]) > 0).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]

        # Pre-computed tumor-type presence label
        label = torch.from_numpy(self._labels[idx])

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        return img, mask, label


class SegTransformWrapper(Dataset):
    """Wrap a Subset to apply segmentation-aware transforms."""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    @property
    def indices(self):
        if hasattr(self.subset, "indices"):
            return self.subset.indices
        return list(range(len(self.subset)))

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, mask, label = self.subset[idx]
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        return img, mask, label


class SegAugmentation:
    """Simple augmentation for segmentation: random flips applied consistently to image and mask."""

    def __call__(self, img, mask):
        if torch.rand(1).item() > 0.5:
            img = img.flip(-1)
            mask = mask.flip(-1)
        if torch.rand(1).item() > 0.5:
            img = img.flip(-2)
            mask = mask.flip(-2)
        return img, mask
