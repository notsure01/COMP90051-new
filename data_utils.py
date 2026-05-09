"""
Data loading, preprocessing, augmentation, corruption utilities, and
normalisation-statistics computation.

Preprocessing pipeline:
  Training : Resize(148×148) → RandomCrop(128×128) → RandomHorizontalFlip
             → ToTensor → [optional corruptions] → Normalize(mean, std)
  Eval     : Resize(128×128) → ToTensor → [optional corruptions]
             → Normalize(mean, std)

Both transforms explicitly resize every image to 128×128 (training gets a
slight oversize then random-crops to introduce spatial variety).

Normalisation statistics are computed from the training subset on the fly
via compute_mean_std() rather than using hardcoded values.

Four corruption types used in Experiment 2:
  1. Gaussian noise
  2. Gaussian blur
  3. Fog (additive whitening)
  4. Random rectangular occlusion
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF

import config


# ---------------------------------------------------------------------------
# Custom transform classes
# ---------------------------------------------------------------------------

class GaussianNoise:
    """Add isotropic Gaussian noise to a tensor already in [0,1] space."""

    def __init__(self, std: float = config.NOISE_STD):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def __repr__(self):
        return f"GaussianNoise(std={self.std})"


class GaussianBlurCustom:
    """Gaussian blur applied to a tensor in [0,1] space."""

    def __init__(self,
                 kernel_size: int = config.BLUR_KERNEL_SIZE,
                 sigma: float = config.BLUR_SIGMA):
        # kernel_size must be odd
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma = sigma

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return TF.gaussian_blur(tensor, self.kernel_size, self.sigma)

    def __repr__(self):
        return f"GaussianBlur(k={self.kernel_size}, sigma={self.sigma})"


class FogEffect:
    """Simulate fog by blending the image with a white (ones) tensor."""

    def __init__(self, intensity: float = config.FOG_INTENSITY):
        self.intensity = intensity

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        fog = torch.ones_like(tensor)
        blended = (1.0 - self.intensity) * tensor + self.intensity * fog
        return torch.clamp(blended, 0.0, 1.0)

    def __repr__(self):
        return f"FogEffect(intensity={self.intensity})"


class RandomOcclusion:
    """Zero out a random square patch (black rectangle) of the image tensor."""

    def __init__(self, ratio: float = config.OCCLUSION_RATIO):
        self.ratio = ratio

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        _, H, W = tensor.shape
        occ_h = max(1, int(H * self.ratio))
        occ_w = max(1, int(W * self.ratio))
        y0 = random.randint(0, H - occ_h)
        x0 = random.randint(0, W - occ_w)
        tensor = tensor.clone()
        tensor[:, y0:y0 + occ_h, x0:x0 + occ_w] = 0.0
        return tensor

    def __repr__(self):
        return f"RandomOcclusion(ratio={self.ratio})"


# ---------------------------------------------------------------------------
# Normalisation statistics — computed from the dataset (not hardcoded)
# ---------------------------------------------------------------------------

def compute_mean_std(raw_dataset,
                     sample_limit: int | None = None,
                     seed: int = config.RANDOM_SEED
                     ) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """
    Compute per-channel mean and standard deviation over *raw_dataset*
    (a dataset of raw PIL images) using only NumPy/PyTorch — no external
    statistics libraries.

    The calculation uses the computational formula:
        std = sqrt(E[X²] - E[X]²)
    accumulated incrementally over all pixels so that large datasets fit
    in memory.

    Parameters
    ----------
    raw_dataset  : dataset whose __getitem__ returns (PIL.Image, label)
    sample_limit : if set, use a random subset of this size to speed things up
    seed         : reproducibility seed for the random sample

    Returns
    -------
    mean : (R_mean, G_mean, B_mean) floats in [0, 1]
    std  : (R_std,  G_std,  B_std)  floats in [0, 1]
    """
    to_tensor = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),   # → float32 in [0, 1]
    ])

    n = len(raw_dataset)
    indices: np.ndarray = np.arange(n)
    if sample_limit is not None and n > sample_limit:
        rng     = np.random.default_rng(seed)
        indices = rng.choice(n, size=sample_limit, replace=False)

    channel_sum    = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    n_pixels       = 0

    for idx in indices:
        img, _ = raw_dataset[int(idx)]
        t      = to_tensor(img)           # (3, H, W)
        channel_sum    += t.sum(dim=[1, 2]).numpy().astype(np.float64)
        channel_sq_sum += (t ** 2).sum(dim=[1, 2]).numpy().astype(np.float64)
        n_pixels       += t.shape[1] * t.shape[2]

    mean  = channel_sum    / n_pixels
    var   = channel_sq_sum / n_pixels - mean ** 2
    std   = np.sqrt(np.maximum(var, 0.0))   # clamp fp rounding negatives

    return tuple(mean.tolist()), tuple(std.tolist())


# ---------------------------------------------------------------------------
# Transform pipelines
# ---------------------------------------------------------------------------
# Every factory takes optional mean/std; falls back to config values so that
# callers before compute_mean_std() don't break.

def get_clean_transform(mean=None, std=None) -> transforms.Compose:
    """
    Training-time augmentation on clean images.

    Spatial pipeline:
      Resize(148×148) → RandomCrop(128×128) → RandomHorizontalFlip

    The Resize step first brings every image (regardless of original aspect
    ratio) to an exact 148×148 grid; RandomCrop then draws a 128×128 patch
    to introduce positional diversity while guaranteeing output shape 128×128.
    """
    mean = mean or config.NORM_MEAN
    std  = std  or config.NORM_STD
    pad  = config.IMG_SIZE + 20          # 148
    return transforms.Compose([
        transforms.Resize((pad, pad)),
        transforms.RandomCrop(config.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_corrupted_transform(mean=None, std=None) -> transforms.Compose:
    """
    Same spatial augmentation as the clean pipeline, then all four corruption
    types are applied in [0,1] space before normalisation.
    """
    mean = mean or config.NORM_MEAN
    std  = std  or config.NORM_STD
    pad  = config.IMG_SIZE + 20
    return transforms.Compose([
        transforms.Resize((pad, pad)),
        transforms.RandomCrop(config.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        GaussianNoise(),
        GaussianBlurCustom(),
        FogEffect(),
        RandomOcclusion(),
        transforms.Normalize(mean, std),
    ])


def get_eval_transform(mean=None, std=None) -> transforms.Compose:
    """
    Deterministic evaluation transform.

    Images are resized directly to 128×128 with no random operations,
    ensuring reproducible, exact 128×128 tensors for every sample.
    """
    mean = mean or config.NORM_MEAN
    std  = std  or config.NORM_STD
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_corrupted_eval_transform(mean=None, std=None) -> transforms.Compose:
    """
    Deterministic corrupted evaluation transform.
    Same resize-to-128×128 as the clean eval, plus all four corruptions.
    """
    mean = mean or config.NORM_MEAN
    std  = std  or config.NORM_STD
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        GaussianNoise(),
        GaussianBlurCustom(),
        FogEffect(),
        RandomOcclusion(),
        transforms.Normalize(mean, std),
    ])


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class TransformDataset(Dataset):
    """
    Wraps a base dataset but applies a (potentially different) transform.
    Used to apply train-time augmentation on train subsets and eval transforms
    on validation/test subsets while sharing the same underlying data.
    """

    def __init__(self, base_dataset: Dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class RawImageFolder(ImageFolder):
    """
    ImageFolder that returns raw PIL images (no transform applied), so that
    TransformDataset can apply different transforms later.
    """

    def __init__(self, root: str):
        super().__init__(root, transform=None)


def load_raw_dataset(root: str = config.DATA_ROOT,
                     subset_per_class: int | None = config.SUBSET_PER_CLASS,
                     seed: int = config.RANDOM_SEED) -> RawImageFolder:
    """
    Load dataset from *root* as an ImageFolder with raw PIL images.
    If *subset_per_class* is given, sample that many images per class
    (stratified) to keep nested-CV runtime manageable.
    Returns a dataset whose __getitem__ returns (PIL.Image, int_label).
    """
    full = RawImageFolder(root)

    if subset_per_class is None:
        return full

    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []

    for class_idx in range(len(full.classes)):
        class_indices = [i for i, (_, lbl) in enumerate(full.samples) if lbl == class_idx]
        n_take = min(subset_per_class, len(class_indices))
        chosen = rng.choice(class_indices, size=n_take, replace=False).tolist()
        selected_indices.extend(chosen)

    # Build a lightweight wrapper that exposes only the selected indices
    subset_ds = _SubsetRawDataset(full, selected_indices)
    return subset_ds


class _SubsetRawDataset(Dataset):
    """Internal: exposes only *indices* from a base RawImageFolder."""

    def __init__(self, base: RawImageFolder, indices: list[int]):
        self.base    = base
        self.indices = indices
        self.classes = base.classes

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


# ---------------------------------------------------------------------------
# K-fold split utility (implemented from scratch — no sklearn)
# ---------------------------------------------------------------------------

def get_kfold_splits(n_samples: int,
                     k: int,
                     seed: int = config.RANDOM_SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Pure-numpy stratification-free k-fold split.

    Returns a list of k tuples (train_indices, test_indices) where every
    index is an integer in [0, n_samples).  Indices within each fold are
    drawn without replacement and every sample appears in exactly one test
    fold.

    Parameters
    ----------
    n_samples : total number of samples
    k         : number of folds
    seed      : reproducibility seed

    Returns
    -------
    List of (train_idx, test_idx) numpy arrays.
    """
    rng     = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[: n_samples % k] += 1   # distribute remainder one-per-fold

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    current = 0
    for fold_size in fold_sizes:
        test_idx  = indices[current : current + fold_size]
        train_idx = np.concatenate([indices[:current],
                                    indices[current + fold_size :]])
        splits.append((train_idx, test_idx))
        current += fold_size

    return splits
