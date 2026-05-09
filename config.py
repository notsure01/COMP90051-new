"""
Central configuration for the COMP90051 project:
  "How does CNN-based scene classification degrade under simulated
   real-world corruptions, and can adversarial training improve robustness?"
"""

import os
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT   = os.path.join("dataset", "seg_train", "seg_train")
RESULTS_DIR = "results"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
NUM_CLASSES  = 6
CLASS_NAMES  = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
IMG_SIZE     = 128          # images are resized/cropped to 128×128
RANDOM_SEED  = 42

# Stratified subsample per class.
# None  → use every image in seg_train (~14 034 total across 6 classes).
# int   → sample that many per class (e.g. 500 → 3 000 total) for fast runs.
SUBSET_PER_CLASS = None     # use the full dataset

# Normalisation statistics — computed from the training subset at runtime
# (see data_utils.compute_mean_std).  Populated by main.py before training.
NORM_MEAN: tuple = (0.5, 0.5, 0.5)   # placeholder; overwritten in main.py
NORM_STD:  tuple = (0.5, 0.5, 0.5)   # placeholder; overwritten in main.py

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE    = 32
INNER_EPOCHS  = 5    # epochs used during inner-CV HP evaluation
OUTER_EPOCHS  = 10   # epochs used to train final model on each outer fold

# ---------------------------------------------------------------------------
# Nested cross-validation
# ---------------------------------------------------------------------------
OUTER_K = 10   # outer folds (requirement: k >= 10)
INNER_K = 3    # inner folds for HP tuning (requirement: k >= 3)

# Fractions of outer-training data used for learning-curve experiment
LEARNING_CURVE_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# ---------------------------------------------------------------------------
# Corruption parameters (Experiment 2)
# ---------------------------------------------------------------------------
NOISE_STD        = 0.08   # Gaussian noise std (in [0,1] pixel space)
BLUR_KERNEL_SIZE = 5      # must be odd
BLUR_SIGMA       = 1.5
FOG_INTENSITY    = 0.35   # additive fog weight in [0,1] pixel space
OCCLUSION_RATIO  = 0.15   # fraction of image side length per occlusion patch

# ---------------------------------------------------------------------------
# Hyperparameter grids  (2 HPs × 3 values each per model)
# Middle value is the well-known default and should be selected most often.
# ---------------------------------------------------------------------------
HP_GRIDS = {
    "CNN": {
        "lr":      [1e-4, 1e-3, 1e-2],   # middle: 1e-3  (standard Adam LR)
        "dropout": [0.2,  0.4,  0.6],    # middle: 0.4   (moderate regularisation)
    },
    "ResNet": {
        "lr":           [1e-4, 1e-3, 1e-2],   # middle: 1e-3
        "weight_decay": [1e-5, 1e-4, 1e-3],   # middle: 1e-4 (standard L2 reg)
    },
    "ViT": {
        "lr":         [1e-5, 1e-4, 1e-3],   # middle: 1e-4 (standard ViT LR)
        "patch_size": [8,    16,   32],      # middle: 16   (original ViT-16)
    },
}

MODEL_NAMES = list(HP_GRIDS.keys())  # ["CNN", "ResNet", "ViT"]

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
