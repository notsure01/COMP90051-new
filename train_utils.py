"""
Training, evaluation, nested cross-validation, and learning-curve utilities.

Design notes
------------
* Nested CV is implemented entirely from scratch (no sklearn).
* The outer loop uses k=OUTER_K folds; the inner loop uses k=INNER_K folds.
* Hyperparameter selection: exhaustive grid search over HP_GRIDS[model_name].
* Learning curves track accuracy vs. fraction of training samples (NOT loss
  vs. epochs, which is excluded by the project guide).
* All metric computations delegate to metrics.py (also scratch).
"""

from __future__ import annotations

import itertools
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import config
import metrics as M
from data_utils import (
    get_kfold_splits,
    TransformDataset,
    compute_mean_std,
    get_clean_transform,
    get_corrupted_transform,
    get_eval_transform,
    get_corrupted_eval_transform,
)
from models import build_model


# ---------------------------------------------------------------------------
# Terminal-only helpers  (progress bar should not pollute the log file)
# ---------------------------------------------------------------------------

def _terminal_write(msg: str) -> None:
    """
    Write *msg* to the real terminal only.
    When sys.stdout is a _Tee (from main.py), the _terminal attribute holds
    the original stdout; otherwise sys.stdout itself is the terminal.
    """
    term = getattr(sys.stdout, "_terminal", sys.stdout)
    term.write(msg)
    term.flush()


def _ascii_bar(done: int, total: int, width: int = 28) -> str:
    """
    Return a fixed-width ASCII progress bar, e.g.:
      [████████████░░░░░░░░]  60%  192/323
    """
    filled = int(width * done / max(total, 1))
    bar    = "█" * filled + "░" * (width - filled)
    pct    = int(100 * done / max(total, 1))
    return f"[{bar}] {pct:3d}%  {done}/{total}"


# ---------------------------------------------------------------------------
# Optimiser factory
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, hp: dict) -> torch.optim.Optimizer:
    """
    Build an Adam optimiser for *model* using lr and weight_decay from *hp*.
    Defaults match the middle values of each HP grid.
    """
    lr           = float(hp.get("lr", 1e-3))
    weight_decay = float(hp.get("weight_decay", 1e-4))
    return torch.optim.Adam(model.parameters(),
                            lr=lr, weight_decay=weight_decay)


# ---------------------------------------------------------------------------
# Single epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device | str,
                    show_bar: bool = False,
                    epoch_num: int = 0,
                    n_epochs: int = 0,
                    bar_prefix: str = "") -> tuple[float, float]:
    """
    Train for one epoch.

    Parameters
    ----------
    show_bar   : if True, print a live batch progress bar to the terminal
                 (uses \\r so it never appears in the log file).
    epoch_num  : current 1-based epoch index, used in the bar label.
    n_epochs   : total epochs for this run, used in the bar label.
    bar_prefix : context string prepended to every bar update, e.g.
                 "HP  2/9 | fold 1/3" for inner-CV runs.

    Returns
    -------
    (mean_loss, train_accuracy) — both floats computed over the whole epoch.
    """
    model.train()
    total_loss    = 0.0
    total_correct = 0
    n_samples     = 0
    n_batches     = len(loader)
    epoch_t0      = time.time()

    for batch_idx, (images, labels) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        n_samples     += images.size(0)

        if show_bar:
            bar      = _ascii_bar(batch_idx, n_batches)
            elapsed  = time.time() - epoch_t0
            run_loss = total_loss / n_samples
            prefix   = f"{bar_prefix}  " if bar_prefix else ""
            _terminal_write(
                f"\r  {prefix}Epoch [{epoch_num:>2}/{n_epochs}]  {bar}"
                f"  loss={run_loss:.4f}  {elapsed:.1f}s"
            )

    mean_loss = total_loss    / max(n_samples, 1)
    train_acc = total_correct / max(n_samples, 1)
    return mean_loss, train_acc


@torch.no_grad()
def get_predictions(model: nn.Module,
                    loader: DataLoader,
                    device: torch.device | str) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on *loader*; return (predictions, true_labels) as
    numpy integer arrays.
    """
    model.eval()
    all_preds  = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def train_model(model: nn.Module,
                train_dataset,
                hp: dict,
                n_epochs: int,
                device: torch.device | str,
                train_transform=None,
                mode: str = "silent",
                hp_header: str = "",
                fold_num: int = 0,
                total_folds: int = 0,
                bar_context: str = "") -> None:
    """
    Train *model* in-place for *n_epochs* epochs.

    Parameters
    ----------
    mode : controls output verbosity —
      "silent"  no output at all (used by learning-curve helper).
      "bar"     live \\r batch progress bar shown on the terminal only;
                no epoch summary lines printed (used for inner-CV runs so
                the user can see activity without flooding the log).
      "full"    full output: HP header + fold banner + per-epoch summary
                lines + bar (used for outer final training).
    hp_header   : banner string, e.g. "CNN: lr=1e-03  dropout=0.4"
                  (shown only in "full" mode).
    fold_num    : 1-based outer fold index (shown in "full" banner).
    total_folds : total outer folds (shown in "full" banner).
    bar_context : short context prefix shown inside the \\r bar for
                  "bar" mode, e.g. "HP  2/9 | fold 1/3".
    """
    if train_transform is not None:
        wrapped = TransformDataset(train_dataset, train_transform)
    else:
        wrapped = train_dataset

    loader    = DataLoader(wrapped,
                           batch_size=config.BATCH_SIZE,
                           shuffle=True,
                           num_workers=0,
                           pin_memory=(str(device) == "cuda"))
    optimizer = build_optimizer(model, hp)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    show_bar         = mode in ("bar", "full")
    show_epoch_lines = mode == "full"
    SEP              = "=" * 60

    if mode == "full" and fold_num > 0:
        print(f"\n--- {hp_header} ---", flush=True)
        print(f"\n{SEP}", flush=True)
        print(f"Fold {fold_num}/{total_folds}", flush=True)
        print(SEP, flush=True)

    fold_t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        epoch_t0 = time.time()
        loss, acc = train_one_epoch(
            model, loader, optimizer, criterion, device,
            show_bar=show_bar,
            epoch_num=epoch,
            n_epochs=n_epochs,
            bar_prefix=bar_context,
        )
        epoch_elapsed = time.time() - epoch_t0

        if show_epoch_lines:
            _terminal_write("\r" + " " * 90 + "\r")   # erase bar line
            print(
                f"  Epoch [{epoch:>2}/{n_epochs}]"
                f"  train_loss={loss:.4f}"
                f"  train_acc={acc:.4f}"
                f"  [{epoch_elapsed:.1f}s]",
                flush=True,
            )

    if mode == "full" and fold_num > 0:
        fold_elapsed = time.time() - fold_t0
        print(f"  Fold {fold_num} completed in {fold_elapsed:.1f}s", flush=True)
        print(SEP, flush=True)


# ---------------------------------------------------------------------------
# HP grid enumeration
# ---------------------------------------------------------------------------

def enumerate_hp_grid(hp_grid: dict[str, list]) -> list[dict]:
    """
    Produce a list of all HP combinations from a grid of the form
    {"param1": [v1, v2, ...], "param2": [v1, v2, ...], ...}.

    Example
    -------
    >>> enumerate_hp_grid({"lr": [1e-4, 1e-3], "dropout": [0.2, 0.4]})
    [{"lr": 1e-4, "dropout": 0.2},
     {"lr": 1e-4, "dropout": 0.4},
     {"lr": 1e-3, "dropout": 0.2},
     {"lr": 1e-3, "dropout": 0.4}]
    """
    keys   = list(hp_grid.keys())
    values = list(hp_grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# ---------------------------------------------------------------------------
# Inner CV – hyperparameter tuning  (from scratch)
# ---------------------------------------------------------------------------

def inner_cv(outer_train_dataset,
             outer_train_indices: np.ndarray,
             model_name: str,
             hp_grid: dict[str, list],
             inner_k: int,
             inner_epochs: int,
             device: str,
             train_transform,
             eval_transform,
             outer_fold_label: str = "",
             seed: int = config.RANDOM_SEED) -> tuple[dict, dict]:
    """
    Run k-fold cross-validation on *outer_train_dataset* for every HP combo
    in *hp_grid*.  Prints one summary line per HP combination.

    Returns
    -------
    best_hp    : dict of HP name → best value
    hp_scores  : dict of str(hp) → mean validation accuracy across inner folds
    """
    n_inner      = len(outer_train_dataset)
    inner_splits = get_kfold_splits(n_inner, inner_k, seed)
    hp_combos    = enumerate_hp_grid(hp_grid)
    n_combos     = len(hp_combos)
    hp_scores: dict[str, float] = {}

    print(f"    {outer_fold_label}  Inner CV: {n_combos} HP combos × "
          f"{inner_k} folds × {inner_epochs} epochs", flush=True)

    for combo_idx, hp in enumerate(hp_combos, 1):
        hp_label  = "  ".join(f"{k}={v}" for k, v in sorted(hp.items()))
        fold_accs: list[float] = []

        for fold_i, (inner_train_local, inner_val_local) in enumerate(inner_splits, 1):
            inner_train_ds = _AbsoluteSubset(outer_train_dataset, inner_train_local)
            inner_val_ds   = _AbsoluteSubset(outer_train_dataset, inner_val_local)

            ctx   = f"HP {combo_idx:>2}/{n_combos} | fold {fold_i}/{inner_k}"
            model = build_model(model_name, config.NUM_CLASSES, **hp)
            train_model(model, inner_train_ds, hp, inner_epochs,
                        device, train_transform,
                        mode="bar",
                        bar_context=ctx)

            val_wrapped = TransformDataset(inner_val_ds, eval_transform)
            val_loader  = DataLoader(val_wrapped,
                                     batch_size=config.BATCH_SIZE * 2,
                                     shuffle=False, num_workers=0)
            preds, labels_arr = get_predictions(model, val_loader, device)
            fold_accs.append(M.accuracy(labels_arr, preds))
            del model

        mean_acc = float(np.mean(fold_accs))
        hp_scores[_hp_key(hp)] = mean_acc
        _terminal_write("\r" + " " * 90 + "\r")
        print(f"      HP {combo_idx:>2}/{n_combos}  [{hp_label}]"
              f"  val_acc={mean_acc:.4f}", flush=True)

    best_hp_key = max(hp_scores, key=hp_scores.__getitem__)
    best_hp     = _key_to_hp(best_hp_key, hp_combos)
    best_label  = "  ".join(f"{k}={v}" for k, v in sorted(best_hp.items()))
    print(f"    {outer_fold_label}  ✓ Best HP: [{best_label}]"
          f"  val_acc={hp_scores[best_hp_key]:.4f}", flush=True)
    return best_hp, hp_scores


# ---------------------------------------------------------------------------
# Outer / nested CV  (from scratch)
# ---------------------------------------------------------------------------

def nested_cv(raw_dataset,
              model_name: str,
              outer_k: int,
              inner_k: int,
              inner_epochs: int,
              outer_epochs: int,
              device: str,
              experiment_type: str = "clean",
              seed: int = config.RANDOM_SEED,
              verbose: bool = True,
              ) -> dict:
    """
    Full nested k-fold cross-validation for one model.

    Returns a dict with fold_metrics, fold_cms, hp_counts, best_hps,
    aggregated stats, and learning-curve data.
    """
    hp_grid      = config.HP_GRIDS[model_name]
    n            = len(raw_dataset)
    outer_splits = get_kfold_splits(n, outer_k, seed)

    fold_metrics: list = []
    fold_cms:     list = []
    hp_counts:    dict = {}
    best_hps:     list = []
    lc_accs:      list = []

    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        t0         = time.time()
        fold_label = f"[{model_name} | Outer {fold_idx+1:>2}/{outer_k}]"

        outer_train_ds = _AbsoluteSubset(raw_dataset, outer_train_idx)
        outer_test_ds  = _AbsoluteSubset(raw_dataset, outer_test_idx)

        # Compute normalisation stats from outer train fold only
        if verbose:
            print(f"\n  {fold_label}  Computing normalisation stats "
                  f"from outer train fold …", flush=True)
        fold_mean, fold_std = compute_mean_std(outer_train_ds)
        if verbose:
            print(f"  {fold_label}  mean=({fold_mean[0]:.4f}, "
                  f"{fold_mean[1]:.4f}, {fold_mean[2]:.4f})  "
                  f"std=({fold_std[0]:.4f}, {fold_std[1]:.4f}, "
                  f"{fold_std[2]:.4f})", flush=True)

        if experiment_type == "corrupted":
            train_transform = get_corrupted_transform(fold_mean, fold_std)
            eval_transform  = get_corrupted_eval_transform(fold_mean, fold_std)
        else:
            train_transform = get_clean_transform(fold_mean, fold_std)
            eval_transform  = get_eval_transform(fold_mean, fold_std)

        # Inner CV for HP selection
        best_hp, _ = inner_cv(
            outer_train_ds, outer_train_idx,
            model_name, hp_grid, inner_k, inner_epochs,
            device, train_transform, eval_transform,
            outer_fold_label=fold_label,
            seed=seed,
        )
        best_hps.append(best_hp)
        key = _hp_key(best_hp)
        hp_counts[key] = hp_counts.get(key, 0) + 1

        # Final training with best HP
        hp_str    = "  ".join(f"{k}={v}" for k, v in sorted(best_hp.items()))
        hp_header = f"{model_name}: {hp_str}"
        model = build_model(model_name, config.NUM_CLASSES, **best_hp)
        train_model(model, outer_train_ds, best_hp, outer_epochs,
                    device, train_transform,
                    mode="full" if verbose else "silent",
                    hp_header=hp_header,
                    fold_num=fold_idx + 1,
                    total_folds=outer_k)

        # Test evaluation
        test_wrapped = TransformDataset(outer_test_ds, eval_transform)
        test_loader  = DataLoader(test_wrapped,
                                  batch_size=config.BATCH_SIZE * 2,
                                  shuffle=False, num_workers=0)
        preds, labels = get_predictions(model, test_loader, device)
        fold_m = M.summarise_metrics(labels, preds, config.NUM_CLASSES)
        fold_metrics.append(fold_m)
        fold_cms.append(M.confusion_matrix(labels, preds, config.NUM_CLASSES).tolist())

        # Learning curve
        if verbose:
            print(f"\n  {fold_label}  Computing learning curve …", flush=True)
        lc_row = _learning_curve_fold(
            raw_dataset, outer_train_idx, outer_test_idx,
            model_name, best_hp, outer_epochs, device,
            train_transform, eval_transform,
        )
        lc_accs.append(lc_row)

        del model
        elapsed = time.time() - t0
        if verbose:
            print(
                f"\n  {fold_label}  ▶ TEST RESULT"
                f"  acc={fold_m['accuracy']:.4f}"
                f"  prec={fold_m['precision']:.4f}"
                f"  rec={fold_m['recall']:.4f}"
                f"  f1={fold_m['f1']:.4f}"
                f"  [total {elapsed/60:.1f} min]",
                flush=True,
            )

    aggregated = M.aggregate_fold_metrics(fold_metrics)
    return {
        "fold_metrics":  fold_metrics,
        "fold_cms":      fold_cms,
        "hp_counts":     hp_counts,
        "best_hps":      best_hps,
        "aggregated":    aggregated,
        "lc_accs":       lc_accs,
        "lc_fractions":  config.LEARNING_CURVE_FRACTIONS,
    }


# ---------------------------------------------------------------------------
# Learning-curve helper (accuracy vs. training-set fraction)
# ---------------------------------------------------------------------------

def _learning_curve_fold(raw_dataset,
                         outer_train_idx: np.ndarray,
                         outer_test_idx:  np.ndarray,
                         model_name: str,
                         best_hp: dict,
                         outer_epochs: int,
                         device: str,
                         train_transform,
                         eval_transform) -> list[float]:
    """
    For a single outer fold, train with increasing fractions of the outer
    training set and record test accuracy.  Returns a list of accuracies,
    one per fraction in config.LEARNING_CURVE_FRACTIONS.

    This produces learning curves as required by the project guide:
    accuracy vs. number of training samples (NOT loss vs. epochs).
    """
    rng           = np.random.default_rng(config.RANDOM_SEED)
    n_train       = len(outer_train_idx)
    test_wrapped  = TransformDataset(
        _AbsoluteSubset(raw_dataset, outer_test_idx), eval_transform)
    test_loader   = DataLoader(test_wrapped, batch_size=config.BATCH_SIZE * 2,
                               shuffle=False, num_workers=0)
    accs: list[float] = []

    for frac in config.LEARNING_CURVE_FRACTIONS:
        n_take    = max(1, int(n_train * frac))
        chosen    = rng.choice(outer_train_idx, size=n_take, replace=False)
        sub_ds    = _AbsoluteSubset(raw_dataset, chosen)

        model = build_model(model_name, config.NUM_CLASSES, **best_hp)
        train_model(model, sub_ds, best_hp, outer_epochs, device, train_transform)

        preds, labels = get_predictions(model, test_loader, device)
        accs.append(M.accuracy(labels, preds))
        del model

    return accs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _AbsoluteSubset:
    """
    Lightweight dataset wrapper exposing only the items at *indices*.
    Works on any dataset whose __getitem__ accepts an integer index.
    """

    def __init__(self, base, indices: np.ndarray):
        self.base    = base
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base[int(self.indices[idx])]


def _hp_key(hp: dict) -> str:
    """Deterministic string key for a HP combo dict."""
    return str(sorted(hp.items()))


def _key_to_hp(key: str, hp_combos: list[dict]) -> dict:
    """Reverse-lookup: find the HP combo dict that matches *key*."""
    for hp in hp_combos:
        if _hp_key(hp) == key:
            return hp
    raise KeyError(f"HP key not found: {key}")
