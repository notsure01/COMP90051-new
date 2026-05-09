"""
Entry point for the COMP90051 project experiments.

Two experiments are run sequentially for all three models (CNN, ResNet, ViT):

  Experiment 1 – Clean baseline
    • Training data  : clean images (standard augmentation)
    • Test data      : clean images (deterministic eval transform)
    • Purpose        : establish baseline accuracy

  Experiment 2 – Corrupted robustness
    • Training data  : images with all four corruptions applied
                       (Gaussian noise, Gaussian blur, fog, random occlusion)
    • Test data      : corrupted images (same four corruptions)
    • Purpose        : measure degradation under real-world conditions and
                       whether training on corrupted data improves robustness

Both experiments use nested cross-validation (outer k=10, inner k=3) with
hyperparameter tuning implemented from scratch.  Results are saved as JSON
files in results/ for later visualisation with generate_vis.py.
"""

import json
import os
import sys
import time
import numpy as np
import torch

import config
from data_utils import load_raw_dataset
from train_utils import nested_cv


# ---------------------------------------------------------------------------
# Logging — tee stdout to both terminal and a timestamped log file
# ---------------------------------------------------------------------------

class _Tee:
    """
    Forwards every write/flush to both the original stdout and a log file,
    so all print() output appears on the terminal AND is saved to disk.
    """

    def __init__(self, log_path: str):
        self._terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._file = open(log_path, "w", encoding="utf-8", buffering=1)

    def write(self, message: str) -> None:
        self._terminal.write(message)
        self._file.write(message)

    def flush(self) -> None:
        self._terminal.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    # Make _Tee behave as a proper file-like object
    def fileno(self):
        return self._terminal.fileno()

    @property
    def encoding(self):
        return self._terminal.encoding


def start_logging(log_path: str) -> _Tee:
    """Replace sys.stdout with a Tee; return the Tee so it can be closed."""
    tee = _Tee(log_path)
    sys.stdout = tee
    return tee


def stop_logging(tee: _Tee) -> None:
    """Restore sys.stdout and close the log file."""
    sys.stdout = tee._terminal
    tee.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = config.RANDOM_SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_results(results: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {path}")


def print_summary(model_name: str, exp_name: str, agg: dict) -> None:
    print(f"\n  {'─'*50}")
    print(f"  {exp_name}  |  {model_name}")
    print(f"  {'─'*50}")
    for metric, vals in agg.items():
        print(f"  {metric:12s}: {vals['mean']:.4f} ± {vals['std']:.4f}")


# ---------------------------------------------------------------------------
# Run a single experiment for all models
# ---------------------------------------------------------------------------

def run_experiment(experiment_name: str,
                   raw_dataset,
                   experiment_type: str,
                   device: str) -> dict:
    """Run nested CV for every model and return {model_name: results}."""
    all_results: dict = {}

    for model_name in config.MODEL_NAMES:
        print(f"\n  ╔══ {experiment_name} │ {model_name} ══╗", flush=True)
        t0 = time.time()

        results = nested_cv(
            raw_dataset     = raw_dataset,
            model_name      = model_name,
            outer_k         = config.OUTER_K,
            inner_k         = config.INNER_K,
            inner_epochs    = config.INNER_EPOCHS,
            outer_epochs    = config.OUTER_EPOCHS,
            device          = device,
            experiment_type = experiment_type,
            seed            = config.RANDOM_SEED,
            verbose         = True,
        )

        elapsed = time.time() - t0
        print(f"\n  ╚══ {model_name} done in {elapsed/60:.1f} min ══╝",
              flush=True)
        print_summary(model_name, experiment_name, results["aggregated"])

        print(f"\n  HP selection counts ({model_name}):")
        for hp_key, cnt in sorted(results["hp_counts"].items(),
                                  key=lambda x: -x[1]):
            print(f"    {hp_key:60s}: {cnt}")

        all_results[model_name] = results

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed()

    log_path = os.path.join(
        config.RESULTS_DIR,
        f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.txt",
    )
    tee = start_logging(log_path)
    try:
        _main_body()
    finally:
        stop_logging(tee)
        print(f"\nTraining log saved → {log_path}")


def _main_body() -> None:
    print("=" * 60)
    print(" COMP90051 Project — Scene Classification Robustness")
    print(f" Device : {config.DEVICE}")
    print(f" Outer k: {config.OUTER_K}  |  Inner k: {config.INNER_K}")
    print(f" Epochs : Inner={config.INNER_EPOCHS}, Outer={config.OUTER_EPOCHS}")
    print("=" * 60)

    # Load raw dataset once (PIL images, no transforms applied yet)
    print("\nLoading dataset …")
    raw_dataset = load_raw_dataset(
        root             = config.DATA_ROOT,
        subset_per_class = config.SUBSET_PER_CLASS,
        seed             = config.RANDOM_SEED,
    )
    print(f"  Dataset size: {len(raw_dataset)} images")

    # =======================================================================
    # EXPERIMENT 1 — Clean baseline
    # =======================================================================
    print("\n" + "=" * 60)
    print(" EXPERIMENT 1 — Clean baseline")
    print("=" * 60)
    exp1_results = run_experiment(
        experiment_name = "Exp1-Clean",
        raw_dataset     = raw_dataset,
        experiment_type = "clean",
        device          = config.DEVICE,
    )
    save_results(exp1_results, os.path.join(config.RESULTS_DIR, "exp1_clean.json"))

    # =======================================================================
    # EXPERIMENT 2 — Corrupted images
    # =======================================================================
    print("\n" + "=" * 60)
    print(" EXPERIMENT 2 — Corrupted images (noise + blur + fog + occlusion)")
    print("=" * 60)
    exp2_results = run_experiment(
        experiment_name = "Exp2-Corrupted",
        raw_dataset     = raw_dataset,
        experiment_type = "corrupted",
        device          = config.DEVICE,
    )
    save_results(exp2_results, os.path.join(config.RESULTS_DIR, "exp2_corrupted.json"))

    # =======================================================================
    # Cross-experiment comparison summary
    # =======================================================================
    print("\n" + "=" * 60)
    print(" FINAL COMPARISON SUMMARY  (mean ± std across 10 outer folds)")
    print("=" * 60)

    header = f"{'Model':<10}  {'Exp':>12}  "
    header += "  ".join(f"{m:>12}" for m in ["Accuracy", "Precision",
                                              "Recall", "F1"])
    print(header)
    print("-" * len(header))

    for model_name in config.MODEL_NAMES:
        for exp_tag, results in [("Clean",     exp1_results),
                                  ("Corrupted", exp2_results)]:
            agg = results[model_name]["aggregated"]
            row = f"{model_name:<10}  {exp_tag:>12}  "
            for m in ["accuracy", "precision", "recall", "f1"]:
                row += f"  {agg[m]['mean']:.4f}±{agg[m]['std']:.4f}"
            print(row)
        print()

    print("\nDone.  Run generate_vis.py to produce plots.")


if __name__ == "__main__":
    main()

