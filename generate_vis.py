"""
Visualisation script — run AFTER main.py has saved results.

Loads results/exp1_clean.json and results/exp2_corrupted.json and produces:

  1. learning_curves_{model}.png
       Accuracy vs. fraction of training samples (with error bars).
       One plot per model, both experiments overlaid.

  2. metric_comparison.png
       Grouped bar chart comparing all metrics across models and experiments.

  3. error_bars_{metric}.png
       Box-and-whisker plots of per-fold values for each metric, split by
       model and experiment.

  4. confusion_matrices.png
       Row-normalised confusion matrices aggregated over all 10 outer folds,
       one subplot per model × experiment (3 models × 2 experiments = 6 plots).

  5. hp_selection.png
       Bar chart showing how often each HP combination was selected per model.

  6. summary_table.txt
       Formatted text table of mean ± std for all metrics.

All outputs go to results/plots/.
"""

from __future__ import annotations
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe in all envs)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config

RESULTS_DIR = config.RESULTS_DIR
PLOTS_DIR   = config.PLOTS_DIR
os.makedirs(PLOTS_DIR, exist_ok=True)

METRICS_DISPLAY = ["accuracy", "precision", "recall", "f1"]
METRIC_LABELS   = ["Accuracy", "Precision", "Recall", "F1"]
COLOURS = {
    "CNN":    "#4C72B0",
    "ResNet": "#DD8452",
    "ViT":    "#55A868",
}
EXP_STYLES = {
    "Clean":     {"linestyle": "-",  "marker": "o"},
    "Corrupted": {"linestyle": "--", "marker": "s"},
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def check_results_exist() -> bool:
    ok = True
    for fname in ["exp1_clean.json", "exp2_corrupted.json"]:
        p = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(p):
            print(f"  [WARN] Missing: {p} — run main.py first.")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# 1. Learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(exp1: dict, exp2: dict) -> None:
    """
    One figure per model: accuracy vs. training-set fraction.
    x-axis = number of training samples (not epochs).
    Error bars = ± 1 std across outer folds.
    """
    fractions = config.LEARNING_CURVE_FRACTIONS
    # Approximate number of training samples per fold:
    #   total dataset * (1 - 1/OUTER_K) * fraction
    n_total    = (config.SUBSET_PER_CLASS or 2333) * config.NUM_CLASSES
    n_train    = int(n_total * (1 - 1 / config.OUTER_K))
    x_samples  = [int(n_train * f) for f in fractions]

    for model_name in config.MODEL_NAMES:
        fig, ax = plt.subplots(figsize=(7, 4.5))

        for exp_tag, exp_data in [("Clean", exp1), ("Corrupted", exp2)]:
            lc_matrix = np.array(exp_data[model_name]["lc_accs"])
            # lc_matrix shape: (outer_k, n_fractions)
            means = lc_matrix.mean(axis=0)
            stds  = lc_matrix.std(axis=0)

            style = EXP_STYLES[exp_tag]
            ax.plot(x_samples, means,
                    color=COLOURS[model_name],
                    label=exp_tag,
                    **style)
            ax.fill_between(x_samples,
                            means - stds, means + stds,
                            color=COLOURS[model_name], alpha=0.2)

        ax.set_xlabel("Number of Training Samples", fontsize=12)
        ax.set_ylabel("Test Accuracy", fontsize=12)
        ax.set_title(f"Learning Curve — {model_name}", fontsize=13)
        ax.legend(title="Experiment", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        path = os.path.join(PLOTS_DIR, f"learning_curves_{model_name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 2. Metric comparison bar chart
# ---------------------------------------------------------------------------

def plot_metric_comparison(exp1: dict, exp2: dict) -> None:
    """
    Grouped bar chart: for each metric × model, show clean vs corrupted
    mean with error bars (± 1 std).
    """
    n_models  = len(config.MODEL_NAMES)
    n_metrics = len(METRICS_DISPLAY)
    bar_width = 0.35
    group_gap = 0.15

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5),
                             sharey=False)

    for ax_idx, (metric, mlabel) in enumerate(zip(METRICS_DISPLAY, METRIC_LABELS)):
        ax = axes[ax_idx]
        positions_clean  = []
        positions_corr   = []
        means_clean  = []
        means_corr   = []
        errs_clean   = []
        errs_corr    = []
        tick_labels  = []

        for m_idx, model_name in enumerate(config.MODEL_NAMES):
            base  = m_idx * (2 * bar_width + group_gap)
            positions_clean.append(base)
            positions_corr.append(base + bar_width)
            tick_labels.append(model_name)

            agg1 = exp1[model_name]["aggregated"][metric]
            agg2 = exp2[model_name]["aggregated"][metric]

            means_clean.append(agg1["mean"])
            means_corr.append(agg2["mean"])
            errs_clean.append(agg1["std"])
            errs_corr.append(agg2["std"])

        ax.bar(positions_clean, means_clean, width=bar_width,
               yerr=errs_clean, capsize=4,
               label="Clean",     color="#4C72B0", alpha=0.85)
        ax.bar(positions_corr,  means_corr,  width=bar_width,
               yerr=errs_corr,  capsize=4,
               label="Corrupted", color="#DD8452", alpha=0.85)

        tick_pos = [p + bar_width / 2 for p in positions_clean]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.set_title(mlabel, fontsize=12)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        if ax_idx == n_metrics - 1:
            ax.legend(fontsize=9)

    fig.suptitle("Metric Comparison: Clean vs. Corrupted", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "metric_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 3. Box plots (error bars per fold)
# ---------------------------------------------------------------------------

def plot_error_bars(exp1: dict, exp2: dict) -> None:
    """
    One figure per metric: box plots showing the distribution of per-fold
    scores across the 10 outer folds, for each model × experiment.
    """
    for metric, mlabel in zip(METRICS_DISPLAY, METRIC_LABELS):
        fig, ax = plt.subplots(figsize=(9, 4.5))

        all_data   = []
        tick_labels = []
        colors     = []

        for model_name in config.MODEL_NAMES:
            clean_vals = [fm[metric]
                          for fm in exp1[model_name]["fold_metrics"]]
            corr_vals  = [fm[metric]
                          for fm in exp2[model_name]["fold_metrics"]]
            all_data.append(clean_vals)
            all_data.append(corr_vals)
            tick_labels.append(f"{model_name}\nClean")
            tick_labels.append(f"{model_name}\nCorrupted")
            colors.extend([COLOURS[model_name], COLOURS[model_name]])

        bp = ax.boxplot(all_data, patch_artist=True, notch=False,
                        medianprops={"color": "black", "linewidth": 2})

        for patch, color, i in zip(bp["boxes"], colors, range(len(all_data))):
            patch.set_facecolor(color)
            patch.set_alpha(0.7 if i % 2 == 0 else 0.4)

        ax.set_xticks(range(1, len(tick_labels) + 1))
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_ylabel(mlabel, fontsize=12)
        ax.set_title(f"{mlabel} Distribution Across Outer Folds", fontsize=13)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        legend_patches = [
            mpatches.Patch(color=COLOURS[m], label=m, alpha=0.8)
            for m in config.MODEL_NAMES
        ]
        ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
        fig.tight_layout()
        path = os.path.join(PLOTS_DIR, f"error_bars_{metric}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 4. Confusion matrices
# ---------------------------------------------------------------------------

def plot_confusion_matrices(exp1: dict, exp2: dict) -> None:
    """
    Aggregate confusion matrices across all outer folds (element-wise sum),
    then normalise each row so entries show the fraction of true-class samples
    predicted as each class.  One subplot grid per experiment:
      rows = models,  cols = [Clean, Corrupted].
    """
    n_models = len(config.MODEL_NAMES)
    fig, axes = plt.subplots(n_models, 2,
                             figsize=(10, 4.5 * n_models),
                             squeeze=False)

    for row, model_name in enumerate(config.MODEL_NAMES):
        for col, (exp_tag, exp_data) in enumerate(
                [("Clean", exp1), ("Corrupted", exp2)]):
            ax = axes[row][col]

            # Sum all per-fold confusion matrices
            cms = exp_data[model_name]["fold_cms"]
            agg_cm = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES),
                              dtype=np.int64)
            for cm in cms:
                agg_cm += np.array(cm, dtype=np.int64)

            # Row-normalise to percentages
            row_sums = agg_cm.sum(axis=1, keepdims=True)
            norm_cm  = np.where(row_sums > 0,
                                agg_cm.astype(float) / row_sums,
                                0.0)

            im = ax.imshow(norm_cm, interpolation="nearest",
                           cmap="Blues", vmin=0.0, vmax=1.0)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ticks = range(config.NUM_CLASSES)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            short = [c[:4] for c in config.CLASS_NAMES]
            ax.set_xticklabels(short, fontsize=8, rotation=45, ha="right")
            ax.set_yticklabels(short, fontsize=8)

            # Annotate each cell with the percentage value
            thresh = 0.5
            for i in range(config.NUM_CLASSES):
                for j in range(config.NUM_CLASSES):
                    val = norm_cm[i, j]
                    color = "white" if val > thresh else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=color)

            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("True", fontsize=9)
            ax.set_title(f"{model_name} | {exp_tag}", fontsize=11)

    fig.suptitle("Normalised Confusion Matrices (aggregated over 10 folds)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 5. HP selection frequency
# ---------------------------------------------------------------------------

def plot_hp_selection(exp1: dict, exp2: dict) -> None:
    """
    Bar charts showing how often each HP combination was selected
    during inner CV, per model and experiment.
    """
    n_models = len(config.MODEL_NAMES)
    fig, axes = plt.subplots(n_models, 2,
                             figsize=(14, 3.5 * n_models),
                             squeeze=False)

    for row, model_name in enumerate(config.MODEL_NAMES):
        for col, (exp_tag, exp_data) in enumerate(
                [("Clean", exp1), ("Corrupted", exp2)]):
            ax     = axes[row][col]
            counts = exp_data[model_name]["hp_counts"]

            # Parse readable labels from the stored key strings
            labels = []
            values = []
            for k, v in sorted(counts.items(), key=lambda x: -x[1]):
                # key looks like "[('dropout', 0.4), ('lr', 0.001)]"
                try:
                    pairs = eval(k)          # list of (param, value) tuples
                    label = ", ".join(f"{p}={val}" for p, val in pairs)
                except Exception:
                    label = k
                labels.append(label)
                values.append(v)

            ax.barh(labels, values, color=COLOURS[model_name], alpha=0.8)
            ax.set_xlabel("Times Selected", fontsize=10)
            ax.set_title(f"{model_name} | {exp_tag}", fontsize=11)
            ax.grid(axis="x", linestyle="--", alpha=0.5)

    fig.suptitle("Hyperparameter Selection Frequency (inner CV)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "hp_selection.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 6. Text summary table
# ---------------------------------------------------------------------------

def write_summary_table(exp1: dict, exp2: dict) -> None:
    lines = []
    header = (f"{'Model':<10}  {'Experiment':>12}  " +
              "  ".join(f"{m:>18}" for m in METRIC_LABELS))
    sep    = "─" * len(header)
    lines  += [sep, header, sep]

    for model_name in config.MODEL_NAMES:
        for exp_tag, exp_data in [("Clean", exp1), ("Corrupted", exp2)]:
            agg = exp_data[model_name]["aggregated"]
            row = f"{model_name:<10}  {exp_tag:>12}  "
            row += "  ".join(
                f"{agg[m]['mean']:.4f} ± {agg[m]['std']:.4f}"
                for m in METRICS_DISPLAY
            )
            lines.append(row)
        lines.append("")

    lines.append(sep)

    path = os.path.join(RESULTS_DIR, "summary_table.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {path}")
    print("\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not check_results_exist():
        return

    print("Loading results …")
    exp1 = load_results(os.path.join(RESULTS_DIR, "exp1_clean.json"))
    exp2 = load_results(os.path.join(RESULTS_DIR, "exp2_corrupted.json"))

    print("\nGenerating plots …")
    plot_learning_curves(exp1, exp2)
    plot_metric_comparison(exp1, exp2)
    plot_error_bars(exp1, exp2)
    plot_confusion_matrices(exp1, exp2)
    plot_hp_selection(exp1, exp2)
    write_summary_table(exp1, exp2)

    print(f"\nAll outputs saved in: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
