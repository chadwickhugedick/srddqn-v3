"""
plot_ablation.py
----------------
Reads logs/ablation_results.csv and generates a thesis-ready
comparison bar chart across all 5 feature modes.

Usage:
    python scripts/plot_ablation.py
    python scripts/plot_ablation.py --results logs/ablation_results.csv
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODE_LABELS = {
    "indicators":      "Indicators\nOnly",
    "raw_ratios":      "Raw Ratios\n(scale-inv.)",
    "raw_prices":      "Raw Prices\n(z-score)",
    "combined_ratios": "Combined\n(+ Ratios)",
    "combined_prices": "Combined\n(+ Prices)",
}

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def plot_ablation(results_path: str = "logs/ablation_results.csv",
                  output_dir: str = "logs/plots"):

    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        print("Run `python training/run_ablation.py` first.")
        return

    df = pd.read_csv(results_path)

    # Keep only rows with valid numeric results
    metric_cols = ['cr', 'ar', 'sr', 'mdd', 'best_val_mse']
    df_valid = df.dropna(subset=[c for c in metric_cols if c in df.columns])

    if df_valid.empty:
        print("No valid rows found in results CSV.")
        return

    # Order modes as defined (only include modes present in the data)
    order = [m for m in MODE_LABELS if m in df_valid['mode'].values]
    df_valid = df_valid.set_index('mode').reindex(order).reset_index()

    labels = [MODE_LABELS.get(m, m) for m in df_valid['mode']]
    x      = np.arange(len(labels))
    colors = PALETTE[:len(labels)]

    os.makedirs(output_dir, exist_ok=True)

    # ── Figure 1: CR / AR / Sharpe / MDD comparison ────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("SRDDQN — 5-Mode Feature Ablation Results", fontsize=15, fontweight='bold')

    metrics = [
        ('cr',  'Cumulative Return (%)',   True,  axes[0, 0]),
        ('ar',  'Annualised Return (%)',   True,  axes[0, 1]),
        ('sr',  'Sharpe Ratio',            True,  axes[1, 0]),
        ('mdd', 'Max Drawdown (%)',        False, axes[1, 1]),  # lower is better
    ]

    for col, ylabel, higher_better, ax in metrics:
        if col not in df_valid.columns:
            ax.set_visible(False)
            continue

        vals = df_valid[col].values
        if col in ('cr', 'ar', 'mdd'):
            vals = vals * 100  # convert to percent

        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.8, width=0.6)
        ax.set_title(ylabel, fontsize=11, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Annotate bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(abs(vals)) * 0.01),
                    f"{val:.1f}",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Highlight best bar
        best_idx = np.argmin(vals) if not higher_better else np.argmax(vals)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(2.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(output_dir, "ablation_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison chart: {out_path}")

    # ── Figure 2: TimesNet val MSE comparison ───────────────────────────────
    if 'best_val_mse' in df_valid.columns:
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        vals = df_valid['best_val_mse'].values
        bars = ax2.bar(x, vals, color=colors, edgecolor='white', linewidth=0.8, width=0.6)
        ax2.set_title("TimesNet Reward Network — Best Validation MSE by Feature Mode",
                      fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=10)
        ax2.set_ylabel("Val MSE (lower = better)", fontsize=10)
        ax2.grid(axis='y', linestyle='--', alpha=0.4)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + (max(vals) * 0.01),
                     f"{val:.4f}",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

        best_idx = np.argmin(vals)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(2.5)

        plt.tight_layout()
        mse_path = os.path.join(output_dir, "ablation_timesnet_mse.png")
        plt.savefig(mse_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved TimesNet MSE chart: {mse_path}")

    # ── Print summary table ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print("ABLATION RESULTS SUMMARY")
    print("="*70)
    display = df_valid[['mode'] + [c for c in ['best_val_mse', 'cr', 'ar', 'sr', 'mdd', 'elapsed_min']
                                   if c in df_valid.columns]].copy()
    for pct_col in ['cr', 'ar', 'mdd']:
        if pct_col in display.columns:
            display[pct_col] = (display[pct_col] * 100).round(2).astype(str) + '%'
    print(display.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="logs/ablation_results.csv",
                        help="Path to ablation results CSV")
    parser.add_argument("--output-dir", default="logs/plots",
                        help="Directory to save plots")
    args = parser.parse_args()
    plot_ablation(results_path=args.results, output_dir=args.output_dir)
