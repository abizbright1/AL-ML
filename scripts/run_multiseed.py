#!/usr/bin/env python3
"""
run_multiseed.py  —  Multi-seed Round 4 runner + aggregator
=============================================================
Runs run_all_options.py three times (seeds 42, 7, 123) then
aggregates mean ± std across seeds and saves:

    results/multiseed/
        seed42/options_results.csv
        seed7/options_results.csv
        seed123/options_results.csv
        aggregated_results.csv          ← mean ± std per metric per model
        multiseed_bars.png              ← bar chart with error bars
        multiseed_table.png             ← publication table figure

Usage
-----
    python3 run_multiseed.py ~/Downloads/BraTS2021_data \\
        --device mps --epochs 30 --seg_epochs 20 \\
        --max_subjects 50 --out results/multiseed

    # Quick smoke-test
    python3 run_multiseed.py ~/Downloads/BraTS2021_data \\
        --device mps --epochs 2 --seg_epochs 2 \\
        --max_subjects 5 --out results/multiseed_smoke
"""

import argparse
import os
import subprocess
import sys
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEEDS  = [42, 7, 123]
SCRIPT = os.path.join(os.path.dirname(__file__), 'run_all_options.py')
METRICS = ['PSNR', 'SSIM', 'NRMSE', 'Dice_WT', 'Dice_TC', 'Dice_ET']


# ── Run one seed ──────────────────────────────────────────────────────────────

def run_seed(data_dir, seed, args, out_root):
    seed_out = os.path.join(out_root, f'seed{seed}')
    os.makedirs(seed_out, exist_ok=True)

    cmd = [
        sys.executable, SCRIPT,
        data_dir,
        '--rounds', '4',
        '--epochs',       str(args.epochs),
        '--seg_epochs',   str(args.seg_epochs),
        '--max_subjects', str(args.max_subjects),
        '--patch_size',   str(args.patch_size),
        '--device',       args.device,
        '--seed',         str(seed),
        '--out',          seed_out,
    ]
    print(f"\n{'='*60}")
    print(f"  Running seed={seed}  →  {seed_out}")
    print(f"{'='*60}\n", flush=True)
    result = subprocess.run(cmd, check=True)
    return os.path.join(seed_out, 'options_results.csv')


# ── Read CSV ──────────────────────────────────────────────────────────────────

def read_csv(path):
    """Returns list of dicts, one per model row."""
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


# ── Aggregate ─────────────────────────────────────────────────────────────────

def aggregate(csv_paths, out_dir):
    """Compute mean ± std across seeds for each model/metric."""
    all_data = defaultdict(lambda: defaultdict(list))

    for path in csv_paths:
        for row in read_csv(path):
            method = row['Method']
            for m in METRICS:
                try:
                    all_data[method][m].append(float(row[m]))
                except (KeyError, ValueError):
                    pass

    methods = list(all_data.keys())
    agg = {}
    for method in methods:
        agg[method] = {}
        for m in METRICS:
            vals = all_data[method][m]
            agg[method][m] = {
                'mean': float(np.mean(vals)),
                'std':  float(np.std(vals)),
                'vals': vals,
            }

    # Save aggregated CSV
    agg_path = os.path.join(out_dir, 'aggregated_results.csv')
    with open(agg_path, 'w', newline='') as f:
        cols = ['Method'] + [f'{m}_mean' for m in METRICS] + [f'{m}_std' for m in METRICS]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for method in methods:
            row = {'Method': method}
            for m in METRICS:
                row[f'{m}_mean'] = f"{agg[method][m]['mean']:.4f}"
                row[f'{m}_std']  = f"{agg[method][m]['std']:.4f}"
            w.writerow(row)
    print(f"\n  Saved {agg_path}")
    return agg, methods


# ── Figures ───────────────────────────────────────────────────────────────────

COLOURS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
PROPOSED_COLOUR = '#D32F2F'

def _short(name):
    return (name.replace(' [PROPOSED]', '')
                .replace('PP-MAE (Swin)', 'PP-MAE')
                .replace('-lite (L1)', '-lite')
                .replace(' + PathologyLoss', '+PathLoss'))


def plot_bars(agg, methods, out_dir):
    """Bar chart with error bars for key metrics."""
    key_metrics = ['PSNR', 'Dice_TC', 'Dice_ET']
    titles      = ['PSNR (dB) ↑', 'Dice_TC ↑', 'Dice_ET ↑']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Round 4 Multi-Seed Results  (mean ± std, n=3 seeds)', fontsize=13, fontweight='bold')

    x = np.arange(len(methods))
    colours = [PROPOSED_COLOUR if 'PROPOSED' in m else COLOURS[i % len(COLOURS)]
               for i, m in enumerate(methods)]

    for ax, metric, title in zip(axes, key_metrics, titles):
        means = [agg[m][metric]['mean'] for m in methods]
        stds  = [agg[m][metric]['std']  for m in methods]
        bars  = ax.bar(x, means, yerr=stds, capsize=5, color=colours,
                       edgecolor='black', linewidth=0.5, error_kw=dict(elinewidth=1.5))
        ax.set_xticks(x)
        ax.set_xticklabels([_short(m) for m in methods], rotation=30, ha='right', fontsize=8)
        ax.set_title(title, fontweight='bold')
        ax.set_ylim(min(means) * 0.95, max(means) * 1.05)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
        # Annotate bars
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.002,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    path = os.path.join(out_dir, 'multiseed_bars.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def plot_table(agg, methods, out_dir):
    """Publication-style table figure."""
    col_labels = ['Method', 'PSNR', 'SSIM', 'Dice_WT', 'Dice_TC', 'Dice_ET']
    rows = []
    for method in methods:
        row = [_short(method)]
        for m in ['PSNR', 'SSIM', 'Dice_WT', 'Dice_TC', 'Dice_ET']:
            mn = agg[method][m]['mean']
            sd = agg[method][m]['std']
            row.append(f'{mn:.3f} ± {sd:.3f}')
        rows.append(row)

    fig, ax = plt.subplots(figsize=(14, 0.5 * len(methods) + 1.5))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # Header style
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor('#1565C0')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')

    # Highlight proposed model
    for i, method in enumerate(methods):
        if 'PROPOSED' in method:
            for j in range(len(col_labels)):
                tbl[(i + 1, j)].set_facecolor('#FFF3E0')
                tbl[(i + 1, j)].set_text_props(fontweight='bold')

    ax.set_title(f'Table: Multi-Seed Mean ± Std  (n={len(SEEDS)} seeds: {SEEDS})',
                 fontsize=11, fontweight='bold', pad=10)
    path = os.path.join(out_dir, 'multiseed_table.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def print_summary(agg, methods):
    w = 35
    print(f"\n{'='*75}")
    print(f"  MULTI-SEED SUMMARY  (seeds: {SEEDS})")
    print(f"{'='*75}")
    header = f"{'Method':<{w}}  {'PSNR':>12}  {'Dice_TC':>12}  {'Dice_ET':>12}"
    print(header)
    print('-' * 75)
    for method in methods:
        psnr  = agg[method]['PSNR']
        dtc   = agg[method]['Dice_TC']
        det   = agg[method]['Dice_ET']
        tag   = '  <-- PP-MAE' if 'PROPOSED' in method else ''
        print(f"  {_short(method):<{w-2}}  "
              f"{psnr['mean']:>6.3f}±{psnr['std']:.3f}  "
              f"{dtc['mean']:>6.3f}±{dtc['std']:.3f}  "
              f"{det['mean']:>6.3f}±{det['std']:.3f}{tag}")
    print('=' * 75)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Multi-seed Round 4 runner',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('data_dir',      help='BraTS data directory')
    ap.add_argument('--device',      default='mps')
    ap.add_argument('--epochs',      type=int, default=30)
    ap.add_argument('--seg_epochs',  type=int, default=20)
    ap.add_argument('--max_subjects',type=int, default=50)
    ap.add_argument('--patch_size',  type=int, default=96)
    ap.add_argument('--out',         default='results/multiseed')
    ap.add_argument('--seeds',       default='42,7,123',
                    help='Comma-separated seeds to run')
    args = ap.parse_args()

    global SEEDS
    SEEDS = [int(s) for s in args.seeds.split(',')]
    os.makedirs(args.out, exist_ok=True)

    csv_paths = []
    for seed in SEEDS:
        csv_path = run_seed(args.data_dir, seed, args, args.out)
        csv_paths.append(csv_path)

    print(f"\n{'='*60}")
    print("  Aggregating results across seeds...")
    print(f"{'='*60}")

    agg, methods = aggregate(csv_paths, args.out)
    plot_bars(agg, methods, args.out)
    plot_table(agg, methods, args.out)
    print_summary(agg, methods)

    print(f"\nAll outputs saved to: {args.out}")
    print("Run complete.\n")


if __name__ == '__main__':
    main()
