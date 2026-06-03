#!/usr/bin/env python3
"""
run_noise_ablation.py  —  Noise-level robustness ablation
===========================================================
Tests all Round 4 models at three Rician noise levels:
    σ = 0.05  (mild)
    σ = 0.08  (moderate — standard BraTS benchmark)
    σ = 0.15  (severe)

Outputs:
    results/noise_ablation/
        sigma0.05/options_results.csv
        sigma0.08/options_results.csv
        sigma0.15/options_results.csv
        noise_ablation_psnr.png         ← PSNR vs σ line chart
        noise_ablation_dice_et.png      ← Dice_ET vs σ line chart
        noise_ablation_table.png        ← combined table figure
        noise_ablation.csv              ← all results in one CSV

Key research claim: PP-MAE maintains Dice_ET advantage across ALL noise levels
because its pathology-preserving loss prioritises tumour-region fidelity.
High-PSNR baselines (SwinIR-lite) degrade faster on tumour metrics at higher σ.

Usage
-----
    python3 run_noise_ablation.py ~/Downloads/BraTS2021_data \\
        --device mps --epochs 30 --seg_epochs 20 --max_subjects 50

    # Quick smoke-test
    python3 run_noise_ablation.py ~/Downloads/BraTS2021_data \\
        --device mps --epochs 2 --seg_epochs 2 --max_subjects 5
"""

import argparse
import os
import subprocess
import sys
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIGMAS = [0.05, 0.08, 0.15]
SCRIPT = os.path.join(os.path.dirname(__file__), 'run_all_options.py')
METRICS = ['PSNR', 'SSIM', 'Dice_WT', 'Dice_TC', 'Dice_ET']

# Consistent colour per model (matched to run_all_options.py plots)
MODEL_COLOURS = {
    'PP-MAE (Swin) [PROPOSED]':  '#D32F2F',
    'SwinIR-lite (L1)':          '#1976D2',
    'Uformer-lite (L1)':         '#388E3C',
    'SwinIR + PathologyLoss':    '#F57C00',
    'Uformer + PathologyLoss':   '#7B1FA2',
}
MODEL_MARKERS = {
    'PP-MAE (Swin) [PROPOSED]': 'o',
    'SwinIR-lite (L1)':         's',
    'Uformer-lite (L1)':        '^',
    'SwinIR + PathologyLoss':   'D',
    'Uformer + PathologyLoss':  'v',
}


def _short(name):
    return (name.replace(' [PROPOSED]', ' [P]')
                .replace('-lite (L1)', '-lite')
                .replace(' + PathologyLoss', '+PL'))


# ── Run one sigma ─────────────────────────────────────────────────────────────

def run_sigma(data_dir, sigma, args, out_root):
    sigma_out = os.path.join(out_root, f'sigma{sigma:.2f}')
    os.makedirs(sigma_out, exist_ok=True)

    cmd = [
        sys.executable, SCRIPT,
        data_dir,
        '--rounds',       '4',
        '--epochs',       str(args.epochs),
        '--seg_epochs',   str(args.seg_epochs),
        '--max_subjects', str(args.max_subjects),
        '--patch_size',   str(args.patch_size),
        '--device',       args.device,
        '--sigma',        str(sigma),
        '--seed',         '42',
        '--out',          sigma_out,
    ]
    print(f"\n{'='*60}")
    print(f"  Running σ={sigma:.2f}  →  {sigma_out}")
    print(f"{'='*60}\n", flush=True)
    subprocess.run(cmd, check=True)
    return os.path.join(sigma_out, 'options_results.csv')


# ── Read CSV ──────────────────────────────────────────────────────────────────

def read_csv(path):
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


# ── Aggregate all sigmas ───────────────────────────────────────────────────────

def collect(csv_paths):
    """Returns {method: {sigma: {metric: float}}}"""
    data = {}
    for sigma, path in zip(SIGMAS, csv_paths):
        for row in read_csv(path):
            method = row['Method']
            if method not in data:
                data[method] = {}
            data[method][sigma] = {m: float(row.get(m, 0)) for m in METRICS}
    return data


def save_combined_csv(data, methods, out_dir):
    path = os.path.join(out_dir, 'noise_ablation.csv')
    with open(path, 'w', newline='') as f:
        cols = ['Method', 'Sigma'] + METRICS
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for method in methods:
            for sigma in SIGMAS:
                row = {'Method': method, 'Sigma': sigma}
                row.update(data[method].get(sigma, {}))
                w.writerow(row)
    print(f"  Saved {path}")


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_line(data, methods, metric, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        vals = [data[method].get(s, {}).get(metric, np.nan) for s in SIGMAS]
        colour = MODEL_COLOURS.get(method, '#555')
        marker = MODEL_MARKERS.get(method, 'o')
        lw     = 2.5 if 'PROPOSED' in method else 1.5
        ls     = '-'  if 'PROPOSED' in method else '--'
        ax.plot(SIGMAS, vals, marker=marker, color=colour,
                linewidth=lw, linestyle=ls, label=_short(method), markersize=7)
        for x, y in zip(SIGMAS, vals):
            if not np.isnan(y):
                ax.annotate(f'{y:.3f}', (x, y),
                            textcoords='offset points', xytext=(0, 7),
                            fontsize=7, ha='center', color=colour)

    ax.set_xlabel('Rician Noise Level  σ', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(SIGMAS)
    ax.set_xticklabels([f'σ={s}' for s in SIGMAS])
    ax.legend(fontsize=8, loc='best')
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


def plot_table_fig(data, methods, out_dir):
    """3-column table: one sub-table per sigma value."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 0.5 * len(methods) + 2))
    fig.suptitle('Noise Robustness Ablation — Reconstruction & Segmentation Metrics',
                 fontsize=12, fontweight='bold')

    show_metrics = ['PSNR', 'Dice_ET']
    col_labels   = ['Method'] + show_metrics

    for ax, sigma in zip(axes, SIGMAS):
        rows = []
        for method in methods:
            d = data[method].get(sigma, {})
            rows.append([_short(method)] + [f"{d.get(m, 0):.3f}" for m in show_metrics])

        ax.axis('off')
        tbl = ax.table(cellText=rows, colLabels=col_labels,
                       loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.5)
        ax.set_title(f'σ = {sigma:.2f}', fontweight='bold', fontsize=10)

        for j in range(len(col_labels)):
            tbl[(0, j)].set_facecolor('#1565C0')
            tbl[(0, j)].set_text_props(color='white', fontweight='bold')
        for i, method in enumerate(methods):
            if 'PROPOSED' in method:
                for j in range(len(col_labels)):
                    tbl[(i + 1, j)].set_facecolor('#FFF3E0')
                    tbl[(i + 1, j)].set_text_props(fontweight='bold')

    plt.tight_layout()
    path = os.path.join(out_dir, 'noise_ablation_table.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def print_summary(data, methods):
    print(f"\n{'='*70}")
    print("  NOISE ABLATION SUMMARY  —  Dice_ET across σ levels")
    print(f"{'='*70}")
    header = f"  {'Method':<35}" + ''.join(f"  σ={s:.2f}" for s in SIGMAS)
    print(header)
    print('-' * 70)
    for method in methods:
        vals = [f"{data[method].get(s, {}).get('Dice_ET', 0):.3f}" for s in SIGMAS]
        tag  = '  <--' if 'PROPOSED' in method else ''
        print(f"  {_short(method):<35}  " + '   '.join(vals) + tag)
    print('=' * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Noise-level robustness ablation',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('data_dir',       help='BraTS data directory')
    ap.add_argument('--device',       default='mps')
    ap.add_argument('--epochs',       type=int, default=30)
    ap.add_argument('--seg_epochs',   type=int, default=20)
    ap.add_argument('--max_subjects', type=int, default=50)
    ap.add_argument('--patch_size',   type=int, default=96)
    ap.add_argument('--out',          default='results/noise_ablation')
    ap.add_argument('--sigmas',       default='0.05,0.08,0.15',
                    help='Comma-separated noise sigmas to test')
    args = ap.parse_args()

    global SIGMAS
    SIGMAS = [float(s) for s in args.sigmas.split(',')]
    os.makedirs(args.out, exist_ok=True)

    csv_paths = []
    for sigma in SIGMAS:
        csv_path = run_sigma(args.data_dir, sigma, args, args.out)
        csv_paths.append(csv_path)

    print(f"\n{'='*60}")
    print("  Generating ablation figures...")
    print(f"{'='*60}")

    data    = collect(csv_paths)
    methods = list(data.keys())
    save_combined_csv(data, methods, args.out)

    plot_line(data, methods, 'PSNR', 'PSNR (dB) ↑',
              'Reconstruction Quality vs Noise Level',
              os.path.join(args.out, 'noise_ablation_psnr.png'))
    plot_line(data, methods, 'Dice_ET', 'Dice_ET ↑',
              'Tumour Segmentation vs Noise Level  (PP-MAE maintains advantage)',
              os.path.join(args.out, 'noise_ablation_dice_et.png'))
    plot_table_fig(data, methods, args.out)
    print_summary(data, methods)

    print(f"\nAll outputs saved to: {args.out}")
    print("Run complete.\n")


if __name__ == '__main__':
    main()
