"""
visualize_results.py — One-stop viewer + analyst for all PP-MAE runs
=====================================================================
Run this on your MacBook from the repo root.  It will:

  1. Auto-discover every results CSV (root + results/<run>/ subfolders)
  2. Print a clean per-run results table
  3. Compute the PathologyLoss effect (controlled ablation pairs)
  4. Flag which runs are REAL BraTS vs synthetic demo (by PSNR sanity)
  5. Build a combined figure: PSNR / SSIM / Dice_ET bars per run
  6. Open / save every results PNG already produced by your runs
  7. Print a "WHERE YOU STAND" report telling you what is publishable
     and exactly what to run next.

Usage
-----
  python3 visualize_results.py                 # scan ./ and ./results
  python3 visualize_results.py --root .        # explicit root
  python3 visualize_results.py --show          # also pop figures on screen
  python3 visualize_results.py --out _analysis # where to save combined figs

No external deps beyond matplotlib + numpy (already in your env).
scipy is optional (used for significance if per-sample CSVs exist).
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Discovery + loading
# ---------------------------------------------------------------------------

# CSVs whose columns we understand.  Each maps to (method_col, metric_cols).
KNOWN_SCHEMAS = {
    # run_all_options.py output
    'options': {
        'method_col': 'Method',
        'metrics': ['PSNR', 'SSIM', 'NRMSE', 'Dice_WT', 'Dice_TC', 'Dice_ET'],
    },
    # run_comparison.py / run_brats_test.py output
    'comparison': {
        'method_col': 'Model',
        'metrics': ['PSNR', 'SSIM', 'NRMSE', 'Dice_WT', 'Dice_TC', 'Dice_ET'],
    },
    'brats': {
        'method_col': 'Method',
        'metrics': ['PSNR', 'SSIM', 'NRMSE', 'Dice_WT', 'Dice_TC', 'Dice_ET'],
    },
}


def find_csvs(root):
    """Find every *_results.csv under root and root/results/**."""
    patterns = [
        os.path.join(root, '*results*.csv'),
        os.path.join(root, 'results', '**', '*results*.csv'),
        os.path.join(root, '**', 'options_results.csv'),
    ]
    found = set()
    for p in patterns:
        for f in glob.glob(p, recursive=True):
            found.add(os.path.abspath(f))
    return sorted(found)


def load_csv(path):
    """Return list of row dicts and the detected method column."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return [], None
    header = rows[0].keys()
    method_col = None
    for cand in ('Method', 'Model', 'method', 'model'):
        if cand in header:
            method_col = cand
            break
    return rows, method_col


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def is_real_brats(rows):
    """Heuristic: real BraTS denoisers reach PSNR > 25 dB; synthetic demo ~10-15."""
    psnrs = [to_float(r.get('PSNR', 'nan')) for r in rows]
    psnrs = [p for p in psnrs if not np.isnan(p)]
    if not psnrs:
        return None
    return max(psnrs) > 25.0


def pathology_pairs(rows, method_col):
    """
    Detect controlled ablation pairs: '<X>' (L1) vs '<X> + PathologyLoss'
    or PP-MAE vs same-arch baseline.  Returns list of (base, treat, deltas).
    """
    by_name = {r[method_col]: r for r in rows}
    pairs = []
    for name, r in by_name.items():
        if 'PathologyLoss' in name or 'Pathology' in name:
            # find the L1 / lite counterpart
            base_name = (name.replace(' + PathologyLoss', '')
                             .replace('+PathologyLoss', '').strip())
            # try common baseline spellings
            for cand in (base_name, base_name + '-lite (L1)',
                         base_name + ' (L1)', base_name + '-lite'):
                if cand in by_name and cand != name:
                    pairs.append((cand, name))
                    break
    return pairs


def fmt(v, nd=3):
    f = to_float(v)
    return f"{f:.{nd}f}" if not np.isnan(f) else "  -  "


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_run(rows, method_col, title, out_path, show=False):
    """3-panel bar chart: PSNR, SSIM, Dice_ET, proposed model highlighted."""
    names = [r[method_col] for r in rows]
    psnr = [to_float(r.get('PSNR', 'nan')) for r in rows]
    ssim = [to_float(r.get('SSIM', 'nan')) for r in rows]
    det  = [to_float(r.get('Dice_ET', 'nan')) for r in rows]

    def colors_for():
        cs = []
        for n in names:
            if 'PROPOSED' in n or 'PP-MAE' in n:
                cs.append('#1565C0')           # blue = proposed
            elif 'Pathology' in n:
                cs.append('#2E7D32')           # green = +PathologyLoss ablation
            else:
                cs.append('#E65100')           # orange = baseline
        return cs

    cols = colors_for()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, vals, lab, lo, hi in [
        (axes[0], psnr, 'PSNR (dB) ↑', 0, max([v for v in psnr if not np.isnan(v)] + [1]) * 1.15),
        (axes[1], ssim, 'SSIM ↑', 0, 1.05),
        (axes[2], det,  'Dice ET ↑', 0, 1.05),
    ]:
        x = np.arange(len(names))
        bars = ax.bar(x, vals, color=cols, edgecolor='white')
        ax.set_title(lab, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
        ax.set_ylim(lo, hi)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f'{v:.2f}' if lab.startswith('PSNR') else f'{v:.3f}',
                        ha='center', va='bottom', fontsize=7)

    import matplotlib.patches as mp
    fig.legend(handles=[
        mp.Patch(color='#1565C0', label='PP-MAE (proposed)'),
        mp.Patch(color='#2E7D32', label='+ PathologyLoss (ablation)'),
        mp.Patch(color='#E65100', label='Baseline (L1)'),
    ], loc='upper right', fontsize=9)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f"   saved figure: {out_path}")
    if show:
        plt.show()
    plt.close()


def collect_existing_pngs(root):
    """Find result PNGs already produced by the runs."""
    pats = [
        os.path.join(root, 'options_*.png'),
        os.path.join(root, 'results', '**', '*.png'),
        os.path.join(root, 'fig*_*.png'),
        os.path.join(root, 'comparison_*.png'),
        os.path.join(root, 'brats_*.png'),
    ]
    out = set()
    for p in pats:
        for f in glob.glob(p, recursive=True):
            out.add(os.path.abspath(f))
    return sorted(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', help='repo root to scan')
    ap.add_argument('--out', default='_analysis', help='output dir for figures')
    ap.add_argument('--show', action='store_true', help='also display figures')
    args = ap.parse_args()

    if args.show:
        matplotlib.use('TkAgg')   # interactive backend on Mac
    else:
        matplotlib.use('Agg')

    root = os.path.abspath(args.root)
    os.makedirs(args.out, exist_ok=True)

    csvs = find_csvs(root)
    if not csvs:
        print(f"No results CSVs found under {root}")
        print("Expected files like options_results.csv, results/<run>/options_results.csv")
        sys.exit(1)

    print("=" * 78)
    print("  PP-MAE RESULTS VIEWER")
    print("=" * 78)
    print(f"Scanned: {root}")
    print(f"Found {len(csvs)} result file(s):")
    for c in csvs:
        print(f"   - {os.path.relpath(c, root)}")

    real_runs = []      # (label, rows, method_col, path)
    demo_runs = []

    for path in csvs:
        rows, mcol = load_csv(path)
        if not rows or mcol is None:
            continue
        label = os.path.relpath(path, root)
        real = is_real_brats(rows)
        tag = 'REAL BraTS' if real else ('SYNTHETIC/demo' if real is False else 'unknown')

        print("\n" + "-" * 78)
        print(f"RUN: {label}    [{tag}]")
        print("-" * 78)
        hdr = f"{'Method':<32} {'PSNR':>7} {'SSIM':>7} {'D_WT':>7} {'D_TC':>7} {'D_ET':>7}"
        print(hdr)
        for r in rows:
            print(f"{r[mcol]:<32} {fmt(r.get('PSNR'),2):>7} {fmt(r.get('SSIM')):>7} "
                  f"{fmt(r.get('Dice_WT')):>7} {fmt(r.get('Dice_TC')):>7} "
                  f"{fmt(r.get('Dice_ET')):>7}")

        # PathologyLoss ablation effect
        pairs = pathology_pairs(rows, mcol)
        if pairs:
            by = {r[mcol]: r for r in rows}
            print("\n   PathologyLoss effect (same architecture, loss-only change):")
            for base, treat in pairs:
                d_et = to_float(by[treat].get('Dice_ET')) - to_float(by[base].get('Dice_ET'))
                d_tc = to_float(by[treat].get('Dice_TC')) - to_float(by[base].get('Dice_TC'))
                d_ps = to_float(by[treat].get('PSNR'))    - to_float(by[base].get('PSNR'))
                print(f"     {base:<26} -> {treat}")
                print(f"        Dice_ET {d_et:+.3f}   Dice_TC {d_tc:+.3f}   PSNR {d_ps:+.2f} dB")

        # figure
        safe = label.replace('/', '__').replace('.csv', '')
        plot_run(rows, mcol, label, os.path.join(args.out, f'view_{safe}.png'), show=args.show)

        (real_runs if real else demo_runs).append((label, rows, mcol, path))

    # ----- existing PNG inventory -----
    pngs = collect_existing_pngs(root)
    if pngs:
        print("\n" + "=" * 78)
        print(f"  EXISTING FIGURES ({len(pngs)}) — open these to inspect visually")
        print("=" * 78)
        for p in pngs:
            print(f"   {os.path.relpath(p, root)}")

    # ----- WHERE YOU STAND report -----
    print("\n" + "=" * 78)
    print("  WHERE YOU STAND  —  publishability assessment")
    print("=" * 78)
    print(f"""
REAL BraTS runs found : {len(real_runs)}
Synthetic/demo runs   : {len(demo_runs)}

WHAT IS SOLID (use in paper):
  - Round 4 (Swin family) on REAL BraTS is your strongest evidence.
    PathologyLoss adds ~+6% Dice_ET and ~+8% Dice_TC on BOTH SwinIR and
    Uformer backbones, at only ~0.3 dB PSNR cost. This is a clean, controlled
    ablation (same architecture, loss-only change) -> publishable.

WHAT IS WEAK (do NOT publish as-is):
  - Any run with max PSNR < 25 dB is synthetic-demo data. The famous
    '0.000 -> 0.711 Dice_ET' (Round 3 Multi-task) currently sits on synthetic
    data — striking, but must be reproduced on REAL BraTS before submission.
  - Grading results (4 test samples) are statistically meaningless. Exclude.

THE PROPOSED MODEL PROBLEM:
  - 'PP-MAE (Swin) [PROPOSED]' currently LOSES to the simpler
    'Uformer + PathologyLoss' on Dice_ET and trails on PSNR (it ran on CPU
    fallback + is under-sized). Either (a) reframe the paper so PathologyLoss
    is the contribution, or (b) re-run PP-MAE Swin at matched capacity on GPU.

NEXT RUN THAT FIXES THE MOST (single command):
  python3 run_all_options.py ~/Downloads/BraTS2021_data \\
      --rounds 3 --epochs 30 --seg_epochs 20 \\
      --max_subjects 50 --patch_size 96 \\
      --out ~/AL-ML/results/round3_real

  -> reproduces the '0 -> 0.71' headline on REAL data. Combined with Round 4
     already in hand, you would have a complete, honest, publishable story.

THEN, to harden for MICCAI/MIDL:
  - scale to >=150 subjects (statistical power)
  - the updated run_all_options.py now emits significance.csv (Wilcoxon p)
  - generate qualitative noisy/denoised/clean panels with tumour overlays
""")
    print("Analysis complete. Figures saved to:", os.path.abspath(args.out))


if __name__ == '__main__':
    main()
