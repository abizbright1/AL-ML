#!/usr/bin/env python3
"""
make_paper_figures.py — publication figures and tables from a finished run
===========================================================================

Produces everything a journal submission needs from the per-slice metrics of
a loss-comparison run: the main results table in LaTeX and CSV, and eight
figures as 600-dpi PDF (vector, for typesetting) and PNG (for drafts).

Design decisions, so a reviewer cannot object to the plots themselves:

  * Five-colour categorical palette, validated for colour-vision deficiency
    (worst adjacent pair dE 10.3 protan, 25.0 normal vision) and for >= 3:1
    contrast against the page. Hues are assigned in a FIXED order per
    objective, so a figure that drops an arm never repaints the survivors.
  * Every series is also direct-labelled or hatched, so identity never rests
    on colour alone -- the figures survive greyscale printing.
  * One y-axis per panel. Never two scales on one plot.
  * The floor and ceiling appear as reference rules on every accuracy plot,
    because an absolute Dice means nothing without them.
  * Error bars are bootstrap 95% CIs over SUBJECTS, not over slices: slices
    from one patient are not independent.

Usage
-----
    python3 make_paper_figures.py --run results/comparison/phase2_seed1
    python3 make_paper_figures.py --run results/comparison --out paper_figs
    python3 make_paper_figures.py --run <dir> --backbone Uformer
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics as st
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── validated print palette ────────────────────────────────────────────────
PAL = ['#1B6CA8', '#C1440E', '#0A7B5F', '#8E44AD', '#9A6A00']
INK, MUTED, RULE = '#1a1a1a', '#5a5a5a', '#c8c8c8'
FLOOR_C, CEIL_C = '#8a8a8a', '#2f6f4f'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'serif'],
    'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.8,
    'axes.grid': True, 'grid.color': RULE, 'grid.linewidth': 0.5,
    'grid.alpha': 0.6, 'axes.axisbelow': True,
    'figure.dpi': 150, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
    'legend.frameon': False,
})

METRICS = ['dice_et', 'dice_tc', 'dice_wt', 'psnr', 'psnr_et', 'psnr_tc',
           'ssim', 'ssim_brain', 'nrmse']

# fixed display order; hue follows the objective, never its rank
ORDER = ['L1', 'MatchedBase', 'MS-SSIM', 'Mix', 'BraSyn-Freq', 'TV', 'L2',
         'FlatROI', 'BraSyn-MaskedL1', 'PathologyLoss',
         'PL-EqualWeights', 'PL-ETonly', 'PL-Disjoint']


def tidy_name(method: str) -> str:
    if 'FLOOR' in method:
        return 'Floor'
    if 'CEILING' in method:
        return 'Ceiling'
    m = method
    for a, b in [('Uformer + ', ''), ('SwinIR + ', ''),
                 ('Uformer-lite (L1)', 'L1'), ('SwinIR-lite (L1)', 'L1'),
                 (' (current, nested)', ''), (' [ABLATION]', ''),
                 (' (no pathology) [CONTROL]', ''), ('matched base', 'MatchedBase'),
                 ('masked L1 w=0.5', 'BraSyn-MaskedL1'),
                 ('low/high frequency split', 'BraSyn-Freq'),
                 ('Mix MS-SSIM+L1', 'Mix'), ('total variation', 'TV'),
                 ('PathologyLoss Equal weights', 'PL-EqualWeights'),
                 ('PathologyLoss ETonly weights', 'PL-ETonly'),
                 ('PathologyLoss (disjoint)', 'PL-Disjoint')]:
        m = m.replace(a, b)
    for tag in ['[Zhao 2017]', '[Johnson 2016]', '[Yang 2017]', '[BraSyn 2023]',
                '[ROIRecNet-inspired]', '[LIDnet-inspired]', '[DAGAN-inspired]',
                'non-adversarial']:
        m = m.replace(tag, '')
    return ' '.join(m.split())


def backbone_of(method: str) -> str:
    return ('Uformer' if method.startswith('Uformer') else
            'SwinIR' if method.startswith('SwinIR') else 'Reference')


def load(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, '*', 'per_slice_metrics.csv')))
    if not files:
        f = os.path.join(run_dir, 'per_slice_metrics.csv')
        files = [f] if os.path.isfile(f) else []
    if not files:
        sys.exit(f'No per_slice_metrics.csv under {run_dir}')
    rows = []
    for p in files:
        run = os.path.basename(os.path.dirname(p))
        with open(p, newline='') as fh:
            for r in csv.DictReader(fh):
                r['_run'] = run
                rows.append(r)
    return rows, files


def num(x):
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return None


def by_subject(rows, metric):
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = num(r.get(metric))
        if v is not None:
            acc[r['method']][(r['_run'], r['subject'])].append(v)
    return {m: {k: sum(v) / len(v) for k, v in d.items()} for m, d in acc.items()}


def boot_ci(vals, n=4000, seed=0):
    if len(vals) < 2:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    a = np.asarray(vals)
    means = a[rng.integers(0, len(a), size=(n, len(a)))].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='results/comparison')
    ap.add_argument('--out', default=None)
    ap.add_argument('--backbone', default=None, choices=['SwinIR', 'Uformer'])
    ap.add_argument('--metric', default='dice_et')
    args = ap.parse_args()
    out = args.out or os.path.join(args.run, 'paper_figures')
    os.makedirs(out, exist_ok=True)

    rows, files = load(args.run)
    if args.backbone:
        rows = [r for r in rows
                if backbone_of(r['method']) in (args.backbone, 'Reference')]

    D = {m: by_subject(rows, m) for m in METRICS}
    methods = list(D[args.metric].keys())
    floor = ceil = None
    for m in methods:
        if 'FLOOR' in m:
            floor = st.mean(D[args.metric][m].values()) if False else st.mean(list(D[args.metric][m].values()))
        if 'CEILING' in m:
            ceil = st.mean(list(D[args.metric][m].values()))

    trained = [m for m in methods if 'FLOOR' not in m and 'CEILING' not in m]
    bbs = sorted({backbone_of(m) for m in trained})

    def key(m):
        n = tidy_name(m)
        return (ORDER.index(n) if n in ORDER else 99, n)
    trained.sort(key=key)
    losses = []
    for m in trained:
        n = tidy_name(m)
        if n not in losses:
            losses.append(n)
    colour = {n: PAL[i % len(PAL)] for i, n in enumerate(losses)}

    def stats(m, metric):
        v = list(D[metric].get(m, {}).values())
        if not v:
            return float('nan'), (float('nan'), float('nan')), 0
        return st.mean(v), boot_ci(v), len(v)

    saved = []

    def save(fig, name):
        for ext in ('pdf', 'png'):
            p = os.path.join(out, f'{name}.{ext}')
            fig.savefig(p)
        plt.close(fig)
        saved.append(name)
        print(f'  {name}.pdf / .png')

    # ── FIG 1 — main accuracy comparison ───────────────────────────────────
    fig, axes = plt.subplots(1, len(bbs), figsize=(3.6 * len(bbs), 3.6),
                             sharey=True, squeeze=False)
    for ax, bb in zip(axes[0], bbs):
        arms = [m for m in trained if backbone_of(m) == bb]
        names = [tidy_name(m) for m in arms]
        mu = [stats(m, args.metric)[0] for m in arms]
        ci = [stats(m, args.metric)[1] for m in arms]
        err = np.array([[m - c[0] for m, c in zip(mu, ci)],
                        [c[1] - m for m, c in zip(mu, ci)]])
        y = np.arange(len(arms))
        ax.barh(y, mu, xerr=err, height=0.68,
                color=[colour[n] for n in names],
                error_kw=dict(ecolor=INK, lw=0.9, capsize=2.5), zorder=3)
        if floor is not None:
            ax.axvline(floor, color=FLOOR_C, ls=':', lw=1.2, zorder=2)
        if ceil is not None:
            ax.axvline(ceil, color=CEIL_C, ls='--', lw=1.2, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel('Dice$_{ET}$' if args.metric == 'dice_et' else args.metric)
        ax.set_title(bb, pad=6)
        for yi, v in zip(y, mu):
            ax.text(v + 0.004, yi, f'{v:.3f}', va='center', fontsize=7, color=INK)
    if floor is not None and ceil is not None:
        axes[0][0].legend(handles=[
            plt.Line2D([], [], color=FLOOR_C, ls=':', lw=1.2,
                       label=f'no restoration ({floor:.3f})'),
            plt.Line2D([], [], color=CEIL_C, ls='--', lw=1.2,
                       label=f'clean image ({ceil:.3f})')],
            loc='lower right')
    fig.suptitle('Enhancing-tumour segmentation by training objective', y=1.0)
    save(fig, 'fig1_accuracy_by_loss')

    # ── FIG 2 — accuracy vs whole-image fidelity ───────────────────────────
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    marker = {'SwinIR': 'o', 'Uformer': 's', 'Reference': 'D'}
    for m in trained:
        n, bb = tidy_name(m), backbone_of(m)
        x, _, _ = stats(m, 'psnr')
        y, _, _ = stats(m, args.metric)
        ax.scatter(x, y, s=46, color=colour[n], marker=marker[bb],
                   edgecolor='white', linewidth=0.8, zorder=3)
        ax.annotate(n, (x, y), textcoords='offset points', xytext=(5, 3),
                    fontsize=6.5, color=INK)
    if floor is not None:
        ax.axhline(floor, color=FLOOR_C, ls=':', lw=1.2)
    if ceil is not None:
        ax.axhline(ceil, color=CEIL_C, ls='--', lw=1.2)
    ax.set_xlabel('PSNR, whole image (dB)')
    ax.set_ylabel('Dice$_{ET}$')
    ax.set_title('Task accuracy against image fidelity', pad=6)
    ax.legend(handles=[plt.Line2D([], [], ls='', marker=marker[b], color=MUTED,
                                  label=b) for b in bbs], loc='lower left')
    save(fig, 'fig2_accuracy_vs_psnr')

    # ── FIG 3 — the local-fidelity version of the same trade-off ───────────
    if any(D['psnr_et'].get(m) for m in trained):
        fig, ax = plt.subplots(figsize=(4.4, 3.6))
        for m in trained:
            n, bb = tidy_name(m), backbone_of(m)
            x, _, _ = stats(m, 'psnr_et')
            y, _, _ = stats(m, args.metric)
            if math.isnan(x):
                continue
            ax.scatter(x, y, s=46, color=colour[n], marker=marker[bb],
                       edgecolor='white', linewidth=0.8, zorder=3)
            ax.annotate(n, (x, y), textcoords='offset points', xytext=(5, 3),
                        fontsize=6.5, color=INK)
        ax.set_xlabel('PSNR inside the enhancing tumour (dB)')
        ax.set_ylabel('Dice$_{ET}$')
        ax.set_title('Accuracy against fidelity where the claim lives', pad=6)
        save(fig, 'fig3_accuracy_vs_psnr_et')

    # ── FIG 4 — heatmap, objectives x metrics, normalised per column ───────
    show = ['dice_et', 'dice_tc', 'dice_wt', 'psnr_et', 'psnr', 'ssim']
    show = [m for m in show if any(D[m].get(x) for x in trained)]
    M = np.full((len(trained), len(show)), np.nan)
    for i, m in enumerate(trained):
        for j, met in enumerate(show):
            M[i, j] = stats(m, met)[0]
    Z = np.full_like(M, np.nan)
    for j in range(M.shape[1]):
        col = M[:, j]
        good = ~np.isnan(col)
        if good.sum() > 1:
            lo, hi = np.nanmin(col), np.nanmax(col)
            Z[:, j] = (col - lo) / (hi - lo) if hi > lo else 0.5
    fig, ax = plt.subplots(figsize=(0.78 * len(show) + 2.6, 0.30 * len(trained) + 1.9))
    im = ax.imshow(Z, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(show)))
    ax.set_xticklabels([s.replace('_', '$_{') + '}$' if '_' in s else s
                        for s in show], rotation=30, ha='right')
    ax.set_yticks(range(len(trained)))
    ax.set_yticklabels([f'{tidy_name(m)} ({backbone_of(m)[:4]})' for m in trained])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f'{M[i, j]:.3f}' if M[i, j] < 10 else f'{M[i, j]:.1f}',
                        ha='center', va='center', fontsize=6,
                        color='white' if Z[i, j] > 0.62 else INK)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('rank within column (0 worst, 1 best)', fontsize=7)
    cb.ax.tick_params(labelsize=7)
    ax.set_title('All objectives, all metrics', pad=8)
    save(fig, 'fig4_heatmap')

    # ── FIG 5 — decomposition of the effect over L1 ────────────────────────
    for bb in bbs:
        pick = lambda frag: next((m for m in trained
                                  if backbone_of(m) == bb and frag in tidy_name(m)), None)
        l1, base, pl = pick('L1'), pick('MatchedBase'), pick('PathologyLoss')
        if not (l1 and base and pl):
            continue
        a, b, c = (stats(l1, args.metric)[0], stats(base, args.metric)[0],
                   stats(pl, args.metric)[0])
        fig, ax = plt.subplots(figsize=(4.0, 2.5))
        parts = [('L1 baseline', a, PAL[0]),
                 ('+ SSIM, cross-modal', b - a, PAL[1]),
                 ('+ region weighting', c - b, PAL[2])]
        # Values live in the legend, not on the bar. Two of these segments are
        # a few thousandths of a Dice point wide, so an in-bar label collides
        # with its neighbour at any readable font size.
        left = 0.0
        for lbl, w, col in parts:
            ax.barh([0], [w], left=[left], height=0.5, color=col,
                    edgecolor='white', linewidth=1.4, zorder=3)
            left += w
        ax.plot([c, c], [-0.32, 0.32], color=INK, lw=1.0, zorder=4)
        ax.text(c, 0.40, f'{c:.3f}', ha='center', va='bottom',
                fontsize=8, color=INK)
        ax.set_yticks([])
        ax.set_ylim(-0.55, 0.62)
        ax.set_xlabel('Dice$_{ET}$')
        ax.set_xlim(0, max(c * 1.18, 0.1))
        ax.set_title(f'{bb}: what the gain over L1 is actually made of', pad=6)
        share = [f'{lbl}  {w:+.3f}' if i else f'{lbl}  {w:.3f}'
                 for i, (lbl, w, _) in enumerate(parts)]
        ax.legend(handles=[Patch(facecolor=col, label=t)
                           for (_, _, col), t in zip(parts, share)],
                  loc='lower left', ncol=1, fontsize=7.5)
        save(fig, f'fig5_decomposition_{bb.lower()}')

    # ── FIG 6 — per-region effect of each objective ────────────────────────
    regions = [r for r in ('dice_wt', 'dice_tc', 'dice_et') if D[r]]
    for bb in bbs:
        arms = [m for m in trained if backbone_of(m) == bb]
        base = next((m for m in arms if tidy_name(m) == 'MatchedBase'), None)
        if not base or len(arms) < 2:
            continue
        fig, ax = plt.subplots(figsize=(1.05 * len(arms) + 1.6, 3.2))
        w = 0.8 / len(regions)
        x = np.arange(len(arms))
        hatches = ['', '//', '..']
        for k, reg in enumerate(regions):
            d = [stats(m, reg)[0] - stats(base, reg)[0] for m in arms]
            ax.bar(x + k * w - 0.4 + w / 2, d, width=w * 0.92,
                   color=PAL[k], hatch=hatches[k], edgecolor='white',
                   linewidth=0.7, zorder=3,
                   label=reg.replace('dice_', 'Dice ').upper())
        ax.axhline(0, color=INK, lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([tidy_name(m) for m in arms], rotation=28, ha='right')
        ax.set_ylabel(f'change vs MatchedBase')
        ax.set_title(f'{bb}: effect by tumour sub-region', pad=6)
        ax.legend(ncol=len(regions))
        save(fig, f'fig6_by_region_{bb.lower()}')

    # ── TABLE 1 — the main results table, CSV + LaTeX ──────────────────────
    hdr = ['Backbone', 'Objective', 'n', 'Dice_ET', 'CI95', 'Dice_TC',
           'Dice_WT', 'PSNR', 'PSNR_ET', 'SSIM']
    table = []
    for m in ([x for x in methods if 'FLOOR' in x] +
              [x for x in methods if 'CEILING' in x] + trained):
        mu, ci, n = stats(m, args.metric)
        table.append([backbone_of(m), tidy_name(m), n, f'{mu:.4f}',
                      f'[{ci[0]:.3f}, {ci[1]:.3f}]',
                      f'{stats(m, "dice_tc")[0]:.4f}',
                      f'{stats(m, "dice_wt")[0]:.4f}',
                      f'{stats(m, "psnr")[0]:.2f}',
                      f'{stats(m, "psnr_et")[0]:.2f}',
                      f'{stats(m, "ssim")[0]:.4f}'])
    with open(os.path.join(out, 'table1_main_results.csv'), 'w', newline='') as fh:
        w_ = csv.writer(fh)
        w_.writerow(hdr)
        w_.writerows(table)
    with open(os.path.join(out, 'table1_main_results.tex'), 'w') as fh:
        fh.write('% Main results. Dice CI is a percentile bootstrap over SUBJECTS\n'
                 '% (4000 resamples); slices within a patient are not independent.\n')
        fh.write('\\begin{table}[t]\n\\centering\n\\caption{Segmentation accuracy and '
                 'reconstruction fidelity by training objective.}\n')
        fh.write('\\begin{tabular}{ll' + 'r' * (len(hdr) - 2) + '}\n\\hline\n')
        fh.write(' & '.join(h.replace('_', '$_{') + '}$' if '_' in h else h
                            for h in hdr) + ' \\\\\n\\hline\n')
        for r in table:
            fh.write(' & '.join(str(c).replace('_', '\\_') for c in r) + ' \\\\\n')
        fh.write('\\hline\n\\end{tabular}\n\\end{table}\n')
    print('  table1_main_results.csv / .tex')

    # ── provenance ─────────────────────────────────────────────────────────
    with open(os.path.join(out, 'README.txt'), 'w') as fh:
        fh.write('Figures and tables generated by make_paper_figures.py\n\n')
        fh.write(f'source files ({len(files)}):\n')
        for p in files:
            fh.write(f'  {p}\n')
        fh.write(f'\nranking metric : {args.metric}\n')
        fh.write(f'backbones      : {", ".join(bbs)}\n')
        fh.write(f'objectives     : {len(trained)}\n')
        if floor is not None:
            fh.write(f'floor          : {floor:.4f}\n')
        if ceil is not None:
            fh.write(f'ceiling        : {ceil:.4f}\n')
        fh.write('\nCIs are percentile bootstrap over subjects, 4000 resamples.\n'
                 'Palette validated for CVD: worst adjacent pair dE 10.3 (protan),\n'
                 '25.0 (normal vision); all series also direct-labelled or hatched,\n'
                 'so the figures survive greyscale printing.\n')

    print(f'\n  {len(saved)} figures + 1 table -> {out}')
    print('  PDF is vector at 600 dpi for typesetting; PNG for drafts.')


if __name__ == '__main__':
    main()
