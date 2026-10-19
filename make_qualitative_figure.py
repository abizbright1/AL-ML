#!/usr/bin/env python3
"""
make_qualitative_figure.py — see the images, not just the numbers
==================================================================

Builds the figure a reviewer looks at first: the same patient slice restored
by every training objective, side by side, with what the segmentor made of
each one.

Three panels per objective:

    RESTORED     the denoised T1ce, windowed identically across all arms so
                 brightness differences are real and not a display artefact
    ERROR        |restored - clean|, one shared colour scale, so a brighter
                 panel genuinely means a worse reconstruction
    SEGMENTATION green where the segmentor found real enhancing tumour,
                 red where it missed, blue where it invented tumour that is
                 not there

The blue is the one to look at. Dice rewards finding tumour and does not
symmetrically punish inventing it, so an arm can win on Dice while painting
false enhancing tumour across healthy brain. That shows up here and nowhere
in the tables.

The slice is chosen automatically as the one where the arms DISAGREE most,
because a slice they all handle identically illustrates nothing. Override
with --slice to pick your own.

Usage
-----
    python3 make_qualitative_figure.py --run results/comparison/phase2_seed1
    python3 make_qualitative_figure.py --run <dir> --backbone Uformer
    python3 make_qualitative_figure.py --run <dir> --slice 7 --modality 1
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

INK, MUTED = '#1a1a1a', '#5a5a5a'
HIT, MISS, FP = '#2f8f4f', '#c1440e', '#1b6ca8'      # green / red / blue
MODALITIES = ['T1', 'T1ce', 'T2', 'FLAIR']

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'font.size': 8, 'axes.titlesize': 8,
    'figure.dpi': 150, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
})

ORDER = ['Noisy', 'Clean', 'L1', 'Base', 'SSIM', 'MSSSIM', 'Mix', 'L2',
         'TV', 'BSFreq', 'HighFreq', 'DAGAN', 'FlatROI', 'BSMaskL1',
         'PLcur', 'PLEqual', 'PLETonly', 'PLdisj']


def pretty(short: str) -> str:
    s = short.replace('plus', '+').replace('_', ' ')
    for a, b in [('Uformer+', ''), ('SwinIR+', ''), ('Uformer-', ''), ('SwinIR-', ''),
                 ('PLcur', 'PathologyLoss'), ('BSMaskL1', 'BraSyn masked L1'),
                 ('BSFreq', 'BraSyn frequency'), ('Base', 'Matched base'),
                 ('PLEqual', 'PL equal weights'), ('PLETonly', 'PL ET only'),
                 ('PLdisj', 'PL disjoint'), ('MSSSIM', 'MS-SSIM')]:
        s = s.replace(a, b)
    return s.strip()


def load_dumps(run_dir, backbone=None):
    out = []
    for p in sorted(glob.glob(os.path.join(run_dir, 'predictions', '*.npz'))):
        d = np.load(p, allow_pickle=True)
        short = str(d['short'][0]) if 'short' in d else os.path.basename(p)[:-4]
        if backbone and not (short.startswith(backbone)
                             or short in ('Noisy', 'Clean')):
            continue
        if 'denoised' not in d or len(d['denoised']) == 0:
            continue
        out.append((short, d))
    if not out:
        sys.exit(f'No usable .npz under {run_dir}/predictions. '
                 f'Was the run launched with --dump_slices > 0?')

    def key(s):
        base = s[0].split('+')[-1].split('-')[-1]
        return (ORDER.index(base) if base in ORDER else 99, s[0])
    out.sort(key=key)
    return out


def pick_slice(dumps, n):
    """The slice on which the arms disagree most -- a slice they all handle
    identically demonstrates nothing."""
    spread = []
    for i in range(n):
        segs = [d['seg_pred'][i] for _, d in dumps if len(d['seg_pred']) > i]
        if len(segs) < 2:
            spread.append(0.0)
            continue
        et = [(s == 3) for s in segs]
        union = np.any(et, axis=0).sum()
        inter = np.all(et, axis=0).sum()
        spread.append(float(union - inter))
    return int(np.argmax(spread)) if spread else 0


def overlay(seg_pred, seg_gt):
    """green hit / red miss / blue false positive, on the ET class."""
    p, g = seg_pred == 3, seg_gt == 3
    rgb = np.zeros((*p.shape, 3), dtype=np.float32)
    for mask, col in ((p & g, HIT), (g & ~p, MISS), (p & ~g, FP)):
        c = np.array([int(col[i:i + 2], 16) / 255 for i in (1, 3, 5)])
        rgb[mask] = c
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--backbone', default=None)
    ap.add_argument('--slice', type=int, default=None)
    ap.add_argument('--modality', type=int, default=1,
                    help='0=T1 1=T1ce 2=T2 3=FLAIR; T1ce shows enhancement')
    args = ap.parse_args()
    out = args.out or os.path.join(args.run, 'paper_figures')
    os.makedirs(out, exist_ok=True)

    dumps = load_dumps(args.run, args.backbone)
    n = min(len(d['denoised']) for _, d in dumps)
    idx = args.slice if args.slice is not None else pick_slice(dumps, n)
    subj = str(dumps[0][1]['subject'][idx])
    mod = args.modality

    clean = dumps[0][1]['clean'][idx][mod]
    seg_gt = dumps[0][1]['seg_gt'][idx]
    vmin, vmax = float(clean.min()), float(np.percentile(clean, 99.5))

    errs = [np.abs(d['denoised'][idx][mod] - d['clean'][idx][mod])
            for _, d in dumps]
    emax = float(np.percentile(np.stack(errs), 99.0)) or 1e-6

    rows = len(dumps)
    fig, axes = plt.subplots(rows, 3, figsize=(6.4, 2.08 * rows),
                             squeeze=False)
    for r, (short, d) in enumerate(dumps):
        img = d['denoised'][idx][mod]
        err = np.abs(img - d['clean'][idx][mod])
        ov = overlay(d['seg_pred'][idx], seg_gt)
        dice = float(d['dice_et'][idx]) if 'dice_et' in d else float('nan')
        psnr = float(d['psnr'][idx]) if 'psnr' in d else float('nan')

        axes[r][0].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        axes[r][0].set_ylabel(pretty(short), rotation=0, ha='right',
                              va='center', labelpad=8, fontsize=8)
        im1 = axes[r][1].imshow(err, cmap='inferno', vmin=0, vmax=emax)
        axes[r][2].imshow(np.clip(np.stack([clean] * 3, -1)
                                  / max(vmax, 1e-6) * 0.55, 0, 1))
        axes[r][2].imshow(ov, alpha=(ov.sum(-1) > 0).astype(float))

        # Stats go INSIDE the panel. As a title with this row spacing they
        # render just under the previous row and read as belonging to it.
        txt = (f'PSNR {psnr:.1f} dB' if not np.isnan(psnr) else '')
        if not np.isnan(dice):
            txt += f'   Dice$_{{ET}}$ {dice:.3f}'
        if txt:
            axes[r][2].text(0.03, 0.97, txt, transform=axes[r][2].transAxes,
                            ha='left', va='top', fontsize=6.5, color='white',
                            bbox=dict(facecolor='black', alpha=0.55,
                                      edgecolor='none', pad=1.6))
        for c in range(3):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
            for sp in axes[r][c].spines.values():
                sp.set_color('#d8d8d8')

    for c, t in enumerate(['restored', '|error|', 'segmentation']):
        axes[0][c].set_title(t, fontsize=9, color=INK, pad=14)

    cax = fig.add_axes([0.40, 0.065, 0.22, 0.008])
    cb = fig.colorbar(im1, cax=cax, orientation='horizontal')
    cb.set_label('reconstruction error', fontsize=7)
    cb.ax.tick_params(labelsize=6)

    fig.legend(handles=[Patch(facecolor=HIT, label='found real ET'),
                        Patch(facecolor=MISS, label='missed ET'),
                        Patch(facecolor=FP, label='invented ET')],
               loc='lower center', ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.012))
    fig.suptitle(f'{subj}, slice {idx}, {MODALITIES[mod]}'
                 + (f' — {args.backbone}' if args.backbone else ''),
                 y=0.995, fontsize=10)
    fig.subplots_adjust(hspace=0.06, wspace=0.03, bottom=0.11)

    tag = f'_{args.backbone.lower()}' if args.backbone else ''
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'fig7_qualitative{tag}.{ext}'))
    plt.close(fig)

    print(f'  subject {subj}, slice {idx}, {MODALITIES[mod]}')
    print(f'  {rows} objectives shown')
    print(f'  fig7_qualitative{tag}.pdf / .png -> {out}')
    print('\n  Look at the blue. Dice rewards finding tumour and does not')
    print('  symmetrically punish inventing it, so an arm can win on Dice')
    print('  while painting false enhancing tumour across healthy brain.')


if __name__ == '__main__':
    main()
