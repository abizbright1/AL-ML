#!/usr/bin/env python3
"""
run_full_experiment.py — Round 4 with checkpointing, prediction dumps, figures
==============================================================================

Standalone. Does not import or modify run_all_options.py.

What it adds over run_all_options.py
------------------------------------
  1. Saves every trained model  (weights + config + metrics + provenance)
  2. Dumps every model's predictions to .npz  (so figures never need retraining)
  3. Generates 9 figures covering ALL FIVE models, not two
  4. --figures_only  redraws everything from saved dumps in seconds
  5. Per-slice metrics carry subject IDs, for subject-level analysis later
  6. Dice on empty regions is NaN, not 1.0  (see note below)

Usage
-----
  # Full run, 100 epochs
  python3 run_full_experiment.py ~/Downloads/BraTS2021_data \\
      --epochs 100 --seg_epochs 20 --max_subjects 100 \\
      --device mps --out results/full_100ep

  # Quick smoke test first (always do this)
  python3 run_full_experiment.py ~/Downloads/BraTS2021_data \\
      --epochs 2 --seg_epochs 2 --max_subjects 5 \\
      --device mps --out results/smoke_full

  # Redraw figures from an existing run — no training, seconds
  python3 run_full_experiment.py ~/Downloads/BraTS2021_data \\
      --out results/full_100ep --figures_only

Outputs
-------
  <out>/checkpoints/*.pt        model weights + config + provenance
  <out>/predictions/*.npz       denoised images, predicted masks, per-slice Dice
  <out>/figures/*.png           9 figures, all five models
  <out>/options_results.csv     summary metrics
  <out>/per_slice_metrics.csv   one row per (model, slice) WITH subject ID
  <out>/run_config.json         every argument, git SHA, timestamp

Note on empty-region Dice
-------------------------
Many slices contain no enhancing tumour. Dice is undefined there (0/0).
run_all_options.py scores those as 1.0 by convention, which inflates mean
Dice_ET. This script records NaN and averages with nanmean, then reports how
many slices were undefined. Expect LOWER Dice_ET numbers here than in
run_all_options.py. That is the point — they are more honest.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, 'pp_mae'))

from option4_swin_pp_mae import SwinPPMAE, SwinPPMAETrainer
from option_baselines import (
    SwinIRLite, SwinIRTrainer, SwinIRPathologyTrainer,
    UformerLite, UformerTrainer, UformerPathologyTrainer,
)
from segmentor import UNetSegmentor, SegTrainer
from brats_loader import BraTSDataset
from evaluation import psnr, ssim_numpy, nrmse


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Round 4 with checkpoints, prediction dumps and figures',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('data_dir', nargs='?', default=None,
                   help='BraTS directory (not needed with --figures_only)')
    p.add_argument('--epochs',       type=int,   default=100)
    p.add_argument('--seg_epochs',   type=int,   default=20)
    p.add_argument('--max_subjects', type=int,   default=100)
    p.add_argument('--patch_size',   type=int,   default=96)
    p.add_argument('--sigma',        type=float, default=0.08)
    p.add_argument('--batch_size',   type=int,   default=4)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--device',       type=str,   default=None,
                   help='mps | cuda | cpu  (default: auto-detect)')
    p.add_argument('--out',          type=str,   default='results/full_run')
    p.add_argument('--dump_slices',  type=int,   default=48,
                   help='Validation slices to store per model for figures')
    p.add_argument('--ckpt_every',   type=int,   default=25,
                   help='Also save an intermediate checkpoint every N epochs')
    p.add_argument('--figures_only', action='store_true',
                   help='Skip training; redraw figures from saved .npz dumps')
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  Provenance — so you can always tell two runs apart
# ═══════════════════════════════════════════════════════════════════════════

def git_sha() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=_DIR,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=_DIR,
            stderr=subprocess.DEVNULL).decode().strip()
        return len(out) > 0
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════════

def region_mask(seg: np.ndarray, region: str) -> np.ndarray:
    """BraTS sub-regions. Loader maps disk label 4 -> 3."""
    if region == 'WT':
        return seg > 0
    if region == 'TC':
        return (seg == 1) | (seg == 3)
    if region == 'ET':
        return seg == 3
    raise ValueError(region)


def dice(pred_seg: np.ndarray, gt_seg: np.ndarray, region: str) -> float:
    """Dice for one region on one slice. NaN when the region is absent in BOTH.

    Returning NaN rather than 1.0 matters: ~40-60% of BraTS slices contain no
    enhancing tumour, and scoring those as perfect inflates mean Dice_ET.
    """
    p = region_mask(pred_seg, region)
    g = region_mask(gt_seg,  region)
    denom = p.sum() + g.sum()
    if denom == 0:
        return float('nan')          # undefined, not perfect
    return float(2.0 * np.logical_and(p, g).sum() / denom)


def nanmean(xs: List[float]) -> float:
    a = np.asarray(xs, dtype=np.float64)
    return float(np.nanmean(a)) if np.any(~np.isnan(a)) else 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Model registry — config kept beside the constructor so it can be saved
# ═══════════════════════════════════════════════════════════════════════════

def build_models(device: str) -> Dict[str, dict]:
    """Round 4 exactly as specified in proposal section 6.5."""
    return {
        'PP-MAE (Swin) [PROPOSED]': {
            'class':  SwinPPMAE,
            'config': dict(in_ch=4, embed_dim=48, depths=(2, 2, 2, 2),
                           n_heads=(3, 3, 6, 6), window_size=4),
            'trainer_fn': lambda m: SwinPPMAETrainer(m, device=device),
            'infer_fn':   lambda m, noisy, seg: m(noisy, seg),
            'uses_mask_at_inference': True,      # <-- the known confound
            'short': 'PP-MAE',
        },
        'SwinIR-lite (L1)': {
            'class':  SwinIRLite,
            'config': dict(in_ch=4, dim=64, n_blocks=4, window_size=4),
            'trainer_fn': lambda m: SwinIRTrainer(m, device=device, lr=1e-4),
            'infer_fn':   lambda m, noisy, seg: m(noisy),
            'uses_mask_at_inference': False,
            'short': 'SwinIR-L1',
        },
        'Uformer-lite (L1)': {
            'class':  UformerLite,
            'config': dict(in_ch=4, dim=32, window_size=4),
            'trainer_fn': lambda m: UformerTrainer(m, device=device, lr=1e-4),
            'infer_fn':   lambda m, noisy, seg: m(noisy),
            'uses_mask_at_inference': False,
            'short': 'Uformer-L1',
        },
        'SwinIR + PathologyLoss': {
            'class':  SwinIRLite,
            'config': dict(in_ch=4, dim=64, n_blocks=4, window_size=4),
            'trainer_fn': lambda m: SwinIRPathologyTrainer(
                m, device=device, lr=1e-4, mode='clinical_risk'),
            'infer_fn':   lambda m, noisy, seg: m(noisy),
            'uses_mask_at_inference': False,
            'short': 'SwinIR+PL',
        },
        'Uformer + PathologyLoss': {
            'class':  UformerLite,
            'config': dict(in_ch=4, dim=32, window_size=4),
            'trainer_fn': lambda m: UformerPathologyTrainer(
                m, device=device, lr=1e-4, mode='clinical_risk'),
            'infer_fn':   lambda m, noisy, seg: m(noisy),
            'uses_mask_at_inference': False,
            'short': 'Uformer+PL',
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Checkpointing
# ═══════════════════════════════════════════════════════════════════════════

def save_checkpoint(model, spec, metrics, history, args, device,
                    ckpt_dir, tag, epoch):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f'{tag}.pt')
    torch.save({
        'model':          model.state_dict(),
        'model_class':    spec['class'].__name__,
        'config':         spec['config'],       # needed to rebuild architecture
        'metrics':        metrics,
        'loss_history':   history,
        'epoch':          epoch,
        'args':           vars(args),
        'git_sha':        git_sha(),
        'git_dirty':      git_dirty(),
        'device':         device,
        'torch_version':  torch.__version__,
        'timestamp':      time.strftime('%Y-%m-%d %H:%M:%S'),
        'uses_mask_at_inference': spec['uses_mask_at_inference'],
    }, path)
    return path


def load_checkpoint(path, device='cpu'):
    """Rebuild architecture from saved config, then pour the weights in."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    registry = {c.__name__: c for c in (SwinPPMAE, SwinIRLite, UformerLite)}
    model = registry[ck['model_class']](**ck['config'])
    model.load_state_dict(ck['model'])
    model.to(device).eval()                     # eval() matters — dropout/norm
    return model, ck


# ═══════════════════════════════════════════════════════════════════════════
#  Train / evaluate / dump
# ═══════════════════════════════════════════════════════════════════════════

def train_model(model, spec, train_loader, args, device, name):
    """Returns (model, loss_history, actual_device)."""
    trainer = spec['trainer_fn'](model.to(device))
    history, cur_device = [], device

    for ep in range(1, args.epochs + 1):
        total = 0.0
        try:
            for b in train_loader:
                total += trainer.step(b).get('total', 0.0)
        except RuntimeError as e:
            # MPS backward bug: fall back to CPU and restart this model
            if cur_device != 'cpu' and ('view size' in str(e) or 'MPS' in str(e)):
                print(f"    !! {type(e).__name__} on {cur_device} — "
                      f"restarting {name} on CPU", flush=True)
                cur_device = 'cpu'
                model = spec['class'](**spec['config']).to('cpu')
                trainer = spec['trainer_fn'](model)
                history = []
                continue
            raise

        history.append(total / max(len(train_loader), 1))
        if ep == 1 or ep % 5 == 0 or ep == args.epochs:
            dev = '' if cur_device == device else f'  ({cur_device})'
            print(f"    Ep {ep:3d}/{args.epochs}  loss={history[-1]:.4f}{dev}",
                  flush=True)

    return model, history, cur_device


@torch.no_grad()
def evaluate_and_dump(model, spec, val_loader, seg_model, device,
                      name, short, dump_path, max_dump):
    """Evaluate on the full val set; store the first `max_dump` slices."""
    model.eval()
    seg_model.eval()
    seg_device = next(seg_model.parameters()).device

    per_slice = []          # dicts, one per validation slice
    keep = {k: [] for k in ('clean', 'noisy', 'denoised', 'seg_gt', 'seg_pred')}
    kept_subjects = []

    for b in val_loader:
        noisy   = b['noisy'].to(device)
        seg     = b['seg'].to(device)
        target  = b['target']
        subs    = b.get('subject', ['?'] * noisy.shape[0])

        pred = spec['infer_fn'](model, noisy, seg)
        if isinstance(pred, dict):
            pred = pred.get('denoised', list(pred.values())[0])
        pred_cpu = pred.detach().cpu()

        logits    = seg_model(pred_cpu.to(seg_device))
        seg_pred  = logits.argmax(1).cpu().numpy()          # (B,H,W)
        seg_gt    = b['seg'][:, 0].numpy().astype(np.int64)  # (B,H,W)

        for i in range(pred_cpu.shape[0]):
            p_np = pred_cpu[i].permute(1, 2, 0).numpy()
            t_np = target[i].permute(1, 2, 0).numpy()
            row = {
                'method':  name,
                'subject': subs[i] if isinstance(subs, (list, tuple)) else str(subs[i]),
                'psnr':    psnr(p_np, t_np),
                'ssim':    ssim_numpy(p_np, t_np),
                'nrmse':   nrmse(p_np, t_np),
                'dice_wt': dice(seg_pred[i], seg_gt[i], 'WT'),
                'dice_tc': dice(seg_pred[i], seg_gt[i], 'TC'),
                'dice_et': dice(seg_pred[i], seg_gt[i], 'ET'),
            }
            per_slice.append(row)

            if len(kept_subjects) < max_dump:
                keep['clean'].append(target[i].numpy().astype(np.float32))
                keep['noisy'].append(b['noisy'][i].numpy().astype(np.float32))
                keep['denoised'].append(pred_cpu[i].numpy().astype(np.float32))
                keep['seg_gt'].append(seg_gt[i].astype(np.int8))
                keep['seg_pred'].append(seg_pred[i].astype(np.int8))
                kept_subjects.append(row['subject'])

    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    n_keep = len(kept_subjects)
    np.savez_compressed(
        dump_path,
        method=np.array([name]), short=np.array([short]),
        subject=np.array(kept_subjects),
        dice_et=np.array([per_slice[i]['dice_et'] for i in range(n_keep)],
                         dtype=np.float32),
        psnr=np.array([per_slice[i]['psnr'] for i in range(n_keep)],
                      dtype=np.float32),
        **{k: np.stack(v) for k, v in keep.items()},
    )

    summary = {
        'PSNR':    nanmean([r['psnr']    for r in per_slice]),
        'SSIM':    nanmean([r['ssim']    for r in per_slice]),
        'NRMSE':   nanmean([r['nrmse']   for r in per_slice]),
        'Dice_WT': nanmean([r['dice_wt'] for r in per_slice]),
        'Dice_TC': nanmean([r['dice_tc'] for r in per_slice]),
        'Dice_ET': nanmean([r['dice_et'] for r in per_slice]),
        'n_slices': len(per_slice),
        'n_et_undefined': int(sum(np.isnan(r['dice_et']) for r in per_slice)),
    }
    return summary, per_slice


# ═══════════════════════════════════════════════════════════════════════════
#  Figures — all five models, drawn from the .npz dumps
# ═══════════════════════════════════════════════════════════════════════════

MODALITIES  = ['T1', 'T1ce', 'T2', 'FLAIR']
SEG_RGBA    = [(0, 0, 0, 0), (1.0, .2, .2, .6), (1.0, .8, 0, .6), (0, 1.0, .5, .6)]
ERR_CMAP    = LinearSegmentedColormap.from_list(
    'err', ['#000033', '#0000FF', '#00FFFF', '#FFFF00', '#FF0000'])


def _overlay(ax, img, seg, alpha_img='gray'):
    ax.imshow(img, cmap=alpha_img, interpolation='nearest')
    rgba = np.zeros((*seg.shape, 4), np.float32)
    for lab, col in enumerate(SEG_RGBA):
        if lab:
            rgba[seg == lab] = col
    ax.imshow(rgba, interpolation='nearest')


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  saved {path}", flush=True)


def load_dumps(pred_dir) -> List[dict]:
    out = []
    for f in sorted(os.listdir(pred_dir)):
        if f.endswith('.npz'):
            d = dict(np.load(os.path.join(pred_dir, f), allow_pickle=True))
            d['method'] = str(d['method'][0])
            d['short']  = str(d['short'][0])
            out.append(d)
    return out


def pick_slice(dumps: List[dict]) -> int:
    """Choose the slice where models DISAGREE most.

    A figure on a slice where every model looks identical proves nothing.
    Maximising the spread in per-slice Dice_ET finds the informative case.
    """
    n = min(len(d['dice_et']) for d in dumps)
    stacked = np.stack([d['dice_et'][:n] for d in dumps])      # (models, n)
    spread = np.nanmax(stacked, 0) - np.nanmin(stacked, 0)
    spread = np.nan_to_num(spread, nan=-1.0)
    return int(np.argmax(spread))


def fig_data_sample(dumps, idx, fig_dir):
    d = dumps[0]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
    fig.suptitle(f"Data sample — subject {d['subject'][idx]}",
                 fontsize=13, fontweight='bold')
    for c in range(4):
        axes[c].imshow(d['clean'][idx, c], cmap='gray')
        axes[c].set_title(MODALITIES[c], fontweight='bold')
        axes[c].axis('off')
    _overlay(axes[4], d['clean'][idx, 1], d['seg_gt'][idx])
    axes[4].set_title('Ground-truth tumour', fontweight='bold')
    axes[4].axis('off')
    fig.legend(handles=[mpatches.Patch(color=SEG_RGBA[i][:3], label=l)
                        for i, l in [(1, 'NCR'), (2, 'Oedema'), (3, 'ET')]],
               loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(.5, -.05))
    _save(fig, os.path.join(fig_dir, 'fig01_data_sample.png'))


def fig_noise(dumps, idx, fig_dir, sigma):
    d = dumps[0]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    fig.suptitle(f'Clean vs noisy input  (Rician, sigma={sigma})',
                 fontsize=13, fontweight='bold')
    for r, c in enumerate([1, 3]):                     # T1ce, FLAIR
        clean, noisy = d['clean'][idx, c], d['noisy'][idx, c]
        for j, (im, t) in enumerate([(clean, 'Clean'), (noisy, f'Noisy sigma={sigma}'),
                                      (np.abs(noisy - clean), 'Noise added')]):
            ax = axes[r, j]
            if j < 2:
                ax.imshow(im, cmap='gray', vmin=0, vmax=1)
            else:
                m = ax.imshow(im, cmap=ERR_CMAP, vmin=0, vmax=.3)
                plt.colorbar(m, ax=ax, fraction=.046)
            ax.set_title(f'{MODALITIES[c]} — {t}', fontweight='bold', fontsize=10)
            ax.axis('off')
        mse = np.mean((noisy - clean) ** 2)
        axes[r, 1].text(.97, .03, f'PSNR={10*np.log10(1/(mse+1e-10)):.1f} dB',
                        transform=axes[r, 1].transAxes, ha='right', va='bottom',
                        color='yellow', fontsize=8,
                        bbox=dict(fc='black', alpha=.5, pad=1))
    _save(fig, os.path.join(fig_dir, 'fig02_noise_added.png'))


def fig_reconstruction(dumps, idx, fig_dir, mod=1):
    n = len(dumps)
    fig, axes = plt.subplots(1, n + 2, figsize=(3.2 * (n + 2), 3.8))
    fig.suptitle(f'Denoising — all models  ({MODALITIES[mod]})',
                 fontsize=13, fontweight='bold')
    clean = dumps[0]['clean'][idx, mod]
    axes[0].imshow(clean, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Clean (target)', fontweight='bold', fontsize=10)
    axes[1].imshow(dumps[0]['noisy'][idx, mod], cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Noisy (input)', fontweight='bold', fontsize=10)
    for k, d in enumerate(dumps):
        img = d['denoised'][idx, mod]
        ax = axes[k + 2]
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        mse = np.mean((img - clean) ** 2)
        ax.set_title(f"{d['short']}\nPSNR={10*np.log10(1/(mse+1e-10)):.1f} dB",
                     fontweight='bold', fontsize=9)
    for ax in axes:
        ax.axis('off')
    _save(fig, os.path.join(fig_dir, 'fig03_denoised_all_models.png'))


def fig_error(dumps, idx, fig_dir, mod=1):
    n = len(dumps)
    fig, axes = plt.subplots(1, n + 1, figsize=(3.2 * (n + 1), 3.8))
    fig.suptitle('Reconstruction error  |prediction - clean|  '
                 '(bright = worse). Look inside vs outside the tumour.',
                 fontsize=12, fontweight='bold')
    clean = dumps[0]['clean'][idx, mod]
    noisy_err = np.abs(dumps[0]['noisy'][idx, mod] - clean)
    vmax = max(float(noisy_err.max()), .3)
    axes[0].imshow(noisy_err, cmap=ERR_CMAP, vmin=0, vmax=vmax)
    axes[0].set_title('Noisy input', fontweight='bold', fontsize=9)
    for k, d in enumerate(dumps):
        err = np.abs(d['denoised'][idx, mod] - clean)
        im = axes[k + 1].imshow(err, cmap=ERR_CMAP, vmin=0, vmax=vmax)
        axes[k + 1].set_title(d['short'], fontweight='bold', fontsize=9)
    for ax in axes:
        ax.axis('off')
    plt.colorbar(im, ax=axes.tolist(), fraction=.02, pad=.01)
    _save(fig, os.path.join(fig_dir, 'fig04_error_maps_all_models.png'))


def fig_segmentation(dumps, idx, fig_dir, mod=1):
    n = len(dumps)
    fig, axes = plt.subplots(2, n + 1, figsize=(3.2 * (n + 1), 7.2))
    fig.suptitle('Segmentation by the frozen segmentor on each reconstruction',
                 fontsize=13, fontweight='bold')
    gt = dumps[0]['seg_gt'][idx]

    _overlay(axes[0, 0], dumps[0]['clean'][idx, mod], gt)
    axes[0, 0].set_title('Ground truth', fontweight='bold', fontsize=9)
    axes[1, 0].axis('off')

    for k, d in enumerate(dumps):
        pred = d['seg_pred'][idx]
        _overlay(axes[0, k + 1], d['denoised'][idx, mod], pred)
        det = d['dice_et'][idx]
        lbl = 'n/a' if np.isnan(det) else f'{det:.3f}'
        axes[0, k + 1].set_title(f"{d['short']}\nDice_ET={lbl}",
                                 fontweight='bold', fontsize=9)

        # agreement map on ET
        p, g = pred == 3, gt == 3
        agree = np.zeros((*gt.shape, 3), np.float32)
        agree[np.logical_and(p, g)]      = [0, .9, 0]    # hit
        agree[np.logical_and(~p, g)]     = [.9, 0, 0]    # missed
        agree[np.logical_and(p, ~g)]     = [0, .4, 1.]   # false positive
        axes[1, k + 1].imshow(agree)
        axes[1, k + 1].set_title('green hit / red miss / blue FP', fontsize=8)

    for ax in axes.flat:
        ax.axis('off')
    _save(fig, os.path.join(fig_dir, 'fig05_segmentation_all_models.png'))


def fig_metrics(results, fig_dir):
    names = [r['Method'] for r in results]
    shorts = [r['Short'] for r in results]
    colours = ['#D32F2F' if 'PROPOSED' in n else '#1976D2' for n in names]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    fig.suptitle('Metrics — all models', fontsize=13, fontweight='bold')
    for ax, key in zip(axes, ['PSNR', 'Dice_WT', 'Dice_TC', 'Dice_ET']):
        vals = [float(r[key]) for r in results]
        ax.bar(range(len(vals)), vals, color=colours, edgecolor='black', lw=.5)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(shorts, rotation=30, ha='right', fontsize=8)
        ax.set_title(key + (' (dB)' if key == 'PSNR' else ''), fontweight='bold')
        ax.set_ylim(min(vals) * .95, max(vals) * 1.03)
        ax.yaxis.grid(True, alpha=.4); ax.set_axisbelow(True)
        for i, v in enumerate(vals):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=7)
    _save(fig, os.path.join(fig_dir, 'fig06_metrics_bars.png'))


def fig_paradox(results, fig_dir):
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    for r in results:
        prop = 'PROPOSED' in r['Method']
        ax.scatter(float(r['PSNR']), float(r['Dice_ET']),
                   s=190 if prop else 110,
                   c='#D32F2F' if prop else '#1976D2',
                   marker='*' if prop else 'o',
                   edgecolors='black', zorder=3)
        ax.annotate(r['Short'], (float(r['PSNR']), float(r['Dice_ET'])),
                    textcoords='offset points', xytext=(7, 5), fontsize=9)
    ax.set_xlabel('PSNR (dB) — higher = cleaner picture')
    ax.set_ylabel('Dice_ET — higher = better tumour finding')
    ax.set_title('The PSNR paradox\nbest picture is not the best for the tumour',
                 fontweight='bold')
    ax.grid(alpha=.35); ax.set_axisbelow(True)
    _save(fig, os.path.join(fig_dir, 'fig07_psnr_paradox.png'))


def fig_curves(histories, fig_dir):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for name, h in histories.items():
        ax.plot(range(1, len(h) + 1), h,
                lw=2.5 if 'PROPOSED' in name else 1.4,
                label=name.replace(' [PROPOSED]', ''))
    ax.set_xlabel('Epoch'); ax.set_ylabel('Training loss')
    ax.set_title('Training curves  (loss scales differ by objective — '
                 'compare shape, not height)', fontweight='bold', fontsize=11)
    ax.set_yscale('log'); ax.legend(fontsize=8); ax.grid(alpha=.35)
    _save(fig, os.path.join(fig_dir, 'fig08_loss_curves.png'))


def fig_regions(dumps, idx, fig_dir):
    d, gt = dumps[0], dumps[0]['seg_gt'][idx]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    fig.suptitle('Tumour sub-regions — note how small ET is',
                 fontsize=13, fontweight='bold')
    axes[0].imshow(d['clean'][idx, 1], cmap='gray'); axes[0].set_title('T1ce')
    for ax, (reg, cm) in zip(axes[1:], [('WT', 'YlOrRd'), ('TC', 'Reds'),
                                        ('ET', 'Greens')]):
        m = region_mask(gt, reg)
        ax.imshow(m, cmap=cm, vmin=0, vmax=1)
        ax.set_title(f'{reg}  ({100 * m.mean():.2f}% of pixels)',
                     fontweight='bold', fontsize=10)
    for ax in axes:
        ax.axis('off')
    _save(fig, os.path.join(fig_dir, 'fig09_tumour_regions.png'))


def make_all_figures(out_dir, sigma, results=None, histories=None):
    pred_dir, fig_dir = os.path.join(out_dir, 'predictions'), os.path.join(out_dir, 'figures')
    dumps = load_dumps(pred_dir)
    if not dumps:
        print(f"  no .npz dumps in {pred_dir} — nothing to draw"); return

    idx = pick_slice(dumps)
    print(f"\n  figures use slice {idx} "
          f"(subject {dumps[0]['subject'][idx]}) — chosen for max disagreement\n")

    fig_data_sample(dumps, idx, fig_dir)
    fig_noise(dumps, idx, fig_dir, sigma)
    fig_reconstruction(dumps, idx, fig_dir)
    fig_error(dumps, idx, fig_dir)
    fig_segmentation(dumps, idx, fig_dir)
    fig_regions(dumps, idx, fig_dir)

    if results is None:
        p = os.path.join(out_dir, 'options_results.csv')
        if os.path.exists(p):
            with open(p) as f:
                results = list(csv.DictReader(f))
    if results:
        fig_metrics(results, fig_dir)
        fig_paradox(results, fig_dir)
    if histories:
        fig_curves(histories, fig_dir)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.figures_only:
        print(f"\nRedrawing figures from {args.out} — no training\n")
        make_all_figures(args.out, args.sigma)
        print("\nDone.\n")
        return

    if not args.data_dir:
        sys.exit("data_dir is required unless --figures_only is passed")

    device = args.device or ('cuda' if torch.cuda.is_available()
                             else 'mps' if getattr(torch.backends, 'mps', None)
                             and torch.backends.mps.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\nDevice   : {device}")
    print(f"Epochs   : {args.epochs}   Seg epochs: {args.seg_epochs}")
    print(f"Git SHA  : {git_sha()[:8]}{'  (DIRTY)' if git_dirty() else ''}")
    print(f"Output   : {args.out}\n")

    with open(os.path.join(args.out, 'run_config.json'), 'w') as f:
        json.dump({'args': vars(args), 'device': device, 'git_sha': git_sha(),
                   'git_dirty': git_dirty(), 'torch': torch.__version__,
                   'started': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)

    # ── Data ──────────────────────────────────────────────────────────────
    ds = BraTSDataset(os.path.expanduser(args.data_dir),
                      patch_size=args.patch_size, sigma=args.sigma,
                      max_subjects=args.max_subjects)
    n_train = int(.8 * len(ds))
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, len(ds) - n_train],
        generator=torch.Generator().manual_seed(args.seed))
    train_loader = torch.utils.data.DataLoader(train_ds, args.batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   args.batch_size, shuffle=False)
    print(f"  train {len(train_ds)} slices | val {len(val_ds)} slices")
    print("  WARNING: split is by SLICE, so patients appear in both sets.\n"
          "           Treat all numbers below as provisional.\n", flush=True)

    # ── Frozen segmentor ──────────────────────────────────────────────────
    print("=== SEGMENTOR (clean images) ===", flush=True)
    seg_model = UNetSegmentor(in_channels=4, n_classes=4, base_ch=32).to(device)
    seg_tr = SegTrainer(seg_model, device=device, lr=5e-4)
    for ep in range(1, args.seg_epochs + 1):
        tot = sum(seg_tr.step(b['target'].to(device),
                              b['seg'][:, 0].long().to(device))
                  for b in train_loader) / max(len(train_loader), 1)
        if ep == 1 or ep % 5 == 0 or ep == args.seg_epochs:
            print(f"  [seg] Ep {ep:2d}/{args.seg_epochs}  loss={tot:.4f}", flush=True)
    seg_model.eval()
    for p in seg_model.parameters():
        p.requires_grad_(False)
    ckpt_dir = os.path.join(args.out, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({'model': seg_model.state_dict(),
                'config': dict(in_channels=4, n_classes=4, base_ch=32),
                'epochs': args.seg_epochs, 'git_sha': git_sha()},
               os.path.join(ckpt_dir, 'segmentor.pt'))
    print("Segmentor frozen.\n", flush=True)

    # ── Train every model ─────────────────────────────────────────────────
    specs, results, histories, all_slices = build_models(device), [], {}, []

    for name, spec in specs.items():
        print(f"\n{'='*64}\n  {name}\n{'='*64}", flush=True)
        t0 = time.time()
        model = spec['class'](**spec['config'])
        model, hist, dev = train_model(model, spec, train_loader, args, device, name)
        histories[name] = hist

        tag = spec['short'].replace(' ', '_').replace('+', 'plus')
        summary, per_slice = evaluate_and_dump(
            model, spec, val_loader, seg_model, dev, name, spec['short'],
            os.path.join(args.out, 'predictions', f'{tag}.npz'), args.dump_slices)

        ck = save_checkpoint(model, spec, summary, hist, args, dev,
                             os.path.join(args.out, 'checkpoints'), tag, args.epochs)

        print(f"    PSNR={summary['PSNR']:.2f}  SSIM={summary['SSIM']:.3f}  "
              f"Dice_TC={summary['Dice_TC']:.3f}  Dice_ET={summary['Dice_ET']:.3f}")
        print(f"    {summary['n_et_undefined']}/{summary['n_slices']} slices "
              f"had no ET (excluded from Dice_ET)")
        print(f"    checkpoint -> {ck}    [{(time.time()-t0)/60:.1f} min]", flush=True)

        results.append({'Method': name, 'Short': spec['short'],
                        **{k: f'{v:.4f}' if isinstance(v, float) else v
                           for k, v in summary.items()},
                        'MaskAtInference': spec['uses_mask_at_inference']})
        all_slices.extend(per_slice)

    # ── CSVs ──────────────────────────────────────────────────────────────
    with open(os.path.join(args.out, 'options_results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys())); w.writeheader()
        w.writerows(results)
    with open(os.path.join(args.out, 'per_slice_metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_slices[0].keys())); w.writeheader()
        w.writerows(all_slices)

    print(f"\n{'='*80}\n  RESULTS\n{'='*80}")
    print(f"{'Method':<30}{'PSNR':>8}{'SSIM':>8}{'Dice_WT':>10}"
          f"{'Dice_TC':>10}{'Dice_ET':>10}{'mask?':>7}")
    print('-' * 83)
    for r in results:
        print(f"  {r['Short']:<28}{float(r['PSNR']):>8.2f}{float(r['SSIM']):>8.3f}"
              f"{float(r['Dice_WT']):>10.3f}{float(r['Dice_TC']):>10.3f}"
              f"{float(r['Dice_ET']):>10.3f}"
              f"{'YES' if r['MaskAtInference'] else '-':>7}")
    print('=' * 83)

    print("\n=== FIGURES ===", flush=True)
    make_all_figures(args.out, args.sigma, results, histories)

    print(f"\nAll outputs in {args.out}")
    print(f"  checkpoints/   reload with load_checkpoint(path)")
    print(f"  predictions/   redraw figures with --figures_only")
    print(f"  figures/       9 figures, all five models\n")


if __name__ == '__main__':
    main()
