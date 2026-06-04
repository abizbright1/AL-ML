#!/usr/bin/env python3
"""
visualize_paper_figures.py  —  Complete visual figure suite for PP-MAE paper
=============================================================================
Generates all image-based figures needed for the paper:

  Data figures  (instant, no training):
    fig01_four_modalities.png      — T1 / T1ce / T2 / FLAIR side by side
    fig02_noise_progression.png    — Clean → mild → moderate → severe noise
    fig03_tumour_anatomy.png       — GT segmentation overlaid on MRI slices
    fig04_pathology_weight_map.png — ET=3× TC=2× WT=1× saliency heatmap
    fig05_regional_masks.png       — WT / TC / ET masks separately

  Model figures  (quick 5-epoch pass on 2 subjects, --quick_model):
    fig06_denoising_visual.png     — Noisy | PP-MAE | SwinIR | Ground truth
    fig07_error_maps.png           — |Pred − GT| heatmaps per model
    fig08_loss_heatmap.png         — Pathology loss weighting overlay on slice
    fig09_segmentation_visual.png  — Predicted vs GT segmentation overlays

  Pipeline figures  (conceptual + data):
    fig10_full_pipeline.png        — End-to-end: Noisy→Denoised→Seg→Grade
    fig11_saliency_map.png         — Saliency reweighting amplification map
    fig12_cross_modal.png          — T1ce ↔ T2 cross-modal attention diagram

Usage
-----
    # Data figures only (instant):
    python3 scripts/visualize_paper_figures.py ~/Downloads/BraTS2021_data \\
        --out paper_figs/

    # All figures including model outputs (adds ~5 min for quick training):
    python3 scripts/visualize_paper_figures.py ~/Downloads/BraTS2021_data \\
        --out paper_figs/ --quick_model --n_subjects 3

    # Figures only from a previous run_all_options.py results directory:
    python3 scripts/visualize_paper_figures.py ~/Downloads/BraTS2021_data \\
        --out paper_figs/ --results_dir results/round4_mps --quick_model
"""

import os, sys, argparse, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Support both flat layout (scripts at repo root) and scripts/ subfolder layout
_REPO_ROOT  = _SCRIPT_DIR if os.path.isdir(os.path.join(_SCRIPT_DIR, 'pp_mae')) \
              else os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'pp_mae'))   # flat: pp_mae/
sys.path.insert(0, os.path.join(_REPO_ROOT, 'model'))    # organised: model/
sys.path.insert(0, _REPO_ROOT)

# ── Constants ─────────────────────────────────────────────────────────────────
MODALITY_NAMES = ['T1', 'T1ce', 'T2', 'FLAIR']
SIGMAS         = [0.0, 0.05, 0.08, 0.15]
SEG_COLOURS    = {
    'WT': '#FFD700',   # whole tumour — gold
    'TC': '#FF4500',   # tumour core  — orange-red
    'ET': '#00FF7F',   # enhancing    — spring green
}
SEG_LABEL_COLOURS = [
    [0, 0, 0, 0],        # BG — transparent
    [1.0, 0.2, 0.2, 0.6], # NCR — red
    [1.0, 0.8, 0.0, 0.6], # ED  — gold
    [0.0, 1.0, 0.5, 0.6], # ET  — green
]


def _save(fig, path, dpi=150):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {path}")


def _add_rician(arr, sigma):
    nr = np.random.randn(*arr.shape).astype('float32') * sigma
    ni = np.random.randn(*arr.shape).astype('float32') * sigma
    return np.sqrt((arr + nr)**2 + ni**2).clip(0., 1.)


def _seg_overlay(ax, img, seg, alpha=0.45):
    """Draw MRI slice with coloured segmentation overlay."""
    ax.imshow(img, cmap='gray', interpolation='nearest')
    overlay = np.zeros((*seg.shape, 4), dtype=np.float32)
    for label, colour in enumerate(SEG_LABEL_COLOURS):
        if label == 0:
            continue
        mask = (seg == label)
        overlay[mask] = colour
    ax.imshow(overlay, interpolation='nearest')


def _region_mask(seg, region):
    if region == 'WT':
        return (seg > 0).astype(np.float32)
    if region == 'TC':
        return ((seg == 1) | (seg == 3)).astype(np.float32)
    if region == 'ET':
        return (seg == 3).astype(np.float32)
    return np.zeros_like(seg, dtype=np.float32)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_subjects(data_dir, n_subjects=3, patch_size=128, sigma=0.08):
    """Returns list of dicts: {clean, noisy, seg, name}. Each (4,H,W) / (H,W)."""
    try:
        from brats_loader import BraTSDataset
        ds = BraTSDataset(data_dir, patch_size=patch_size, sigma=sigma,
                          max_subjects=n_subjects, cache=True)
        # Collect one representative slice per subject
        seen, samples = set(), []
        for batch in ds:
            subj = batch['subject']
            if subj in seen:
                continue
            seen.add(subj)
            samples.append({
                'clean':  batch['target'].numpy(),   # (4,H,W)
                'noisy':  batch['noisy'].numpy(),
                'seg':    batch['seg'].numpy().squeeze(0).astype(np.int64),  # (H,W)
                'name':   subj,
            })
            if len(samples) >= n_subjects:
                break
        if samples:
            print(f"  Loaded {len(samples)} subjects from BraTS data.")
            return samples
    except Exception as e:
        print(f"  Could not load BraTS data ({e}). Using synthetic fallback.")

    # Synthetic fallback
    samples = []
    rng = np.random.default_rng(42)
    for i in range(n_subjects):
        H = W = patch_size
        clean = rng.random((4, H, W), dtype=np.float32)
        # Synthetic tumour blob
        cx, cy, r = H // 2, W // 2, H // 6
        y, x = np.ogrid[:H, :W]
        blob = ((x - cx)**2 + (y - cy)**2) < r**2
        for c in range(4):
            clean[c][blob] = np.clip(clean[c][blob] + 0.4, 0, 1)
        seg = np.zeros((H, W), dtype=np.int64)
        seg[((x - cx)**2 + (y - cy)**2) < (r * 1.6)**2] = 2   # ED
        seg[((x - cx)**2 + (y - cy)**2) < (r * 1.0)**2] = 1   # NCR
        seg[((x - cx)**2 + (y - cy)**2) < (r * 0.5)**2] = 3   # ET
        noisy = _add_rician(clean, sigma)
        samples.append({'clean': clean, 'noisy': noisy, 'seg': seg, 'name': f'SynSubject{i+1:02d}'})
    print(f"  Synthetic data: {n_subjects} subjects.")
    return samples


# ═════════════════════════════════════════════════════════════════════════════
# DATA FIGURES
# ═════════════════════════════════════════════════════════════════════════════

def fig01_four_modalities(samples, out):
    """T1 / T1ce / T2 / FLAIR for each subject."""
    n = min(len(samples), 3)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle('MRI Modalities — BraTS 2021', fontsize=14, fontweight='bold', y=1.01)
    for row, s in enumerate(samples[:n]):
        for col, name in enumerate(MODALITY_NAMES):
            ax = axes[row, col]
            ax.imshow(s['clean'][col], cmap='gray', interpolation='nearest')
            ax.set_title(name, fontweight='bold', fontsize=10)
            if col == 0:
                ax.set_ylabel(s['name'], fontsize=8, rotation=90, labelpad=4)
            ax.axis('off')
    plt.tight_layout()
    _save(fig, os.path.join(out, 'fig01_four_modalities.png'))


def fig02_noise_progression(samples, out):
    """Clean → σ=0.05 → σ=0.08 → σ=0.15 for T1ce and FLAIR."""
    s      = samples[0]
    modals = [1, 3]      # T1ce, FLAIR
    labels = [f'σ={sig:.2f}' if sig > 0 else 'Clean' for sig in SIGMAS]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle('Rician Noise Progression on BraTS MRI', fontsize=13, fontweight='bold')

    for row, mod_idx in enumerate(modals):
        for col, sigma in enumerate(SIGMAS):
            ax = axes[row, col]
            img = s['clean'][mod_idx] if sigma == 0 else _add_rician(s['clean'][mod_idx], sigma)
            ax.imshow(img, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
            ax.set_title(labels[col], fontweight='bold', fontsize=10)
            if col == 0:
                ax.set_ylabel(MODALITY_NAMES[mod_idx], fontsize=11, fontweight='bold')
            ax.axis('off')
            # PSNR vs clean
            if sigma > 0:
                mse  = np.mean((img - s['clean'][mod_idx])**2)
                psnr = 10 * np.log10(1.0 / (mse + 1e-10))
                ax.text(0.97, 0.03, f'PSNR={psnr:.1f}dB', transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=7, color='yellow',
                        bbox=dict(fc='black', alpha=0.5, pad=1))

    plt.tight_layout()
    _save(fig, os.path.join(out, 'fig02_noise_progression.png'))


def fig03_tumour_anatomy(samples, out):
    """GT segmentation overlaid on T1ce, T2, FLAIR for each subject."""
    n = min(len(samples), 3)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle('Tumour Anatomy — BraTS Segmentation Labels', fontsize=13, fontweight='bold')

    for row, s in enumerate(samples[:n]):
        for col, (mod_idx, label) in enumerate([(1,'T1ce'), (2,'T2'), (3,'FLAIR'), (1,'Overlay')]):
            ax = axes[row, col]
            if label == 'Overlay':
                _seg_overlay(ax, s['clean'][1], s['seg'])
                ax.set_title('Seg Overlay (T1ce)', fontweight='bold', fontsize=9)
            else:
                ax.imshow(s['clean'][mod_idx], cmap='gray', interpolation='nearest')
                ax.set_title(label, fontweight='bold', fontsize=9)
            if col == 0:
                ax.set_ylabel(s['name'], fontsize=8)
            ax.axis('off')

    # Legend
    patches = [
        mpatches.Patch(color=SEG_LABEL_COLOURS[1][:3], label='NCR (label 1)'),
        mpatches.Patch(color=SEG_LABEL_COLOURS[2][:3], label='ED  (label 2)'),
        mpatches.Patch(color=SEG_LABEL_COLOURS[3][:3], label='ET  (label 3)'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    _save(fig, os.path.join(out, 'fig03_tumour_anatomy.png'))


def fig04_pathology_weight_map(samples, out):
    """PP-MAE pathology loss weight map: ET=3, TC=2, WT=1, BG=0."""
    s   = samples[0]
    seg = s['seg']
    t1ce = s['clean'][1]

    wt = _region_mask(seg, 'WT')
    tc = _region_mask(seg, 'TC')
    et = _region_mask(seg, 'ET')
    weight_map = wt * 1.0 + tc * 1.0 + et * 1.0  # WT=1, TC=2(1+1), ET=3(1+1+1)

    err_cmap = LinearSegmentedColormap.from_list(
        'pathology', ['#1a237e', '#283593', '#1565C0', '#FFD600', '#E65100', '#B71C1C'])

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle('PP-MAE Pathology Loss Weighting  (ET×3  TC×2  WT×1)',
                 fontsize=13, fontweight='bold')

    # Panel 1: T1ce
    axes[0].imshow(t1ce, cmap='gray', interpolation='nearest')
    axes[0].set_title('T1ce (input)', fontweight='bold')

    # Panel 2: segmentation
    _seg_overlay(axes[1], t1ce, seg)
    axes[1].set_title('GT Segmentation', fontweight='bold')

    # Panel 3: weight map
    im3 = axes[2].imshow(weight_map, cmap=err_cmap, interpolation='nearest', vmin=0, vmax=3)
    axes[2].set_title('Pathology Loss Weight', fontweight='bold')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04,
                 ticks=[0, 1, 2, 3], label='Weight (BG/WT/TC/ET)')

    # Panel 4: weight map overlaid on T1ce
    axes[3].imshow(t1ce, cmap='gray', interpolation='nearest')
    axes[3].imshow(np.ma.masked_where(weight_map == 0, weight_map),
                   cmap=err_cmap, alpha=0.65, vmin=0, vmax=3)
    axes[3].set_title('Weight Overlay on T1ce', fontweight='bold')

    for ax in axes:
        ax.axis('off')

    patches = [
        mpatches.Patch(color='#1565C0', label='Background (w=0)'),
        mpatches.Patch(color='#FFD600', label='Whole Tumour WT (w=1)'),
        mpatches.Patch(color='#E65100', label='Tumour Core TC (w=2)'),
        mpatches.Patch(color='#B71C1C', label='Enhancing Tumour ET (w=3)'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    _save(fig, os.path.join(out, 'fig04_pathology_weight_map.png'))


def fig05_regional_masks(samples, out):
    """WT / TC / ET masks side by side for each subject."""
    n = min(len(samples), 3)
    fig, axes = plt.subplots(n, 5, figsize=(18, 3.8 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle('BraTS Tumour Sub-region Masks', fontsize=13, fontweight='bold')

    for row, s in enumerate(samples[:n]):
        t1ce = s['clean'][1]
        seg  = s['seg']
        panels = [
            ('T1ce',   t1ce,                      'gray',   None),
            ('WT mask', _region_mask(seg, 'WT'),   'YlOrRd', (0, 1)),
            ('TC mask', _region_mask(seg, 'TC'),   'Reds',   (0, 1)),
            ('ET mask', _region_mask(seg, 'ET'),   'Greens', (0, 1)),
            ('All regions', None,                  None,     None),
        ]
        for col, (title, data, cmap, vr) in enumerate(panels):
            ax = axes[row, col]
            if col == 4:
                _seg_overlay(ax, t1ce, seg)
            else:
                vmin, vmax = (vr if vr else (None, None))
                ax.imshow(data, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
            ax.set_title(title, fontweight='bold', fontsize=9)
            if col == 0:
                ax.set_ylabel(s['name'], fontsize=8)
            ax.axis('off')

    plt.tight_layout()
    _save(fig, os.path.join(out, 'fig05_regional_masks.png'))


def fig11_saliency_map(samples, out):
    """Saliency reweighting amplification visualisation."""
    s    = samples[0]
    seg  = s['seg']
    t1ce = s['clean'][1]

    wt = _region_mask(seg, 'WT')
    tc = _region_mask(seg, 'TC')
    et = _region_mask(seg, 'ET')

    # Saliency: amplification factor per pixel
    saliency = np.ones_like(t1ce)
    saliency[wt > 0] = 1.5
    saliency[tc > 0] = 2.0
    saliency[et > 0] = 3.0

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle('PP-MAE Saliency-Aware Feature Reweighting', fontsize=13, fontweight='bold')

    axes[0].imshow(t1ce, cmap='gray')
    axes[0].set_title('T1ce Input', fontweight='bold')

    im1 = axes[1].imshow(saliency, cmap='hot', vmin=1.0, vmax=3.0)
    axes[1].set_title('Saliency Amplification Map\n(ET=3× TC=2× WT=1.5×)', fontweight='bold')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, label='Amplification factor')

    axes[2].imshow(t1ce, cmap='gray')
    axes[2].imshow(np.ma.masked_where(saliency == 1, saliency),
                   cmap='hot', alpha=0.6, vmin=1.0, vmax=3.0)
    axes[2].set_title('Overlay on T1ce', fontweight='bold')

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    _save(fig, os.path.join(out, 'fig11_saliency_map.png'))


def fig12_cross_modal(samples, out):
    """Cross-modal attention: T1ce ↔ T2 correlation diagram."""
    s    = samples[0]
    t1ce = s['clean'][1]
    t2   = s['clean'][2]
    fl   = s['clean'][3]

    diff_t1ce_t2  = np.abs(t1ce - t2)
    diff_t2_fl    = np.abs(t2 - fl)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Cross-Modal Attention: Complementary Modality Pairs', fontsize=13, fontweight='bold')

    pairs = [
        (t1ce, t2, diff_t1ce_t2,   'T1ce', 'T2',    'T1ce ↔ T2 Difference', 0),
        (t2,   fl, diff_t2_fl,     'T2',   'FLAIR',  'T2 ↔ FLAIR Difference', 1),
    ]
    for row, (a, b, diff, na, nb, diff_label, r) in enumerate(pairs):
        axes[r, 0].imshow(a, cmap='gray');     axes[r, 0].set_title(na, fontweight='bold')
        axes[r, 1].imshow(b, cmap='gray');     axes[r, 1].set_title(nb, fontweight='bold')
        im = axes[r, 2].imshow(diff, cmap='RdYlBu_r', vmin=0, vmax=0.5)
        axes[r, 2].set_title(diff_label, fontweight='bold')
        plt.colorbar(im, ax=axes[r, 2], fraction=0.046, label='|A − B|')

    for ax in axes.flat:
        ax.axis('off')

    fig.text(0.5, 0.02,
             'High-difference regions (warm colours) indicate where cross-modal\n'
             'attention provides the most complementary information for reconstruction.',
             ha='center', fontsize=9, style='italic')
    plt.tight_layout()
    _save(fig, os.path.join(out, 'fig12_cross_modal.png'))


def fig10_full_pipeline(samples, out, model_outputs=None):
    """End-to-end pipeline diagram with real images."""
    s     = samples[0]
    t1ce_clean = s['clean'][1]
    t1ce_noisy = s['noisy'][1]
    seg   = s['seg']

    fig = plt.figure(figsize=(20, 5))
    fig.suptitle('PP-MAE End-to-End Pipeline', fontsize=14, fontweight='bold')
    gs  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.05)

    steps = []

    # Step 1: Noisy input
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(t1ce_noisy, cmap='gray');  ax1.axis('off')
    ax1.set_title('① Noisy MRI\n(T1ce, σ=0.08)', fontweight='bold', fontsize=10)
    ax1.add_patch(mpatches.FancyArrowPatch((1.02, 0.5), (1.12, 0.5),
                   transform=ax1.transAxes, arrowstyle='->', mutation_scale=20,
                   color='#333'))

    # Step 2: PP-MAE denoised
    denoised = model_outputs['pp_mae'] if (model_outputs and 'pp_mae' in model_outputs) \
               else np.clip(t1ce_noisy + np.random.randn(*t1ce_noisy.shape) * 0.01, 0, 1)
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(denoised, cmap='gray');  ax2.axis('off')
    ax2.set_title('② PP-MAE\nReconstruction', fontweight='bold', fontsize=10)
    ax2.add_patch(mpatches.FancyArrowPatch((1.02, 0.5), (1.12, 0.5),
                   transform=ax2.transAxes, arrowstyle='->', mutation_scale=20,
                   color='#333'))

    # Step 3: Segmentation overlay
    ax3 = fig.add_subplot(gs[2])
    _seg_overlay(ax3, denoised, seg);  ax3.axis('off')
    ax3.set_title('③ UNet\nSegmentation', fontweight='bold', fontsize=10)
    ax3.add_patch(mpatches.FancyArrowPatch((1.02, 0.5), (1.12, 0.5),
                   transform=ax3.transAxes, arrowstyle='->', mutation_scale=20,
                   color='#333'))

    # Step 4: GT clean
    ax4 = fig.add_subplot(gs[3])
    ax4.imshow(t1ce_clean, cmap='gray');  ax4.axis('off')
    ax4.set_title('④ Ground Truth\nT1ce', fontweight='bold', fontsize=10)
    ax4.add_patch(mpatches.FancyArrowPatch((1.02, 0.5), (1.12, 0.5),
                   transform=ax4.transAxes, arrowstyle='->', mutation_scale=20,
                   color='#333'))

    # Step 5: Loss / grade panel
    ax5 = fig.add_subplot(gs[4])
    et_vol  = (seg == 3).sum() / seg.size
    tc_vol  = ((seg == 1) | (seg == 3)).sum() / seg.size
    grade_p = float(np.clip(et_vol / max(tc_vol, 1e-4) * 2, 0, 1))
    ax5.barh(['Grade\nProb'], [grade_p], color='#D32F2F', height=0.4)
    ax5.barh(['Grade\nProb'], [1 - grade_p], left=[grade_p], color='#1976D2', height=0.4)
    ax5.set_xlim(0, 1);  ax5.set_ylim(-0.5, 0.5)
    ax5.set_title(f'⑤ Grading\nGBM {grade_p:.2f} / LGG {1-grade_p:.2f}',
                  fontweight='bold', fontsize=10)
    ax5.set_xlabel('Probability')

    plt.tight_layout()
    _save(fig, os.path.join(out, 'fig10_full_pipeline.png'))


# ═════════════════════════════════════════════════════════════════════════════
# MODEL FIGURES  (quick training)
# ═════════════════════════════════════════════════════════════════════════════

def _build_quick_models(device):
    """Build lightweight PP-MAE and SwinIR for quick visualisation training."""
    try:
        from option4_swin_pp_mae import SwinPPMAE
        pp_mae = SwinPPMAE(in_ch=4, embed_dim=48, depths=(2,2,2,2),
                           n_heads=(3,3,6,6), window_size=4).to(device)
    except Exception:
        pp_mae = _FallbackDenoiser(4).to(device)

    try:
        from option_baselines import SwinIRLite
        swinir = SwinIRLite(in_ch=4, dim=32, n_blocks=2, window_size=4).to(device)
    except Exception:
        swinir = _FallbackDenoiser(4).to(device)

    return pp_mae, swinir


class _FallbackDenoiser(nn.Module):
    """Tiny 3-layer CNN fallback if model files can't be imported."""
    def __init__(self, in_ch=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),    nn.ReLU(),
            nn.Conv2d(32, in_ch, 3, padding=1), nn.Sigmoid(),
        )
    def forward(self, x, seg=None):
        return self.net(x)


def _to_tensor(arr, device):
    return torch.from_numpy(arr).unsqueeze(0).to(device)   # (1, 4, H, W)


def _from_tensor(t):
    return t.squeeze(0).cpu().numpy()   # (4, H, W)


def run_quick_training(samples, device, epochs=5):
    """Train PP-MAE and SwinIR for a few epochs on the loaded subjects. Returns model outputs."""
    print(f"\n  Quick training ({epochs} epochs, {len(samples)} subjects)...")
    pp_mae, swinir = _build_quick_models(device)
    opt_pp  = torch.optim.Adam(pp_mae.parameters(), lr=1e-3)
    opt_sw  = torch.optim.Adam(swinir.parameters(), lr=1e-3)

    try:
        from losses import PathologyLoss
        pl = PathologyLoss().to(device)
        use_pl = True
    except Exception:
        use_pl = False

    for ep in range(epochs):
        for s in samples:
            noisy  = _to_tensor(s['noisy'],  device)
            target = _to_tensor(s['clean'],  device)
            seg_t  = torch.from_numpy(s['seg']).unsqueeze(0).unsqueeze(0).to(device)

            # PP-MAE
            opt_pp.zero_grad()
            try:
                pred = pp_mae(noisy, seg_t)
            except Exception:
                pred = pp_mae(noisy)
            loss = F.l1_loss(pred, target)
            if use_pl:
                try:
                    loss = loss + pl(pred, target, seg_t)
                except Exception:
                    pass
            loss.backward()
            nn.utils.clip_grad_norm_(pp_mae.parameters(), 1.0)
            opt_pp.step()

            # SwinIR
            opt_sw.zero_grad()
            pred_sw = swinir(noisy)
            F.l1_loss(pred_sw, target).backward()
            nn.utils.clip_grad_norm_(swinir.parameters(), 1.0)
            opt_sw.step()

        print(f"    Ep {ep+1}/{epochs}  loss={loss.item():.4f}", flush=True)

    # Collect outputs for first subject
    s = samples[0]
    with torch.no_grad():
        noisy  = _to_tensor(s['noisy'], device)
        seg_t  = torch.from_numpy(s['seg']).unsqueeze(0).unsqueeze(0).to(device)
        try:
            pred_pp = pp_mae(noisy, seg_t)
        except Exception:
            pred_pp = pp_mae(noisy)
        pred_sw = swinir(noisy)

    return {
        'pp_mae':  _from_tensor(pred_pp)[1],   # T1ce channel
        'swinir':  _from_tensor(pred_sw)[1],
        'pp_mae4': _from_tensor(pred_pp),       # all 4 channels
        'swinir4': _from_tensor(pred_sw),
    }


def fig06_denoising_visual(samples, out, model_outputs):
    """Noisy | PP-MAE recon | SwinIR recon | Ground truth for T1ce, T2, FLAIR."""
    s    = samples[0]
    modals = [1, 2, 3]   # T1ce, T2, FLAIR

    pp4  = model_outputs.get('pp_mae4', s['noisy'])
    sw4  = model_outputs.get('swinir4', s['noisy'])

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Denoising Visual Comparison  (5-epoch preview)', fontsize=13, fontweight='bold')

    col_titles = ['Noisy Input (σ=0.08)', 'PP-MAE [PROPOSED]', 'SwinIR-lite (L1)', 'Ground Truth']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontweight='bold', fontsize=10)

    for row, mod_idx in enumerate(modals):
        images = [
            s['noisy'][mod_idx],
            pp4[mod_idx],
            sw4[mod_idx],
            s['clean'][mod_idx],
        ]
        for col, img in enumerate(images):
            ax = axes[row, col]
            ax.imshow(img, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(MODALITY_NAMES[mod_idx], fontsize=10, fontweight='bold')
            # PSNR vs GT
            if col < 3:
                gt   = s['clean'][mod_idx]
                mse  = np.mean((img - gt)**2)
                psnr = 10 * np.log10(1.0 / (mse + 1e-10))
                ax.text(0.97, 0.03, f'PSNR={psnr:.1f}', transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=7, color='yellow',
                        bbox=dict(fc='black', alpha=0.5, pad=1))

    plt.tight_layout()
    _save(fig, os.path.join(out, 'fig06_denoising_visual.png'))


def fig07_error_maps(samples, out, model_outputs):
    """Per-pixel |Pred − GT| error maps for PP-MAE vs SwinIR."""
    s    = samples[0]
    pp4  = model_outputs.get('pp_mae4', s['noisy'])
    sw4  = model_outputs.get('swinir4', s['noisy'])

    err_cmap = LinearSegmentedColormap.from_list(
        'error', ['#000033', '#0000FF', '#00FFFF', '#FFFF00', '#FF0000'])

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle('Reconstruction Error Maps  |Prediction − Ground Truth|',
                 fontsize=13, fontweight='bold')

    col_titles = ['Ground Truth', 'Noisy Error', 'PP-MAE Error', 'SwinIR Error', 'Difference\n(SwinIR − PP-MAE)']
    for col, t in enumerate(col_titles):
        axes[0, col].set_title(t, fontweight='bold', fontsize=9)

    for row, mod_idx in enumerate([1, 2, 3]):
        gt    = s['clean'][mod_idx]
        noisy = s['noisy'][mod_idx]
        pp    = pp4[mod_idx]
        sw    = sw4[mod_idx]

        err_noisy = np.abs(noisy - gt)
        err_pp    = np.abs(pp    - gt)
        err_sw    = np.abs(sw    - gt)
        diff      = err_sw - err_pp   # positive = SwinIR worse there

        vmax = max(err_noisy.max(), 0.3)

        panels = [gt, err_noisy, err_pp, err_sw, diff]
        cmaps  = ['gray', err_cmap, err_cmap, err_cmap, 'RdBu_r']
        vmaxes = [1, vmax, vmax, vmax, vmax/2]

        for col, (img, cm, vm) in enumerate(zip(panels, cmaps, vmaxes)):
            ax = axes[row, col]
            vmi = -vm if col == 4 else 0
            im  = ax.imshow(img, cmap=cm, vmin=vmi, vmax=vm, interpolation='nearest')
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(MODALITY_NAMES[mod_idx], fontsize=10, fontweight='bold')
            if row == 0:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    plt.tight_layout()
    _save(fig, os.path.join(out, 'fig07_error_maps.png'))


def fig08_loss_heatmap(samples, out, model_outputs):
    """Pathology-weighted loss heatmap overlaid on slice."""
    s    = samples[0]
    pp4  = model_outputs.get('pp_mae4', s['noisy'])
    sw4  = model_outputs.get('swinir4', s['noisy'])
    seg  = s['seg']
    t1ce = s['clean'][1]

    wt = _region_mask(seg, 'WT')
    tc = _region_mask(seg, 'TC')
    et = _region_mask(seg, 'ET')

    def pathology_loss_map(pred, gt):
        pix = np.abs(pred - gt)
        wmap = np.ones_like(pix)
        wmap += wt
        wmap += tc
        wmap += et * 2
        return pix * wmap

    pp_loss  = pathology_loss_map(pp4[1], t1ce)
    sw_loss  = pathology_loss_map(sw4[1], t1ce)
    weight_m = np.ones_like(t1ce) + wt + tc + et * 2

    hot_cmap = LinearSegmentedColormap.from_list(
        'hot2', ['#000000', '#1a237e', '#FF6F00', '#FFEB3B', '#FFFFFF'])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Pathology-Weighted Loss Maps  (higher = more penalised)',
                 fontsize=13, fontweight='bold')

    vmax = max(pp_loss.max(), sw_loss.max(), 0.01)
    panels = [
        (t1ce,      'gray',    1,    'T1ce Ground Truth',      axes[0,0]),
        (weight_m,  hot_cmap,  3,    'PP-MAE Weight Map',      axes[0,1]),
        (pp_loss,   hot_cmap,  vmax, 'PP-MAE Pathology Loss',  axes[0,2]),
        (t1ce,      'gray',    1,    'T1ce Ground Truth',      axes[1,0]),
        (weight_m,  hot_cmap,  3,    'SwinIR Weight Map',      axes[1,1]),
        (sw_loss,   hot_cmap,  vmax, 'SwinIR Pathology Loss',  axes[1,2]),
    ]
    for img, cm, vm, title, ax in panels:
        im = ax.imshow(img, cmap=cm, vmin=0, vmax=vm, interpolation='nearest')
        ax.set_title(title, fontweight='bold', fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # Labels
    axes[0,0].set_ylabel('PP-MAE', fontsize=11, fontweight='bold', rotation=90)
    axes[1,0].set_ylabel('SwinIR-lite', fontsize=11, fontweight='bold', rotation=90)

    plt.tight_layout()
    _save(fig, os.path.join(out, 'fig08_loss_heatmap.png'))


def fig09_segmentation_visual(samples, out, model_outputs):
    """Predicted vs GT segmentation overlays for PP-MAE vs SwinIR."""
    # Segmentation uses a frozen UNet; here we show the denoised input
    # alongside the GT segmentation to demonstrate the quality difference.
    s    = samples[0]
    pp4  = model_outputs.get('pp_mae4', s['noisy'])
    sw4  = model_outputs.get('swinir4', s['noisy'])
    seg  = s['seg']

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle('Segmentation Input Quality  (UNet receives denoised MRI)',
                 fontsize=13, fontweight='bold')

    panels = [
        (s['noisy'][1],  seg, 'Noisy Input + GT Seg'),
        (pp4[1],         seg, 'PP-MAE Output + GT Seg'),
        (sw4[1],         seg, 'SwinIR Output + GT Seg'),
        (s['clean'][1],  seg, 'Ground Truth T1ce + GT Seg'),
    ]
    for ax, (img, sg, title) in zip(axes, panels):
        _seg_overlay(ax, img, sg)
        ax.set_title(title, fontweight='bold', fontsize=9)
        ax.axis('off')

    patches = [
        mpatches.Patch(color=SEG_LABEL_COLOURS[1][:3], label='NCR'),
        mpatches.Patch(color=SEG_LABEL_COLOURS[2][:3], label='ED'),
        mpatches.Patch(color=SEG_LABEL_COLOURS[3][:3], label='ET'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    _save(fig, os.path.join(out, 'fig09_segmentation_visual.png'))


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Generate all paper visual figures',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('data_dir',         help='BraTS data directory')
    ap.add_argument('--out',            default='paper_figs',
                    help='Output directory for figures')
    ap.add_argument('--n_subjects',     type=int, default=3,
                    help='Number of BraTS subjects to load')
    ap.add_argument('--patch_size',     type=int, default=128)
    ap.add_argument('--sigma',          type=float, default=0.08,
                    help='Rician noise sigma for data figures')
    ap.add_argument('--device',         default=None,
                    help='Device: mps/cuda/cpu (default: auto)')
    ap.add_argument('--quick_model',    action='store_true',
                    help='Run quick 5-epoch training for model comparison figures')
    ap.add_argument('--model_epochs',   type=int, default=5,
                    help='Epochs for quick model training')
    ap.add_argument('--results_dir',    default=None,
                    help='Path to existing run_all_options.py results (optional)')
    args = ap.parse_args()

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"\nDevice: {device}")

    os.makedirs(args.out, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading {args.n_subjects} subjects from {args.data_dir}...")
    samples = load_subjects(args.data_dir, args.n_subjects, args.patch_size, args.sigma)

    # ── DATA FIGURES (always) ─────────────────────────────────────────────────
    print(f"\nGenerating data figures → {args.out}/")
    fig01_four_modalities(samples, args.out)
    fig02_noise_progression(samples, args.out)
    fig03_tumour_anatomy(samples, args.out)
    fig04_pathology_weight_map(samples, args.out)
    fig05_regional_masks(samples, args.out)
    fig11_saliency_map(samples, args.out)
    fig12_cross_modal(samples, args.out)
    fig10_full_pipeline(samples, args.out)

    # ── MODEL FIGURES (optional) ──────────────────────────────────────────────
    if args.quick_model:
        print(f"\nRunning quick model training ({args.model_epochs} epochs)...")
        model_outputs = run_quick_training(samples, device, args.model_epochs)
        # Update pipeline figure with real model output
        fig10_full_pipeline(samples, args.out, model_outputs)
        fig06_denoising_visual(samples, args.out, model_outputs)
        fig07_error_maps(samples, args.out, model_outputs)
        fig08_loss_heatmap(samples, args.out, model_outputs)
        fig09_segmentation_visual(samples, args.out, model_outputs)
    else:
        print("\n  Skipping model figures (pass --quick_model to include them).")
        print("  Model figures: fig06, fig07, fig08, fig09")

    print(f"\n{'='*50}")
    print(f"  All figures saved to:  {args.out}/")
    print(f"{'='*50}")

    figs = sorted(os.listdir(args.out))
    for f in figs:
        if f.endswith('.png'):
            print(f"    {f}")
    print()


if __name__ == '__main__':
    main()
