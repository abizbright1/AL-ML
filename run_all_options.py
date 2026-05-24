"""
run_all_options.py — Unified runner comparing all 4 PP-MAE options vs baselines
=================================================================================

Runs 4 rounds of experiments, each comparing one PP-MAE option against
architecture-matched baselines:

  Round 1 — CNN Family      (Option 1 vs DnCNN / UNet-L1 / Noise2Noise / REDNet)
  Round 2 — ViT/MAE Family  (Option 2 2D vs VanillaMAE / SparK-CNN)
  Round 3 — Multi-task      (Option 3 vs MultiTaskUNet / TransUNet-lite)
  Round 4 — Swin Family     (Option 4 vs SwinIR-lite / Uformer-lite)

Final head-to-head: the best PP-MAE from each round compared directly.

CLI usage
---------
  # Demo mode (no BraTS data needed):
  python3 run_all_options.py

  # Real BraTS data:
  python3 run_all_options.py /path/to/BraTS2021_Training_Data

  # Full options:
  python3 run_all_options.py [brats_root] \\
      --epochs 20 --seg_epochs 20 --patch_size 96 --sigma 0.08 \\
      --out /path/to/output --max_subjects 20 --rounds 1,2,3,4

Output files
------------
  options_round1.png   — CNN family PSNR + Dice ET
  options_round2.png   — ViT/MAE family PSNR + Dice ET
  options_round3.png   — Multi-task family PSNR + Dice ET
  options_round4.png   — Swin family PSNR + Dice ET
  options_headtohead.png — All 4 PP-MAE options, 6-metric comparison
  options_table.png    — Full results table across all methods and rounds
  options_results.csv  — Numeric results
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import sys
import os
import csv
import argparse
import numpy as np
from typing import Optional, Dict, Tuple, List, Callable

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, 'pp_mae'))

# ── PP-MAE Option imports ─────────────────────────────────────────────────────
from option1_cnn_pp_mae import CNNPPMAE, PPMAETrainer
from option3_full_pipeline import PPMAEPipeline, PipelineTrainer
from option4_swin_pp_mae import SwinPPMAE, SwinPPMAETrainer

# ── Baseline imports ──────────────────────────────────────────────────────────
from baselines import (
    DnCNN, DnCNNTrainer,
    StandardUNet, StandardUNetTrainer,
    Noise2Noise, Noise2NoiseTrainer,
    REDNet, REDNetTrainer,
)
from option_baselines import (
    VanillaMAE2D, VanillaMAETrainer,
    ViTPPMAE2D, ViTPPMAE2DTrainer,
    SparKCNN, SparKCNNTrainer,
    MultiTaskUNet, MultiTaskUNetTrainer,
    TransUNetLite, TransUNetTrainer,
    SwinIRLite, SwinIRTrainer,
    UformerLite, UformerTrainer,
)

# ── Shared utilities ──────────────────────────────────────────────────────────
from segmentor import UNetSegmentor, SegTrainer, seg_metrics
from evaluation import psnr, ssim_numpy, nrmse
from brats_loader import BraTSDataset, make_demo_brats

# ── CLI ───────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(
    description='PP-MAE all-options comparison runner',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
_parser.add_argument('brats_root', nargs='?', default=None,
                     help='Path to BraTS training folder (omit for demo mode)')
_parser.add_argument('--epochs',       type=int,   default=20,
                     help='Denoising training epochs per model')
_parser.add_argument('--seg_epochs',   type=int,   default=20,
                     help='Segmentor training epochs')
_parser.add_argument('--patch_size',   type=int,   default=96,
                     help='Spatial crop size in pixels (must be divisible by 8)')
_parser.add_argument('--sigma',        type=float, default=0.08,
                     help='Rician noise sigma')
_parser.add_argument('--out',          type=str,   default=None,
                     help='Output directory for plots and CSV')
_parser.add_argument('--max_subjects', type=int,   default=None,
                     help='Limit number of BraTS subjects (None = all; demo uses 6)')
_parser.add_argument('--rounds',       type=str,   default='1,2,3,4',
                     help='Comma-separated list of rounds to run (e.g. "1,3")')
_args = _parser.parse_args()

# ── Configuration ─────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
OUT        = _args.out if _args.out else _SCRIPT_DIR
os.makedirs(OUT, exist_ok=True)

D_EPOCHS   = _args.epochs
S_EPOCHS   = _args.seg_epochs
PATCH_SIZE = _args.patch_size
SIGMA      = _args.sigma
SEED       = 42
BATCH_SIZE = 4
MAX_SUBJ   = _args.max_subjects
ROUNDS     = [int(r.strip()) for r in _args.rounds.split(',') if r.strip()]

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"\nDevice : {DEVICE}")
print(f"Rounds : {ROUNDS}")
print(f"Epochs : {D_EPOCHS}  |  Seg epochs: {S_EPOCHS}", flush=True)

# ── Data loading ──────────────────────────────────────────────────────────────
BRATS_ROOT = _args.brats_root
USE_REAL   = False

if BRATS_ROOT and os.path.isdir(BRATS_ROOT):
    try:
        ds_full = BraTSDataset(
            BRATS_ROOT,
            slice_axis=2,
            patch_size=PATCH_SIZE,
            sigma=SIGMA,
            min_tumour_frac=0.01,
            cache=True,
            max_subjects=MAX_SUBJ,
        )
        USE_REAL   = True
        DATA_LABEL = f'Real BraTS  ({MAX_SUBJ} subjects, {len(ds_full)} slices)'
        print(f"\n Using REAL BraTS data from: {BRATS_ROOT}", flush=True)
        print(f"   {len(ds_full)} tumour slices loaded", flush=True)
    except Exception as e:
        print(f"  Could not load BraTS data: {e}", flush=True)
        print("   Falling back to demo mode.\n", flush=True)

if not USE_REAL:
    n_subj    = MAX_SUBJ if MAX_SUBJ else 6
    ds_full   = make_demo_brats(n_subjects=n_subj, patch_size=PATCH_SIZE,
                                sigma=SIGMA, slices_per_subject=24)
    DATA_LABEL = 'Synthetic BraTS-style demo  (no real files)'
    print("\n DEMO MODE — no real BraTS files found.", flush=True)
    print("   To use real data:  python3 run_all_options.py /path/to/BraTS2021_Training_Data\n",
          flush=True)

n_total = len(ds_full)
n_train = int(0.8 * n_total)
n_val   = n_total - n_train
train_ds, val_ds = torch.utils.data.random_split(
    ds_full, [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED))

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"   Train: {n_train} slices  |  Val: {n_val} slices", flush=True)
print(f"   Data: {DATA_LABEL}\n", flush=True)

# =============================================================================
# SHARED INFRASTRUCTURE
# =============================================================================

# ── Frozen segmentor (shared across all rounds) ───────────────────────────────
print("=== TRAINING SHARED SEGMENTOR (on clean images) ===", flush=True)
seg_model   = UNetSegmentor(in_channels=4, n_classes=4, base_ch=32).to(DEVICE)
seg_trainer = SegTrainer(seg_model, device=DEVICE, lr=5e-4)
seg_hist    = []

for ep in range(1, S_EPOCHS + 1):
    ep_loss = 0.0
    for b in train_loader:
        clean  = b['target'].to(DEVICE)
        labels = b['seg'][:, 0].long().to(DEVICE)
        ep_loss += seg_trainer.step(clean, labels)
    ep_loss /= len(train_loader)
    seg_hist.append(ep_loss)
    if ep % 5 == 0 or ep == 1:
        print(f"  [segmentor] Ep {ep:2d}/{S_EPOCHS}  loss={ep_loss:.4f}", flush=True)

seg_model.eval()
print("Segmentor trained and frozen.\n", flush=True)


def run_epoch(trainer: object, loader: torch.utils.data.DataLoader) -> float:
    """
    Generic epoch runner — works for all trainers with .step(batch) -> dict.
    Also handles PipelineTrainer by calling .stage1_step instead of .step.
    Returns mean total loss for the epoch.
    """
    total = 0.0
    for b in loader:
        if hasattr(trainer, 'stage1_step'):
            # PipelineTrainer uses stage1_step for denoiser pre-training
            m = trainer.stage1_step(b)
        else:
            m = trainer.step(b)
        total += m.get('total', 0.0)
    return total / max(len(loader), 1)


def evaluate_model(
    model:       nn.Module,
    infer_fn:    Callable,
    val_loader:  torch.utils.data.DataLoader,
    seg_model_:  nn.Module,
    device:      str,
) -> Dict[str, float]:
    """
    Evaluate a denoiser on val_loader.

    Args:
        model:      trained denoising model
        infer_fn:   callable(model, noisy, seg) -> denoised tensor  (on device)
        val_loader: validation DataLoader
        seg_model_: frozen segmentor for downstream Dice evaluation
        device:     torch device string

    Returns:
        dict: {psnr, ssim, nrmse, dice_wt, dice_tc, dice_et}
    """
    model.eval()
    seg_model_.eval()

    ps_, ss_, nr_ = [], [], []
    dw_, dt_, de_ = [], [], []

    with torch.no_grad():
        for b in val_loader:
            noisy  = b['noisy'].to(device)
            target = b['target']
            seg    = b['seg'].to(device)

            # Get denoised output
            try:
                pred = infer_fn(model, noisy, seg)
            except TypeError:
                pred = infer_fn(model, noisy, torch.zeros_like(seg))

            if isinstance(pred, dict):
                pred = pred.get('denoised', list(pred.values())[0])

            pred_cpu = pred.detach().cpu()

            # Image quality per sample in batch
            for i in range(pred_cpu.shape[0]):
                p_np = pred_cpu[i].permute(1, 2, 0).numpy()
                t_np = target[i].permute(1, 2, 0).numpy()
                ps_.append(psnr(p_np, t_np))
                ss_.append(ssim_numpy(p_np, t_np))
                nr_.append(nrmse(p_np, t_np))

            # Downstream segmentation on denoised output
            logits = seg_model_(pred_cpu.to(device))
            gt_lab = b['seg'][:, 0].long()
            m      = seg_metrics(logits.cpu(), gt_lab)
            dw_.append(m['dice_wt'])
            dt_.append(m['dice_tc'])
            de_.append(m['dice_et'])

    return {
        'psnr':    float(np.mean(ps_)),
        'ssim':    float(np.mean(ss_)),
        'nrmse':   float(np.mean(nr_)),
        'dice_wt': float(np.mean(dw_)),
        'dice_tc': float(np.mean(dt_)),
        'dice_et': float(np.mean(de_)),
    }


def train_and_eval(
    round_name:   str,
    models_cfg:   Dict[str, Dict],
    train_loader_: torch.utils.data.DataLoader,
    val_loader_:   torch.utils.data.DataLoader,
    seg_model_:    nn.Module,
    epochs:        int,
    device:        str,
) -> Dict[str, Dict[str, float]]:
    """
    Train and evaluate a collection of models for one round.

    models_cfg: dict of {
        'model_name': {
            'model':      nn.Module,
            'trainer_fn': callable(model) -> trainer object,
            'infer_fn':   callable(model, noisy, seg) -> denoised tensor,
        }
    }

    Returns:
        dict of {model_name: metrics_dict}
    """
    results   = {}
    histories = {}

    print(f"\n{'='*60}", flush=True)
    print(f"  {round_name}", flush=True)
    print(f"{'='*60}", flush=True)

    for name, cfg in models_cfg.items():
        model    = cfg['model']
        trainer  = cfg['trainer_fn'](model)
        infer_fn = cfg['infer_fn']
        hist     = []

        print(f"\n  [{name}]", flush=True)
        for ep in range(1, epochs + 1):
            loss = run_epoch(trainer, train_loader_)
            hist.append(loss)
            if hasattr(trainer, 'scheduler'):
                trainer.scheduler.step()
            if ep % 5 == 0 or ep == 1:
                print(f"    Ep {ep:2d}/{epochs}  loss={loss:.4f}", flush=True)

        histories[name] = hist

        # Evaluate
        metrics = evaluate_model(model, infer_fn, val_loader_, seg_model_, device)
        results[name] = metrics
        print(
            f"    PSNR={metrics['psnr']:.2f}  SSIM={metrics['ssim']:.3f}  "
            f"Dice_ET={metrics['dice_et']:.3f}",
            flush=True,
        )

    return results, histories


# =============================================================================
# ROUND DEFINITIONS
# =============================================================================

all_results: Dict[str, Dict[str, Dict[str, float]]] = {}  # round_name -> results
all_histories: Dict[str, Dict] = {}

# ── Round 1: CNN Family ───────────────────────────────────────────────────────
if 1 in ROUNDS:
    r1_cfg = {
        'PP-MAE CNN (clinical_risk)': {
            'model': CNNPPMAE(4, base_ch=32, depth=3),
            'trainer_fn': lambda m: PPMAETrainer(m, device=DEVICE, mode='clinical_risk', lr=3e-4),
            'infer_fn': lambda m, noisy, seg: m(noisy, seg),
        },
        'PP-MAE CNN (fixed)': {
            'model': CNNPPMAE(4, base_ch=32, depth=3),
            'trainer_fn': lambda m: PPMAETrainer(m, device=DEVICE, mode='fixed', lr=3e-4),
            'infer_fn': lambda m, noisy, seg: m(noisy, seg),
        },
        'DnCNN': {
            'model': DnCNN(4, features=64, num_layers=17),
            'trainer_fn': lambda m: DnCNNTrainer(m, device=DEVICE, lr=1e-3),
            'infer_fn': lambda m, noisy, seg: m(noisy),
        },
        'UNet-L1': {
            'model': StandardUNet(4, base_ch=32),
            'trainer_fn': lambda m: StandardUNetTrainer(m, device=DEVICE, lr=1e-4),
            'infer_fn': lambda m, noisy, seg: m(noisy),
        },
        'Noise2Noise': {
            'model': Noise2Noise(4, features=64, num_layers=17),
            'trainer_fn': lambda m: Noise2NoiseTrainer(m, device=DEVICE, lr=1e-3),
            'infer_fn': lambda m, noisy, seg: m(noisy),
        },
        'REDNet': {
            'model': REDNet(4, features=64, num_layers=6),
            'trainer_fn': lambda m: REDNetTrainer(m, device=DEVICE, lr=1e-4),
            'infer_fn': lambda m, noisy, seg: m(noisy),
        },
    }
    r1_results, r1_hist = train_and_eval(
        'Round 1 — CNN Family (Option 1 vs CNN baselines)',
        r1_cfg, train_loader, val_loader, seg_model, D_EPOCHS, DEVICE,
    )
    all_results['Round 1 — CNN'] = r1_results
    all_histories['Round 1 — CNN'] = r1_hist

# ── Round 2: ViT/MAE Family ───────────────────────────────────────────────────
if 2 in ROUNDS:
    r2_cfg = {
        'ViT PP-MAE 2D': {
            'model': ViTPPMAE2D(
                img_size=PATCH_SIZE, patch=8, embed=128, depth=4,
                n_heads=4, decoder_dim=64, decoder_depth=2,
            ),
            'trainer_fn': lambda m: ViTPPMAE2DTrainer(m, device=DEVICE, lr=1e-4),
            'infer_fn': lambda m, noisy, seg: m(noisy, seg),
        },
        'VanillaMAE': {
            'model': VanillaMAE2D(
                img_size=PATCH_SIZE, patch=8, embed=128, depth=4,
                n_heads=4, decoder_dim=64, decoder_depth=2,
            ),
            'trainer_fn': lambda m: VanillaMAETrainer(m, device=DEVICE, lr=1e-4),
            'infer_fn': lambda m, noisy, seg: m(noisy),
        },
        'SparK-CNN': {
            'model': SparKCNN(in_channels=4, base_ch=32),
            'trainer_fn': lambda m: SparKCNNTrainer(m, device=DEVICE, lr=1e-4),
            'infer_fn': lambda m, noisy, seg: m(noisy),
        },
    }
    r2_results, r2_hist = train_and_eval(
        'Round 2 — ViT/MAE Family (Option 2 vs MAE baselines)',
        r2_cfg, train_loader, val_loader, seg_model, D_EPOCHS, DEVICE,
    )
    all_results['Round 2 — ViT/MAE'] = r2_results
    all_histories['Round 2 — ViT/MAE'] = r2_hist

# ── Round 3: Multi-task Family ────────────────────────────────────────────────
if 3 in ROUNDS:
    _pipeline_model = PPMAEPipeline({'in_channels': 4, 'base_ch': 32, 'depth': 3})

    r3_cfg = {
        'PP-MAE Pipeline': {
            'model': _pipeline_model,
            'trainer_fn': lambda m: PipelineTrainer(m, device=DEVICE),
            'infer_fn': lambda m, noisy, seg: m.denoiser(noisy, seg),
        },
        'MultiTask-UNet': {
            'model': MultiTaskUNet(in_ch=4, base_ch=16, depth=3),
            'trainer_fn': lambda m: MultiTaskUNetTrainer(m, device=DEVICE, lr=1e-4),
            'infer_fn': lambda m, noisy, seg: m(noisy)['denoised'],
        },
        'TransUNet-lite': {
            'model': TransUNetLite(in_ch=4),
            'trainer_fn': lambda m: TransUNetTrainer(m, device=DEVICE, lr=1e-4),
            'infer_fn': lambda m, noisy, seg: m(noisy),
        },
    }
    r3_results, r3_hist = train_and_eval(
        'Round 3 — Multi-task Family (Option 3 vs Multi-task baselines)',
        r3_cfg, train_loader, val_loader, seg_model, D_EPOCHS, DEVICE,
    )
    all_results['Round 3 — Multi-task'] = r3_results
    all_histories['Round 3 — Multi-task'] = r3_hist

# ── Round 4: Swin Family ──────────────────────────────────────────────────────
if 4 in ROUNDS:
    r4_cfg = {
        'Swin PP-MAE': {
            'model': SwinPPMAE(
                in_ch=4, embed_dim=48, depths=(2, 2, 2, 2),
                n_heads=(3, 3, 6, 6), window_size=4,
            ),
            'trainer_fn': lambda m: SwinPPMAETrainer(m, device=DEVICE),
            'infer_fn': lambda m, noisy, seg: m(noisy, seg),
        },
        'SwinIR-lite': {
            'model': SwinIRLite(in_ch=4, dim=64, n_blocks=4, window_size=4),
            'trainer_fn': lambda m: SwinIRTrainer(m, device=DEVICE, lr=1e-4),
            'infer_fn': lambda m, noisy, seg: m(noisy),
        },
        'Uformer-lite': {
            'model': UformerLite(in_ch=4, dim=32, window_size=4),
            'trainer_fn': lambda m: UformerTrainer(m, device=DEVICE, lr=1e-4),
            'infer_fn': lambda m, noisy, seg: m(noisy),
        },
    }
    r4_results, r4_hist = train_and_eval(
        'Round 4 — Swin Family (Option 4 vs Swin baselines)',
        r4_cfg, train_loader, val_loader, seg_model, D_EPOCHS, DEVICE,
    )
    all_results['Round 4 — Swin'] = r4_results
    all_histories['Round 4 — Swin'] = r4_hist


# =============================================================================
# SUMMARY PRINTING
# =============================================================================

ROUND_LABELS = {
    'Round 1 — CNN':        ('ROUND 1', 'PP-MAE CNN (clinical_risk)'),
    'Round 2 — ViT/MAE':    ('ROUND 2', 'ViT PP-MAE 2D'),
    'Round 3 — Multi-task': ('ROUND 3', 'PP-MAE Pipeline'),
    'Round 4 — Swin':       ('ROUND 4', 'Swin PP-MAE'),
}

best_per_round: Dict[str, Tuple[str, Dict[str, float]]] = {}

print("\n", flush=True)
for round_key, results in all_results.items():
    label, ppmae_key = ROUND_LABELS.get(round_key, (round_key, None))
    print(f"\n=== {label} RESULTS ===", flush=True)
    print(f"{'Method':<32}  {'PSNR':>6}  {'SSIM':>6}  {'Dice_WT':>8}  {'Dice_TC':>8}  {'Dice_ET':>8}",
          flush=True)
    print('-' * 80, flush=True)

    best_psnr      = -1.0
    best_name      = None
    best_metrics   = None

    for name, m in results.items():
        is_best = (ppmae_key and name == ppmae_key)
        marker  = '  <-- PP-MAE' if is_best else ''
        print(
            f"  {name:<30}  {m['psnr']:6.2f}  {m['ssim']:6.3f}  "
            f"{m['dice_wt']:8.3f}  {m['dice_tc']:8.3f}  {m['dice_et']:8.3f}{marker}",
            flush=True,
        )
        if m['psnr'] > best_psnr:
            best_psnr    = m['psnr']
            best_name    = name
            best_metrics = m

    if ppmae_key and ppmae_key in results:
        best_per_round[round_key] = (ppmae_key, results[ppmae_key])
    elif best_name:
        best_per_round[round_key] = (best_name, best_metrics)

# ── All 4 PP-MAE head-to-head ─────────────────────────────────────────────────
if len(best_per_round) > 1:
    print(f"\n=== FINAL: ALL 4 OPTIONS HEAD-TO-HEAD ===", flush=True)
    print(f"{'Option':<32}  {'PSNR':>6}  {'SSIM':>6}  {'NRMSE':>7}  "
          f"{'Dice_WT':>8}  {'Dice_TC':>8}  {'Dice_ET':>8}", flush=True)
    print('-' * 88, flush=True)
    for rnd, (name, m) in best_per_round.items():
        print(
            f"  {name:<30}  {m['psnr']:6.2f}  {m['ssim']:6.3f}  {m['nrmse']:7.4f}  "
            f"{m['dice_wt']:8.3f}  {m['dice_tc']:8.3f}  {m['dice_et']:8.3f}",
            flush=True,
        )


# =============================================================================
# CSV EXPORT
# =============================================================================

csv_path = os.path.join(OUT, 'options_results.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Round', 'Method', 'PSNR', 'SSIM', 'NRMSE', 'Dice_WT', 'Dice_TC', 'Dice_ET'])
    for round_key, results in all_results.items():
        for name, m in results.items():
            w.writerow([
                round_key, name,
                f"{m['psnr']:.4f}", f"{m['ssim']:.4f}", f"{m['nrmse']:.4f}",
                f"{m['dice_wt']:.4f}", f"{m['dice_tc']:.4f}", f"{m['dice_et']:.4f}",
            ])
print(f"\nSaved {csv_path}", flush=True)


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

# Colour palettes
PPMAE_BLUE    = '#1565C0'
BASELINE_ORANGES = ['#E65100', '#F57C00', '#FFA726', '#FFCC80', '#FF8F00']
OPTION_COLORS = {
    'Round 1 — CNN':        '#1565C0',   # Option 1 — blue
    'Round 2 — ViT/MAE':    '#6A0DAD',   # Option 2 — purple
    'Round 3 — Multi-task': '#007C7C',   # Option 3 — teal
    'Round 4 — Swin':       '#C2185B',   # Option 4 — deep pink
}

PPMAE_NAME_MAP = {
    'Round 1 — CNN':        'PP-MAE CNN (clinical_risk)',
    'Round 2 — ViT/MAE':    'ViT PP-MAE 2D',
    'Round 3 — Multi-task': 'PP-MAE Pipeline',
    'Round 4 — Swin':       'Swin PP-MAE',
}


def _bar_chart_round(
    ax_psnr:   plt.Axes,
    ax_dice:   plt.Axes,
    results:   Dict[str, Dict[str, float]],
    ppmae_name: str,
    title:     str,
) -> None:
    """Draw PSNR + Dice ET grouped bars for one round."""
    names  = list(results.keys())
    psnrs  = [results[n]['psnr']    for n in names]
    dices  = [results[n]['dice_et'] for n in names]

    colors = []
    for n in names:
        if n == ppmae_name:
            colors.append(PPMAE_BLUE)
        else:
            idx = sum(1 for prev in names[:names.index(n)] if prev != ppmae_name)
            colors.append(BASELINE_ORANGES[idx % len(BASELINE_ORANGES)])

    x = np.arange(len(names))
    w = 0.6

    # PSNR
    bars1 = ax_psnr.bar(x, psnrs, width=w, color=colors, edgecolor='white', linewidth=0.8)
    ax_psnr.set_title(f'{title}\nPSNR (dB) ↑', fontsize=10)
    ax_psnr.set_xticks(x)
    ax_psnr.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
    ax_psnr.set_ylabel('PSNR (dB)')
    for bar, v in zip(bars1, psnrs):
        ax_psnr.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f'{v:.2f}', ha='center', va='bottom', fontsize=7)

    # Dice ET
    bars2 = ax_dice.bar(x, dices, width=w, color=colors, edgecolor='white', linewidth=0.8)
    ax_dice.set_title(f'{title}\nDice ET ↑', fontsize=10)
    ax_dice.set_xticks(x)
    ax_dice.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
    ax_dice.set_ylabel('Dice ET')
    ax_dice.set_ylim(0, 1.05)
    for bar, v in zip(bars2, dices):
        ax_dice.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{v:.3f}', ha='center', va='bottom', fontsize=7)


# =============================================================================
# PER-ROUND FIGURES
# =============================================================================

ROUND_FIGURE_MAP = {
    'Round 1 — CNN':        ('options_round1.png', 'Round 1 — CNN Family'),
    'Round 2 — ViT/MAE':    ('options_round2.png', 'Round 2 — ViT/MAE Family'),
    'Round 3 — Multi-task': ('options_round3.png', 'Round 3 — Multi-task Family'),
    'Round 4 — Swin':       ('options_round4.png', 'Round 4 — Swin Family'),
}

for round_key, results in all_results.items():
    fname, title_str = ROUND_FIGURE_MAP.get(
        round_key, (f'options_{round_key.replace(" ", "_")}.png', round_key))
    ppmae_name = PPMAE_NAME_MAP.get(round_key, '')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title_str, fontsize=13, fontweight='bold')
    _bar_chart_round(ax1, ax2, results, ppmae_name, title_str)

    # Legend
    patches = [mpatches.Patch(color=PPMAE_BLUE, label='PP-MAE (proposed)'),
               mpatches.Patch(color=BASELINE_ORANGES[0], label='Baselines')]
    ax1.legend(handles=patches, fontsize=8, loc='lower right')

    plt.tight_layout()
    fig_path = os.path.join(OUT, fname)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_path}", flush=True)


# =============================================================================
# HEAD-TO-HEAD FIGURE (all 4 PP-MAE options)
# =============================================================================

if len(best_per_round) >= 2:
    metrics_6 = ['psnr', 'ssim', 'nrmse', 'dice_wt', 'dice_tc', 'dice_et']
    metric_labels = ['PSNR', 'SSIM', 'NRMSE↓', 'Dice WT', 'Dice TC', 'Dice ET']

    round_keys   = list(best_per_round.keys())
    option_names = [best_per_round[k][0] for k in round_keys]
    option_mvals = {
        k: best_per_round[k][1] for k in round_keys
    }

    n_metrics = len(metrics_6)
    n_options = len(round_keys)
    x         = np.arange(n_metrics)
    bar_w     = 0.8 / max(n_options, 1)

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle('All 4 PP-MAE Options — Head-to-Head (6 Metrics)', fontsize=13,
                 fontweight='bold')

    for i, rk in enumerate(round_keys):
        name   = option_names[i]
        vals   = [option_mvals[rk][mk] for mk in metrics_6]
        color  = list(OPTION_COLORS.values())[i % len(OPTION_COLORS)]
        offset = (i - n_options / 2.0 + 0.5) * bar_w
        bars   = ax.bar(x + offset, vals, width=bar_w * 0.9, label=name,
                        color=color, edgecolor='white', linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=6, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel('Score')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    hth_path = os.path.join(OUT, 'options_headtohead.png')
    plt.savefig(hth_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {hth_path}", flush=True)


# =============================================================================
# FULL TABLE FIGURE
# =============================================================================

# Gather all rows
table_rows: List[List[str]] = []
table_header = ['Round', 'Method', 'PSNR', 'SSIM', 'NRMSE', 'Dice WT', 'Dice TC', 'Dice ET']
highlight_rows: List[int] = []  # row indices for PP-MAE methods

for round_key, results in all_results.items():
    ppmae_name = PPMAE_NAME_MAP.get(round_key, '')
    for name, m in results.items():
        row = [
            round_key.split('—')[-1].strip(),
            name,
            f"{m['psnr']:.2f}",
            f"{m['ssim']:.3f}",
            f"{m['nrmse']:.4f}",
            f"{m['dice_wt']:.3f}",
            f"{m['dice_tc']:.3f}",
            f"{m['dice_et']:.3f}",
        ]
        if name == ppmae_name:
            highlight_rows.append(len(table_rows))
        table_rows.append(row)

if table_rows:
    n_rows = len(table_rows)
    n_cols = len(table_header)

    fig_h  = max(4, 0.35 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.axis('off')

    tbl = ax.table(
        cellText=table_rows,
        colLabels=table_header,
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(col=list(range(n_cols)))

    # Style header
    for j in range(n_cols):
        tbl[0, j].set_facecolor('#1A237E')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    # Style PP-MAE rows
    for ri in highlight_rows:
        for j in range(n_cols):
            tbl[ri + 1, j].set_facecolor('#E3F2FD')
            tbl[ri + 1, j].set_text_props(fontweight='bold')

    # Alternate row colours for readability
    for ri in range(n_rows):
        if ri not in highlight_rows:
            color = '#FAFAFA' if ri % 2 == 0 else '#F0F0F0'
            for j in range(n_cols):
                tbl[ri + 1, j].set_facecolor(color)

    fig.suptitle('All Methods — All Rounds Summary Table\n(blue rows = PP-MAE)',
                 fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()
    tbl_path = os.path.join(OUT, 'options_table.png')
    plt.savefig(tbl_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {tbl_path}", flush=True)


# =============================================================================
# DONE
# =============================================================================
print(f"\nAll outputs saved to: {OUT}", flush=True)
print("Run complete.", flush=True)
