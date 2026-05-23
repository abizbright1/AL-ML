"""
run_brats_test.py
=================
End-to-end test on real BraTS data (or demo mode if no files found).

How to use with REAL BraTS data
--------------------------------
1. Download BraTS 2021 training set from:
   https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1
   (or from https://www.synapse.org/#!Synapse:syn27046444)

2. Unzip so the layout is:
   /path/to/BraTS2021_Training_Data/
     BraTS2021_00000/
       BraTS2021_00000_t1.nii.gz
       BraTS2021_00000_t1ce.nii.gz
       BraTS2021_00000_t2.nii.gz
       BraTS2021_00000_flair.nii.gz
       BraTS2021_00000_seg.nii.gz
     ...

3. Run with optional flags:
   python3 run_brats_test.py /path/to/BraTS2021_Training_Data
   python3 run_brats_test.py /path/to/BraTS2021_Training_Data --max_subjects 20
   python3 run_brats_test.py /path/to/BraTS2021_Training_Data --max_subjects 50 --epochs 30
   python3 run_brats_test.py /path/to/BraTS2021_Training_Data --max_subjects 10 --out ~/results

Demo mode (no files needed)
---------------------------
   python3 run_brats_test.py          # auto-detects no data, uses demo

All command-line flags
----------------------
  brats_root             Path to BraTS training folder (positional, optional)
  --max_subjects N       Use only the first N subjects  [default: 5]
  --epochs N             Denoising training epochs      [default: 15]
  --seg_epochs N         Segmentor training epochs      [default: 20]
  --patch_size N         Spatial crop size (px)         [default: 96]
  --sigma F              Rician noise sigma             [default: 0.08]
  --out DIR              Output directory               [default: script dir]

Output files
------------
  brats_train_loss.png     — training loss curves (real or demo data)
  brats_seg_metrics.png    — segmentation Dice WT/TC/ET bar chart
  brats_visual.png         — side-by-side denoising (all 4 modalities)
  brats_results.csv        — numeric results table
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch, sys, os, csv, argparse
import numpy as np
sys.path.insert(0, '/home/user/AL-ML/pp_mae')

from option1_cnn_pp_mae import CNNPPMAE, PPMAETrainer
from baselines import (DnCNN, DnCNNTrainer,
                       StandardUNet, StandardUNetTrainer,
                       Noise2Noise, Noise2NoiseTrainer,
                       REDNet, REDNetTrainer)
from segmentor  import UNetSegmentor, SegTrainer, seg_metrics
from evaluation import psnr, ssim_numpy, nrmse
from brats_loader import BraTSDataset, make_demo_brats

# ── Parse command-line arguments ──────────────────────────────────────────────
_parser = argparse.ArgumentParser(
    description='PP-MAE vs baselines on BraTS data',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
_parser.add_argument('brats_root',   nargs='?', default=None,
                     help='Path to BraTS training folder (omit for demo mode)')
_parser.add_argument('--max_subjects', type=int,   default=5,
                     help='Use only the first N subjects (None = all)')
_parser.add_argument('--epochs',       type=int,   default=15,
                     help='Denoising training epochs')
_parser.add_argument('--seg_epochs',   type=int,   default=20,
                     help='Segmentor training epochs')
_parser.add_argument('--patch_size',   type=int,   default=96,
                     help='Spatial crop size in pixels (must be divisible by 8)')
_parser.add_argument('--sigma',        type=float, default=0.08,
                     help='Rician noise sigma')
_parser.add_argument('--out',          type=str,   default=None,
                     help='Output directory for plots and CSV')
_args = _parser.parse_args()

# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE     = 'cpu'
OUT        = _args.out if _args.out else os.path.dirname(os.path.abspath(__file__))
D_EPOCHS   = _args.epochs
S_EPOCHS   = _args.seg_epochs
PATCH_SIZE = _args.patch_size
SIGMA      = _args.sigma
SEED       = 42
BATCH_SIZE = 4
MAX_SUBJ   = _args.max_subjects

torch.manual_seed(SEED); np.random.seed(SEED)

# ── Detect data source ────────────────────────────────────────────────────────
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
        USE_REAL = True
        DATA_LABEL = f'Real BraTS  ({MAX_SUBJ} subjects, {len(ds_full)} slices)'
        print(f"\n✅ Using REAL BraTS data from: {BRATS_ROOT}", flush=True)
        print(f"   {len(ds_full)} tumour slices loaded", flush=True)
    except Exception as e:
        print(f"⚠️  Could not load BraTS data: {e}", flush=True)
        print("   Falling back to demo mode.\n", flush=True)

if not USE_REAL:
    ds_full   = make_demo_brats(n_subjects=6, patch_size=PATCH_SIZE,
                                sigma=SIGMA, slices_per_subject=24)
    DATA_LABEL = 'Synthetic BraTS-style demo  (no real files)'
    print("\n📌 DEMO MODE — no real BraTS files found.", flush=True)
    print("   To use real data:  python3 run_brats_test.py /path/to/BraTS2021_Training_Data\n", flush=True)

# Train / val split (80/20)
n_total = len(ds_full)
n_train = int(0.8 * n_total)
n_val   = n_total - n_train
train_ds, val_ds = torch.utils.data.random_split(
    ds_full, [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED))

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

print(f"   Train: {n_train} slices  |  Val: {n_val} slices", flush=True)
print(f"   Data: {DATA_LABEL}\n", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Define and train all models
# ══════════════════════════════════════════════════════════════════════════════
C = 4  # modalities

MODELS = {
    'PP-MAE (clinical_risk)': (
        CNNPPMAE(C, base_ch=16, depth=3),
        lambda m: PPMAETrainer(m, device=DEVICE, mode='clinical_risk', lr=1e-4)),
    'PP-MAE (fixed)': (
        CNNPPMAE(C, base_ch=16, depth=3),
        lambda m: PPMAETrainer(m, device=DEVICE, mode='fixed', lr=1e-4)),
    'DnCNN': (
        DnCNN(C, features=32, num_layers=10),
        lambda m: DnCNNTrainer(m, device=DEVICE, lr=1e-3)),
    'UNet-L1': (
        StandardUNet(C, base_ch=16),
        lambda m: StandardUNetTrainer(m, device=DEVICE, lr=1e-4)),
    'Noise2Noise': (
        Noise2Noise(C, features=32, num_layers=10),
        lambda m: Noise2NoiseTrainer(m, device=DEVICE, lr=1e-3)),
    'REDNet': (
        REDNet(C, features=32, num_layers=4),
        lambda m: REDNetTrainer(m, device=DEVICE, lr=1e-4)),
}

def run_epoch(trainer, loader):
    total = 0.
    for b in loader:
        m = trainer.step(b)
        total += m.get('total', 0.)
    return total / len(loader)

histories  = {}
trained    = {}

print("=== TRAINING DENOISERS ===", flush=True)
for name, (model, make_trainer) in MODELS.items():
    trainer = make_trainer(model)
    hist = []
    print(f"\n  [{name}]", flush=True)
    for ep in range(1, D_EPOCHS + 1):
        loss = run_epoch(trainer, train_loader)
        hist.append(loss)
        if ep % 5 == 0 or ep == 1:
            print(f"    Ep {ep:2d}/{D_EPOCHS}  loss={loss:.4f}", flush=True)
    histories[name] = hist
    trained[name]   = model

# ══════════════════════════════════════════════════════════════════════════════
# 2. Train segmentor on CLEAN images
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== TRAINING SEGMENTOR (on clean BraTS images) ===", flush=True)
seg_model   = UNetSegmentor(in_channels=C, n_classes=4, base_ch=32)
seg_trainer = SegTrainer(seg_model, device=DEVICE, lr=5e-4)
seg_hist    = []

for ep in range(1, S_EPOCHS + 1):
    ep_loss = 0.
    for b in train_loader:
        clean  = b['target']
        labels = b['seg'][:, 0].long()
        ep_loss += seg_trainer.step(clean, labels)
    ep_loss /= len(train_loader)
    seg_hist.append(ep_loss)
    if ep % 5 == 0 or ep == 1:
        print(f"  [segmentor] Ep {ep:2d}/{S_EPOCHS}  loss={ep_loss:.4f}", flush=True)

print("Segmentor trained.\n", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3. Evaluate on validation set
# ══════════════════════════════════════════════════════════════════════════════
print("=== EVALUATING ON VALIDATION SET ===", flush=True)

def eval_denoiser(model, val_loader, seg_model, device='cpu'):
    model.eval(); seg_model.eval()
    ps_, ss_, nr_, dw_, dt_, de_ = [], [], [], [], [], []
    with torch.no_grad():
        for b in val_loader:
            x   = b['noisy'].to(device)
            tgt = b['target']
            seg = b['seg']
            try:    pred = model(x, seg.to(device)).cpu()
            except: pred = model(x).cpu()
            # Image quality per sample in batch
            for i in range(pred.shape[0]):
                p_np = pred[i].permute(1,2,0).numpy()
                t_np = tgt[i].permute(1,2,0).numpy()
                ps_.append(psnr(p_np, t_np))
                ss_.append(ssim_numpy(p_np, t_np))
                nr_.append(nrmse(p_np, t_np))
            # Segmentation on denoised output
            logits = seg_model(pred.to(device))
            gt_lab = seg[:, 0].long()
            m      = seg_metrics(logits.cpu(), gt_lab)
            dw_.append(m['dice_wt']); dt_.append(m['dice_tc']); de_.append(m['dice_et'])
    return {
        'psnr': float(np.mean(ps_)), 'ssim': float(np.mean(ss_)),
        'nrmse': float(np.mean(nr_)),
        'dice_wt': float(np.mean(dw_)), 'dice_tc': float(np.mean(dt_)),
        'dice_et': float(np.mean(de_)),
    }

def eval_noisy_baseline(val_loader, seg_model, device='cpu'):
    seg_model.eval()
    ps_, ss_, nr_, dw_, dt_, de_ = [], [], [], [], [], []
    with torch.no_grad():
        for b in val_loader:
            noisy = b['noisy']; tgt = b['target']; seg = b['seg']
            for i in range(noisy.shape[0]):
                p_np = noisy[i].permute(1,2,0).numpy()
                t_np = tgt[i].permute(1,2,0).numpy()
                ps_.append(psnr(p_np, t_np))
                ss_.append(ssim_numpy(p_np, t_np))
                nr_.append(nrmse(p_np, t_np))
            logits = seg_model(noisy.to(device))
            gt_lab = seg[:, 0].long()
            m      = seg_metrics(logits.cpu(), gt_lab)
            dw_.append(m['dice_wt']); dt_.append(m['dice_tc']); de_.append(m['dice_et'])
    return {
        'psnr': float(np.mean(ps_)), 'ssim': float(np.mean(ss_)),
        'nrmse': float(np.mean(nr_)),
        'dice_wt': float(np.mean(dw_)), 'dice_tc': float(np.mean(dt_)),
        'dice_et': float(np.mean(de_)),
    }

results = {'No Denoising': eval_noisy_baseline(val_loader, seg_model)}
r = results['No Denoising']
print(f"  {'No Denoising':<28}  PSNR={r['psnr']:.2f}  SSIM={r['ssim']:.3f}  "
      f"Dice ET={r['dice_et']:.3f}", flush=True)

for name, model in trained.items():
    r = eval_denoiser(model, val_loader, seg_model)
    results[name] = r
    print(f"  {name:<28}  PSNR={r['psnr']:.2f}  SSIM={r['ssim']:.3f}  "
          f"Dice ET={r['dice_et']:.3f}", flush=True)

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = f'{OUT}/brats_results.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Method','PSNR','SSIM','NRMSE','Dice_WT','Dice_TC','Dice_ET'])
    for name, r in results.items():
        w.writerow([name,
                    f'{r["psnr"]:.4f}', f'{r["ssim"]:.4f}', f'{r["nrmse"]:.4f}',
                    f'{r["dice_wt"]:.4f}', f'{r["dice_tc"]:.4f}', f'{r["dice_et"]:.4f}'])
print(f'Saved {csv_path}', flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 4. Plots
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== GENERATING PLOTS ===", flush=True)

ALL_NAMES  = list(results.keys())
pp_names   = [n for n in ALL_NAMES if 'PP-MAE' in n]
bl_names   = [n for n in ALL_NAMES if 'PP-MAE' not in n and n != 'No Denoising']

ppmae_col  = '#1565C0'
base_col   = '#E65100'
nd_col     = '#9E9E9E'
col_list   = []
for n in ALL_NAMES:
    if n == 'No Denoising': col_list.append(nd_col)
    elif 'PP-MAE' in n:     col_list.append(ppmae_col if 'clinical' in n else '#42A5F5')
    elif n == 'DnCNN':      col_list.append('#E65100')
    elif n == 'UNet-L1':    col_list.append('#F57C00')
    elif n == 'Noise2Noise':col_list.append('#EF6C00')
    else:                   col_list.append('#FF8F00')

# ── Figure 1: Global metrics ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (key, label, better) in zip(axes, [
        ('psnr',  'PSNR (dB)',  '↑ higher'), ('ssim', 'SSIM', '↑ higher'),
        ('nrmse', 'NRMSE',      '↓ lower')]):
    vals = [results[n][key] for n in ALL_NAMES]
    bars = ax.bar(range(len(ALL_NAMES)), vals, color=col_list, edgecolor='white', lw=1.2)
    ax.set_xticks(range(len(ALL_NAMES)))
    ax.set_xticklabels(ALL_NAMES, rotation=30, ha='right', fontsize=9)
    ax.set_title(f'{label}  {better}', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.001,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
fig.suptitle(f'PP-MAE vs. Baselines — Global Denoising Metrics\n({DATA_LABEL})',
             fontweight='bold', fontsize=13, y=1.02)
pp_p = mpatches.Patch(color=ppmae_col, label='PP-MAE (ours)')
bl_p = mpatches.Patch(color=base_col,  label='Baselines')
nd_p = mpatches.Patch(color=nd_col,    label='No Denoising')
fig.legend(handles=[pp_p, bl_p, nd_p], loc='lower center', ncol=3,
           fontsize=10, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
plt.savefig(f'{OUT}/brats_train_loss.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved brats_train_loss.png', flush=True)

# ── Figure 2: Tumour-region Dice ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (key, label) in zip(axes, [
        ('dice_wt', 'Dice — Whole Tumour'),
        ('dice_tc', 'Dice — Tumour Core'),
        ('dice_et', 'Dice — Enhancing Tumour')]):
    vals = [results[n][key] for n in ALL_NAMES]
    bars = ax.bar(range(len(ALL_NAMES)), vals, color=col_list, edgecolor='white', lw=1.2)
    ax.set_xticks(range(len(ALL_NAMES))); ax.set_ylim(0, 1.1)
    ax.set_xticklabels(ALL_NAMES, rotation=30, ha='right', fontsize=9)
    ax.set_title(label, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
fig.suptitle(f'Downstream Segmentation — Tumour-Region Dice\n({DATA_LABEL})',
             fontweight='bold', fontsize=13, y=1.02)
fig.legend(handles=[pp_p, bl_p, nd_p], loc='lower center', ncol=3,
           fontsize=10, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
plt.savefig(f'{OUT}/brats_seg_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved brats_seg_metrics.png', flush=True)

# ── Figure 3: Training loss curves ────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))
pp_colors = ['#1565C0', '#42A5F5']
bl_colors = ['#E65100', '#F57C00', '#EF6C00', '#FF8F00']
for (name, hist), col in zip(
        [(n,h) for n,h in histories.items() if 'PP-MAE' in n], pp_colors):
    ax1.plot(range(1, len(hist)+1), hist, label=name, color=col, lw=2)
ax1.set_title('PP-MAE Training Loss', fontweight='bold')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(alpha=0.3)
for (name, hist), col in zip(
        [(n,h) for n,h in histories.items() if 'PP-MAE' not in n], bl_colors):
    ax2.plot(range(1, len(hist)+1), hist, label=name, color=col, lw=2)
ax2.set_title('Baseline Training Loss', fontweight='bold')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.legend(); ax2.grid(alpha=0.3)
fig.suptitle(f'Convergence — {DATA_LABEL}', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/brats_loss_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved brats_loss_curves.png', flush=True)

# ── Figure 4: Visual comparison on one val sample ─────────────────────────────
sample = val_ds[0]
noisy_t = sample['noisy']; clean_t = sample['target']; seg_t = sample['seg']
seg_np  = seg_t[0].numpy()
seg_con = (seg_np > 0).astype(float)

def get_denoised(model, sample):
    model.eval()
    with torch.no_grad():
        x = sample['noisy'].unsqueeze(0)
        s = sample['seg'].unsqueeze(0)
        try:    return model(x, s)[0]
        except: return model(x)[0]

show = ['PP-MAE (clinical_risk)', 'PP-MAE (fixed)', 'DnCNN', 'Noise2Noise', 'REDNet']
preds = {n: get_denoised(trained[n], sample) for n in show}

# 4 rows (one per modality) × (2 + len(show)) cols
mod_labels = ['T1W', 'T1CE', 'T2W', 'FLAIR']
n_cols = 2 + len(show)
fig, axes = plt.subplots(4, n_cols, figsize=(n_cols*3, 14))

for row, (mod_idx, mod_name) in enumerate(zip(range(4), mod_labels)):
    # col 0: noisy
    axes[row, 0].imshow(noisy_t[mod_idx].numpy(), cmap='gray', vmin=0, vmax=1)
    axes[row, 0].contour(seg_con, levels=[.5], colors='lime', linewidths=.8)
    axes[row, 0].set_title(f'Noisy\n({mod_name})' if row == 0 else '', fontsize=9, fontweight='bold')
    axes[row, 0].set_ylabel(mod_name, fontsize=10, fontweight='bold')
    axes[row, 0].axis('off')
    # col 1: ground truth
    axes[row, 1].imshow(clean_t[mod_idx].numpy(), cmap='gray', vmin=0, vmax=1)
    axes[row, 1].contour(seg_con, levels=[.5], colors='lime', linewidths=.8)
    if row == 0: axes[row, 1].set_title('Ground Truth', fontsize=9, fontweight='bold')
    axes[row, 1].axis('off')
    # denoised outputs
    for col, name in enumerate(show, start=2):
        img = preds[name][mod_idx].numpy()
        axes[row, col].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[row, col].contour(seg_con, levels=[.5], colors='lime', linewidths=.8)
        if row == 0:
            axes[row, col].set_title(name.replace(' (', '\n('), fontsize=8, fontweight='bold')
        axes[row, col].axis('off')

fig.suptitle(f'Visual Denoising — All 4 Modalities  (lime = tumour boundary)\n{DATA_LABEL}',
             fontweight='bold', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/brats_visual.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved brats_visual.png', flush=True)

# ── Figure 5: Summary table ────────────────────────────────────────────────────
col_labels = ['Method', 'PSNR↑', 'SSIM↑', 'NRMSE↓', 'Dice WT↑', 'Dice TC↑', 'Dice ET↑']
cell_data  = []
for name in ALL_NAMES:
    r = results[name]
    cell_data.append([name,
                      f'{r["psnr"]:.2f}', f'{r["ssim"]:.3f}', f'{r["nrmse"]:.4f}',
                      f'{r["dice_wt"]:.3f}', f'{r["dice_tc"]:.3f}', f'{r["dice_et"]:.3f}'])

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.axis('off')
tbl = ax.table(cellText=cell_data, colLabels=col_labels, cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.2)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#1A237E')
        cell.set_text_props(color='white', fontweight='bold')
    elif any(n in cell_data[row-1][0] for n in ['PP-MAE']):
        cell.set_facecolor('#BBDEFB' if 'clinical' in cell_data[row-1][0] else '#E3F2FD')
    elif row == 1:
        cell.set_facecolor('#F5F5F5')
    else:
        cell.set_facecolor('#FFF3E0' if row % 2 else '#FFFFFF')
    cell.set_edgecolor('#BDBDBD')
mode_label = "REAL BraTS Data" if USE_REAL else "Synthetic BraTS-style Demo"
fig.suptitle(f'PP-MAE vs. Baselines — {mode_label}\n(Blue = PP-MAE  |  Orange = baselines)',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/brats_table.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved brats_table.png', flush=True)

# ── Final summary ─────────────────────────────────────────────────────────────
print('\n' + '='*68, flush=True)
print(f'BRATS TEST COMPLETE  —  {mode_label}', flush=True)
print('='*68, flush=True)
print(f'\n{"Method":<28}  {"PSNR":>6}  {"SSIM":>6}  {"Dice_WT":>8}  {"Dice_ET":>8}')
print('-'*65)
for name in ALL_NAMES:
    r = results[name]
    mark = ' ◀' if name == max(
        [n for n in ALL_NAMES if n != 'No Denoising'],
        key=lambda n: results[n]['dice_et']) else ''
    print(f'{name:<28}  {r["psnr"]:>6.2f}  {r["ssim"]:>6.3f}  '
          f'{r["dice_wt"]:>8.3f}  {r["dice_et"]:>8.3f}{mark}')

print(f'\n📁 Output files in {OUT}/')
for fn in ['brats_train_loss.png','brats_seg_metrics.png','brats_loss_curves.png',
           'brats_visual.png','brats_table.png','brats_results.csv']:
    print(f'   {fn}', flush=True)

if not USE_REAL:
    print('\n' + '─'*68, flush=True)
    print('📌 This was a DEMO RUN on synthetic data.', flush=True)
    print('   To run on real BraTS 2021 data:', flush=True)
    print('   1. Download from: https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1', flush=True)
    print('   2. python3 run_brats_test.py /path/to/BraTS2021_Training_Data', flush=True)
    print('─'*68, flush=True)
