"""
run_segmentation.py
===================
Downstream segmentation benchmark:
  How well do denoised MRI images support automatic tumour segmentation?

Pipeline
--------
  1. Build synthetic glioma dataset (same seed as run_comparison.py)
  2. Train all denoising models (PP-MAE × 4 modes  +  4 baselines)
  3. Train a U-Net segmentor on CLEAN images  (ground-truth upper bound)
  4. Apply segmentor to:
       • noisy input (lower bound — no denoising)
       • each denoised output (8 models)
       • clean image  (oracle upper bound)
  5. Report Dice_WT / Dice_TC / Dice_ET / IoU / PixelAcc for all
  6. Generate 4 publication-quality figures:
       seg_metrics.png      — grouped bar chart (Dice + IoU)
       seg_curves.png       — segmentor training curve
       seg_visual.png       — visual: seg masks from each method
       seg_table.png        — styled summary table
       seg_radar.png        — radar / spider chart

Output: /home/user/AL-ML/seg_*.png, seg_results.csv
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import torch, sys, csv
sys.path.insert(0, '/home/user/AL-ML/pp_mae')

from option1_cnn_pp_mae import CNNPPMAE, PPMAETrainer
from baselines import (DnCNN, DnCNNTrainer, StandardUNet, StandardUNetTrainer,
                       Noise2Noise, Noise2NoiseTrainer, REDNet, REDNetTrainer)
from segmentor import UNetSegmentor, SegTrainer, seg_metrics

DEVICE   = 'cpu'
OUT      = '/home/user/AL-ML'
D_EPOCHS = 20      # denoising epochs
S_EPOCHS = 30      # segmentation epochs
SEED     = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Dataset
# ══════════════════════════════════════════════════════════════════════════════
class SynGliomaMRI(torch.utils.data.Dataset):
    def __init__(self, n=64, sigma=0.08, size=64, seed=0):
        rng = np.random.default_rng(seed)
        imgs, segs = [], []
        for _ in range(n):
            base = rng.random((4, size, size)).astype('float32') * 0.4 + 0.1
            cy, cx = rng.integers(16, size-16), rng.integers(16, size-16)
            Y, X   = np.ogrid[:size, :size]
            r_wt   = rng.integers(10, 16); r_tc = rng.integers(5, 10); r_et = rng.integers(2, 5)
            wt = ((Y-cy)**2+(X-cx)**2) < r_wt**2
            tc = ((Y-cy)**2+(X-cx)**2) < r_tc**2
            et = ((Y-cy)**2+(X-cx)**2) < r_et**2
            base[0][wt] = rng.uniform(0.6, 0.9); base[1][tc] = rng.uniform(0.7, 1.0)
            base[2][wt] = rng.uniform(0.5, 0.85); base[3][wt] = rng.uniform(0.65, 0.95)
            base = base.clip(0., 1.)
            seg = np.zeros((1, size, size), dtype='int64')
            seg[0][wt] = 2; seg[0][tc] = 1; seg[0][et] = 3
            imgs.append(base); segs.append(seg)
        clean = torch.from_numpy(np.stack(imgs))
        noisy = (clean + sigma * torch.randn_like(clean)).clamp(0., 1.)
        self.x = noisy; self.t = clean
        self.s = torch.from_numpy(np.stack(segs))   # (N,1,H,W)

    def __len__(self): return len(self.t)
    def __getitem__(self, i):
        return {'noisy': self.x[i], 'target': self.t[i], 'seg': self.s[i]}


print("Building dataset …", flush=True)
ds     = SynGliomaMRI(n=64, sigma=0.08, seed=SEED)
loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)

# ══════════════════════════════════════════════════════════════════════════════
# 2. Train all denoising models
# ══════════════════════════════════════════════════════════════════════════════
def train_denoiser(trainer, loader, epochs, label):
    for ep in range(1, epochs+1):
        for b in loader: trainer.step(b)
        if ep % 10 == 0 or ep == 1:
            acc = {}
            for b in loader:
                m = trainer.step(b)
                for k,v in m.items(): acc[k] = acc.get(k,0)+v
            total = acc.get('total', 0) / len(loader)
            print(f'  [denoiser {label:>22s}] Ep {ep:2d}/{epochs} loss={total:.4f}', flush=True)

C = 4
PPmae_MODES    = ['PP-MAE (fixed)', 'PP-MAE (adaptive)', 'PP-MAE (clinical_risk)', 'PP-MAE (combined)']
BASELINE_NAMES = ['DnCNN', 'UNet-L1', 'Noise2Noise', 'REDNet']

print("\n=== TRAINING DENOISERS ===", flush=True)
denoisers = {}

for mode in ['fixed', 'adaptive', 'clinical_risk', 'combined']:
    m = CNNPPMAE(in_channels=C, base_ch=16, depth=3)
    t = PPMAETrainer(m, device=DEVICE, mode=mode, lr=1e-4)
    print(f'\n  PP-MAE ({mode}) …', flush=True)
    train_denoiser(t, loader, D_EPOCHS, f'PP-MAE ({mode})')
    denoisers[f'PP-MAE ({mode})'] = m

m_dn = DnCNN(C, features=32, num_layers=10)
m_un = StandardUNet(C, base_ch=16)
m_n2 = Noise2Noise(C, features=32, num_layers=10)
m_rd = REDNet(C, features=32, num_layers=4)
for name, model, trainer_cls in [
        ('DnCNN',       m_dn, DnCNNTrainer),
        ('UNet-L1',     m_un, StandardUNetTrainer),
        ('Noise2Noise', m_n2, Noise2NoiseTrainer),
        ('REDNet',      m_rd, REDNetTrainer)]:
    t = trainer_cls(model, device=DEVICE)
    print(f'\n  {name} …', flush=True)
    train_denoiser(t, loader, D_EPOCHS, name)
    denoisers[name] = model

# ══════════════════════════════════════════════════════════════════════════════
# 3. Train U-Net Segmentor on CLEAN images
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== TRAINING SEGMENTOR (on clean images) ===", flush=True)
seg_model   = UNetSegmentor(in_channels=C, n_classes=4, base_ch=32)
seg_trainer = SegTrainer(seg_model, device=DEVICE, lr=1e-3)
seg_hist    = []

for ep in range(1, S_EPOCHS + 1):
    ep_loss = 0.
    for b in loader:
        clean  = b['target']
        labels = b['seg'][:, 0].long()   # (B, H, W)
        ep_loss += seg_trainer.step(clean, labels)
    ep_loss /= len(loader)
    seg_hist.append(ep_loss)
    if ep % 5 == 0 or ep == 1:
        print(f'  [segmentor] Ep {ep:2d}/{S_EPOCHS}  loss={ep_loss:.4f}', flush=True)

print("Segmentor trained.", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 4. Evaluate segmentation quality on each denoised output
# ══════════════════════════════════════════════════════════════════════════════
def denoise_dataset(denoiser, ds):
    """Run denoiser on entire dataset, return denoised tensor (N,C,H,W)."""
    denoiser.eval()
    outs = []
    with torch.no_grad():
        for i in range(len(ds)):
            b = ds[i]
            x   = b['noisy'].unsqueeze(0)
            seg = b['seg'].unsqueeze(0)
            try:    pred = denoiser(x, seg)
            except: pred = denoiser(x)
            outs.append(pred[0])
    return torch.stack(outs)   # (N,C,H,W)


def eval_segmentation(images, ds, seg_model, device='cpu', n=None):
    """
    images : (N, C, H, W) tensor
    Returns averaged seg_metrics dict.
    """
    seg_model.eval()
    all_metrics = []
    N = len(ds) if n is None else min(n, len(ds))
    with torch.no_grad():
        for i in range(N):
            img  = images[i].unsqueeze(0).to(device)
            gt   = ds[i]['seg'][0].unsqueeze(0).long()  # (1,H,W)
            logits = seg_model(img)                      # (1,4,H,W)
            m = seg_metrics(logits.cpu(), gt)
            all_metrics.append(m)
    avg = {k: float(np.mean([x[k] for x in all_metrics])) for k in all_metrics[0]}
    return avg


print("\n=== EVALUATING SEGMENTATION ===", flush=True)

# Build image sets
clean_imgs = torch.stack([ds[i]['target'] for i in range(len(ds))])
noisy_imgs = torch.stack([ds[i]['noisy']  for i in range(len(ds))])

# Evaluate oracle (clean) and noisy baseline first
results = {}
results['Oracle (clean)'] = eval_segmentation(clean_imgs, ds, seg_model)
results['Noisy (no denoise)'] = eval_segmentation(noisy_imgs, ds, seg_model)

print(f'  Oracle (clean):        '
      f'Dice WT={results["Oracle (clean)"]["dice_wt"]:.3f} '
      f'TC={results["Oracle (clean)"]["dice_tc"]:.3f} '
      f'ET={results["Oracle (clean)"]["dice_et"]:.3f}', flush=True)
print(f'  Noisy (no denoise):    '
      f'Dice WT={results["Noisy (no denoise)"]["dice_wt"]:.3f} '
      f'TC={results["Noisy (no denoise)"]["dice_tc"]:.3f} '
      f'ET={results["Noisy (no denoise)"]["dice_et"]:.3f}', flush=True)

# Evaluate each denoised output
for name, model in denoisers.items():
    print(f'  Evaluating {name} …', flush=True)
    denoised = denoise_dataset(model, ds)
    m = eval_segmentation(denoised, ds, seg_model)
    results[name] = m
    print(f'    → Dice WT={m["dice_wt"]:.3f} TC={m["dice_tc"]:.3f} ET={m["dice_et"]:.3f} '
          f'Acc={m["pixel_acc"]:.3f} mIoU={m["mean_iou"]:.3f}', flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 5. Save CSV
# ══════════════════════════════════════════════════════════════════════════════
csv_path = f'{OUT}/seg_results.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    first = list(results.values())[0]
    w.writerow(['Method'] + list(first.keys()))
    for name, m in results.items():
        w.writerow([name] + [f'{v:.4f}' for v in m.values()])
print(f'Saved {csv_path}', flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 6. PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== GENERATING PLOTS ===", flush=True)

# ── Colour scheme ─────────────────────────────────────────────────────────────
oracle_color = '#4CAF50'      # green
noisy_color  = '#9E9E9E'      # grey
ppmae_colors = ['#1565C0', '#1E88E5', '#42A5F5', '#90CAF9']
base_colors  = ['#E65100', '#F57C00', '#EF6C00', '#FF8F00']

ALL_NAMES = (['Oracle (clean)', 'Noisy (no denoise)']
             + PPmae_MODES + BASELINE_NAMES)

color_map = {
    'Oracle (clean)':     oracle_color,
    'Noisy (no denoise)': noisy_color,
}
color_map.update(dict(zip(PPmae_MODES,     ppmae_colors)))
color_map.update(dict(zip(BASELINE_NAMES,  base_colors)))
get_c = lambda names: [color_map[n] for n in names]

# Legend patches
or_patch = mpatches.Patch(color=oracle_color, label='Oracle (clean input)')
nd_patch = mpatches.Patch(color=noisy_color,  label='No Denoising')
pp_patch = mpatches.Patch(color='#1E88E5',    label='PP-MAE (ours)')
bl_patch = mpatches.Patch(color='#F57C00',    label='Baselines')

# ── Figure 1: Main segmentation metrics bar chart ────────────────────────────
metrics_to_plot = [
    ('dice_wt',   'Dice — Whole Tumour',      '↑ higher is better'),
    ('dice_tc',   'Dice — Tumour Core',        '↑ higher is better'),
    ('dice_et',   'Dice — Enhancing Tumour',   '↑ higher is better'),
    ('iou_wt',    'IoU — Whole Tumour',        '↑ higher is better'),
    ('mean_iou',  'Mean IoU (all classes)',    '↑ higher is better'),
    ('pixel_acc', 'Pixel Accuracy',            '↑ higher is better'),
]

fig, axes = plt.subplots(2, 3, figsize=(21, 11))
for ax, (key, title, better) in zip(axes.flat, metrics_to_plot):
    vals = [results[n][key] for n in ALL_NAMES]
    cols = get_c(ALL_NAMES)
    bars = ax.bar(range(len(ALL_NAMES)), vals, color=cols, edgecolor='white', linewidth=1.1)
    ax.set_xticks(range(len(ALL_NAMES)))
    ax.set_xticklabels(ALL_NAMES, rotation=35, ha='right', fontsize=8)
    ax.set_title(f'{title}\n{better}', fontweight='bold', fontsize=10)
    ax.set_ylim(0, 1.12); ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
fig.legend(handles=[or_patch, nd_patch, pp_patch, bl_patch],
           loc='lower center', ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.01))
fig.suptitle('Downstream Segmentation Quality — After Denoising\n'
             'U-Net Segmentor applied to each model\'s output  (higher = better segmentation)',
             fontweight='bold', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/seg_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved seg_metrics.png', flush=True)

# ── Figure 2: Segmentor training curve ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(range(1, S_EPOCHS+1), seg_hist, color='#1565C0', lw=2.5)
ax.fill_between(range(1, S_EPOCHS+1), seg_hist, alpha=0.15, color='#1565C0')
ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-Entropy Loss')
ax.set_title('Segmentor Training Curve (trained on clean images)', fontweight='bold')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/seg_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved seg_curves.png', flush=True)

# ── Figure 3: Visual segmentation comparison ─────────────────────────────────
idx    = 7
b      = ds[idx]
clean  = b['target']
noisy  = b['noisy']
gt_seg = b['seg'][0].numpy()   # (H,W)

# colour map: 0=BG(black), 1=NCR(red), 2=ED(yellow), 3=ET(blue)
SEG_CMAP = mcolors.ListedColormap(['#000000', '#E53935', '#FFB300', '#1E88E5'])
SEG_NORM = mcolors.BoundaryNorm([0, 1, 2, 3, 4], SEG_CMAP.N)

def predict_seg(image_tensor, seg_model):
    seg_model.eval()
    with torch.no_grad():
        logits = seg_model(image_tensor.unsqueeze(0))
        return logits.argmax(1)[0].numpy()   # (H,W)

# Collect predictions
show_models = {
    'Oracle\n(clean)':         clean,
    'Noisy input\n(no denoise)': noisy,
    'PP-MAE\n(clinical_risk)':  None,
    'PP-MAE\n(fixed)':          None,
    'DnCNN':                    None,
    'UNet-L1':                  None,
    'Noise2Noise':              None,
    'REDNet':                   None,
}
# fill denoiser outputs
for key in ['PP-MAE (clinical_risk)', 'PP-MAE (fixed)', 'DnCNN', 'UNet-L1', 'Noise2Noise', 'REDNet']:
    model = denoisers[key]
    model.eval()
    with torch.no_grad():
        x = noisy.unsqueeze(0); sg = b['seg'].unsqueeze(0)
        try:    d = model(x, sg)[0]
        except: d = model(x)[0]
    label = key.replace(' (', '\n(')
    show_models[label if '(' in key else key] = d

# Separate key into short label for plotting
panels = []
for label, img_t in show_models.items():
    if img_t is None: continue
    pred_s = predict_seg(img_t, seg_model)
    panels.append((label, img_t[1].numpy(), pred_s))   # T1Wce channel

n_panels = len(panels)
fig, axes = plt.subplots(2, n_panels, figsize=(n_panels*2.8, 6))

for col, (label, mri_ch, pred_seg) in enumerate(panels):
    # Row 0: MRI image
    axes[0, col].imshow(mri_ch, cmap='gray', vmin=0, vmax=1)
    axes[0, col].set_title(label, fontsize=8.5, fontweight='bold')
    axes[0, col].axis('off')
    # Row 1: predicted seg map
    axes[1, col].imshow(pred_seg, cmap=SEG_CMAP, norm=SEG_NORM, interpolation='nearest')
    axes[1, col].axis('off')
    # Mark Dice_ET on bottom row
    m = results.get(label.replace('\n', ' ').replace('(', '(').strip(), {})
    det = m.get('dice_et', None)
    if det is not None:
        axes[1, col].set_xlabel(f'Dice ET={det:.3f}', fontsize=8)

# GT seg overlay on last panel area (axis after the models)
axes[0, 0].set_title(panels[0][0]+'\n(GT contour)', fontsize=8.5, fontweight='bold')
# Add GT contour to both rows
for col in range(n_panels):
    for row in [0, 1]:
        axes[row, col].contour((gt_seg > 0).astype(float), levels=[0.5],
                                colors='lime', linewidths=0.8)

# Colour legend
labels_legend = [mpatches.Patch(color='#000000', label='BG'),
                 mpatches.Patch(color='#E53935', label='NCR'),
                 mpatches.Patch(color='#FFB300', label='Edema'),
                 mpatches.Patch(color='#1E88E5', label='ET')]
fig.legend(handles=labels_legend, loc='lower center', ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle('Segmentation Maps — T1Wce channel  (lime = GT tumour boundary)\n'
             'Row 1: MRI  │  Row 2: predicted labels  [black=BG, red=NCR, yellow=ED, blue=ET]',
             fontweight='bold', fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/seg_visual.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved seg_visual.png', flush=True)

# ── Figure 4: Styled Summary Table ────────────────────────────────────────────
col_labels = ['Method', 'Dice WT↑', 'Dice TC↑', 'Dice ET↑',
              'IoU WT↑', 'mIoU↑',  'Pixel Acc↑']
cell_data  = []
for name in ALL_NAMES:
    m = results[name]
    cell_data.append([name,
                      f'{m["dice_wt"]:.3f}',
                      f'{m["dice_tc"]:.3f}',
                      f'{m["dice_et"]:.3f}',
                      f'{m["iou_wt"]:.3f}',
                      f'{m["mean_iou"]:.3f}',
                      f'{m["pixel_acc"]:.3f}'])

fig, ax = plt.subplots(figsize=(15, 5))
ax.axis('off')
tbl = ax.table(cellText=cell_data, colLabels=col_labels, cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 2.1)

pp_row_set = set(range(3, 7))    # rows 3-6 are PP-MAE
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#1A237E')
        cell.set_text_props(color='white', fontweight='bold')
    elif row == 1:   # Oracle
        cell.set_facecolor('#E8F5E9')
    elif row == 2:   # Noisy
        cell.set_facecolor('#F5F5F5')
    elif row in pp_row_set:
        cell.set_facecolor('#E3F2FD')
        if row == 5:   # clinical_risk
            cell.set_facecolor('#BBDEFB')
    else:
        cell.set_facecolor('#FFF3E0' if row % 2 else '#FFFFFF')
    cell.set_edgecolor('#BDBDBD')

fig.suptitle('Downstream Segmentation — Summary Table\n'
             '(Green = oracle  |  Blue = PP-MAE ours  |  Orange = baselines)',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/seg_table.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved seg_table.png', flush=True)

# ── Figure 5: Radar Chart ─────────────────────────────────────────────────────
radar_models = ['Oracle (clean)', 'Noisy (no denoise)',
                'PP-MAE (clinical_risk)', 'PP-MAE (fixed)',
                'DnCNN', 'UNet-L1', 'Noise2Noise', 'REDNet']
radar_keys   = ['dice_wt', 'dice_tc', 'dice_et', 'iou_wt', 'mean_iou', 'pixel_acc']
radar_labels = ['Dice WT', 'Dice TC', 'Dice ET', 'IoU WT', 'mIoU', 'Pixel Acc']

N_axes = len(radar_keys)
angles = np.linspace(0, 2*np.pi, N_axes, endpoint=False).tolist()
angles += angles[:1]  # close

radar_colors = [oracle_color, noisy_color,
                '#1565C0', '#42A5F5',
                '#E65100', '#F57C00', '#EF6C00', '#FF8F00']
radar_styles = ['-', '--', '-', '--', '-', '--', ':', '-.']

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'], fontsize=8)
ax.grid(color='grey', alpha=0.3)

for name, col, ls in zip(radar_models, radar_colors, radar_styles):
    vals = [results[name][k] for k in radar_keys]
    vals += vals[:1]
    ax.plot(angles, vals, color=col, linewidth=2, linestyle=ls, label=name)
    ax.fill(angles, vals, color=col, alpha=0.05)

ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15), fontsize=9)
ax.set_title('Segmentation Quality — Radar Chart\n'
             '(Larger area = better segmentation after denoising)',
             fontweight='bold', fontsize=12, pad=20)
plt.tight_layout()
plt.savefig(f'{OUT}/seg_radar.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved seg_radar.png', flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 7. Final summary
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*72, flush=True)
print('SEGMENTATION COMPARISON — FINAL RESULTS', flush=True)
print('='*72, flush=True)
hdr = f'{"Method":<30}  {"Dice WT":>8}  {"Dice TC":>8}  {"Dice ET":>8}  {"mIoU":>8}  {"Acc":>6}'
print(hdr); print('-'*72)
for name in ALL_NAMES:
    m = results[name]
    mark = ' ◀ best denoised' if name == max(
        [n for n in ALL_NAMES if n not in ('Oracle (clean)', 'Noisy (no denoise)')],
        key=lambda n: results[n]['dice_et']) else ''
    print(f'{name:<30}  {m["dice_wt"]:>8.3f}  {m["dice_tc"]:>8.3f}  '
          f'{m["dice_et"]:>8.3f}  {m["mean_iou"]:>8.3f}  {m["pixel_acc"]:>6.3f}{mark}')

# Find best PP-MAE vs best baseline
best_pp  = max(PPmae_MODES,    key=lambda n: results[n]['dice_et'])
best_bl  = max(BASELINE_NAMES, key=lambda n: results[n]['dice_et'])
print(f'\nBest PP-MAE variant:  {best_pp}  Dice_ET={results[best_pp]["dice_et"]:.3f}')
print(f'Best Baseline:        {best_bl}  Dice_ET={results[best_bl]["dice_et"]:.3f}')
delta = results[best_pp]['dice_et'] - results[best_bl]['dice_et']
print(f'ΔDice_ET (PP-MAE vs best baseline): {delta:+.3f}')
print(f'\nOutput files saved to {OUT}/')
for f in ['seg_metrics.png','seg_curves.png','seg_visual.png','seg_table.png','seg_radar.png','seg_results.csv']:
    print(f'  {f}', flush=True)
