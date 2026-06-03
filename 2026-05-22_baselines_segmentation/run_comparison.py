"""
run_comparison.py
=================
Side-by-side benchmark: PP-MAE (4 modes) vs. 4 standard baselines
on synthetic glioma-like 2D MRI data.

Metrics
-------
  Global  : PSNR (dB), SSIM, NRMSE
  Regional: Dice_WT, Dice_TC, Dice_ET  — tumour region denoising fidelity

Output files (all saved to /home/user/AL-ML/)
  comparison_global.png     — PSNR / SSIM / NRMSE grouped bar chart
  comparison_regional.png   — Dice WT / TC / ET grouped bar chart
  comparison_curves.png     — training loss curves for every model
  comparison_visual.png     — visual denoising side by side
  comparison_table.png      — styled summary table
  comparison_results.csv    — raw numbers
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch, sys, numpy as np, csv
sys.path.insert(0, '/home/user/AL-ML/pp_mae')

from option1_cnn_pp_mae import CNNPPMAE, PPMAETrainer
from evaluation         import psnr, ssim_numpy, nrmse
from baselines          import (DnCNN,        DnCNNTrainer,
                                StandardUNet,  StandardUNetTrainer,
                                Noise2Noise,   Noise2NoiseTrainer,
                                REDNet,        REDNetTrainer,
                                dice_score)

DEVICE = 'cpu'
OUT    = '/home/user/AL-ML'
EPOCHS = 20          # enough to show convergence
SEED   = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ── Synthetic dataset ─────────────────────────────────────────────────────────
class SynGliomaMRI(torch.utils.data.Dataset):
    """
    Synthetic 4-channel MRI slices (64×64) with BraTS-style seg labels.
    Circular tumour blobs ensure Dice scores are meaningful.
    """
    def __init__(self, n=64, sigma=0.08, size=64, seed=0):
        rng = np.random.default_rng(seed)
        imgs, segs = [], []
        for _ in range(n):
            base = rng.random((4, size, size)).astype('float32') * 0.4 + 0.1
            cy, cx = rng.integers(16, size-16), rng.integers(16, size-16)
            Y, X   = np.ogrid[:size, :size]
            r_wt   = rng.integers(10, 16)
            r_tc   = rng.integers(5,  10)
            r_et   = rng.integers(2,   5)
            wt_m = ((Y-cy)**2 + (X-cx)**2) < r_wt**2
            tc_m = ((Y-cy)**2 + (X-cx)**2) < r_tc**2
            et_m = ((Y-cy)**2 + (X-cx)**2) < r_et**2
            base[0][wt_m] = rng.uniform(0.6, 0.9)
            base[1][tc_m] = rng.uniform(0.7, 1.0)
            base[2][wt_m] = rng.uniform(0.5, 0.85)
            base[3][wt_m] = rng.uniform(0.65, 0.95)
            base = base.clip(0., 1.)
            seg = np.zeros((1, size, size), dtype='int64')
            seg[0][wt_m] = 2
            seg[0][tc_m] = 1
            seg[0][et_m] = 3
            imgs.append(base); segs.append(seg)
        clean = torch.from_numpy(np.stack(imgs))
        noisy = (clean + sigma * torch.randn_like(clean)).clamp(0., 1.)
        self.x = noisy; self.t = clean
        self.s = torch.from_numpy(np.stack(segs))
    def __len__(self): return len(self.t)
    def __getitem__(self, i):
        return {'noisy': self.x[i], 'target': self.t[i], 'seg': self.s[i]}


print("Building dataset …", flush=True)
ds     = SynGliomaMRI(n=64, sigma=0.08, seed=SEED)
loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)

# ── Training loop ─────────────────────────────────────────────────────────────
def train_model(trainer, loader, epochs, label):
    hist = []
    for ep in range(1, epochs + 1):
        acc = {}
        for b in loader:
            m = trainer.step(b)
            for k, v in m.items(): acc[k] = acc.get(k, 0) + v
        row = {k: v/len(loader) for k,v in acc.items()}
        row['epoch'] = ep
        hist.append(row)
        if ep % 5 == 0 or ep == 1:
            print(f'  [{label:>22s}] Ep {ep:2d}/{epochs}  total={row["total"]:.4f}', flush=True)
    return hist


# ── Define all models ─────────────────────────────────────────────────────────
C = 4

PPmae_MODES    = ['PP-MAE (fixed)', 'PP-MAE (adaptive)', 'PP-MAE (clinical_risk)', 'PP-MAE (combined)']
BASELINE_NAMES = ['DnCNN', 'UNet-L1', 'Noise2Noise', 'REDNet']

models_trainers = {}

for mode in ['fixed', 'adaptive', 'clinical_risk', 'combined']:
    m = CNNPPMAE(in_channels=C, base_ch=16, depth=3)   # lighter model for speed
    models_trainers[f'PP-MAE ({mode})'] = (m, PPMAETrainer(m, device=DEVICE, mode=mode, lr=1e-4))

m_dncnn = DnCNN(C, features=32, num_layers=10)
m_unet  = StandardUNet(C, base_ch=16)
m_n2n   = Noise2Noise(C, features=32, num_layers=10)
m_red   = REDNet(C, features=32, num_layers=4)

models_trainers['DnCNN']       = (m_dncnn, DnCNNTrainer(m_dncnn,  device=DEVICE))
models_trainers['UNet-L1']     = (m_unet,  StandardUNetTrainer(m_unet,   device=DEVICE))
models_trainers['Noise2Noise'] = (m_n2n,   Noise2NoiseTrainer(m_n2n,    device=DEVICE))
models_trainers['REDNet']      = (m_red,   REDNetTrainer(m_red,    device=DEVICE))

# ── Train everything ──────────────────────────────────────────────────────────
print("\n=== TRAINING ALL MODELS ===", flush=True)
histories = {}
for name, (model, trainer) in models_trainers.items():
    print(f'\n--- {name} ---', flush=True)
    histories[name] = train_model(trainer, loader, EPOCHS, name)

# ── Evaluation ────────────────────────────────────────────────────────────────
def eval_model(model, ds, n_eval=64):
    model.eval()
    ps_, ss_, nr_, dw_, dt_, de_ = [], [], [], [], [], []
    with torch.no_grad():
        for i in range(min(n_eval, len(ds))):
            b = ds[i]
            x   = b['noisy'].unsqueeze(0)
            tgt = b['target'].unsqueeze(0)
            seg = b['seg'].unsqueeze(0)
            try:    pred = model(x, seg)
            except: pred = model(x)
            p_np = pred[0].permute(1,2,0).numpy()
            t_np = tgt[0].permute(1,2,0).numpy()
            ps_.append(psnr(p_np, t_np))
            ss_.append(ssim_numpy(p_np, t_np))
            nr_.append(nrmse(p_np, t_np))
            dc = dice_score(pred, seg)
            dw_.append(dc['wt']); dt_.append(dc['tc']); de_.append(dc['et'])
    return (float(np.mean(ps_)), float(np.mean(ss_)), float(np.mean(nr_)),
            float(np.mean(dw_)), float(np.mean(dt_)), float(np.mean(de_)))

def eval_noisy(ds, n_eval=64):
    ps_, ss_, nr_, dw_, dt_, de_ = [], [], [], [], [], []
    for i in range(min(n_eval, len(ds))):
        b = ds[i]
        p_np = b['noisy'].permute(1,2,0).numpy()
        t_np = b['target'].permute(1,2,0).numpy()
        ps_.append(psnr(p_np, t_np)); ss_.append(ssim_numpy(p_np, t_np)); nr_.append(nrmse(p_np, t_np))
        dc = dice_score(b['noisy'].unsqueeze(0), b['seg'].unsqueeze(0))
        dw_.append(dc['wt']); dt_.append(dc['tc']); de_.append(dc['et'])
    return (float(np.mean(ps_)), float(np.mean(ss_)), float(np.mean(nr_)),
            float(np.mean(dw_)), float(np.mean(dt_)), float(np.mean(de_)))

print("\n=== EVALUATING ALL MODELS ===", flush=True)
results = {'No Denoising': eval_noisy(ds)}
for name, (model, _) in models_trainers.items():
    r = eval_model(model, ds)
    results[name] = r
    print(f'  {name:<26s}  PSNR={r[0]:.2f}  SSIM={r[1]:.3f}  NRMSE={r[2]:.4f}'
          f'  Dice WT={r[3]:.3f} TC={r[4]:.3f} ET={r[5]:.3f}', flush=True)

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = f'{OUT}/comparison_results.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Model','PSNR','SSIM','NRMSE','Dice_WT','Dice_TC','Dice_ET'])
    for name, r in results.items():
        w.writerow([name, *[f'{v:.4f}' for v in r]])
print(f'Saved {csv_path}', flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== GENERATING PLOTS ===", flush=True)

ALL_MODELS   = ['No Denoising'] + PPmae_MODES + BASELINE_NAMES
ppmae_colors = ['#1565C0', '#1E88E5', '#42A5F5', '#90CAF9']
base_colors  = ['#E65100', '#F57C00', '#EF6C00', '#FF8F00']
color_map    = dict(zip(PPmae_MODES, ppmae_colors))
color_map.update(dict(zip(BASELINE_NAMES, base_colors)))
color_map['No Denoising'] = '#9E9E9E'
get_colors = lambda names: [color_map[n] for n in names]

pp_patch = mpatches.Patch(color='#1E88E5', label='PP-MAE (ours)')
bl_patch = mpatches.Patch(color='#F57C00', label='Baselines')
nd_patch = mpatches.Patch(color='#9E9E9E', label='No Denoising')

# ── Figure 1: Global Metrics ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (mi, label, better) in zip(axes, [
        (0, 'PSNR (dB)',  '↑ higher is better'),
        (1, 'SSIM',       '↑ higher is better'),
        (2, 'NRMSE',      '↓ lower is better')]):
    vals = [results[n][mi] for n in ALL_MODELS]
    bars = ax.bar(range(len(ALL_MODELS)), vals, color=get_colors(ALL_MODELS),
                  edgecolor='white', linewidth=1.2)
    ax.set_xticks(range(len(ALL_MODELS)))
    ax.set_xticklabels(ALL_MODELS, rotation=32, ha='right', fontsize=8)
    ax.set_title(f'{label}\n{better}', fontweight='bold', fontsize=11)
    ax.set_ylabel(label); ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
fig.legend(handles=[pp_patch, bl_patch, nd_patch], loc='lower center',
           ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('PP-MAE vs. Baselines — Global Image Quality Metrics\n'
             '(PSNR, SSIM, NRMSE — Synthetic Glioma MRI, σ=0.08 Rician noise)',
             fontweight='bold', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/comparison_global.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved comparison_global.png', flush=True)

# ── Figure 2: Tumour-Region Dice ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (ri, rn, rd) in zip(axes, [
        (3, 'Dice WT', 'Whole Tumour  (ED + NCR + ET)'),
        (4, 'Dice TC', 'Tumour Core   (NCR + ET)'),
        (5, 'Dice ET', 'Enhancing Tumour  (ET)')]):
    vals = [results[n][ri] for n in ALL_MODELS]
    bars = ax.bar(range(len(ALL_MODELS)), vals, color=get_colors(ALL_MODELS),
                  edgecolor='white', linewidth=1.2)
    ax.set_xticks(range(len(ALL_MODELS))); ax.set_ylim(0, 1.12)
    ax.set_xticklabels(ALL_MODELS, rotation=32, ha='right', fontsize=8)
    ax.set_title(f'{rn}\n{rd}', fontweight='bold', fontsize=11)
    ax.set_ylabel('Dice Score'); ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
fig.legend(handles=[pp_patch, bl_patch, nd_patch], loc='lower center',
           ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('PP-MAE vs. Baselines — Tumour-Region Dice Scores\n'
             '(Whole Tumour · Tumour Core · Enhancing Tumour)',
             fontweight='bold', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/comparison_regional.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved comparison_regional.png', flush=True)

# ── Figure 3: Training Loss Curves ────────────────────────────────────────────
fig, (ax_pp, ax_bl) = plt.subplots(2, 1, figsize=(14, 10))
for name, col in zip(PPmae_MODES, ppmae_colors):
    h = histories[name]
    ax_pp.plot([r['epoch'] for r in h], [r['total'] for r in h],
               label=name, color=col, lw=2)
ax_pp.set_title('PP-MAE: Training Loss Curves (4 Loss Modes)', fontweight='bold')
ax_pp.set_xlabel('Epoch'); ax_pp.set_ylabel('Total Loss')
ax_pp.legend(fontsize=9); ax_pp.grid(alpha=0.3)
for name, col in zip(BASELINE_NAMES, base_colors):
    h = histories[name]
    ax_bl.plot([r['epoch'] for r in h], [r['total'] for r in h],
               label=name, color=col, lw=2)
ax_bl.set_title('Baselines: Training Loss Curves', fontweight='bold')
ax_bl.set_xlabel('Epoch'); ax_bl.set_ylabel('Total Loss')
ax_bl.legend(fontsize=9); ax_bl.grid(alpha=0.3)
fig.suptitle('PP-MAE vs. Baselines — Convergence Comparison', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/comparison_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved comparison_curves.png', flush=True)

# ── Figure 4: Visual Denoising Comparison ─────────────────────────────────────
idx = 7; b = ds[idx]
noisy_np = b['noisy'].numpy(); clean_np = b['target'].numpy()
seg_np   = b['seg'][0].numpy()
seg_contour = (seg_np > 0).astype(float)

def get_pred_np(model, b):
    model.eval()
    with torch.no_grad():
        x = b['noisy'].unsqueeze(0); seg = b['seg'].unsqueeze(0)
        try:    return model(x, seg)[0].numpy()
        except: return model(x)[0].numpy()

preds = {name: get_pred_np(model, b) for name, (model, _) in models_trainers.items()}

ch = 1   # T1Wce — tumour most visible
show_models = ['PP-MAE (clinical_risk)', 'DnCNN', 'UNet-L1', 'Noise2Noise', 'REDNet']

fig, axes = plt.subplots(2, 5, figsize=(22, 9))
row1 = [('Noisy Input\n(σ=0.08)',  noisy_np[ch]),
        ('Ground Truth',            clean_np[ch])] + \
       [(f'{n}\n(denoised)', preds[n][ch]) for n in ['PP-MAE (clinical_risk)', 'DnCNN', 'UNet-L1']]
row2 = [(f'{n}\nError (pred−GT)', preds[n][ch] - clean_np[ch]) for n in show_models]
# pad row2 to 5 items (add ground truth error=0 for first cell)
row2 = [('GT Error\n(= 0)', np.zeros_like(clean_np[ch]))] + row2[:4]  # 5 items

for ax, (title, img) in zip(axes[0], row1):
    is_err = 'Error' in title
    im = ax.imshow(img, cmap='RdBu_r' if is_err else 'gray',
                   vmin=-0.2 if is_err else 0, vmax=0.2 if is_err else 1)
    ax.contour(seg_contour, levels=[0.5], colors='lime', linewidths=0.9)
    ax.set_title(title, fontsize=8.5, fontweight='bold'); ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for ax, (title, img) in zip(axes[1], row2):
    im = ax.imshow(img, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
    ax.contour(seg_contour, levels=[0.5], colors='lime', linewidths=0.9)
    ax.set_title(title, fontsize=8.5, fontweight='bold'); ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle('Visual Comparison — T1Wce channel  (lime = tumour boundary)\n'
             'Row 1: predictions  │  Row 2: error maps (pred − GT, blue=under, red=over)',
             fontweight='bold', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/comparison_visual.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved comparison_visual.png', flush=True)

# ── Figure 5: Summary Comparison Table ───────────────────────────────────────
def param_count(m): return sum(p.numel() for p in m.parameters())

row_order = ['No Denoising'] + PPmae_MODES + BASELINE_NAMES
refs = {
    'No Denoising': '—',
    'PP-MAE (fixed)': 'Ours', 'PP-MAE (adaptive)': 'Ours',
    'PP-MAE (clinical_risk)': 'Ours', 'PP-MAE (combined)': 'Ours',
    'DnCNN': 'Zhang 2017', 'UNet-L1': 'Ronneberger 2015',
    'Noise2Noise': 'Lehtinen 2018', 'REDNet': 'Mao 2016',
}
param_map = {'No Denoising': '—'}
for mode in ['fixed', 'adaptive', 'clinical_risk', 'combined']:
    m_tmp = CNNPPMAE(4, 16, 3)
    param_map[f'PP-MAE ({mode})'] = f'{param_count(m_tmp):,}'
param_map.update({
    'DnCNN': f'{param_count(m_dncnn):,}',
    'UNet-L1': f'{param_count(m_unet):,}',
    'Noise2Noise': f'{param_count(m_n2n):,}',
    'REDNet': f'{param_count(m_red):,}',
})

col_labels = ['Model', '#Params', 'PSNR↑', 'SSIM↑', 'NRMSE↓', 'Dice WT↑', 'Dice TC↑', 'Dice ET↑', 'Ref']
cell_data = []
for name in row_order:
    r = results[name]
    cell_data.append([name, param_map.get(name,'—'),
                      f'{r[0]:.2f}', f'{r[1]:.3f}', f'{r[2]:.4f}',
                      f'{r[3]:.3f}', f'{r[4]:.3f}', f'{r[5]:.3f}',
                      refs.get(name,'—')])

fig, ax = plt.subplots(figsize=(18, 5.5))
ax.axis('off')
tbl = ax.table(cellText=cell_data, colLabels=col_labels, cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 2.2)

pp_rows = set(range(1, 5))   # PP-MAE rows
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#1565C0'); cell.set_text_props(color='white', fontweight='bold')
    elif row in pp_rows:
        cell.set_facecolor('#BBDEFB' if row == 3 else '#E3F2FD')  # row 3 = clinical_risk (best)
    elif row == len(row_order):
        cell.set_facecolor('#FFF9C4')
    else:
        cell.set_facecolor('#FFF3E0' if row % 2 else '#FFFFFF')
    cell.set_edgecolor('#BDBDBD')

fig.suptitle('PP-MAE vs. Baselines — Complete Comparison Table\n'
             '(Blue shading = PP-MAE ours, darker blue = best variant  |  Orange = baselines)',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/comparison_table.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved comparison_table.png', flush=True)

# ── Final summary ─────────────────────────────────────────────────────────────
print('\n' + '='*65, flush=True)
print('COMPARISON COMPLETE — SUMMARY', flush=True)
print('='*65, flush=True)
r_pp = results['PP-MAE (clinical_risk)']
r_dn = results['DnCNN']
r_no = results['No Denoising']
print(f'\n{"Model":<26}  {"PSNR":>7}  {"SSIM":>6}  {"NRMSE":>7}  '
      f'{"Dice_WT":>8}  {"Dice_ET":>8}')
print('-'*70)
for name in row_order:
    r = results[name]
    mark = ' ◀ best' if name == 'PP-MAE (clinical_risk)' else ''
    print(f'{name:<26}  {r[0]:>7.2f}  {r[1]:>6.3f}  {r[2]:>7.4f}  '
          f'{r[3]:>8.3f}  {r[5]:>8.3f}{mark}')
print(f'\n  ΔPSNR   PP-MAE (clinical_risk) vs DnCNN: {r_pp[0]-r_dn[0]:+.2f} dB')
print(f'  ΔSSIM   PP-MAE (clinical_risk) vs DnCNN: {r_pp[1]-r_dn[1]:+.3f}')
print(f'  ΔDice_ET PP-MAE (clinical_risk) vs DnCNN: {r_pp[5]-r_dn[5]:+.3f}')
print(f'\nOutput files saved to {OUT}/', flush=True)
print('  comparison_global.png', flush=True)
print('  comparison_regional.png', flush=True)
print('  comparison_curves.png', flush=True)
print('  comparison_visual.png', flush=True)
print('  comparison_table.png', flush=True)
print('  comparison_results.csv', flush=True)
