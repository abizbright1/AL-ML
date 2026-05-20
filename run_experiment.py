import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, sys, time, numpy as np
sys.path.insert(0, '/home/user/AL-ML/pp_mae')

from losses import PPMAELoss
from option1_cnn_pp_mae  import CNNPPMAE, PPMAETrainer
from option2_vit_pp_mae  import ViTPPMAE, ViTPPMAETrainer
from option4_swin_pp_mae import SwinPPMAE, SwinPPMAETrainer
from evaluation import psnr, ssim_numpy, nrmse

DEVICE = 'cpu'
OUT    = '/home/user/AL-ML'

# ── Datasets ──────────────────────────────────────────────────────────────
class SynDS2D(torch.utils.data.Dataset):
    def __init__(self, n=64, sigma=0.08):
        self.t = torch.rand(n,4,128,128)
        self.x = (self.t + sigma*torch.randn_like(self.t)).clamp(0,1)
        self.s = torch.randint(0,4,(n,1,128,128))
    def __len__(self): return len(self.t)
    def __getitem__(self,i):
        return {'noisy':self.x[i],'target':self.t[i],'seg':self.s[i]}

class SynDS3D(torch.utils.data.Dataset):
    def __init__(self, n=32, sigma=0.08):
        self.t = torch.rand(n,4,16,64,64)
        self.x = (self.t + sigma*torch.randn_like(self.t)).clamp(0,1)
        self.s = torch.randint(0,4,(n,1,16,64,64))
    def __len__(self): return len(self.t)
    def __getitem__(self,i):
        return {'noisy':self.x[i],'target':self.t[i],'seg':self.s[i]}

loader2d = torch.utils.data.DataLoader(SynDS2D(64), batch_size=4, shuffle=True)
loader3d = torch.utils.data.DataLoader(SynDS3D(32), batch_size=2, shuffle=True)

def train(trainer, loader, epochs, label):
    hist = []
    for ep in range(1, epochs+1):
        acc={}
        for b in loader:
            m = trainer.step(b)
            for k,v in m.items(): acc[k]=acc.get(k,0)+v
        row={k:v/len(loader) for k,v in acc.items()}; row['epoch']=ep
        hist.append(row)
        if ep%5==0 or ep==1:
            print(f'  [{label}] Ep {ep:2d}/{epochs}  total={row.get("total",0):.4f}', flush=True)
    return hist

# ── Ablation ──────────────────────────────────────────────────────────────
print("=== ABLATION (CNN, 4 loss modes) ===", flush=True)
ablation={}
for mode in ['fixed','adaptive','clinical_risk','combined']:
    m = CNNPPMAE(in_channels=4, base_ch=32, depth=3)
    ablation[mode] = train(PPMAETrainer(m, device=DEVICE, mode=mode, lr=1e-4), loader2d, 20, mode)

# ── Architectures ─────────────────────────────────────────────────────────
print("\n=== ARCHITECTURES ===", flush=True)
arch_res={}; param_counts={}

m1 = CNNPPMAE(in_channels=4, base_ch=32, depth=3)
arch_res['CNN'] = train(PPMAETrainer(m1, device=DEVICE, lr=1e-4), loader2d, 20, 'CNN')
param_counts['CNN'] = sum(p.numel() for p in m1.parameters())

m2  = ViTPPMAE(vol_size=(16,64,64), patch_size=16, in_chans=4,
               embed_dim=96, depth=4, n_heads=4, decoder_dim=48, decoder_depth=2)
opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-4, weight_decay=0.05)
arch_res['ViT'] = train(ViTPPMAETrainer(m2, opt2, device=DEVICE), loader3d, 20, 'ViT')
param_counts['ViT'] = sum(p.numel() for p in m2.parameters())

m4  = SwinPPMAE(in_ch=4, embed_dim=32, depths=(2,2,4,2), n_heads=(2,4,8,16), window_size=4)
opt4 = torch.optim.AdamW(m4.parameters(), lr=1e-4, weight_decay=0.05)
arch_res['Swin'] = train(SwinPPMAETrainer(m4, opt4, device=DEVICE), loader2d, 20, 'Swin')
param_counts['Swin'] = sum(p.numel() for p in m4.parameters())

# ── Figure 1: Ablation loss curves ────────────────────────────────────────
print("\n=== GENERATING PLOTS ===", flush=True)
colors=['#2196F3','#4CAF50','#FF9800','#E91E63']
fig,axes=plt.subplots(1,2,figsize=(13,4))
for (mode,h),col in zip(ablation.items(),colors):
    ep=[x['epoch'] for x in h]
    axes[0].plot(ep,[x.get('total',0) for x in h],label=mode,color=col,lw=2)
    axes[1].plot(ep,[x.get('pathology',0) for x in h],label=mode,color=col,lw=2)
for ax,t in zip(axes,['Total Loss','Pathology Loss']):
    ax.set_title(t,fontweight='bold'); ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=0.3)
fig.suptitle('PP-MAE Ablation Study — CNN U-Net, 4 Loss Modes',fontweight='bold',y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/fig1_ablation.png',dpi=150,bbox_inches='tight'); plt.close()
print("Saved fig1_ablation.png", flush=True)

# ── Figure 2: Architecture comparison ─────────────────────────────────────
colors2=['#2196F3','#4CAF50','#9C27B0']
fig,axes=plt.subplots(1,2,figsize=(13,4))
for (arch,h),col in zip(arch_res.items(),colors2):
    ep=[x['epoch'] for x in h]
    axes[0].plot(ep,[x.get('total',0) for x in h],
                 label=f"{arch} ({param_counts[arch]:,} params)",color=col,lw=2)
    axes[1].plot(ep,[x.get('pathology',0) for x in h],label=arch,color=col,lw=2)
for ax,t in zip(axes,['Total Loss','Pathology Loss']):
    ax.set_title(t,fontweight='bold'); ax.set_xlabel('Epoch'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.suptitle('PP-MAE Architecture Comparison — CNN / ViT / Swin',fontweight='bold',y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/fig2_architectures.png',dpi=150,bbox_inches='tight'); plt.close()
print("Saved fig2_architectures.png", flush=True)

# ── Figure 3: PSNR / SSIM / NRMSE bar chart ───────────────────────────────
rng=np.random.default_rng(42); N=20
t_np=rng.random((N,128,128,4)).astype('float32')
n_np=(t_np+0.08*rng.standard_normal(t_np.shape)).clip(0,1).astype('float32')

def eval_2d(model, n_np, t_np):
    model.eval(); ps,ss,ns=[],[],[]
    with torch.no_grad():
        for i in range(len(n_np)):
            x=torch.from_numpy(n_np[i]).permute(2,0,1).unsqueeze(0)
            s=torch.zeros(1,1,128,128,dtype=torch.long)
            try:    pred=model(x,s)
            except: pred=model(x)
            p2=pred[0].permute(1,2,0).numpy()
            ps.append(psnr(p2,t_np[i])); ss.append(ssim_numpy(p2,t_np[i])); ns.append(nrmse(p2,t_np[i]))
    return np.mean(ps), np.mean(ss), np.mean(ns)

names=['No Denoising','CNN','Swin']
results=[
    (psnr(n_np[0],t_np[0]), ssim_numpy(n_np[0],t_np[0]), nrmse(n_np[0],t_np[0])),
    eval_2d(m1,n_np,t_np),
    eval_2d(m4,n_np,t_np),
]
psnrs=[r[0] for r in results]; ssims=[r[1] for r in results]; nrmses=[r[2] for r in results]

fig,axes=plt.subplots(1,3,figsize=(13,4))
bar_colors=['#90A4AE','#2196F3','#9C27B0']
for ax,(vals,title,better) in zip(axes,[
        (psnrs,'PSNR (dB)','↑ higher is better'),
        (ssims,'SSIM','↑ higher is better'),
        (nrmses,'NRMSE','↓ lower is better')]):
    bars=ax.bar(names,vals,color=bar_colors,edgecolor='white',linewidth=1.2)
    ax.set_title(f'{title}\n{better}',fontweight='bold'); ax.set_ylabel(title)
    ax.tick_params(axis='x',rotation=10)
    for bar,val in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
fig.suptitle('Image Quality Metrics — PP-MAE vs No Denoising Baseline',fontweight='bold',y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/fig3_metrics.png',dpi=150,bbox_inches='tight'); plt.close()
print("Saved fig3_metrics.png", flush=True)

# ── Figure 4: Visual denoising comparison ─────────────────────────────────
def get_pred(model,i):
    model.eval()
    with torch.no_grad():
        x=torch.from_numpy(n_np[i]).permute(2,0,1).unsqueeze(0)
        s=torch.zeros(1,1,128,128,dtype=torch.long)
        try:    return model(x,s)[0,0].numpy()
        except: return model(x)[0,0].numpy()

idx=3
panels=[
    ('Noisy input\n(σ=0.08)',   n_np[idx,:,:,0]),
    ('Ground truth',            t_np[idx,:,:,0]),
    ('CNN denoised',            get_pred(m1,idx)),
    ('Swin denoised',           get_pred(m4,idx)),
    ('CNN error\n(pred − GT)',   get_pred(m1,idx) - t_np[idx,:,:,0]),
]
fig,axes=plt.subplots(1,5,figsize=(20,4))
cmaps=['gray','gray','gray','gray','RdBu_r']
for ax,(title,img),cmap in zip(axes,panels,cmaps):
    vmin,vmax=(-0.3,0.3) if 'error' in title else (0,1)
    im=ax.imshow(img,cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set_title(title,fontweight='bold',fontsize=10); ax.axis('off')
    plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
fig.suptitle('T1W MRI Denoising — Visual Comparison (σ=0.08 Rician noise)',
             fontweight='bold',fontsize=13,y=1.03)
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_visual.png',dpi=150,bbox_inches='tight'); plt.close()
print("Saved fig4_visual.png", flush=True)

# ── Figure 5: Styled summary table ────────────────────────────────────────
fig,ax=plt.subplots(figsize=(12,3.5))
ax.axis('off')
col_labels=['Loss Mode','Total Loss','Global Loss','Pathology Loss','Epoch 1 → 20']
cell_data=[[mode,
            f"{h[-1].get('total',0):.4f}",
            f"{h[-1].get('global',0):.4f}",
            f"{h[-1].get('pathology',0):.4f}",
            f"{h[0].get('total',0):.3f} → {h[-1].get('total',0):.3f}"]
           for mode,h in ablation.items()]
tbl=ax.table(cellText=cell_data, colLabels=col_labels, cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1,2.2)
for (r,c),cell in tbl.get_celld().items():
    if r==0:
        cell.set_facecolor('#1565C0'); cell.set_text_props(color='white',fontweight='bold')
    elif r%2==0:
        cell.set_facecolor('#E3F2FD')
    cell.set_edgecolor('#90CAF9')
fig.suptitle('PP-MAE Ablation Results — CNN U-Net, 20 Epochs, Synthetic Data',
             fontweight='bold',fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_table.png',dpi=150,bbox_inches='tight'); plt.close()
print("Saved fig5_table.png", flush=True)

print("\nALL DONE", flush=True)
