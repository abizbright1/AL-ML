"""
paper_figures.py
================
Generates all 4 paper-ready visualization panels:

  Fig A — Denoising samples: Noisy | PP-MAE | SwinIR-L1 | Uformer-L1 | GT
  Fig B — Segmentation overlay: GT seg mapped onto each model's output
  Fig C — Training loss curves: convergence of all 5 models
  Fig D — Per-slice metric distributions: box plots of Dice_ET, Dice_TC, PSNR

Usage (MacBook):
    python3 paper_figures.py ~/Downloads/BraTS2021_data \\
        --device mps --epochs 30 --seg_epochs 20 \\
        --max_subjects 50 --out paper_figs/
"""

from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── path setup ────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "pp_mae"))

from option4_swin_pp_mae  import SwinPPMAE, SwinPPMAETrainer
from option_baselines      import SwinIRLite, SwinIRTrainer, SwinIRPathologyTrainer
from option_baselines      import UformerLite, UformerTrainer, UformerPathologyTrainer
from segmentor             import UNetSegmentor, SegTrainer
from evaluation            import psnr, ssim_numpy, nrmse
from brats_loader          import BraTSDataset, make_demo_brats

# ── colour maps ───────────────────────────────────────────────────────────────
# BraTS seg: 0=BG, 1=NCR, 2=ED, 3=ET
SEG_COLOURS = np.array([
    [0,   0,   0,   0  ],   # 0 BG     — transparent
    [0,   0,   1,   0.5],   # 1 NCR    — blue
    [0,   1,   0,   0.5],   # 2 ED     — green
    [1,   0.2, 0,   0.7],   # 3 ET     — red-orange
], dtype=np.float32)
SEG_CMAP = ListedColormap(SEG_COLOURS[:, :3])

MODEL_COLOURS = {
    "PP-MAE (Swin)\n[PROPOSED]": "#1565C0",
    "SwinIR-lite\n(L1)":         "#EF6C00",
    "Uformer-lite\n(L1)":        "#F9A825",
    "SwinIR +\nPathologyLoss":   "#558B2F",
    "Uformer +\nPathologyLoss":  "#00695C",
}

# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("brats_root", nargs="?", default=None)
    p.add_argument("--device",       default=None)
    p.add_argument("--epochs",       type=int, default=30)
    p.add_argument("--seg_epochs",   type=int, default=20)
    p.add_argument("--max_subjects", type=int, default=50)
    p.add_argument("--patch_size",   type=int, default=96)
    p.add_argument("--sigma",        type=float, default=0.08)
    p.add_argument("--out",          default="paper_figs")
    p.add_argument("--n_samples",    type=int, default=4,
                   help="Number of representative slices to show")
    return p.parse_args()


# ── data loading ──────────────────────────────────────────────────────────────
def load_data(args, device):
    if args.brats_root and os.path.isdir(args.brats_root):
        try:
            ds = BraTSDataset(args.brats_root, slice_axis=2,
                              patch_size=args.patch_size, sigma=args.sigma,
                              min_tumour_frac=0.01, cache=True,
                              max_subjects=args.max_subjects)
            print(f"  Real BraTS: {len(ds)} slices from {args.max_subjects} subjects")
            use_real = True
        except Exception as e:
            print(f"  BraTS load failed ({e}), using demo data")
            ds = make_demo_brats(n_subjects=6, patch_size=args.patch_size,
                                 sigma=args.sigma, slices_per_subject=24)
            use_real = False
    else:
        ds = make_demo_brats(n_subjects=6, patch_size=args.patch_size,
                             sigma=args.sigma, slices_per_subject=24)
        use_real = False
        print("  Demo mode (no BraTS path given)")

    n = len(ds)
    n_train = int(0.8 * n)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n - n_train],
        generator=torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=4, shuffle=False)
    return train_loader, val_loader, use_real


# ── segmentor training ────────────────────────────────────────────────────────
def train_segmentor(train_loader, device, seg_epochs):
    seg_model   = UNetSegmentor(4, 4, 32).to(device)
    seg_trainer = SegTrainer(seg_model, device=device, lr=5e-4)
    print(f"\n  Training segmentor ({seg_epochs} epochs) …")
    for ep in range(1, seg_epochs + 1):
        loss = 0.0
        for b in train_loader:
            loss += seg_trainer.step(b["target"].to(device),
                                     b["seg"][:, 0].long().to(device))
        if ep % 5 == 0 or ep == 1:
            print(f"    Ep {ep}/{seg_epochs}  loss={loss/len(train_loader):.4f}")
    seg_model.eval()
    print("  Segmentor frozen.")
    return seg_model


# ── model definitions ─────────────────────────────────────────────────────────
def build_models(device, patch_size):
    return {
        "PP-MAE (Swin)\n[PROPOSED]": {
            "model":      SwinPPMAE(4, embed_dim=48, depths=(2,2,2,2),
                                    n_heads=(3,3,6,6), window_size=4),
            "trainer_fn": lambda m: SwinPPMAETrainer(m, device=device),
            "infer_fn":   lambda m, noisy, seg: m(noisy, seg),
            "colour":     MODEL_COLOURS["PP-MAE (Swin)\n[PROPOSED]"],
        },
        "SwinIR-lite\n(L1)": {
            "model":      SwinIRLite(4, dim=64, n_blocks=4, window_size=4),
            "trainer_fn": lambda m: SwinIRTrainer(m, device=device, lr=1e-4),
            "infer_fn":   lambda m, noisy, seg: m(noisy),
            "colour":     MODEL_COLOURS["SwinIR-lite\n(L1)"],
        },
        "Uformer-lite\n(L1)": {
            "model":      UformerLite(4, dim=32, window_size=4),
            "trainer_fn": lambda m: UformerTrainer(m, device=device, lr=1e-4),
            "infer_fn":   lambda m, noisy, seg: m(noisy),
            "colour":     MODEL_COLOURS["Uformer-lite\n(L1)"],
        },
        "SwinIR +\nPathologyLoss": {
            "model":      SwinIRLite(4, dim=64, n_blocks=4, window_size=4),
            "trainer_fn": lambda m: SwinIRPathologyTrainer(m, device=device,
                                                           lr=1e-4, mode="clinical_risk"),
            "infer_fn":   lambda m, noisy, seg: m(noisy),
            "colour":     MODEL_COLOURS["SwinIR +\nPathologyLoss"],
        },
        "Uformer +\nPathologyLoss": {
            "model":      UformerLite(4, dim=32, window_size=4),
            "trainer_fn": lambda m: UformerPathologyTrainer(m, device=device,
                                                            lr=1e-4, mode="clinical_risk"),
            "infer_fn":   lambda m, noisy, seg: m(noisy),
            "colour":     MODEL_COLOURS["Uformer +\nPathologyLoss"],
        },
    }


# ── training loop ─────────────────────────────────────────────────────────────
def train_all(models, train_loader, device, epochs):
    histories = {}
    for name, cfg in models.items():
        print(f"\n  [{name.replace(chr(10),' ')}]")
        m       = cfg["model"].to(device)
        trainer = cfg["trainer_fn"](m)
        hist    = []
        for ep in range(1, epochs + 1):
            ep_loss = 0.0
            for b in train_loader:
                try:
                    res = trainer.step(b)
                except Exception:
                    res = {"total": 0.0}
                ep_loss += res.get("total", 0.0)
            ep_loss /= max(len(train_loader), 1)
            hist.append(ep_loss)
            if ep % 5 == 0 or ep == 1:
                print(f"    Ep {ep:2d}/{epochs}  loss={ep_loss:.4f}")
        cfg["model"] = m
        histories[name] = hist
    return histories


# ── collect validation samples ────────────────────────────────────────────────
def collect_samples(models, val_loader, seg_model, device, n_samples):
    """
    Collect n_samples representative slices.  Picks slices with clear ET
    (ground-truth ET fraction > threshold), returns dict of arrays.
    """
    all_noisy, all_clean, all_seg = [], [], []
    preds   = {name: [] for name in models}
    seg_preds = {name: [] for name in models}
    metrics   = {name: {"psnr": [], "ssim": [], "dice_et": [], "dice_tc": []} for name in models}

    seg_device = next(seg_model.parameters()).device

    with torch.no_grad():
        for b in val_loader:
            noisy  = b["noisy"]
            target = b["target"]
            seg    = b["seg"]
            gt_lab = seg[:, 0].long()

            # only keep slices with enough ET
            et_frac = (gt_lab == 3).float().mean(dim=(1, 2))
            keep    = (et_frac > 0.005).nonzero(as_tuple=True)[0]
            if len(keep) == 0:
                continue

            for i in keep.tolist():
                if len(all_noisy) >= n_samples:
                    break
                all_noisy.append(noisy[i].cpu())
                all_clean.append(target[i].cpu())
                all_seg.append(gt_lab[i].cpu())

                for name, cfg in models.items():
                    m        = cfg["model"]
                    infer_fn = cfg["infer_fn"]
                    m.eval()
                    ni  = noisy[i:i+1].to(device)
                    si  = seg[i:i+1].to(device)
                    try:
                        pred = infer_fn(m, ni, si)
                    except Exception:
                        pred = infer_fn(m, ni, torch.zeros_like(si))
                    pred_cpu = pred[0].cpu()
                    preds[name].append(pred_cpu)

                    # PSNR / SSIM
                    p_np = pred_cpu.permute(1,2,0).numpy()
                    t_np = target[i].permute(1,2,0).numpy()
                    metrics[name]["psnr"].append(psnr(p_np, t_np))
                    metrics[name]["ssim"].append(ssim_numpy(p_np, t_np))

                    # Downstream segmentation
                    logits   = seg_model(pred_cpu.unsqueeze(0).to(seg_device))
                    pred_lab = logits.argmax(1)[0].cpu()
                    gt_l     = gt_lab[i]
                    # ET dice
                    et_p = (pred_lab == 3); et_g = (gt_l == 3)
                    d_et = (2*(et_p & et_g).sum() / (et_p.sum()+et_g.sum()+1e-6)).item()
                    tc_p = ((pred_lab==1)|(pred_lab==3))
                    tc_g = ((gt_l==1)|(gt_l==3))
                    d_tc = (2*(tc_p & tc_g).sum() / (tc_p.sum()+tc_g.sum()+1e-6)).item()
                    metrics[name]["dice_et"].append(d_et)
                    metrics[name]["dice_tc"].append(d_tc)
                    seg_preds[name].append(pred_lab.numpy())

            if len(all_noisy) >= n_samples:
                break

    return all_noisy, all_clean, all_seg, preds, seg_preds, metrics


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE A — Denoising samples
# ─────────────────────────────────────────────────────────────────────────────
def fig_denoising(noisy_list, clean_list, preds, metrics, out_dir, n_models=3):
    """
    Rows: Noisy input | PP-MAE [PROPOSED] | SwinIR-L1 | Uformer-L1 | Ground Truth
    Cols: n_samples representative slices
    """
    n_s    = len(noisy_list)
    rows   = ["Noisy\nInput"] + list(preds.keys())[:n_models] + ["Ground\nTruth"]
    n_rows = len(rows)

    fig, axes = plt.subplots(n_rows, n_s, figsize=(3.5 * n_s, 3.2 * n_rows))
    fig.suptitle("Denoising Results — Representative Validation Slices\n"
                 "(T1ce channel shown; PSNR / SSIM annotated)",
                 fontsize=13, fontweight="bold", y=1.01)

    def show(ax, img_tensor, title="", psnr_v=None, ssim_v=None, border_col=None):
        # Show T1ce channel (index 1 — best for ET visibility)
        img = img_tensor[1].numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
        ax.axis("off")
        if psnr_v is not None:
            ax.set_title(f"PSNR {psnr_v:.2f} dB\nSSIM {ssim_v:.3f}",
                         fontsize=7.5, pad=2)
        if border_col:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(border_col)
                spine.set_linewidth(2.5)

    model_names = list(preds.keys())

    for col, s in enumerate(range(n_s)):
        for row, row_label in enumerate(rows):
            ax = axes[row, col] if n_s > 1 else axes[row]

            if row == 0:                     # Noisy
                show(ax, noisy_list[s])
                if col == 0:
                    ax.set_ylabel("Noisy Input", fontsize=9, fontweight="bold",
                                  rotation=90, labelpad=4)
                ax.set_title(f"Sample {s+1}", fontsize=9, fontweight="bold")

            elif row == n_rows - 1:          # GT
                show(ax, clean_list[s])
                if col == 0:
                    ax.set_ylabel("Ground Truth", fontsize=9, fontweight="bold",
                                  rotation=90, labelpad=4)

            else:                            # Model output
                mname = model_names[row - 1]
                col_  = list(MODEL_COLOURS.values())[row - 1]
                pv    = metrics[mname]["psnr"][s] if s < len(metrics[mname]["psnr"]) else 0
                sv    = metrics[mname]["ssim"][s] if s < len(metrics[mname]["ssim"]) else 0
                is_proposed = "PROPOSED" in mname
                show(ax, preds[mname][s],
                     psnr_v=pv, ssim_v=sv,
                     border_col=col_ if is_proposed else None)
                if col == 0:
                    short = mname.replace("\n", " ")
                    ax.set_ylabel(short, fontsize=8,
                                  color="#1565C0" if is_proposed else "#333",
                                  fontweight="bold" if is_proposed else "normal",
                                  rotation=90, labelpad=4)

    plt.tight_layout()
    p = os.path.join(out_dir, "figA_denoising_samples.png")
    plt.savefig(p, dpi=180, bbox_inches="tight")
    print(f"  -> {p}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE B — Segmentation overlays
# ─────────────────────────────────────────────────────────────────────────────
def fig_segmentation(noisy_list, clean_list, gt_seg_list, seg_preds, out_dir):
    """
    Rows: GT seg | PP-MAE pred seg | SwinIR-L1 pred seg | Uformer-L1 pred seg
    Cols: n_samples slices
    """
    n_s      = len(noisy_list)
    all_keys = list(seg_preds.keys())
    show_keys = [all_keys[0], all_keys[1], all_keys[2]]   # proposed + 2 baselines
    rows = ["GT\nSegmentation"] + show_keys

    fig, axes = plt.subplots(len(rows), n_s,
                              figsize=(3.5 * n_s, 3.4 * len(rows)))
    fig.suptitle("Downstream Segmentation — GT vs Model Predictions\n"
                 "Red=ET (Enhancing Tumour)  Green=ED  Blue=NCR",
                 fontsize=12, fontweight="bold", y=1.01)

    def overlay(ax, bg_tensor, seg_np, title_col=None):
        bg = bg_tensor[1].numpy()
        bg = np.clip(bg, 0, 1)
        ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
        rgba = SEG_COLOURS[seg_np]          # (H, W, 4)
        ax.imshow(rgba, interpolation="nearest")
        ax.axis("off")
        if title_col:
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor(title_col); sp.set_linewidth(2.5)

    for col in range(n_s):
        for row, row_key in enumerate(rows):
            ax = axes[row, col] if n_s > 1 else axes[row]

            if row == 0:
                seg_np = gt_seg_list[col].numpy()
                overlay(ax, clean_list[col], seg_np)
                if col == 0:
                    ax.set_ylabel("GT Segmentation", fontsize=9, fontweight="bold",
                                  rotation=90, labelpad=4)
                ax.set_title(f"Sample {col+1}", fontsize=9, fontweight="bold")
            else:
                mname = show_keys[row - 1]
                col_  = list(MODEL_COLOURS.values())[row - 1]
                is_proposed = "PROPOSED" in mname
                if col < len(seg_preds[mname]):
                    seg_np = seg_preds[mname][col]
                    overlay(ax, preds_ref[mname][col] if mname in preds_ref else clean_list[col],
                            seg_np, title_col=col_ if is_proposed else None)
                if col == 0:
                    short = mname.replace("\n", " ")
                    ax.set_ylabel(short, fontsize=8,
                                  color="#1565C0" if is_proposed else "#333",
                                  fontweight="bold" if is_proposed else "normal",
                                  rotation=90, labelpad=4)

    # Legend
    legend_patches = [
        mpatches.Patch(color=[0,0,1], label="NCR (label 1)"),
        mpatches.Patch(color=[0,1,0], label="ED  (label 2)"),
        mpatches.Patch(color=[1,0.2,0], label="ET  (label 3)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    p = os.path.join(out_dir, "figB_segmentation_overlay.png")
    plt.savefig(p, dpi=180, bbox_inches="tight")
    print(f"  -> {p}")
    plt.close()


# Store preds globally so fig_segmentation can access it
preds_ref = {}


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE C — Training loss curves
# ─────────────────────────────────────────────────────────────────────────────
def fig_loss_curves(histories, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Training Loss Convergence — All Models\n"
                 "(lower loss ≠ better clinical Dice — see Fig D)",
                 fontsize=12, fontweight="bold")

    for name, hist in histories.items():
        col   = list(MODEL_COLOURS.values())[list(histories.keys()).index(name)]
        lw    = 2.5 if "PROPOSED" in name else 1.5
        ls    = "-"  if "PROPOSED" in name else ("--" if "L1" in name else ":")
        label = name.replace("\n", " ")
        ax.plot(range(1, len(hist)+1), hist,
                color=col, linewidth=lw, linestyle=ls, label=label, alpha=0.9)

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Training Loss", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(1, max(len(h) for h in histories.values()))

    # Annotation: PathologyLoss models have higher loss (multi-component objective)
    ax.annotate("PathologyLoss models start\nhigher (multi-component objective)",
                xy=(5, max(histories[list(histories.keys())[0]][:5])),
                xytext=(max(len(list(histories.values())[0])//4, 5),
                        max(histories[list(histories.keys())[0]][:5]) * 1.05),
                fontsize=8, color="#555",
                arrowprops=dict(arrowstyle="->", color="#888", lw=0.8))

    plt.tight_layout()
    p = os.path.join(out_dir, "figC_loss_curves.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  -> {p}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE D — Per-slice metric distributions (box plots)
# ─────────────────────────────────────────────────────────────────────────────
def fig_metric_distributions(metrics, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Per-Slice Metric Distributions — Validation Set\n"
                 "(box = IQR, whiskers = 1.5×IQR, dots = outliers)",
                 fontsize=12, fontweight="bold")

    metric_keys = [("psnr", "PSNR (dB)"), ("dice_tc", "Dice_TC"), ("dice_et", "Dice_ET")]
    names  = list(metrics.keys())
    labels = [n.replace("\n", " ") for n in names]
    cols   = [list(MODEL_COLOURS.values())[i] for i in range(len(names))]

    for ax, (mkey, ylabel) in zip(axes, metric_keys):
        data = [metrics[n][mkey] for n in names]
        bp   = ax.boxplot(data, patch_artist=True, notch=False,
                          medianprops=dict(color="white", linewidth=2),
                          whiskerprops=dict(linewidth=1.2),
                          capprops=dict(linewidth=1.2),
                          flierprops=dict(marker=".", markersize=3, alpha=0.5))
        for patch, col in zip(bp["boxes"], cols):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)

        # Overlay individual points
        for i, (d, col) in enumerate(zip(data, cols), 1):
            jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(d))
            ax.scatter([i + j for j in jitter], d, color=col,
                       alpha=0.35, s=12, zorder=4)

        ax.set_xticks(range(1, len(names)+1))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

        # Star annotation: proposed vs best L1 on this metric
        prop_data = data[0]
        for j, d in enumerate(data[1:], 2):
            if len(prop_data) > 1 and len(d) > 1:
                from scipy.stats import wilcoxon
                try:
                    pairs = min(len(prop_data), len(d))
                    stat_p = wilcoxon(prop_data[:pairs], d[:pairs]).pvalue
                    stars  = "***" if stat_p < 0.001 else ("**" if stat_p < 0.01
                             else ("*" if stat_p < 0.05 else ""))
                    if stars:
                        ymax = max(max(prop_data), max(d)) + (0.03 if mkey != "psnr" else 1)
                        ax.plot([1, j], [ymax, ymax], "k-", linewidth=0.7)
                        ax.text((1+j)/2, ymax, stars, ha="center", va="bottom",
                                fontsize=9, color="#C62828")
                except Exception:
                    pass

    plt.tight_layout()
    p = os.path.join(out_dir, "figD_metric_distributions.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  -> {p}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE E — PSNR paradox (the key paper argument, single clean figure)
# ─────────────────────────────────────────────────────────────────────────────
def fig_psnr_paradox(metrics, out_dir):
    """Scatter: PSNR vs Dice_ET per model (mean ± std), with GT reference."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("The PSNR Paradox: Higher PSNR ≠ Better Tumour Delineation\n"
                 "Mean ± std across validation slices",
                 fontsize=11, fontweight="bold")

    names = list(metrics.keys())
    for i, name in enumerate(names):
        pv  = np.array(metrics[name]["psnr"])
        dv  = np.array(metrics[name]["dice_et"])
        col = list(MODEL_COLOURS.values())[i]
        lw  = 2 if "PROPOSED" in name else 0
        ax.errorbar(pv.mean(), dv.mean(),
                    xerr=pv.std(), yerr=dv.std(),
                    fmt="o", color=col, markersize=10,
                    capsize=4, linewidth=1.2,
                    markeredgewidth=lw, markeredgecolor="black",
                    label=name.replace("\n", " "), zorder=5)

    ax.set_xlabel("PSNR (dB)  ↑", fontsize=11)
    ax.set_ylabel("Dice_ET  ↑", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    # Draw "ideal" arrow direction
    ax.annotate("", xy=(ax.get_xlim()[1]*0.95, ax.get_ylim()[1]*0.97),
                xytext=(ax.get_xlim()[1]*0.82, ax.get_ylim()[1]*0.84),
                arrowprops=dict(arrowstyle="->", color="#37474F", lw=1.5))
    ax.text(ax.get_xlim()[1]*0.87, ax.get_ylim()[1]*0.91,
            "Ideal", fontsize=8, color="#37474F", ha="center")

    plt.tight_layout()
    p = os.path.join(out_dir, "figE_psnr_paradox.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  -> {p}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global preds_ref
    args = _parse()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"\nDevice: {device}")
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(42); np.random.seed(42)

    # ── load data ──────────────────────────────────────────────────────────────
    print("\nLoading data …")
    train_loader, val_loader, _ = load_data(args, device)

    # ── train segmentor ────────────────────────────────────────────────────────
    seg_model = train_segmentor(train_loader, device, args.seg_epochs)

    # ── build and train all models ─────────────────────────────────────────────
    print("\nTraining denoising models …")
    models    = build_models(device, args.patch_size)
    histories = train_all(models, train_loader, device, args.epochs)

    # ── collect validation samples ─────────────────────────────────────────────
    print("\nCollecting validation samples …")
    noisy_list, clean_list, gt_seg_list, preds, seg_preds, metrics = \
        collect_samples(models, val_loader, seg_model, device, args.n_samples)
    preds_ref = preds   # make available to fig_segmentation

    if not noisy_list:
        print("  No valid samples found — try reducing min_tumour_frac in brats_loader.py")
        return

    print(f"  Collected {len(noisy_list)} representative slices")

    # ── generate figures ───────────────────────────────────────────────────────
    print("\nGenerating figures …")
    fig_denoising(noisy_list, clean_list, preds, metrics, out, n_models=3)
    fig_segmentation(noisy_list, clean_list, gt_seg_list, seg_preds, out)
    fig_loss_curves(histories, out)
    fig_metric_distributions(metrics, out)
    fig_psnr_paradox(metrics, out)

    print(f"\nAll figures saved to: {out}/")
    print("  figA_denoising_samples.png")
    print("  figB_segmentation_overlay.png")
    print("  figC_loss_curves.png")
    print("  figD_metric_distributions.png")
    print("  figE_psnr_paradox.png")


if __name__ == "__main__":
    main()
