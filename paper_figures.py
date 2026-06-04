"""
paper_figures.py  —  fast paper-ready figures
==============================================

Metric figures (no training, reads from pre-computed CSVs):
  Fig D — Bar chart of Dice_WT / Dice_TC / Dice_ET for all 5 models
  Fig E — PSNR vs Dice_ET scatter (the PSNR paradox)
  Fig S — Significance summary table

Visual figures (optional, needs 2-5 BraTS subjects, ~5-min quick train):
  Fig A — Denoising: Noisy | PP-MAE | SwinIR-L1 | Uformer-L1 | GT
  Fig B — Segmentation overlay: GT vs PP-MAE vs SwinIR-L1

Usage — metric figures only (instant, no data dir needed):
    python3 paper_figures.py --out paper_figs/

Usage — add visual proof with 3 BraTS subjects:
    python3 paper_figures.py ~/Downloads/BraTS2021_data \\
        --device mps --n_samples 3 --out paper_figs/
"""

from __future__ import annotations
import argparse, os, sys, csv
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "pp_mae"))

# ── canonical results (from results/round4_mps/options_results.csv) ──────────
RESULTS = {
    "PP-MAE (Swin)\n[PROPOSED]":  dict(PSNR=28.0622, SSIM=0.9565, Dice_WT=0.8788, Dice_TC=0.8383, Dice_ET=0.7903),
    "SwinIR-lite\n(L1)":          dict(PSNR=31.4469, SSIM=0.9796, Dice_WT=0.8788, Dice_TC=0.7943, Dice_ET=0.7601),
    "Uformer-lite\n(L1)":         dict(PSNR=31.6952, SSIM=0.9801, Dice_WT=0.8819, Dice_TC=0.8101, Dice_ET=0.7707),
    "SwinIR +\nPathologyLoss":    dict(PSNR=31.1997, SSIM=0.9784, Dice_WT=0.8811, Dice_TC=0.7670, Dice_ET=0.7198),
    "Uformer +\nPathologyLoss":   dict(PSNR=31.6827, SSIM=0.9802, Dice_WT=0.8836, Dice_TC=0.8258, Dice_ET=0.7780),
}

# significance stars vs PP-MAE (Swin) [PROPOSED]
SIG = {
    ("SwinIR-lite\n(L1)",       "Dice_TC"): "***",
    ("SwinIR-lite\n(L1)",       "Dice_ET"): "***",
    ("Uformer-lite\n(L1)",      "Dice_TC"): "***",
    ("Uformer-lite\n(L1)",      "Dice_ET"): "*",
    ("SwinIR +\nPathologyLoss", "Dice_TC"): "***",
    ("SwinIR +\nPathologyLoss", "Dice_ET"): "***",
    ("Uformer +\nPathologyLoss","Dice_TC"): "ns",
    ("Uformer +\nPathologyLoss","Dice_ET"): "**",
}

METHODS   = list(RESULTS.keys())
COLOURS   = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
PROPOSED  = "PP-MAE (Swin)\n[PROPOSED]"

# BraTS seg colour map: 0=BG, 1=NCR(blue), 2=ED(green), 3=ET(red)
SEG_CMAP  = ListedColormap(["black", "blue", "lime", "red"])

# ─────────────────────────────────────────────────────────────────────────────
# Fig D — Dice bar chart
# ─────────────────────────────────────────────────────────────────────────────
def fig_dice_bars(out_dir: str):
    metrics = ["Dice_WT", "Dice_TC", "Dice_ET"]
    labels  = ["Whole Tumour", "Tumour Core", "Enhancing Tumour"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    fig.suptitle("Segmentation Dice Scores — Round 4 (BraTS 2021, n=50)", fontsize=13, fontweight="bold")

    x = np.arange(len(METHODS))
    w = 0.6

    for ax, met, lab in zip(axes, metrics, labels):
        vals = [RESULTS[m][met] for m in METHODS]
        bars = ax.bar(x, vals, width=w, color=COLOURS, edgecolor="black", linewidth=0.6)

        # bold outline on proposed
        bars[0].set_linewidth(2.2)
        bars[0].set_edgecolor("black")

        # add value labels
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.002, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold" if v == max(vals) else "normal")

        # significance stars above baseline bars vs proposed
        proposed_val = RESULTS[PROPOSED][met]
        for i, m in enumerate(METHODS[1:], 1):
            star = SIG.get((m, met), "")
            if star and star != "ns":
                ymax = max(vals[i], proposed_val)
                ax.annotate("", xy=(x[i], ymax + 0.012), xytext=(x[0], ymax + 0.012),
                            arrowprops=dict(arrowstyle="-", color="grey", lw=0.8))
                ax.text((x[0] + x[i]) / 2, ymax + 0.014, star,
                        ha="center", va="bottom", fontsize=8, color="black")

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("\n", "\n") for m in METHODS], fontsize=8)
        ax.set_ylabel("Dice Score")
        ax.set_title(lab, fontsize=11)
        lo = min(vals) - 0.03
        ax.set_ylim(lo, max(vals) + 0.06)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(out_dir, "figD_dice_bars.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig E — PSNR paradox scatter
# ─────────────────────────────────────────────────────────────────────────────
def fig_psnr_paradox(out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (m, c) in enumerate(zip(METHODS, COLOURS)):
        r = RESULTS[m]
        ax.scatter(r["PSNR"], r["Dice_ET"], s=180, color=c, zorder=3,
                   edgecolors="black", linewidths=1.5,
                   marker="*" if m == PROPOSED else "o")
        offset_x = 0.1 if m != PROPOSED else -0.3
        offset_y = 0.003 if i % 2 == 0 else -0.006
        ax.annotate(m.replace("\n", " "), (r["PSNR"] + offset_x, r["Dice_ET"] + offset_y),
                    fontsize=8, ha="left" if m != PROPOSED else "right")

    # arrow annotation: "higher PSNR ≠ better tumour detection"
    ax.annotate("Higher PSNR ≠\nbetter tumour detection",
                xy=(31.2, 0.772), fontsize=9, color="grey",
                style="italic",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="grey", alpha=0.8))

    ax.set_xlabel("Mean PSNR (dB)", fontsize=11)
    ax.set_ylabel("Mean Dice — Enhancing Tumour (ET)", fontsize=11)
    ax.set_title("PSNR Paradox: Proposed Model Sacrifices PSNR for Tumour Sensitivity",
                 fontsize=11, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.35)

    legend_els = [mpatches.Patch(color=c, label=m.replace("\n", " "))
                  for m, c in zip(METHODS, COLOURS)]
    ax.legend(handles=legend_els, fontsize=8, loc="lower right")

    plt.tight_layout()
    path = os.path.join(out_dir, "figE_psnr_paradox.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig S — Significance summary table
# ─────────────────────────────────────────────────────────────────────────────
def fig_significance_table(out_dir: str):
    rows = [
        ["PP-MAE vs SwinIR-L1",        "Dice_TC", "+0.0440", "4.37e-10", "***"],
        ["PP-MAE vs SwinIR-L1",        "Dice_ET", "+0.0302", "1.94e-04", "***"],
        ["PP-MAE vs Uformer-L1",       "Dice_TC", "+0.0282", "1.37e-05", "***"],
        ["PP-MAE vs Uformer-L1",       "Dice_ET", "+0.0196", "1.61e-02", "*"],
        ["PP-MAE vs SwinIR+PathLoss",  "Dice_TC", "+0.0713", "1.93e-05", "***"],
        ["PP-MAE vs SwinIR+PathLoss",  "Dice_ET", "+0.0705", "4.93e-04", "***"],
        ["PP-MAE vs Uformer+PathLoss", "Dice_TC", "+0.0125", "7.91e-02", "ns"],
        ["PP-MAE vs Uformer+PathLoss", "Dice_ET", "+0.0123", "3.16e-03", "**"],
    ]
    col_headers = ["Comparison", "Region", "Δ Mean", "p-value (Wilcoxon)", "Sig."]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis("off")

    star_colours = {"***": "#1a7a1a", "**": "#5577cc", "*": "#cc8800", "ns": "#888888"}
    cell_colours = []
    for row in rows:
        row_c = ["white"] * 4 + [star_colours.get(row[4], "white")]
        cell_colours.append(row_c)

    tbl = ax.table(
        cellText=rows,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
        cellColours=cell_colours,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")

    ax.set_title("Statistical Significance — PP-MAE vs Baselines (Wilcoxon signed-rank)",
                 fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "figS_significance_table.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Visual figures — only run when data_dir is provided
# ─────────────────────────────────────────────────────────────────────────────
def _load_brats_samples(data_dir: str, n_subjects: int, device):
    from brats_loader import BraTSDataset
    ds = BraTSDataset(data_dir, max_subjects=n_subjects, mode="val")
    samples = []
    for idx in range(len(ds)):
        item = ds[idx]
        # item: dict with "input" (C,H,W), "target" (C,H,W), "seg" (H,W)
        inp   = item["input"].unsqueeze(0).to(device)   # (1,C,H,W)
        tgt   = item["target"].unsqueeze(0).to(device)
        seg   = item["seg"]                              # (H,W) numpy or tensor
        if isinstance(seg, torch.Tensor):
            seg = seg.numpy()
        et_frac = (seg == 3).mean()
        if et_frac > 0.003:                              # only slices with visible ET
            samples.append(dict(inp=inp, tgt=tgt, seg=seg))
        if len(samples) >= 3:
            break
    if not samples:                                      # fallback: take first 3
        for idx in range(min(3, len(ds))):
            item = ds[idx]
            inp = item["input"].unsqueeze(0).to(device)
            tgt = item["target"].unsqueeze(0).to(device)
            seg = item["seg"]
            if isinstance(seg, torch.Tensor):
                seg = seg.numpy()
            samples.append(dict(inp=inp, tgt=tgt, seg=seg))
    return samples


def _quick_train(model, samples, epochs: int, device):
    """Train model for a handful of epochs on the provided samples (for visuals only)."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.L1Loss()
    model.train()
    for _ in range(epochs):
        for s in samples:
            opt.zero_grad()
            out = model(s["inp"])
            if isinstance(out, (tuple, list)):
                out = out[0]
            loss = crit(out, s["tgt"])
            loss.backward()
            opt.step()
    model.eval()
    return model


def _infer(model, inp):
    with torch.no_grad():
        out = model(inp)
        if isinstance(out, (tuple, list)):
            out = out[0]
    return out.squeeze(0).cpu().numpy()   # (C,H,W)


def _psnr_np(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))


def fig_denoising_samples(samples, out_dir: str, device, quick_epochs: int = 8):
    from option4_swin_pp_mae import SwinPPMAE
    from option_baselines    import SwinIRLite, UformerLite

    in_ch  = samples[0]["inp"].shape[1]
    out_ch = samples[0]["tgt"].shape[1]

    models = {
        "PP-MAE\n[PROPOSED]": SwinPPMAE(in_channels=in_ch, out_channels=out_ch).to(device),
        "SwinIR-L1":          SwinIRLite(in_channels=in_ch, out_channels=out_ch).to(device),
        "Uformer-L1":         UformerLite(in_channels=in_ch, out_channels=out_ch).to(device),
    }

    print("  Quick-training visual models …")
    for name, mdl in models.items():
        print(f"    {name.splitlines()[0]} …", end=" ", flush=True)
        _quick_train(mdl, samples, quick_epochs, device)
        print("done")

    n_samples = len(samples)
    col_names = ["Noisy Input"] + list(models.keys()) + ["Ground Truth"]
    n_cols = len(col_names)

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(3.2 * n_cols, 3.2 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row, s in enumerate(samples):
        noisy = s["inp"].squeeze(0).cpu().numpy()  # (C,H,W)
        gt    = s["tgt"].squeeze(0).cpu().numpy()
        t1ce_ch = min(1, noisy.shape[0] - 1)       # use T1ce channel (idx 1) or fallback

        outputs = {name: _infer(mdl, s["inp"]) for name, mdl in models.items()}

        panels = [noisy] + [outputs[k] for k in models] + [gt]
        for col, (panel, cname) in enumerate(zip(panels, col_names)):
            ax = axes[row, col]
            img = panel[t1ce_ch]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)

            if row == 0:
                ax.set_title(cname, fontsize=9, fontweight="bold" if "PROPOSED" in cname else "normal")

            if col > 0 and col < n_cols - 1:
                p = _psnr_np(panel, gt)
                ax.set_xlabel(f"PSNR {p:.1f} dB", fontsize=7.5)

            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Denoising Comparison — T1ce Channel (illustrative samples)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "figA_denoising_samples.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return models   # reuse for segmentation figure


def fig_segmentation_overlay(samples, models_dict, out_dir: str, device, quick_epochs: int = 5):
    from segmentor import UNetSegmentor

    in_ch  = samples[0]["inp"].shape[1]
    out_ch = samples[0]["tgt"].shape[1]

    seg_models = {
        name: UNetSegmentor(in_channels=out_ch, num_classes=4).to(device)
        for name in ["PP-MAE\n[PROPOSED]", "SwinIR-L1"]
    }

    # build pseudo-labelled data: denoise first, then use GT seg as label
    print("  Quick-training segmentation models …")
    seg_loss_fn = nn.CrossEntropyLoss()
    for sname, smodel in seg_models.items():
        denoiser = models_dict[sname]
        opt = torch.optim.Adam(smodel.parameters(), lr=1e-3)
        smodel.train()
        for _ in range(quick_epochs):
            for s in samples:
                with torch.no_grad():
                    denoised = denoiser(s["inp"])
                    if isinstance(denoised, (tuple, list)):
                        denoised = denoised[0]
                seg_gt = torch.from_numpy(s["seg"]).long().unsqueeze(0).to(device)
                opt.zero_grad()
                pred = smodel(denoised)
                loss = seg_loss_fn(pred, seg_gt)
                loss.backward()
                opt.step()
        smodel.eval()
        print(f"    {sname.splitlines()[0]} seg done")

    n_samples = len(samples)
    col_names = ["Input (T1ce)", "GT Segmentation", "PP-MAE Seg", "SwinIR-L1 Seg"]
    n_cols = len(col_names)

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(3.2 * n_cols, 3.2 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row, s in enumerate(samples):
        noisy = s["inp"].squeeze(0).cpu().numpy()
        t1ce  = noisy[min(1, noisy.shape[0]-1)]
        gt_seg = s["seg"]

        # get denoised + seg predictions
        preds = {}
        for sname, smodel in seg_models.items():
            denoiser = models_dict[sname]
            with torch.no_grad():
                denoised = denoiser(s["inp"])
                if isinstance(denoised, (tuple, list)):
                    denoised = denoised[0]
                seg_pred = smodel(denoised).argmax(dim=1).squeeze(0).cpu().numpy()
            preds[sname] = seg_pred

        panels = [
            ("Input (T1ce)", t1ce,      None),
            ("GT Seg",       t1ce,      gt_seg),
            ("PP-MAE Seg",   t1ce,      preds["PP-MAE\n[PROPOSED]"]),
            ("SwinIR-L1",    t1ce,      preds["SwinIR-L1"]),
        ]

        for col, (cname, bg, seg_mask) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(bg, cmap="gray", vmin=0, vmax=1)
            if seg_mask is not None:
                masked = np.ma.masked_where(seg_mask == 0, seg_mask)
                ax.imshow(masked, cmap=SEG_CMAP, vmin=0, vmax=3, alpha=0.55)
            if row == 0:
                ax.set_title(cname, fontsize=9, fontweight="bold" if "GT" in cname else "normal")
            ax.set_xticks([]); ax.set_yticks([])

    # legend
    legend_els = [
        mpatches.Patch(color="blue",  alpha=0.6, label="NCR (1)"),
        mpatches.Patch(color="lime",  alpha=0.6, label="ED (2)"),
        mpatches.Patch(color="red",   alpha=0.6, label="ET (3)"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Segmentation Overlay — GT vs Model Predictions (illustrative samples)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "figB_segmentation_overlay.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default=None,
                    help="BraTS2021 root dir (optional — only needed for visual figures A & B)")
    ap.add_argument("--device",    default="cpu", help="mps | cuda | cpu")
    ap.add_argument("--n_samples", type=int, default=3,
                    help="Number of BraTS subjects for visual figures (2-5)")
    ap.add_argument("--vis_epochs", type=int, default=8,
                    help="Quick-train epochs for visual figures (default 8, ~3 min on MPS)")
    ap.add_argument("--out",       default="paper_figs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    print("\n── Metric figures (from pre-computed CSV results) ──")
    fig_dice_bars(args.out)
    fig_psnr_paradox(args.out)
    fig_significance_table(args.out)

    if args.data_dir:
        print(f"\n── Visual figures ({args.n_samples} BraTS subjects, {args.vis_epochs} quick epochs) ──")
        samples = _load_brats_samples(args.data_dir, n_subjects=args.n_samples, device=device)
        print(f"  Loaded {len(samples)} slices with visible ET tumour")
        models = fig_denoising_samples(samples, args.out, device, quick_epochs=args.vis_epochs)
        fig_segmentation_overlay(samples, models, args.out, device, quick_epochs=args.vis_epochs // 2 + 1)
    else:
        print("\n  (Skipping visual figures — pass a BraTS data dir to generate Fig A & B)")

    print(f"\nDone. Figures saved to: {os.path.abspath(args.out)}/")


if __name__ == "__main__":
    main()
