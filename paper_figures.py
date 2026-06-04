"""
paper_figures.py  —  Complete Paper Visualization (Self-Contained)
===================================================================

ALL metric/statistical figures are built from hard-coded results —
no CSV files, no training, no external dependencies beyond matplotlib.

Visual figures (Fig 6 & 7) optionally load 2-5 real BraTS slices
for illustrative proof — quick 8-epoch pass, ~3-5 min on MPS.

Outputs
-------
  fig1_results_table.png    Full metrics table (all models, all metrics)
  fig2_dice_bars.png        Grouped Dice bar chart with significance brackets
  fig3_all_metrics.png      PSNR / SSIM / NRMSE comparison bars
  fig4_psnr_paradox.png     PSNR vs Dice_ET scatter — the key paper argument
  fig5_ablation.png         Component ablation: PathLoss alone hurts vs full PP-MAE
  fig6_significance.png     Wilcoxon signed-rank p-value table
  fig7_denoising.png        [needs --data_dir]  Denoising visual samples
  fig8_segmentation.png     [needs --data_dir]  Segmentation overlay samples

Usage
-----
Metric figures only (instant, no data needed):
    python3 paper_figures.py --out paper_figs/

All figures including visuals (~5 min on MPS):
    python3 paper_figures.py --data_dir ~/Downloads/BraTS2021_data \\
        --device mps --n_samples 3 --out paper_figs/
"""

from __future__ import annotations
import argparse, os, sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

# ══════════════════════════════════════════════════════════════════════════════
#  ALL RESULTS — hard-coded from BraTS 2021 MPS run (Round 4, 50 subjects,
#  30 denoising epochs + 20 segmentation epochs, Swin backbone)
# ══════════════════════════════════════════════════════════════════════════════
METHODS = [
    "PP-MAE (Swin) [PROPOSED]",
    "SwinIR-lite (L1)",
    "Uformer-lite (L1)",
    "SwinIR + PathologyLoss",
    "Uformer + PathologyLoss",
]

# Exact values from MacBook MPS run
RESULTS = {
    "PP-MAE (Swin) [PROPOSED]": {
        "PSNR": 28.0622, "SSIM": 0.9565, "NRMSE": 0.1369,
        "Dice_WT": 0.8788, "Dice_TC": 0.8383, "Dice_ET": 0.7903,
    },
    "SwinIR-lite (L1)": {
        "PSNR": 31.4469, "SSIM": 0.9796, "NRMSE": 0.0918,
        "Dice_WT": 0.8788, "Dice_TC": 0.7943, "Dice_ET": 0.7601,
    },
    "Uformer-lite (L1)": {
        "PSNR": 31.6952, "SSIM": 0.9801, "NRMSE": 0.0900,
        "Dice_WT": 0.8819, "Dice_TC": 0.8101, "Dice_ET": 0.7707,
    },
    "SwinIR + PathologyLoss": {
        "PSNR": 31.1997, "SSIM": 0.9784, "NRMSE": 0.0942,
        "Dice_WT": 0.8811, "Dice_TC": 0.7670, "Dice_ET": 0.7198,
    },
    "Uformer + PathologyLoss": {
        "PSNR": 31.6827, "SSIM": 0.9802, "NRMSE": 0.0899,
        "Dice_WT": 0.8836, "Dice_TC": 0.8258, "Dice_ET": 0.7780,
    },
}

# Wilcoxon signed-rank p-values (proposed vs each baseline)
SIGNIFICANCE = [
    ("PP-MAE (Swin) [PROPOSED]", "SwinIR-lite (L1)",      "Dice_WT", 0.8788, 0.8788, -0.0000, 9.17e-4,  "***"),
    ("PP-MAE (Swin) [PROPOSED]", "SwinIR-lite (L1)",      "Dice_TC", 0.8383, 0.7943, +0.0440, 4.37e-10, "***"),
    ("PP-MAE (Swin) [PROPOSED]", "SwinIR-lite (L1)",      "Dice_ET", 0.7903, 0.7601, +0.0302, 1.94e-4,  "***"),
    ("PP-MAE (Swin) [PROPOSED]", "Uformer-lite (L1)",     "Dice_WT", 0.8788, 0.8819, -0.0031, 4.21e-3,  "**"),
    ("PP-MAE (Swin) [PROPOSED]", "Uformer-lite (L1)",     "Dice_TC", 0.8383, 0.8101, +0.0282, 1.37e-5,  "***"),
    ("PP-MAE (Swin) [PROPOSED]", "Uformer-lite (L1)",     "Dice_ET", 0.7903, 0.7707, +0.0196, 1.61e-2,  "*"),
    ("PP-MAE (Swin) [PROPOSED]", "SwinIR + PathologyLoss","Dice_WT", 0.8788, 0.8811, -0.0023, 1.57e-8,  "***"),
    ("PP-MAE (Swin) [PROPOSED]", "SwinIR + PathologyLoss","Dice_TC", 0.8383, 0.7670, +0.0713, 1.93e-5,  "***"),
    ("PP-MAE (Swin) [PROPOSED]", "SwinIR + PathologyLoss","Dice_ET", 0.7903, 0.7198, +0.0705, 4.93e-4,  "***"),
    ("PP-MAE (Swin) [PROPOSED]", "Uformer + PathologyLoss","Dice_WT",0.8788, 0.8836, -0.0048, 5.72e-4,  "***"),
    ("PP-MAE (Swin) [PROPOSED]", "Uformer + PathologyLoss","Dice_TC",0.8383, 0.8258, +0.0125, 7.91e-2,  "ns"),
    ("PP-MAE (Swin) [PROPOSED]", "Uformer + PathologyLoss","Dice_ET",0.7903, 0.7780, +0.0123, 3.16e-3,  "**"),
]

COLOURS = {
    "PP-MAE (Swin) [PROPOSED]": "#1f77b4",
    "SwinIR-lite (L1)":         "#ff7f0e",
    "Uformer-lite (L1)":        "#2ca02c",
    "SwinIR + PathologyLoss":   "#d62728",
    "Uformer + PathologyLoss":  "#9467bd",
}

SHORT = {
    "PP-MAE (Swin) [PROPOSED]": "PP-MAE\n[PROPOSED]",
    "SwinIR-lite (L1)":         "SwinIR\n-lite (L1)",
    "Uformer-lite (L1)":        "Uformer\n-lite (L1)",
    "SwinIR + PathologyLoss":   "SwinIR\n+PathLoss",
    "Uformer + PathologyLoss":  "Uformer\n+PathLoss",
}

PROPOSED = "PP-MAE (Swin) [PROPOSED]"

# significance stars lookup: (baseline, metric) → star
SIG_LOOKUP = {(r[1], r[2]): r[8] for r in SIGNIFICANCE}

SEG_CMAP = ListedColormap(["black", "blue", "lime", "red"])


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  {path}")


def _best(metric: str, higher_is_better: bool = True) -> str:
    vals = {m: RESULTS[m][metric] for m in METHODS}
    return max(vals, key=vals.get) if higher_is_better else min(vals, key=vals.get)


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 1 — Full results table
# ══════════════════════════════════════════════════════════════════════════════
def fig1_results_table(out: str):
    metrics = ["PSNR", "SSIM", "NRMSE", "Dice_WT", "Dice_TC", "Dice_ET"]
    best_high = {m: _best(m, True)  for m in ["PSNR", "SSIM", "Dice_WT", "Dice_TC", "Dice_ET"]}
    best_low  = {m: _best(m, False) for m in ["NRMSE"]}

    rows, cell_colors = [], []
    for method in METHODS:
        row, rcolors = [], []
        for met in metrics:
            v = RESULTS[method][met]
            row.append(f"{v:.4f}")
            is_best = (met in best_high and best_high[met] == method) or \
                      (met in best_low  and best_low[met]  == method)
            is_proposed = (method == PROPOSED)
            if is_best and is_proposed:
                rcolors.append("#d4edda")   # green — proposed AND best
            elif is_best:
                rcolors.append("#fff3cd")   # yellow — best but not proposed
            elif is_proposed:
                rcolors.append("#cce5ff")   # blue — proposed
            else:
                rcolors.append("white")
        rows.append(row)
        cell_colors.append(rcolors)

    short_methods = [SHORT[m] for m in METHODS]
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        rowLabels=short_methods,
        colLabels=metrics,
        cellColours=cell_colors,
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # header row style
    for c in range(len(metrics)):
        tbl[0, c].set_facecolor("#2c3e50")
        tbl[0, c].set_text_props(color="white", fontweight="bold")
    # row label style
    for r in range(1, len(METHODS) + 1):
        tbl[r, -1].set_facecolor(COLOURS[METHODS[r-1]])
        tbl[r, -1].set_text_props(color="white", fontweight="bold", fontsize=8)

    ax.set_title(
        "Table I — Quantitative Results: BraTS 2021 (Round 4, n=50 subjects, MPS GPU)\n"
        "Green = proposed model best  |  Yellow = baseline best  |  Blue = proposed",
        fontsize=9, pad=10,
    )
    _save(fig, os.path.join(out, "fig1_results_table.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 2 — Dice grouped bar chart
# ══════════════════════════════════════════════════════════════════════════════
def fig2_dice_bars(out: str):
    dice_metrics = ["Dice_WT", "Dice_TC", "Dice_ET"]
    labels       = ["Whole Tumour (WT)", "Tumour Core (TC)", "Enhancing Tumour (ET)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(
        "Dice Score Comparison — PP-MAE vs Baselines\n"
        "(BraTS 2021, Round 4, n=50, Wilcoxon signed-rank; *** p<0.001, ** p<0.01, * p<0.05)",
        fontsize=11, fontweight="bold",
    )

    x = np.arange(len(METHODS))
    for ax, met, lab in zip(axes, dice_metrics, labels):
        vals  = [RESULTS[m][met] for m in METHODS]
        cols  = [COLOURS[m] for m in METHODS]
        bars  = ax.bar(x, vals, color=cols, edgecolor="black", linewidth=0.7, width=0.65)
        bars[0].set_linewidth(2.5)

        for b, v, m in zip(bars, vals, METHODS):
            weight = "bold" if v == max(vals) else "normal"
            ax.text(b.get_x() + b.get_width()/2, v + 0.001, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=7, fontweight=weight)

        # significance brackets vs proposed
        prop_v = RESULTS[PROPOSED][met]
        for i, m in enumerate(METHODS):
            if m == PROPOSED:
                continue
            star = SIG_LOOKUP.get((m, met), "")
            if not star or star == "ns":
                continue
            top = max(vals[i], prop_v) + 0.015
            ax.annotate("", xy=(x[i], top), xytext=(x[0], top),
                        arrowprops=dict(arrowstyle="-", color="#555", lw=0.8))
            ax.text((x[0]+x[i])/2, top+0.002, star,
                    ha="center", va="bottom", fontsize=9, color="#333")

        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in METHODS], fontsize=8)
        ax.set_ylabel("Dice Score")
        ax.set_title(lab, fontsize=11, fontweight="bold")
        lo = min(vals) - 0.035
        ax.set_ylim(lo, max(vals) + 0.07)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    legend_els = [mpatches.Patch(color=COLOURS[m], label=m) for m in METHODS]
    fig.legend(handles=legend_els, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    _save(fig, os.path.join(out, "fig2_dice_bars.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 3 — PSNR / SSIM / NRMSE bars
# ══════════════════════════════════════════════════════════════════════════════
def fig3_image_quality_bars(out: str):
    panels = [
        ("PSNR", "PSNR (dB) ↑ higher is better",         True),
        ("SSIM", "SSIM ↑ higher is better",               True),
        ("NRMSE","NRMSE ↓ lower is better (image error)", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Image Reconstruction Quality — PP-MAE vs Baselines",
                 fontsize=12, fontweight="bold")

    x = np.arange(len(METHODS))
    for ax, (met, ylab, high) in zip(axes, panels):
        vals = [RESULTS[m][met] for m in METHODS]
        best_val = max(vals) if high else min(vals)
        cols = [COLOURS[m] for m in METHODS]
        bars = ax.bar(x, vals, color=cols, edgecolor="black", linewidth=0.7, width=0.65)
        bars[0].set_linewidth(2.5)

        for b, v in zip(bars, vals):
            weight = "bold" if v == best_val else "normal"
            ax.text(b.get_x() + b.get_width()/2, v * 1.002, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=7.5, fontweight=weight)

        # Proposed arrow annotation
        ax.annotate("↑ Proposed\n(PP-MAE)",
                    xy=(x[0], vals[0]),
                    xytext=(x[0], vals[0] * 0.96 if high else vals[0] * 1.04),
                    ha="center", fontsize=7, color="#1f77b4",
                    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2))

        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in METHODS], fontsize=8)
        ax.set_title(ylab, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        lo = min(vals) * 0.97
        hi_lim = max(vals) * 1.04
        ax.set_ylim(lo, hi_lim)

    legend_els = [mpatches.Patch(color=COLOURS[m], label=m) for m in METHODS]
    fig.legend(handles=legend_els, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    _save(fig, os.path.join(out, "fig3_image_quality.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 4 — PSNR paradox scatter
# ══════════════════════════════════════════════════════════════════════════════
def fig4_psnr_paradox(out: str):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    offsets = {
        PROPOSED:                    (-0.45,  0.002),
        "SwinIR-lite (L1)":          ( 0.08,  0.001),
        "Uformer-lite (L1)":         ( 0.08, -0.004),
        "SwinIR + PathologyLoss":    ( 0.08,  0.002),
        "Uformer + PathologyLoss":   ( 0.08,  0.001),
    }

    for m in METHODS:
        r  = RESULTS[m]
        ox, oy = offsets[m]
        mk = "*" if m == PROPOSED else "o"
        sz = 280 if m == PROPOSED else 150
        ax.scatter(r["PSNR"], r["Dice_ET"], s=sz, color=COLOURS[m],
                   marker=mk, edgecolors="black", linewidths=1.5, zorder=4)
        ax.annotate(m, (r["PSNR"] + ox, r["Dice_ET"] + oy),
                    fontsize=8.5,
                    fontweight="bold" if m == PROPOSED else "normal",
                    ha="right" if m == PROPOSED else "left",
                    color=COLOURS[m])

    # trend annotation
    ax.annotate(
        "PathologyLoss alone\nhurts Dice_ET by 5.3%\n(SwinIR-L1 → SwinIR+PathLoss)",
        xy=(31.20, 0.7198), xytext=(29.8, 0.725),
        fontsize=8, color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="#d62728", alpha=0.9),
    )
    ax.annotate(
        "Full PP-MAE:\nlowest PSNR but\nhighest Dice_ET",
        xy=(28.06, 0.7903), xytext=(28.9, 0.785),
        fontsize=8, color="#1f77b4",
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="#e8f4fd", ec="#1f77b4", alpha=0.9),
    )

    ax.set_xlabel("Mean PSNR (dB)  [image reconstruction quality]", fontsize=11)
    ax.set_ylabel("Mean Dice — Enhancing Tumour (ET)  [clinical utility]", fontsize=11)
    ax.set_title(
        "The PSNR Paradox: Higher Reconstruction Quality ≠ Better Tumour Detection\n"
        "PP-MAE sacrifices PSNR to optimise tumour-region sensitivity",
        fontsize=11, fontweight="bold",
    )
    ax.grid(linestyle="--", alpha=0.35)
    legend_els = [mpatches.Patch(color=COLOURS[m], label=m) for m in METHODS]
    ax.legend(handles=legend_els, fontsize=8.5, loc="lower left",
              framealpha=0.9, edgecolor="grey")
    plt.tight_layout()
    _save(fig, os.path.join(out, "fig4_psnr_paradox.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 5 — Ablation: component contribution
# ══════════════════════════════════════════════════════════════════════════════
def fig5_ablation(out: str):
    """
    Key ablation story:
      PathologyLoss ALONE hurts Dice_ET (SwinIR-L1 → SwinIR+PathLoss: -5.3%)
      Only the FULL PP-MAE (cross-modal + saliency + PathologyLoss) wins (+3.9%)
    """
    ablation_steps = [
        ("SwinIR-L1\n(Backbone only)", 0.7601, "#ff7f0e"),
        ("+ PathologyLoss\n(No arch change)", 0.7198, "#d62728"),
        ("PP-MAE Full\n(All components)", 0.7903, "#1f77b4"),
    ]
    ablation_swin = ablation_steps

    ablation_uformer = [
        ("Uformer-L1\n(Backbone only)", 0.7707, "#2ca02c"),
        ("+ PathologyLoss\n(No arch change)", 0.7780, "#9467bd"),
        ("PP-MAE Full\n(All components)", 0.7903, "#1f77b4"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(
        "Ablation Study — Enhancing Tumour (ET) Dice Score\n"
        "PathologyLoss alone is insufficient; full PP-MAE architecture required",
        fontsize=11, fontweight="bold",
    )

    for ax, ablation, backbone in zip(axes, [ablation_swin, ablation_uformer],
                                       ["SwinIR Backbone", "Uformer Backbone"]):
        labels = [a[0] for a in ablation]
        vals   = [a[1] for a in ablation]
        cols   = [a[2] for a in ablation]
        x = np.arange(len(ablation))
        bars = ax.bar(x, vals, color=cols, edgecolor="black", linewidth=0.8, width=0.55)
        bars[-1].set_linewidth(2.5)  # highlight proposed

        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.001, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        # delta arrows
        for i in range(1, len(ablation)):
            prev_v = vals[i-1]
            curr_v = vals[i]
            delta  = curr_v - prev_v
            sign   = "+" if delta >= 0 else ""
            col    = "green" if delta >= 0 else "red"
            mid_x  = (x[i-1] + x[i]) / 2
            ax.annotate(f"{sign}{delta:.4f}",
                        xy=(mid_x, max(prev_v, curr_v) + 0.008),
                        ha="center", fontsize=9, color=col, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.8))

        # shaded danger zone (PathologyLoss alone worse than baseline)
        if backbone == "SwinIR Backbone":
            ax.axhspan(vals[1], vals[0], alpha=0.08, color="red",
                       label="PathLoss hurts without arch support")
            ax.legend(fontsize=8, loc="lower right")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Dice — Enhancing Tumour (ET)")
        ax.set_title(backbone, fontsize=11, fontweight="bold")
        lo = min(vals) - 0.04
        ax.set_ylim(lo, max(vals) + 0.05)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    plt.tight_layout()
    _save(fig, os.path.join(out, "fig5_ablation.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 6 — Significance table
# ══════════════════════════════════════════════════════════════════════════════
def fig6_significance_table(out: str):
    rows = []
    for r in SIGNIFICANCE:
        _, baseline, region, m_prop, m_base, delta, pval, star = r
        sign = "+" if delta >= 0 else ""
        rows.append([
            f"PP-MAE vs {baseline}",
            region,
            f"{sign}{delta:.4f}",
            f"{pval:.2e}",
            star,
        ])

    col_headers = ["Comparison", "Region", "Δ (PP-MAE − Baseline)", "p-value (Wilcoxon)", "Significance"]
    star_bg = {"***": "#c3e6cb", "**": "#b8daff", "*": "#ffeeba", "ns": "#f8d7da"}

    cell_colours = []
    for row in rows:
        bg = star_bg.get(row[4], "white")
        cell_colours.append(["white", "white", "white", "white", bg])

    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=col_headers,
        cellColours=cell_colours,
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.55)

    for c in range(len(col_headers)):
        tbl[0, c].set_facecolor("#2c3e50")
        tbl[0, c].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        "Table II — Statistical Significance (Wilcoxon Signed-Rank, BraTS 2021 validation set)\n"
        "*** p < 0.001   ** p < 0.01   * p < 0.05   ns = not significant",
        fontsize=10, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    _save(fig, os.path.join(out, "fig6_significance_table.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  VISUAL FIGURES (optional — needs BraTS data dir)
# ══════════════════════════════════════════════════════════════════════════════
def _load_visual_samples(data_dir: str, n: int, device):
    import torch
    _DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_DIR, "pp_mae"))
    from brats_loader import BraTSDataset

    ds = BraTSDataset(data_dir, max_subjects=max(n * 3, 10), mode="val")
    samples = []
    for idx in range(len(ds)):
        item = ds[idx]
        inp = item["input"].unsqueeze(0).to(device)
        tgt = item["target"].unsqueeze(0).to(device)
        seg = item["seg"]
        if hasattr(seg, "numpy"):
            seg = seg.numpy()
        if (seg == 3).mean() > 0.003:
            samples.append({"inp": inp, "tgt": tgt, "seg": seg})
        if len(samples) >= n:
            break
    if not samples:
        for idx in range(min(n, len(ds))):
            item = ds[idx]
            inp = item["input"].unsqueeze(0).to(device)
            tgt = item["target"].unsqueeze(0).to(device)
            seg = item["seg"]
            if hasattr(seg, "numpy"):
                seg = seg.numpy()
            samples.append({"inp": inp, "tgt": tgt, "seg": seg})
    return samples


def _quick_train(model, samples, epochs, loss_fn=None):
    import torch, torch.nn as nn
    if loss_fn is None:
        loss_fn = nn.L1Loss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for s in samples:
            opt.zero_grad()
            out = model(s["inp"])
            if isinstance(out, (list, tuple)):
                out = out[0]
            loss = loss_fn(out, s["tgt"])
            loss.backward()
            opt.step()
    model.eval()
    return model


def _infer(model, inp):
    import torch
    with torch.no_grad():
        out = model(inp)
        if isinstance(out, (list, tuple)):
            out = out[0]
    return out.squeeze(0).cpu().numpy()


def _psnr(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    return 100.0 if mse < 1e-10 else 20 * np.log10(1.0 / np.sqrt(mse))


def _ssim(pred, gt):
    from skimage.metrics import structural_similarity
    ch = pred.shape[0]
    scores = [structural_similarity(pred[c], gt[c], data_range=1.0) for c in range(ch)]
    return np.mean(scores)


def fig7_denoising(samples, out: str, device, epochs: int = 8):
    import torch
    import torch.nn as nn
    _DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_DIR, "pp_mae"))
    from option4_swin_pp_mae import SwinPPMAE
    from option_baselines    import SwinIRLite, UformerLite

    in_ch  = samples[0]["inp"].shape[1]
    out_ch = samples[0]["tgt"].shape[1]

    model_defs = [
        ("PP-MAE [PROPOSED]", SwinPPMAE(in_channels=in_ch, out_channels=out_ch).to(device)),
        ("SwinIR-lite (L1)",  SwinIRLite(in_channels=in_ch, out_channels=out_ch).to(device)),
        ("Uformer-lite (L1)", UformerLite(in_channels=in_ch, out_channels=out_ch).to(device)),
    ]

    print("  Training denoising models (visual only) …")
    for name, mdl in model_defs:
        print(f"    {name} …", end=" ", flush=True)
        _quick_train(mdl, samples, epochs)
        print("done")

    col_labels = ["Noisy Input"] + [n for n, _ in model_defs] + ["Ground Truth"]
    n_rows = len(samples)
    n_cols = len(col_labels)
    t1ce = min(1, samples[0]["inp"].shape[1] - 1)   # T1ce channel index

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows + 0.6))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, s in enumerate(samples):
        noisy = s["inp"].squeeze(0).cpu().numpy()
        gt    = s["tgt"].squeeze(0).cpu().numpy()
        outs  = {name: _infer(mdl, s["inp"]) for name, mdl in model_defs}

        panels = [noisy] + [outs[n] for n, _ in model_defs] + [gt]
        for col, (panel, clabel) in enumerate(zip(panels, col_labels)):
            ax = axes[row, col]
            ax.imshow(panel[t1ce], cmap="gray", vmin=0, vmax=1)
            if row == 0:
                ax.set_title(clabel, fontsize=8.5,
                             fontweight="bold" if "PROPOSED" in clabel else "normal")
            if 0 < col < n_cols - 1:
                p = _psnr(panel, gt)
                ax.set_xlabel(f"PSNR {p:.2f} dB", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        "Fig 7 — Denoising Comparison (T1ce channel, illustrative samples)\n"
        "Quick-trained on 2-5 subjects for visual proof only",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, os.path.join(out, "fig7_denoising_samples.png"))
    return dict(model_defs)   # return trained denoisers for seg figure


def fig8_segmentation(samples, denoisers: dict, out: str, device, epochs: int = 5):
    import torch
    import torch.nn as nn
    _DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_DIR, "pp_mae"))
    from segmentor import UNetSegmentor

    out_ch = samples[0]["tgt"].shape[1]
    seg_loss = nn.CrossEntropyLoss()

    seg_models = {}
    print("  Training segmentation models (visual only) …")
    for name in ["PP-MAE [PROPOSED]", "SwinIR-lite (L1)"]:
        smodel = UNetSegmentor(in_channels=out_ch, num_classes=4).to(device)
        opt    = torch.optim.Adam(smodel.parameters(), lr=1e-3)
        denoiser = denoisers[name]
        smodel.train()
        for _ in range(epochs):
            for s in samples:
                import torch
                with torch.no_grad():
                    dn = denoiser(s["inp"])
                    if isinstance(dn, (list, tuple)):
                        dn = dn[0]
                seg_gt = torch.from_numpy(s["seg"]).long().unsqueeze(0).to(device)
                opt.zero_grad()
                pred = smodel(dn)
                loss = seg_loss(pred, seg_gt)
                loss.backward()
                opt.step()
        smodel.eval()
        seg_models[name] = smodel
        print(f"    {name} seg done")

    n_rows = len(samples)
    col_labels = ["T1ce Input", "GT Segmentation", "PP-MAE Seg", "SwinIR-L1 Seg"]
    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 3.2 * n_rows + 0.6))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    t1ce = min(1, samples[0]["inp"].shape[1] - 1)

    for row, s in enumerate(samples):
        inp_np = s["inp"].squeeze(0).cpu().numpy()
        bg     = inp_np[t1ce]
        gt_seg = s["seg"]

        preds = {}
        import torch
        for name, sm in seg_models.items():
            dn = denoisers[name]
            with torch.no_grad():
                out_dn = dn(s["inp"])
                if isinstance(out_dn, (list, tuple)):
                    out_dn = out_dn[0]
                p = sm(out_dn).argmax(dim=1).squeeze(0).cpu().numpy()
            preds[name] = p

        panels = [
            ("T1ce Input",      bg, None),
            ("GT Seg",          bg, gt_seg),
            ("PP-MAE Seg",      bg, preds["PP-MAE [PROPOSED]"]),
            ("SwinIR-L1 Seg",   bg, preds["SwinIR-lite (L1)"]),
        ]

        for col, (clabel, background, seg_mask) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(background, cmap="gray", vmin=0, vmax=1)
            if seg_mask is not None:
                masked = np.ma.masked_where(seg_mask == 0, seg_mask)
                ax.imshow(masked, cmap=SEG_CMAP, vmin=0, vmax=3, alpha=0.55)
            if row == 0:
                ax.set_title(clabel, fontsize=9,
                             fontweight="bold" if "GT" in clabel or "PP-MAE" in clabel else "normal")
            ax.set_xticks([]); ax.set_yticks([])

    legend_els = [
        mpatches.Patch(color="blue",  alpha=0.7, label="NCR / Necrosis (label 1)"),
        mpatches.Patch(color="lime",  alpha=0.7, label="Oedema / ED  (label 2)"),
        mpatches.Patch(color="red",   alpha=0.7, label="Enhancing Tumour / ET (label 3)"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        "Fig 8 — Segmentation Overlay: GT vs PP-MAE vs SwinIR-L1\n"
        "(Illustrative samples — quick-trained on 2-5 subjects for visual proof)",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, os.path.join(out, "fig8_segmentation_overlay.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Generate all paper figures")
    ap.add_argument("--data_dir",   default=None,
                    help="BraTS2021 root dir (only for visual Figs 7 & 8)")
    ap.add_argument("--device",     default="cpu", help="mps | cuda | cpu")
    ap.add_argument("--n_samples",  type=int, default=3,
                    help="BraTS slices for visual figures (2-5)")
    ap.add_argument("--vis_epochs", type=int, default=8,
                    help="Quick-train epochs for visual figures")
    ap.add_argument("--out",        default="paper_figs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("\n── Metric figures (pre-computed results, no training) ──")
    fig1_results_table(args.out)
    fig2_dice_bars(args.out)
    fig3_image_quality_bars(args.out)
    fig4_psnr_paradox(args.out)
    fig5_ablation(args.out)
    fig6_significance_table(args.out)

    if args.data_dir:
        import torch
        device = torch.device(args.device)
        print(f"\n── Visual figures ({args.n_samples} BraTS subjects, "
              f"{args.vis_epochs} quick epochs on {args.device}) ──")
        samples = _load_visual_samples(args.data_dir, args.n_samples, device)
        print(f"  Loaded {len(samples)} slices with visible ET tumour")
        denoisers = fig7_denoising(samples, args.out, device, epochs=args.vis_epochs)
        fig8_segmentation(samples, denoisers, args.out, device, epochs=max(3, args.vis_epochs // 2))
    else:
        print("\n  (Visual Figs 7 & 8 skipped — add --data_dir ~/Downloads/BraTS2021_data)")

    print(f"\n  All figures saved to: {os.path.abspath(args.out)}/\n")


if __name__ == "__main__":
    main()
