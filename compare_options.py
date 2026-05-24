"""
compare_options.py — Head-to-head comparison of all 4 PP-MAE options
=====================================================================
Reads options_results.csv and generates 4 dedicated figures:

  1. options_compare_radar.png   — Spider/radar chart (6 metrics, normalised)
  2. options_compare_bars.png    — Grouped bar chart (all 6 metrics side by side)
  3. options_compare_rank.png    — Per-metric ranking heatmap + scorecard
  4. options_compare_table.png   — Summary table with winner highlighted per metric

Usage
-----
  python3 compare_options.py
  python3 compare_options.py --csv path/to/options_results.csv --out ./out/
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import csv, argparse, os, sys

# ── CLI ───────────────────────────────────────────────────────────────────────
_p = argparse.ArgumentParser()
_p.add_argument('--csv', default=None)
_p.add_argument('--out', default=None)
_args = _p.parse_args()

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = _args.csv or os.path.join(_SCRIPT_DIR, 'options_results.csv')
OUT      = _args.out  or _SCRIPT_DIR
os.makedirs(OUT, exist_ok=True)

# ── Load the 4 PP-MAE option rows from CSV ────────────────────────────────────
PP_MAE_NAMES = {
    'PP-MAE CNN (clinical_risk)': 'Option 1\nCNN',
    'ViT PP-MAE 2D':              'Option 2\nViT',
    'PP-MAE Pipeline':            'Option 3\nPipeline',
    'Swin PP-MAE':                'Option 4\nSwin',
}

METRICS      = ['PSNR', 'SSIM', 'NRMSE', 'Dice_WT', 'Dice_TC', 'Dice_ET']
METRIC_LABEL = ['PSNR (dB)↑', 'SSIM↑', 'NRMSE↓', 'Dice WT↑', 'Dice TC↑', 'Dice ET↑']
# For NRMSE: lower is better → we invert for radar (plot 1-NRMSE)

data = {}   # {short_name: {metric: value}}
with open(CSV_PATH, newline='') as f:
    for row in csv.DictReader(f):
        if row['Method'] in PP_MAE_NAMES:
            short = PP_MAE_NAMES[row['Method']]
            data[short] = {m: float(row[m]) for m in METRICS}

OPTION_NAMES  = list(data.keys())           # ['Option 1\nCNN', ...]
FULL_NAMES    = list(PP_MAE_NAMES.keys())   # original CSV names

# Colour per option
COLORS = {
    'Option 1\nCNN':      '#1565C0',   # deep blue
    'Option 2\nViT':      '#6A1B9A',   # purple
    'Option 3\nPipeline': '#00695C',   # teal
    'Option 4\nSwin':     '#AD1457',   # deep pink
}

print("\n" + "="*60, flush=True)
print("  PP-MAE Options — Head-to-Head Comparison", flush=True)
print("="*60, flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Radar / Spider Chart
# ─────────────────────────────────────────────────────────────────────────────
# Normalise each metric to [0,1] so they fit on the same radar.
# NRMSE is inverted (1 - normalised) so that "outward = better" for all axes.

radar_metrics = ['PSNR', 'SSIM', 'Dice_WT', 'Dice_TC', 'Dice_ET', 'NRMSE']
radar_labels  = ['PSNR↑', 'SSIM↑', 'Dice WT↑', 'Dice TC↑', 'Dice ET↑', '1−NRMSE↑']

# Collect values and normalise
vals = {name: [] for name in OPTION_NAMES}
for m in radar_metrics:
    col = [data[n][m] for n in OPTION_NAMES]
    mn, mx = min(col), max(col)
    rng = mx - mn if mx != mn else 1.0
    for i, n in enumerate(OPTION_NAMES):
        v = (col[i] - mn) / rng          # 0..1
        if m == 'NRMSE':
            v = 1.0 - v                  # invert: lower NRMSE → closer to 1
        vals[n].append(v)

N = len(radar_metrics)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]   # close the polygon

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

for name in OPTION_NAMES:
    v = vals[name] + vals[name][:1]
    ax.plot(angles, v, 'o-', linewidth=2.5, color=COLORS[name],
            label=name.replace('\n', ' '))
    ax.fill(angles, v, alpha=0.08, color=COLORS[name])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, size=11, fontweight='bold')
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['25%', '50%', '75%', '100%'], size=8, color='grey')
ax.set_ylim(0, 1)
ax.grid(color='grey', alpha=0.3)

ax.set_title('PP-MAE Options — Radar Comparison\n(all metrics normalised to [0,1])',
             fontweight='bold', fontsize=13, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
          fontsize=10, framealpha=0.9)
plt.tight_layout()
path = os.path.join(OUT, 'options_compare_radar.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved options_compare_radar.png", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Grouped Bar Chart — all 6 metrics
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for ax, m, label in zip(axes, METRICS, METRIC_LABEL):
    vals_bar = [data[n][m] for n in OPTION_NAMES]
    colors   = [COLORS[n] for n in OPTION_NAMES]
    short    = [n.replace('\n', ' ') for n in OPTION_NAMES]
    bars = ax.bar(range(4), vals_bar, color=colors,
                  edgecolor='white', linewidth=1.2, width=0.6)

    # Annotate values
    for bar, v in zip(bars, vals_bar):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005 * max(vals_bar),
                f'{v:.3f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    # Highlight winner
    if m == 'NRMSE':
        winner_idx = int(np.argmin(vals_bar))
    else:
        winner_idx = int(np.argmax(vals_bar))
    bars[winner_idx].set_edgecolor('gold')
    bars[winner_idx].set_linewidth(3)

    ax.set_xticks(range(4))
    ax.set_xticklabels(short, fontsize=9)
    ax.set_title(label, fontweight='bold', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ypad = max(vals_bar) * 0.12
    ax.set_ylim(0, max(vals_bar) + ypad)

fig.suptitle(
    'PP-MAE Options 1–4 — All Metrics Head-to-Head\n'
    '(gold border = winner per metric  |  Demo: 6 subjects × 24 slices, 10 epochs)',
    fontweight='bold', fontsize=13)

legend_patches = [mpatches.Patch(color=COLORS[n], label=n.replace('\n', ' '))
                  for n in OPTION_NAMES]
fig.legend(handles=legend_patches, loc='lower center', ncol=4,
           fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))
plt.tight_layout(rect=[0, 0.04, 1, 1])
path = os.path.join(OUT, 'options_compare_bars.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved options_compare_bars.png", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Ranking Heatmap
# ─────────────────────────────────────────────────────────────────────────────
# rank_matrix[metric][option] = rank 1..4 (1=best)
rank_matrix = np.zeros((len(METRICS), 4), dtype=int)
for mi, m in enumerate(METRICS):
    vals_m = [data[n][m] for n in OPTION_NAMES]
    if m == 'NRMSE':
        order = np.argsort(vals_m)        # lower is better
    else:
        order = np.argsort(vals_m)[::-1]  # higher is better
    for rank, idx in enumerate(order):
        rank_matrix[mi, idx] = rank + 1

fig, ax = plt.subplots(figsize=(10, 5))
cmap = plt.cm.RdYlGn_r   # rank 1=green, rank 4=red
im = ax.imshow(rank_matrix, cmap=cmap, aspect='auto', vmin=1, vmax=4)

short_names = [n.replace('\n', ' ') for n in OPTION_NAMES]
ax.set_xticks(range(4));     ax.set_xticklabels(short_names, fontsize=11, fontweight='bold')
ax.set_yticks(range(len(METRICS))); ax.set_yticklabels(METRIC_LABEL, fontsize=11)

# Annotate rank values + actual metric values
for mi, m in enumerate(METRICS):
    for oi, name in enumerate(OPTION_NAMES):
        rank = rank_matrix[mi, oi]
        v    = data[name][m]
        medal = {1: '🥇', 2: '🥈', 3: '🥉', 4: '4'}.get(rank, str(rank))
        ax.text(oi, mi, f'{medal}\n{v:.3f}',
                ha='center', va='center',
                fontsize=9.5, fontweight='bold',
                color='white' if rank >= 3 else 'black')

ax.set_title('Per-Metric Ranking — All 4 PP-MAE Options\n'
             '(🥇 = best, 🥉 = 3rd  |  green = top rank, red = bottom)',
             fontweight='bold', fontsize=12, pad=12)
plt.colorbar(im, ax=ax, label='Rank (1=best)', shrink=0.6)
plt.tight_layout()
path = os.path.join(OUT, 'options_compare_rank.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved options_compare_rank.png", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Summary Table with Overall Score
# ─────────────────────────────────────────────────────────────────────────────
# Overall score = sum of (6 - rank) across all metrics → max=30, min=6
overall = {}
for oi, name in enumerate(OPTION_NAMES):
    overall[name] = sum(6 - rank_matrix[mi, oi] for mi in range(len(METRICS)))
sorted_options = sorted(OPTION_NAMES, key=lambda n: overall[n], reverse=True)

col_labels = ['Option', 'PSNR↑', 'SSIM↑', 'NRMSE↓', 'Dice WT↑', 'Dice TC↑', 'Dice ET↑', 'Score /30']
cell_data  = []
for name in sorted_options:
    d   = data[name]
    row = [
        name.replace('\n', ' '),
        f'{d["PSNR"]:.2f}',  f'{d["SSIM"]:.3f}',  f'{d["NRMSE"]:.3f}',
        f'{d["Dice_WT"]:.3f}', f'{d["Dice_TC"]:.3f}', f'{d["Dice_ET"]:.3f}',
        f'{overall[name]}',
    ]
    cell_data.append(row)

fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')
tbl = ax.table(cellText=cell_data, colLabels=col_labels,
               cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 2.5)

opt_colors = {n.replace('\n', ' '): COLORS[n] for n in OPTION_NAMES}
rank_bg    = ['#FFD700', '#E8E8E8', '#CD7F32', '#FFFFFF']   # gold/silver/bronze/plain

for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#1A237E')
        cell.set_text_props(color='white', fontweight='bold')
    elif row > 0:
        opt_name = cell_data[row - 1][0]
        pos  = [i for i, r in enumerate(cell_data) if r[0] == opt_name]
        rank_pos = pos[0] if pos else 3
        base_col = rank_bg[rank_pos]

        if col == 0:
            # Name cell: colour-coded by option
            c = opt_colors.get(opt_name, '#EEEEEE')
            cell.set_facecolor(c)
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor(base_col)

        # Bold the best value in each metric column
        if col > 0 and col < len(col_labels) - 1:
            m = METRICS[col - 1]
            best_val = (min if m == 'NRMSE' else max)(data[n][m] for n in OPTION_NAMES)
            cur_val  = data.get(
                next((n for n in OPTION_NAMES if n.replace('\n',' ') == opt_name), ''),
                {}).get(m, None)
            if cur_val is not None and abs(cur_val - best_val) < 1e-6:
                cell.set_text_props(fontweight='bold', color='#1565C0')

    cell.set_edgecolor('#BDBDBD')

fig.suptitle(
    'PP-MAE Options 1–4 — Summary Scorecard\n'
    '(sorted by overall score  |  Score = sum of rank points across 6 metrics)',
    fontweight='bold', fontsize=12)
plt.tight_layout()
path = os.path.join(OUT, 'options_compare_table.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved options_compare_table.png", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Print final summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print("  PP-MAE OPTIONS — FINAL SCORECARD", flush=True)
print("="*60, flush=True)
print(f"\n  {'Option':<22}  {'PSNR':>6}  {'SSIM':>6}  {'NRMSE':>6}  "
      f"{'D_WT':>6}  {'D_TC':>6}  {'D_ET':>6}  {'Score':>6}")
print("  " + "-"*78)

medals = ['🥇', '🥈', '🥉', '  ']
for rank_pos, name in enumerate(sorted_options):
    d    = data[name]
    mark = medals[rank_pos]
    print(f"  {mark} {name.replace(chr(10),' '):<20}  "
          f"{d['PSNR']:>6.2f}  {d['SSIM']:>6.3f}  {d['NRMSE']:>6.3f}  "
          f"{d['Dice_WT']:>6.3f}  {d['Dice_TC']:>6.3f}  {d['Dice_ET']:>6.3f}  "
          f"{overall[name]:>5}/30", flush=True)

print("\n  Per-metric winners:", flush=True)
for m, label in zip(METRICS, METRIC_LABEL):
    col = [data[n][m] for n in OPTION_NAMES]
    if m == 'NRMSE':
        winner_idx = int(np.argmin(col))
    else:
        winner_idx = int(np.argmax(col))
    winner = OPTION_NAMES[winner_idx].replace('\n', ' ')
    print(f"    {label:<15}  →  {winner}  ({col[winner_idx]:.3f})", flush=True)

print(f"\n  📁 Figures saved to {OUT}/", flush=True)
print(f"     options_compare_radar.png", flush=True)
print(f"     options_compare_bars.png", flush=True)
print(f"     options_compare_rank.png", flush=True)
print(f"     options_compare_table.png", flush=True)
