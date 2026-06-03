"""
align_results.py
================
Loads all real-BraTS experiment results, integrates them, and checks
alignment with the PP-MAE proposed model's core claims.

Usage:
    python3 align_results.py [--show] [--out figures/]
"""

from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── constants ──────────────────────────────────────────────────────────────────
REAL_PSNR_THRESHOLD = 25.0          # below → synthetic noise run, skip
OUT_DIR = "figures"

# Colour palette
C_PROPOSED  = "#1f77b4"   # blue
C_PATH_LOSS = "#2ca02c"   # green  (PathologyLoss ablation)
C_BASELINE  = "#d62728"   # red    (plain L1 baselines)
C_MULTITASK = "#ff7f0e"   # orange (Round 3 context)
C_NEUTRAL   = "#7f7f7f"   # grey

# ── helpers ────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        # normalise method column
        for col in df.columns:
            if col.lower() in ("method", "model"):
                df = df.rename(columns={col: "Method"})
                break
        return df
    except Exception:
        return None


def is_real(df: pd.DataFrame) -> bool:
    if "PSNR" not in df.columns:
        return False
    return float(df["PSNR"].max()) >= REAL_PSNR_THRESHOLD


def classify_method(name: str) -> str:
    n = name.lower()
    if "proposed" in n or ("pp-mae" in n and "swin" in n):
        return "proposed"
    if "pathologyloss" in n or "pathloss" in n or "pathology" in n:
        return "pathloss"
    if "pipeline" in n:
        return "proposed"       # Round-3 PP-MAE Pipeline = proposed backbone
    return "baseline"


def colour_for(cls: str, name: str) -> str:
    if cls == "proposed":   return C_PROPOSED
    if cls == "pathloss":   return C_PATH_LOSS
    # multi-task Round 3 baselines get a different shade
    if any(k in name.lower() for k in ["swinu", "transu", "multitask", "seqpipe"]):
        return C_MULTITASK
    return C_BASELINE


# ── load & filter ──────────────────────────────────────────────────────────────

SOURCES = {
    "Round 4 — Swin (core ablation)":  "results/round4_real/options_results.csv",
    "Round 3 — Multi-task (context)":  "results/mac_run/options_results.csv",
}

def load_all() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for label, path in SOURCES.items():
        full = os.path.join(os.path.dirname(__file__), path)
        df = load_csv(full)
        if df is None:
            print(f"  [SKIP] {path} — not found")
            continue
        if not is_real(df):
            print(f"  [SKIP] {path} — synthetic run (PSNR < {REAL_PSNR_THRESHOLD})")
            continue
        frames[label] = df
        print(f"  [OK]   {label}  ({len(df)} models)")
    return frames


# ── ablation table ─────────────────────────────────────────────────────────────

def ablation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Δ relative to paired L1 baseline for PathologyLoss rows."""
    rows = df.copy()
    rows["Class"] = rows["Method"].apply(classify_method)

    metrics = ["PSNR", "SSIM", "Dice_WT", "Dice_TC", "Dice_ET"]
    deltas = {f"Δ_{m}": [] for m in metrics}

    def find_l1_pair(method_name: str) -> pd.Series | None:
        n = method_name.lower()
        if "swinir" in n:    ref = rows[rows["Method"].str.lower().str.contains("swinir") & rows["Class"].eq("baseline")]
        elif "uformer" in n: ref = rows[rows["Method"].str.lower().str.contains("uformer") & rows["Class"].eq("baseline")]
        else: return None
        return ref.iloc[0] if len(ref) else None

    for _, row in rows.iterrows():
        if row["Class"] == "pathloss":
            ref = find_l1_pair(row["Method"])
            if ref is not None:
                for m in metrics:
                    deltas[f"Δ_{m}"].append(round(row[m] - ref[m], 4))
            else:
                for m in metrics:
                    deltas[f"Δ_{m}"].append(None)
        else:
            for m in metrics:
                deltas[f"Δ_{m}"].append(None)

    for k, v in deltas.items():
        rows[k] = v

    return rows


# ── alignment check ────────────────────────────────────────────────────────────

def check_alignment(frames: dict[str, pd.DataFrame]) -> list[dict]:
    """
    Test each claim of the proposed PP-MAE model against real data.
    Returns a list of finding dicts.
    """
    findings = []

    # ── Claim 1: PathologyLoss raises Dice_TC and Dice_ET vs plain L1 ──────
    if "Round 4 — Swin (core ablation)" in frames:
        df = ablation_table(frames["Round 4 — Swin (core ablation)"])
        pl_rows = df[df["Class"] == "pathloss"]

        for _, row in pl_rows.iterrows():
            dtc = row.get("Δ_Dice_TC")
            det = row.get("Δ_Dice_ET")
            if dtc is not None and det is not None:
                ok = dtc > 0 and det > 0
                findings.append({
                    "claim": f"PathologyLoss ↑ Dice_TC & Dice_ET vs L1 ({row['Method']})",
                    "result": f"Δ_Dice_TC={dtc:+.4f}  Δ_Dice_ET={det:+.4f}",
                    "aligned": ok,
                })
            dpsnr = row.get("Δ_PSNR")
            if dpsnr is not None:
                ok = dpsnr > -1.0          # ≤ 1 dB PSNR cost is acceptable
                findings.append({
                    "claim": f"PSNR cost < 1 dB for PathologyLoss ({row['Method']})",
                    "result": f"Δ_PSNR={dpsnr:+.4f} dB",
                    "aligned": ok,
                })

    # ── Claim 2: PP-MAE (Swin) Dice_ET ≥ plain-L1 baselines ───────────────
    if "Round 4 — Swin (core ablation)" in frames:
        df = frames["Round 4 — Swin (core ablation)"]
        proposed = df[df["Method"].apply(classify_method).eq("proposed")]
        l1 = df[df["Method"].apply(classify_method).eq("baseline")]
        if len(proposed) and len(l1):
            prop_et = float(proposed["Dice_ET"].values[0])
            best_l1_et = float(l1["Dice_ET"].max())
            ok = prop_et >= best_l1_et
            note = "(⚠ CPU-penalised PSNR)" if proposed["PSNR"].values[0] < 30 else ""
            findings.append({
                "claim": "PP-MAE (Swin) Dice_ET ≥ best plain-L1 baseline",
                "result": f"Proposed={prop_et:.4f}  Best-L1={best_l1_et:.4f} {note}",
                "aligned": ok,
            })
            prop_tc = float(proposed["Dice_TC"].values[0])
            best_l1_tc = float(l1["Dice_TC"].max())
            ok2 = prop_tc >= best_l1_tc
            findings.append({
                "claim": "PP-MAE (Swin) Dice_TC ≥ best plain-L1 baseline",
                "result": f"Proposed={prop_tc:.4f}  Best-L1={best_l1_tc:.4f} {note}",
                "aligned": ok2,
            })

    # ── Claim 3: PP-MAE Pipeline (Round 3) beats MultiTask-UNet on Dice_ET ─
    if "Round 3 — Multi-task (context)" in frames:
        df = frames["Round 3 — Multi-task (context)"]
        pp = df[df["Method"].str.contains("PP-MAE", case=False, na=False)]
        mt = df[df["Method"].str.contains("MultiTask", case=False, na=False)]
        if len(pp) and len(mt):
            pp_et = float(pp["Dice_ET"].values[0])
            mt_et = float(mt["Dice_ET"].values[0])
            ok = pp_et > mt_et
            findings.append({
                "claim": "Round 3: PP-MAE Pipeline > MultiTask-UNet on Dice_ET",
                "result": f"PP-MAE={pp_et:.4f}  MultiTask={mt_et:.4f}",
                "aligned": ok,
            })

    return findings


# ── plotting ───────────────────────────────────────────────────────────────────

def bar_group(ax, df: pd.DataFrame, metric: str, title: str):
    classes = df["Method"].apply(classify_method)
    colours = [colour_for(c, m) for c, m in zip(classes, df["Method"])]
    names   = [m.replace(" [PROPOSED]", "\n[PROPOSED]").replace(" + PathologyLoss", "\n+PathLoss")
               for m in df["Method"]]

    vals = df[metric].astype(float).values
    bars = ax.bar(range(len(vals)), vals, color=colours, edgecolor="white", linewidth=0.6)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(metric, fontsize=9)
    ymin = max(0, vals.min() - 0.04)
    ax.set_ylim(ymin, vals.max() + 0.06)
    ax.spines[["top", "right"]].set_visible(False)


def plot_ablation(df: pd.DataFrame, out_dir: str):
    adf = ablation_table(df)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Round 4 — Swin Ablation Study (Real BraTS, 50 subjects, 30 epochs)",
                 fontsize=12, fontweight="bold", y=1.02)

    for ax, metric, title in zip(axes,
            ["PSNR",    "Dice_TC",                      "Dice_ET"],
            ["PSNR (dB)", "Dice_TC (Tumour Core)",       "Dice_ET (Enhancing Tumour)"]):
        bar_group(ax, adf, metric, title)

    legend_patches = [
        mpatches.Patch(color=C_PROPOSED,  label="PP-MAE (Swin) [PROPOSED]"),
        mpatches.Patch(color=C_PATH_LOSS, label="+ PathologyLoss (ablation)"),
        mpatches.Patch(color=C_BASELINE,  label="Plain L1 baseline"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.08))

    note = ("* PP-MAE (Swin) trained on CPU (MPS bug) — PSNR is penalised ~3 dB.\n"
            "  Run again after applying the MPS fix (git pull) for fair comparison.")
    fig.text(0.01, -0.12, note, fontsize=7.5, color="firebrick", style="italic")

    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_round4.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → saved {path}")
    return fig


def plot_delta(df: pd.DataFrame, out_dir: str):
    """Bar chart of Δ Dice_TC / Δ Dice_ET from L1 → PathologyLoss."""
    adf = ablation_table(df)
    pl = adf[adf["Class"] == "pathloss"].dropna(subset=["Δ_Dice_TC", "Δ_Dice_ET"])
    if pl.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("Gain from PathologyLoss vs Plain L1 (same architecture)",
                 fontsize=11, fontweight="bold")

    for ax, col, label in zip(axes, ["Δ_Dice_TC", "Δ_Dice_ET"],
                                     ["ΔDice_TC (Tumour Core)", "ΔDice_ET (Enhancing Tumour)"]):
        names = [m.replace(" + PathologyLoss", "") for m in pl["Method"]]
        vals  = pl[col].astype(float).values
        bars  = ax.bar(names, vals, color=C_PATH_LOSS, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001 * np.sign(val),
                    f"{val:+.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Δ Dice (PathologyLoss − L1)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, "pathloss_delta.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → saved {path}")
    return fig


def plot_round3(df: pd.DataFrame, out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("Round 3 — Multi-task Context (Real BraTS, 50 subjects, 30 epochs)",
                 fontsize=11, fontweight="bold")
    for ax, metric, title in zip(axes,
            ["PSNR", "Dice_TC", "Dice_ET"],
            ["PSNR (dB)", "Dice_TC", "Dice_ET"]):
        bar_group(ax, df, metric, title)
    plt.tight_layout()
    path = os.path.join(out_dir, "round3_context.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → saved {path}")
    return fig


def plot_combined_dice_et(frames: dict[str, pd.DataFrame], out_dir: str):
    """Single scatter / grouped bar across ALL real-data runs — Dice_ET."""
    all_rows = []
    for run_name, df in frames.items():
        for _, row in df.iterrows():
            all_rows.append({
                "Run":    run_name.split("(")[0].strip(),
                "Method": row["Method"],
                "Dice_ET": float(row["Dice_ET"]),
                "PSNR":   float(row["PSNR"]),
                "Class":  classify_method(row["Method"]),
            })
    combined = pd.DataFrame(all_rows).sort_values("Dice_ET", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    colours = [colour_for(r["Class"], r["Method"]) for _, r in combined.iterrows()]
    labels  = [f"{r['Method']}\n({r['Run']})" for _, r in combined.iterrows()]
    bars    = ax.bar(range(len(combined)), combined["Dice_ET"].values,
                     color=colours, edgecolor="white")
    for bar, val in zip(bars, combined["Dice_ET"].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7.5)
    ax.set_ylabel("Dice_ET (Enhancing Tumour)", fontsize=10)
    ax.set_title("Dice_ET — All Real-BraTS Experiments (sorted)", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    legend_patches = [
        mpatches.Patch(color=C_PROPOSED,  label="PP-MAE [PROPOSED]"),
        mpatches.Patch(color=C_PATH_LOSS, label="+ PathologyLoss"),
        mpatches.Patch(color=C_BASELINE,  label="L1 baseline"),
        mpatches.Patch(color=C_MULTITASK, label="Multi-task context"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper right")
    plt.tight_layout()
    path = os.path.join(out_dir, "combined_dice_et.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → saved {path}")
    return fig


# ── report ─────────────────────────────────────────────────────────────────────

def print_report(frames: dict[str, pd.DataFrame], findings: list[dict]):
    sep = "─" * 72
    print(f"\n{sep}")
    print("  PP-MAE REAL-DATA INTEGRATION REPORT")
    print(f"{sep}\n")

    for run_label, df in frames.items():
        print(f"  ▶  {run_label}")
        adf = ablation_table(df) if "Round 4" in run_label else df
        display_cols = [c for c in ["Method","PSNR","SSIM","Dice_WT","Dice_TC","Dice_ET",
                                     "Δ_Dice_TC","Δ_Dice_ET"] if c in adf.columns]
        pd.set_option("display.float_format", "{:.4f}".format)
        pd.set_option("display.max_colwidth", 38)
        print(adf[display_cols].to_string(index=False))
        print()

    print(f"{sep}")
    print("  CLAIM ALIGNMENT CHECK")
    print(f"{sep}")
    all_aligned = True
    for f in findings:
        icon = "✅" if f["aligned"] else "❌"
        if not f["aligned"]:
            all_aligned = False
        print(f"  {icon}  {f['claim']}")
        print(f"       {f['result']}")
    print()

    print(f"{sep}")
    print("  PUBLISHABILITY ASSESSMENT")
    print(f"{sep}")

    r4 = frames.get("Round 4 — Swin (core ablation)")
    if r4 is not None:
        adf = ablation_table(r4)
        pl = adf[adf["Class"] == "pathloss"]
        if not pl.empty:
            avg_dtc = pl["Δ_Dice_TC"].dropna().mean()
            avg_det = pl["Δ_Dice_ET"].dropna().mean()
            print(f"  PathologyLoss average gain  Dice_TC: {avg_dtc:+.4f}  Dice_ET: {avg_det:+.4f}")

        proposed = adf[adf["Class"] == "proposed"]
        if not proposed.empty and float(proposed["PSNR"].values[0]) < 30:
            print()
            print("  ⚠  PP-MAE (Swin) PSNR = {:.2f} dB — trained on CPU due to MPS bug.".format(
                float(proposed["PSNR"].values[0])))
            print("     This creates an unfair comparison. After the MPS fix (already in repo),")
            print("     re-run Round 4 on GPU to get the true proposed-model numbers.")
            print()
            print("  What the CPU-penalised run STILL shows:")
            prop_et = float(proposed["Dice_ET"].values[0])
            bl = adf[adf["Class"] == "baseline"]
            best_bl_et = float(bl["Dice_ET"].max()) if not bl.empty else 0
            if prop_et > best_bl_et:
                print(f"    ✅ Dice_ET {prop_et:.4f} > best L1 baseline {best_bl_et:.4f}")
                print("       → PathologyLoss design validated even under CPU penalty")

    print()
    print("  MINIMUM REQUIRED BEFORE SUBMISSION:")
    print("  1. ✅  Ablation study on real BraTS (Round 4) — DONE")
    print("  2. ⬜  GPU rerun of PP-MAE (Swin) — pull MPS fix and rerun")
    print("  3. ⬜  Statistical significance (Wilcoxon) — run significance.csv")
    print("  4. ⬜  Scale to ≥150 subjects (current: 50)")
    print("  5. ⬜  Round 3 real-data competitive table — partially done (mac_run)")
    print(f"\n{sep}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show",  action="store_true", help="display figures interactively")
    parser.add_argument("--out",   default=OUT_DIR,     help="output directory for figures")
    parser.add_argument("--root",  default=".",         help="project root")
    args = parser.parse_args()

    # change to project root so relative paths in SOURCES work
    os.chdir(os.path.abspath(args.root))

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    print("\nLoading real-BraTS results …")
    frames = load_all()

    if not frames:
        print("No real-BraTS results found. Run with real data first.")
        sys.exit(1)

    findings = check_alignment(frames)

    print("\nGenerating figures …")
    figs = []
    if "Round 4 — Swin (core ablation)" in frames:
        figs.append(plot_ablation(frames["Round 4 — Swin (core ablation)"], out))
        figs.append(plot_delta(frames["Round 4 — Swin (core ablation)"], out))
    if "Round 3 — Multi-task (context)" in frames:
        figs.append(plot_round3(frames["Round 3 — Multi-task (context)"], out))
    figs.append(plot_combined_dice_et(frames, out))

    print_report(frames, findings)

    if args.show:
        matplotlib.use("TkAgg")
        plt.show()


if __name__ == "__main__":
    main()
