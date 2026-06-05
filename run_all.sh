#!/usr/bin/env bash
# run_all.sh  —  Run the complete PP-MAE experiment pipeline
# =============================================================
# Usage:
#   bash run_all.sh /path/to/BraTS2021_data [--device mps|cuda|cpu]
#
# Full run (default):
#   bash run_all.sh ~/Downloads/BraTS2021_data
#
# Quick smoke-test (finishes in ~10 min):
#   bash run_all.sh ~/Downloads/BraTS2021_data --smoke
#
# GPU server:
#   bash run_all.sh /data/BraTS2021 --device cuda

set -euo pipefail

# ── Parse arguments ────────────────────────────────────────────────────────────
DATA_DIR="${1:?Usage: bash run_all.sh <data_dir> [--device mps|cuda|cpu] [--smoke]}"
DEVICE="mps"
SMOKE=0
EPOCHS=30
SEG_EPOCHS=20
MAX_SUBJ=50

for arg in "${@:2}"; do
    case "$arg" in
        --device)  DEVICE="${!OPTIND}"; ;;
        mps|cuda|cpu)  DEVICE="$arg" ;;
        --smoke)   SMOKE=1 ;;
    esac
done

# If --device X pattern (two-token form)
for i in "${!@}"; do
    if [[ "${@:$i:1}" == "--device" ]]; then
        DEVICE="${@:$((i+1)):1}"
    fi
done

N_SUBJ_VIZ=3
if [[ $SMOKE -eq 1 ]]; then
    EPOCHS=2; SEG_EPOCHS=2; MAX_SUBJ=5; N_SUBJ_VIZ=1
    echo "  [SMOKE TEST mode: epochs=2, max_subjects=5]"
fi

# ── Helpers ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$(command -v python3 || command -v python)"
STEP=0

step() {
    STEP=$((STEP+1))
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  Step $STEP: $1"
    echo "══════════════════════════════════════════════════════════════"
}

ok() { echo "  ✓ $1"; }

# ── Step 1: Round 4 + Round 6 (Option 4 and Option 5 with grading) ────────────
step "Round 4 + Round 6 — PP-MAE Option 4 (reconstruction+seg) and Option 5 (+ grading)"
"$PY" "$SCRIPT_DIR/run_all_options.py" "$DATA_DIR" \
    --rounds 4,6 \
    --epochs "$EPOCHS" \
    --seg_epochs "$SEG_EPOCHS" \
    --max_subjects "$MAX_SUBJ" \
    --device "$DEVICE" \
    --seed 42 \
    --out "$SCRIPT_DIR/results/round4"
ok "Saved → results/round4/ (options_results.csv + grading_results.csv)"

# ── Step 2: Multi-seed (reproducibility) ──────────────────────────────────────
step "Multi-seed — 3 seeds → mean ± std"
"$PY" "$SCRIPT_DIR/run_multiseed.py" "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --seg_epochs "$SEG_EPOCHS" \
    --max_subjects "$MAX_SUBJ" \
    --device "$DEVICE" \
    --out "$SCRIPT_DIR/results/multiseed"
ok "Saved → results/multiseed/aggregated_results.csv, multiseed_bars.png, multiseed_table.png"

# ── Step 3: Noise ablation ─────────────────────────────────────────────────────
step "Noise ablation — σ = 0.05 / 0.08 / 0.15"
"$PY" "$SCRIPT_DIR/run_noise_ablation.py" "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --seg_epochs "$SEG_EPOCHS" \
    --max_subjects "$MAX_SUBJ" \
    --device "$DEVICE" \
    --out "$SCRIPT_DIR/results/noise_ablation"
ok "Saved → results/noise_ablation/noise_ablation.csv, psnr chart, Dice_ET chart"

# ── Step 4: Grading figures ───────────────────────────────────────────────────
step "Grading figures — 8 figures (ROC, confusion, radar, scatter)"
"$PY" "$SCRIPT_DIR/grading_visuals.py" \
    --data_dir "$DATA_DIR" \
    --results_dir "$SCRIPT_DIR/results/round4" \
    --device "$DEVICE" \
    --n_subjects "$N_SUBJ_VIZ" \
    --out "$SCRIPT_DIR/paper_figs/grading"
ok "Saved → paper_figs/grading/ (8 grading figures)"

# ── Step 5: Paper figures ──────────────────────────────────────────────────────
step "Paper figures — 12 figures (data + model)"
N_SUBJ=3
if [[ $SMOKE -eq 1 ]]; then N_SUBJ=1; fi
"$PY" "$SCRIPT_DIR/visualize_paper_figures.py" "$DATA_DIR" \
    --out "$SCRIPT_DIR/paper_figs" \
    --n_subjects "$N_SUBJ" \
    --device "$DEVICE" \
    --quick_model
ok "Saved → paper_figs/ (12 PNG figures)"

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  ALL DONE"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "  results/round4/          Round 4+6 CSV, bar charts + grading_results.csv"
echo "  results/multiseed/       Mean ± std across 3 seeds"
echo "  results/noise_ablation/  PSNR & Dice_ET vs noise level"
echo "  paper_figs/              12 reconstruction/seg figures"
echo "  paper_figs/grading/      8 grading figures (ROC, confusion, radar)"
echo ""
echo "  To push results to GitHub:"
echo "    git add results/ paper_figs/"
echo "    git commit -m 'Add experimental results'"
echo "    git push origin HEAD"
