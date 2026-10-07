#!/usr/bin/env bash
# run_comparison_sweep.sh — benchmark PathologyLoss against published alternatives
# ==============================================================================
#
# Answers the question Dr. Liu raised: what are the benchmark models, and does
# PathologyLoss beat them?
#
# PHASE 1 (default) — the decisive comparison, ~27 h over 3 seeds
#   Noisy            floor: does restoration help segmentation at all?
#   Clean            ceiling: how much headroom exists?
#   SwinIR/Uformer-L1        unweighted control
#   SwinIR/Uformer+PL3       PathologyLoss as Equation (3) states it
#   SwinIR/Uformer+ROI       flat binary tumour weighting (Sun et al. 2019)
#
#   If +PL3 does not beat +ROI, the nested multi-region design contributes
#   nothing over a 2019 method and the novelty claim does not survive. Run
#   this before spending anything on Phase 2.
#
# PHASE 2 — LIDnet-style arms, ~18 h more over 3 seeds
#   SwinIR/Uformer+ROIfeat   ROI-restricted perceptual loss
#   SwinIR/Uformer+task      downstream task loss in the objective
#
# PHASE ALL — everything at once.
#
# Prerequisites (once):
#   python3 patch_seed_support.py        # if not already applied
#   python3 patch_comparison_arms.py
#   python3 -c "import ast; ast.parse(open('run_full_experiment.py').read())"
#
# Usage:
#   bash run_comparison_sweep.sh ~/Downloads/BraTS2021_data
#   bash run_comparison_sweep.sh ~/Downloads/BraTS2021_data --dry-run
#   PHASE=2 bash run_comparison_sweep.sh ~/Downloads/BraTS2021_data
#   SEEDS="1 2 3 4 5" bash run_comparison_sweep.sh ~/Downloads/BraTS2021_data
#   EPOCHS=2 MAX_SUBJ=5 SEEDS="1" bash run_comparison_sweep.sh ~/data   # smoke
#
# Resumable: a seed counts as done only once it has written per_slice_metrics.csv.
# Re-run the same command to continue after any interruption.
#
# Each seed is ONE invocation that trains all requested arms together, so the
# two segmentors are trained once per seed rather than once per arm, and every
# arm within a seed is scored by the identical frozen segmentor.

set -uo pipefail

DATA_DIR="${1:?Usage: bash run_comparison_sweep.sh <data_dir> [--dry-run]}"
DRY_RUN=0
[[ "${2:-}" == "--dry-run" ]] && DRY_RUN=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$(command -v python3 || command -v python)"

PHASE="${PHASE:-1}"
case "$PHASE" in
  1)   MODELS="${MODELS:-Noisy,Clean,SwinIR-L1,Uformer-L1,SwinIR+PL3,Uformer+PL3,SwinIR+ROI,Uformer+ROI}" ;;
  2)   MODELS="${MODELS:-Noisy,Clean,SwinIR+ROIfeat,Uformer+ROIfeat,SwinIR+task,Uformer+task}" ;;
  all) MODELS="${MODELS:-Noisy,Clean,SwinIR-L1,Uformer-L1,SwinIR+PL3,Uformer+PL3,SwinIR+ROI,Uformer+ROI,SwinIR+ROIfeat,Uformer+ROIfeat,SwinIR+task,Uformer+task}" ;;
  *)   echo "PHASE must be 1, 2 or all (got '$PHASE')"; exit 1 ;;
esac

SEEDS="${SEEDS:-1 2 3}"
EPOCHS="${EPOCHS:-10}"
SEG_EPOCHS="${SEG_EPOCHS:-10}"
MAX_SUBJ="${MAX_SUBJ:-100}"
PATCH="${PATCH:-96}"
SIGMA="${SIGMA:-0.08}"
BATCH="${BATCH:-4}"
DEVICE="${DEVICE:-mps}"
DUMP="${DUMP:-48}"
SWEEP_DIR="${SWEEP_DIR:-$SCRIPT_DIR/results/comparison}"
LOG_DIR="$SWEEP_DIR/logs"

mkdir -p "$SWEEP_DIR" "$LOG_DIR"

# ── Preflight ────────────────────────────────────────────────────────────────
if ! grep -q 'comparison_losses' "$SCRIPT_DIR/run_full_experiment.py"; then
    echo "ERROR: comparison arms are not registered."
    echo "       Run:  python3 patch_comparison_arms.py"
    exit 1
fi
if ! grep -q 'def set_all_seeds' "$SCRIPT_DIR/run_full_experiment.py"; then
    echo "ERROR: run_full_experiment.py is not seed-patched."
    echo "       Run:  python3 patch_seed_support.py"
    exit 1
fi
if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: data directory not found: $DATA_DIR"
    exit 1
fi

N_SEEDS=0; for _ in $SEEDS; do N_SEEDS=$((N_SEEDS+1)); done
N_ARMS=$(awk -F, '{print NF}' <<< "$MODELS")

echo "══════════════════════════════════════════════════════════════════════"
echo "  Benchmark comparison sweep — phase $PHASE"
echo "──────────────────────────────────────────────────────────────────────"
echo "  data      : $DATA_DIR"
echo "  arms      : $N_ARMS"
echo "              $MODELS"
echo "  seeds     : $SEEDS  ($N_SEEDS)"
echo "  epochs    : $EPOCHS (recon) / $SEG_EPOCHS (seg, x2 segmentors)"
echo "  subjects  : $MAX_SUBJ    device: $DEVICE"
echo "  output    : $SWEEP_DIR"
[[ $DRY_RUN -eq 1 ]] && echo "  MODE      : DRY RUN — nothing will be trained"
echo "══════════════════════════════════════════════════════════════════════"

N=0; RAN=0; SKIPPED=0; FAILED=0; FAILED_SEEDS=""
T_ALL=$(date +%s)

for SEED in $SEEDS; do
    N=$((N+1))
    OUT="$SWEEP_DIR/phase${PHASE}_seed${SEED}"
    LOG="$LOG_DIR/phase${PHASE}_seed${SEED}.log"
    HDR="[$N/$N_SEEDS] phase $PHASE  seed=$SEED"

    if [[ -f "$OUT/per_slice_metrics.csv" ]]; then
        echo "  SKIP    $HDR  (already complete)"
        SKIPPED=$((SKIPPED+1))
        continue
    fi
    [[ -d "$OUT" ]] && echo "  RETRY   $HDR  (directory exists but incomplete)"

    echo ""
    echo "── $HDR ──────────────────────────────────────────────"
    echo "   out: $OUT"
    echo "   log: $LOG"
    if [[ $DRY_RUN -eq 1 ]]; then echo "   (dry run — skipping)"; continue; fi

    T0=$(date +%s)
    "$PY" "$SCRIPT_DIR/run_full_experiment.py" "$DATA_DIR" \
        --models "$MODELS" \
        --seed "$SEED" \
        --epochs "$EPOCHS" \
        --seg_epochs "$SEG_EPOCHS" \
        --max_subjects "$MAX_SUBJ" \
        --patch_size "$PATCH" \
        --sigma "$SIGMA" \
        --batch_size "$BATCH" \
        --device "$DEVICE" \
        --dump_slices "$DUMP" \
        --out "$OUT" 2>&1 | tee "$LOG"
    RC=${PIPESTATUS[0]}
    DT=$(( $(date +%s) - T0 ))

    if [[ $RC -eq 0 && -f "$OUT/per_slice_metrics.csv" ]]; then
        echo "   ✓ done in $((DT/60)) min"
        RAN=$((RAN+1))
    else
        echo "   ✗ FAILED (exit $RC) after $((DT/60)) min — see $LOG"
        FAILED=$((FAILED+1))
        FAILED_SEEDS="$FAILED_SEEDS\n     seed=$SEED  ($LOG)"
    fi
done

DT_ALL=$(( $(date +%s) - T_ALL ))
echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  PHASE $PHASE FINISHED  —  $((DT_ALL/3600))h $(((DT_ALL%3600)/60))m"
echo "    completed this session : $RAN"
echo "    skipped (already done) : $SKIPPED"
echo "    failed                 : $FAILED"
[[ $FAILED -gt 0 ]] && echo -e "  Failed:$FAILED_SEEDS"
echo "══════════════════════════════════════════════════════════════════════"
echo ""
echo "  Aggregate:"
echo "    python3 aggregate_seeds.py --sweep_dir $SWEEP_DIR --out $SWEEP_DIR/tidy_results.csv"
echo ""
echo "  THE DECISIVE TEST — does nested multi-region weighting beat flat binary?"
echo "    python3 analyze_multiseed.py --tidy $SWEEP_DIR/tidy_results.csv \\"
echo "        --margin 0.05 --baseline BinaryROI --treatment PathologyLoss-Eq3"
echo ""
echo "  Does restoration help segmentation at all? (compare against the floor)"
echo "    python3 analyze_multiseed.py --tidy $SWEEP_DIR/tidy_results.csv \\"
echo "        --margin 0.05 --baseline noisy --treatment L1"
echo ""
echo "  Each remaining contrast:"
echo "    --baseline L1        --treatment PathologyLoss-Eq3"
echo "    --baseline L1        --treatment BinaryROI"
echo "    --baseline L1        --treatment ROIFeature      # phase 2"
echo "    --baseline L1        --treatment TaskFeedback    # phase 2"
echo "    --baseline PathologyLoss-Eq3 --treatment TaskFeedback"

[[ $FAILED -gt 0 ]] && exit 1
exit 0
