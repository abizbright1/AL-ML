#!/usr/bin/env bash
# run_seed_sweep.sh — 4 configurations x 5 seeds = 20 sequential training runs
# ============================================================================
#
# Addresses reviewer ISSUE A: every result reported so far comes from a single
# run at seed 42, so no variance estimate exists and no effect can be separated
# from run-to-run noise.
#
# One invocation of run_full_experiment.py per (model, seed) pair.  Each pair
# gets its own output directory, so:
#   * a completed pair is never recomputed  (resumable after Ctrl-C, sleep,
#     crash, or an OOM kill)
#   * nothing can overwrite anything already on disk
#
# PREREQUISITE — apply the seeding patch once:
#     python3 patch_seed_support.py
#     python3 -c "import ast; ast.parse(open('run_full_experiment.py').read())"
#
# Usage:
#     bash run_seed_sweep.sh ~/Downloads/BraTS2021_data
#     bash run_seed_sweep.sh ~/Downloads/BraTS2021_data --dry-run
#     SEEDS="1 2 3" bash run_seed_sweep.sh ~/Downloads/BraTS2021_data
#     EPOCHS=2 MAX_SUBJ=5 bash run_seed_sweep.sh ~/Downloads/BraTS2021_data
#
# Estimated wall-clock at the defaults below (100 subjects, 10 epochs, MPS),
# extrapolated from the clean_100subj_10ep run: ~2.1 h per pair, ~42 h total.
# Budget four overnight sessions.  Interrupt freely; re-run to continue.
#
# To resume: just run the same command again.  Finished pairs are skipped.
# To force one pair to re-run: delete its directory under results/sweep/.

set -uo pipefail   # NOT -e: one failed pair must not abort the remaining 19

DATA_DIR="${1:?Usage: bash run_seed_sweep.sh <data_dir> [--dry-run]}"
DRY_RUN=0
[[ "${2:-}" == "--dry-run" ]] && DRY_RUN=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$(command -v python3 || command -v python)"

# ── Knobs (override from the environment) ────────────────────────────────────
SEEDS="${SEEDS:-1 2 3 4 5}"
MODELS="${MODELS:-SwinIR-L1 Uformer-L1 SwinIR+PL Uformer+PL}"
EPOCHS="${EPOCHS:-10}"
SEG_EPOCHS="${SEG_EPOCHS:-10}"
MAX_SUBJ="${MAX_SUBJ:-100}"
PATCH="${PATCH:-96}"
SIGMA="${SIGMA:-0.08}"
BATCH="${BATCH:-4}"
DEVICE="${DEVICE:-mps}"
DUMP="${DUMP:-48}"
SWEEP_DIR="${SWEEP_DIR:-$SCRIPT_DIR/results/sweep}"
LOG_DIR="$SWEEP_DIR/logs"

mkdir -p "$SWEEP_DIR" "$LOG_DIR"

# ── Preflight ────────────────────────────────────────────────────────────────
if ! grep -q 'def set_all_seeds' "$SCRIPT_DIR/run_full_experiment.py"; then
    echo "ERROR: run_full_experiment.py is not seed-patched."
    echo "       Run:  python3 patch_seed_support.py"
    exit 1
fi
if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: data directory not found: $DATA_DIR"
    exit 1
fi

# Filesystem-safe tag for a model label ('SwinIR+PL' -> 'SwinIR_plus_PL')
tag_of() { printf '%s' "${1//+/_plus_}" | tr -c 'A-Za-z0-9_.-' '_'; }

N_TOTAL=0
for _ in $MODELS; do for _ in $SEEDS; do N_TOTAL=$((N_TOTAL+1)); done; done

echo "══════════════════════════════════════════════════════════════════════"
echo "  Multi-seed sweep"
echo "──────────────────────────────────────────────────────────────────────"
echo "  data      : $DATA_DIR"
echo "  models    : $MODELS"
echo "  seeds     : $SEEDS"
echo "  pairs     : $N_TOTAL"
echo "  epochs    : $EPOCHS (recon) / $SEG_EPOCHS (seg)"
echo "  subjects  : $MAX_SUBJ    device: $DEVICE"
echo "  output    : $SWEEP_DIR"
echo "  logs      : $LOG_DIR"
[[ $DRY_RUN -eq 1 ]] && echo "  MODE      : DRY RUN — nothing will be trained"
echo "══════════════════════════════════════════════════════════════════════"

N=0; RAN=0; SKIPPED=0; FAILED=0
SWEEP_T0=$(date +%s)
FAILED_PAIRS=""

for MODEL in $MODELS; do
  for SEED in $SEEDS; do
    N=$((N+1))
    TAG="$(tag_of "$MODEL")"
    OUT="$SWEEP_DIR/${TAG}_seed${SEED}"
    LOG="$LOG_DIR/${TAG}_seed${SEED}.log"
    HDR="[$N/$N_TOTAL] $MODEL  seed=$SEED"

    # A run counts as done only if it produced its per-slice metrics file.
    # A bare directory means an interrupted run and is retried.
    if [[ -f "$OUT/per_slice_metrics.csv" ]]; then
        echo "  SKIP    $HDR  (already complete: $OUT)"
        SKIPPED=$((SKIPPED+1))
        continue
    fi
    if [[ -d "$OUT" ]]; then
        echo "  RETRY   $HDR  (directory exists but incomplete)"
    fi

    echo ""
    echo "── $HDR ──────────────────────────────────────────────"
    echo "   out: $OUT"
    echo "   log: $LOG"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "   (dry run — skipping)"
        continue
    fi

    T0=$(date +%s)
    "$PY" "$SCRIPT_DIR/run_full_experiment.py" "$DATA_DIR" \
        --models "$MODEL" \
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
        FAILED_PAIRS="$FAILED_PAIRS\n     $MODEL seed=$SEED  ($LOG)"
    fi
  done
done

SWEEP_DT=$(( $(date +%s) - SWEEP_T0 ))
echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  SWEEP FINISHED  —  $((SWEEP_DT/3600))h $(((SWEEP_DT%3600)/60))m"
echo "    completed this session : $RAN"
echo "    skipped (already done) : $SKIPPED"
echo "    failed                 : $FAILED"
[[ $FAILED -gt 0 ]] && echo -e "  Failed pairs:$FAILED_PAIRS"
echo "══════════════════════════════════════════════════════════════════════"
echo ""
echo "  Next:"
echo "    python3 aggregate_seeds.py --sweep_dir $SWEEP_DIR --out $SWEEP_DIR/tidy_results.csv"
echo "    python3 analyze_multiseed.py --tidy $SWEEP_DIR/tidy_results.csv --margin 0.02"

[[ $FAILED -gt 0 ]] && exit 1
exit 0
