#!/usr/bin/env bash
# progress.sh — one view of where every run stands
#
#   bash progress.sh              summary
#   bash progress.sh -w           refresh every 60s until you press Ctrl-C
#
# Safe to run at any time. Reads only; never touches a running job.

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WATCH=0
[[ "${1:-}" == "-w" ]] && WATCH=1

show() {
# Only wipe the screen in watch mode. A one-shot check should leave your
# scrollback intact.
[[ "$WATCH" -eq 1 ]] && { clear 2>/dev/null || true; }
echo "══════════════════════════════════════════════════════════════════════"
echo "  RUN STATUS                                     $(date '+%a %d %b  %H:%M')"
echo "══════════════════════════════════════════════════════════════════════"

# ── 1. is anything training right now ───────────────────────────────────────
NSWEEP=$(pgrep -f run_comparison_sweep.sh 2>/dev/null | wc -l | tr -d ' ')
NTRAIN=$(pgrep -f run_full_experiment.py 2>/dev/null | wc -l | tr -d ' ')
echo ""
if [[ "$NTRAIN" -eq 0 && "$NSWEEP" -eq 0 ]]; then
    echo "  NOTHING IS RUNNING"
elif [[ "$NSWEEP" -gt 1 || "$NTRAIN" -gt 1 ]]; then
    echo "  !! $NSWEEP sweep process(es), $NTRAIN training process(es)"
    echo "     More than one of either means they are fighting over the same"
    echo "     output directory. Stop them all and relaunch exactly one:"
    echo "       pkill -9 -f run_comparison_sweep.sh"
    echo "       pkill -9 -f run_full_experiment.py"
else
    echo "  RUNNING   $NSWEEP sweep, $NTRAIN training     (this is the healthy state)"
    for p in $(pgrep -f run_full_experiment.py 2>/dev/null); do
        ET=$(ps -o etime= -p "$p" 2>/dev/null | tr -d ' ')
        CPU=$(ps -o %cpu= -p "$p" 2>/dev/null | tr -d ' ')
        echo "            pid $p   up $ET   cpu ${CPU}%"
    done
fi

# ── 2. what the live log says ───────────────────────────────────────────────
LOG=$(ls -t ./*.log results/comparison/logs/*.log 2>/dev/null | head -1)
if [[ -n "${LOG:-}" ]]; then
    echo ""
    echo "  ── newest log: $LOG"
    AGE=$(( $(date +%s) - $(stat -f %m "$LOG" 2>/dev/null || stat -c %Y "$LOG" 2>/dev/null) ))
    echo "     last written $((AGE/60)) min ago"
    ARM=$(grep -E '^  [A-Za-z].*(PathologyLoss|L1|FlatROI|masked|base|Noisy|Clean|SSIM|Mix)' "$LOG" 2>/dev/null | tail -1 | sed 's/^ *//')
    EP=$(grep -oE 'Ep +[0-9]+/[0-9]+' "$LOG" 2>/dev/null | tail -1)
    [[ -n "$ARM" ]] && echo "     current arm   : $ARM"
    [[ -n "$EP"  ]] && echo "     current epoch : $EP"
    LAST=$(grep -E 'PSNR=|done in|FAILED' "$LOG" 2>/dev/null | tail -1 | sed 's/^ *//')
    [[ -n "$LAST" ]] && echo "     last result   : $LAST"
fi

# ── 3. finished runs ────────────────────────────────────────────────────────
echo ""
echo "  ── completed runs"
FOUND=0
for d in results/comparison/*/ results/sweep/*/; do
    [[ -d "$d" ]] || continue
    [[ "$(basename "$d")" == "logs" ]] && continue
    NAME=$(basename "$d")
    if [[ -f "$d/per_slice_metrics.csv" ]]; then
        ROWS=$(( $(wc -l < "$d/per_slice_metrics.csv") - 1 ))
        ARMS=$(tail -n +2 "$d/per_slice_metrics.csv" | cut -d, -f1 | sort -u | wc -l | tr -d ' ')
        CK=$(ls "$d/checkpoints" 2>/dev/null | wc -l | tr -d ' ')
        NPZ=$(ls "$d/predictions" 2>/dev/null | wc -l | tr -d ' ')
        printf "     %-26s DONE   %6s rows  %2s arms  %2s ckpt  %2s npz\n" \
               "$NAME" "$ROWS" "$ARMS" "$CK" "$NPZ"
    else
        CK=$(ls "$d/checkpoints" 2>/dev/null | wc -l | tr -d ' ')
        printf "     %-26s PARTIAL  %s model(s) trained, no metrics file yet\n" "$NAME" "${CK:-0}"
    fi
    FOUND=1
done
[[ "$FOUND" -eq 0 ]] && echo "     none yet"

# ── 4. disk, since a full disk kills a run silently ─────────────────────────
echo ""
AVAIL=$(df -h . 2>/dev/null | awk 'NR==2{print $4}')
SIZE=$(du -sh results 2>/dev/null | cut -f1)
echo "  ── disk: $AVAIL free, results/ is $SIZE"

echo ""
echo "══════════════════════════════════════════════════════════════════════"
if [[ "$NTRAIN" -eq 0 && "$NSWEEP" -eq 0 ]]; then
  echo "  To start the publication run:"
  echo "    nohup env PHASE=final bash run_comparison_sweep.sh \\"
  echo "      ~/Downloads/BraTS2021_data > final.log 2>&1 &"
else
  echo "  Leave it alone. Re-run this script any time to check in."
fi
echo "══════════════════════════════════════════════════════════════════════"
}

if [[ "$WATCH" -eq 1 ]]; then
    while true; do show; sleep 60; done
else
    show
fi
