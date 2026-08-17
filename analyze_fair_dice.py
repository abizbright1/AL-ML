#!/usr/bin/env python3
"""
analyze_fair_dice.py — equal-denominator Dice plus a false-positive rate
========================================================================

Why this exists
---------------
run_full_experiment.py records Dice as NaN when a region is absent from BOTH
the prediction and the ground truth, because 0/0 is undefined. That is correct,
but it makes the DENOMINATOR model-dependent: a model that over-predicts is
scored on more slices (collecting extra zeros) than one that does not.

On the 100-subject / 10-epoch run this understated PP-MAE-v2 by 0.097 Dice_ET.

How ground truth is recovered from the CSV
------------------------------------------
Dice is NaN only when prediction AND ground truth are both empty. So if the
ground truth contains the region, NO model can report NaN on that slice.
Therefore:

    a slice is REGION-POSITIVE  <=>  no model reports NaN on it

Rows are aligned across models because val_loader runs with shuffle=False, so
row k of every model's block is the same slice. The script asserts this.

Reports
-------
  Dice on region-positive slices   -- same denominator for every model
  False-positive rate              -- share of region-NEGATIVE slices on which
                                      the model predicted the region anyway

Report both. Dice alone rewards over-prediction; the FP rate exposes it.

Usage
-----
    python3 analyze_fair_dice.py results/clean_100subj_10ep/per_slice_metrics.csv
    python3 analyze_fair_dice.py <csv> --region dice_tc
    python3 analyze_fair_dice.py <csv> --out fair_metrics.csv
"""

import argparse
import csv
import sys
from collections import defaultdict


def load_aligned(path):
    """Group rows by method and verify the slice ordering matches across models."""
    rows = list(csv.DictReader(open(path)))
    by = defaultdict(list)
    for r in rows:
        by[r['method']].append(r)

    methods = list(by)
    if not methods:
        sys.exit(f"no rows found in {path}")

    n = len(by[methods[0]])
    for m in methods:
        if len(by[m]) != n:
            sys.exit(f"row counts differ ({m} has {len(by[m])}, expected {n}) "
                     "- slices cannot be aligned")
    for i in range(n):
        s0 = by[methods[0]][i]['subject']
        for m in methods:
            if by[m][i]['subject'] != s0:
                sys.exit(f"subject order differs at row {i} - cannot align")
    return by, methods, n


def analyse(by, methods, n, region):
    # Ground truth is model-independent: positive iff nobody reports NaN.
    gt_pos = [all(by[m][i][region] != 'nan' for m in methods) for i in range(n)]
    n_pos = sum(gt_pos)
    n_neg = n - n_pos

    results = []
    for m in methods:
        rs = by[m]
        scored = [float(rs[i][region]) for i in range(n) if rs[i][region] != 'nan']
        reported = sum(scored) / len(scored) if scored else float('nan')

        pos = [float(rs[i][region]) for i in range(n) if gt_pos[i]]
        fair = sum(pos) / len(pos) if pos else float('nan')

        fp = sum(1 for i in range(n) if not gt_pos[i] and rs[i][region] != 'nan')
        fp_rate = fp / n_neg if n_neg else 0.0

        results.append({
            'Method':        m,
            'Reported':      round(reported, 4),
            'Fair':          round(fair, 4),
            'Delta':         round(fair - reported, 4),
            'FP_rate':       round(fp_rate, 4),
            'N_scored':      len(scored),
            'N_positive':    n_pos,
            'N_negative':    n_neg,
            'N_false_pos':   fp,
        })
    return results, n_pos, n_neg


def main():
    ap = argparse.ArgumentParser(
        description='Equal-denominator Dice and false-positive rate')
    ap.add_argument('csv_path', help='per_slice_metrics.csv from a run')
    ap.add_argument('--region', default='dice_et',
                    choices=['dice_et', 'dice_tc', 'dice_wt'])
    ap.add_argument('--out', default=None, help='optional CSV to write')
    args = ap.parse_args()

    by, methods, n = load_aligned(args.csv_path)
    print(f"\naligned: {len(methods)} models x {n} slices, identical order")

    results, n_pos, n_neg = analyse(by, methods, n, args.region)
    reg = args.region.replace('dice_', '').upper()
    print(f"ground truth: {n_pos} slices contain {reg}, "
          f"{n_neg} do not  ({100 * n_neg / n:.1f}% empty)\n")

    print(f"{'model':<26}{'reported':>10}{'fair':>9}{'change':>9}{'FP rate':>10}")
    print('-' * 64)
    for r in results:
        short = r['Method'].split(' (')[0]
        print(f"{short:<26}{r['Reported']:>10.4f}{r['Fair']:>9.4f}"
              f"{r['Delta']:>+9.4f}{100 * r['FP_rate']:>9.1f}%")

    print(f"\n  'fair'    = mean Dice over the {n_pos} {reg}-positive slices only,")
    print( "              identical denominator for every model")
    print(f"  'FP rate' = share of the {n_neg} {reg}-negative slices on which the")
    print( "              model predicted the region anyway")
    print( "\n  Report BOTH. Dice alone rewards over-prediction.\n")

    if args.out:
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"  saved {args.out}\n")


if __name__ == '__main__':
    main()
