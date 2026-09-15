#!/usr/bin/env python3
"""
aggregate_seeds.py — collect a multi-seed sweep into one tidy CSV
=================================================================

Reads every ``<sweep_dir>/*/per_slice_metrics.csv`` produced by
``run_seed_sweep.sh`` and emits a single long-format ("tidy") table:

    seed, backbone, loss, subject_id, metric, value

One row per (seed, backbone, loss, subject, metric).  That is the shape the
mixed-effects model, the bootstrap and the TOST in ``analyze_multiseed.py``
expect, and it is the shape a reviewer can re-analyse without needing any of
this code.

Slice → subject aggregation
---------------------------
``per_slice_metrics.csv`` has one row per validation *slice*.  Treating those
rows as independent samples is pseudo-replication: slices from one patient are
correlated, so slice-level p-values are anticonservative.  This script
therefore averages slices within a subject before anything else, giving one
value per subject per metric.  ``--level slice`` keeps the raw rows instead,
for diagnostics only — do not run inference on them.

Undefined Dice
--------------
A Dice value is NaN when the region is absent from a slice's ground truth
(reported as NaN rather than 1.0 by ``run_full_experiment.py``).  NaNs are
skipped in the subject mean; a subject whose every slice is NaN for a region
is omitted for that metric, and the count is reported in the summary. It is
never silently replaced by 0.

Usage
-----
    python3 aggregate_seeds.py --sweep_dir results/sweep \\
                               --out results/sweep/tidy_results.csv

    # include the existing single-seed runs alongside the sweep
    python3 aggregate_seeds.py --sweep_dir results/sweep \\
                               --extra results/clean_100subj_10ep \\
                               --out results/sweep/tidy_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict


METRICS = ['psnr', 'ssim', 'nrmse', 'dice_wt', 'dice_tc', 'dice_et']

# Method name (as written into per_slice_metrics.csv) -> (backbone, loss).
# `loss` is the factor under test; `backbone` is the factor it must generalise
# across.  PP-MAE rows carry their ablation in `loss` so the same tidy file can
# hold them, but the paper's contrast is L1 vs PathologyLoss.
METHOD_MAP = {
    'SwinIR-lite (L1)':                 ('SwinIR',  'L1'),
    'SwinIR + PathologyLoss':           ('SwinIR',  'PathologyLoss'),
    'Uformer-lite (L1)':                ('Uformer', 'L1'),
    'Uformer + PathologyLoss':          ('Uformer', 'PathologyLoss'),
    'PP-MAE-v1 (mask-gated)':           ('PP-MAE',  'v1-mask-gated'),
    'PP-MAE-v2 (mask-free) [PROPOSED]': ('PP-MAE',  'v2-CMA+SAR'),
    'PP-MAE-v2 (CMA only, no SAR)':     ('PP-MAE',  'v2-CMA'),
    'PP-MAE-v2 (SAR only, no CMA)':     ('PP-MAE',  'v2-SAR'),
    'PP-MAE-v2 (neither module)':       ('PP-MAE',  'v2-none'),
}


def parse_args():
    p = argparse.ArgumentParser(
        description='Collect a multi-seed sweep into one tidy CSV.')
    p.add_argument('--sweep_dir', default='results/sweep',
                   help='Directory holding one subdirectory per run '
                        '(default: results/sweep)')
    p.add_argument('--extra', nargs='*', default=[],
                   help='Additional run directories to fold in, e.g. an '
                        'earlier single-seed run.')
    p.add_argument('--out', default=None,
                   help='Output CSV (default: <sweep_dir>/tidy_results.csv)')
    p.add_argument('--level', choices=['subject', 'slice'], default='subject',
                   help="'subject' (default) averages slices within a subject. "
                        "'slice' keeps raw slice rows — diagnostics only, they "
                        "are not independent samples.")
    p.add_argument('--strict', action='store_true',
                   help='Exit non-zero if any run directory is unreadable or '
                        'has an unknown method name.')
    return p.parse_args()


def seed_of(run_dir: str) -> int | None:
    """Seed from run_config.json; fall back to the _seed{N} directory suffix."""
    cfg_path = os.path.join(run_dir, 'run_config.json')
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path) as fh:
                return int(json.load(fh)['args']['seed'])
        except Exception:
            pass
    m = re.search(r'seed[_-]?(\d+)', os.path.basename(run_dir), re.I)
    return int(m.group(1)) if m else None


def read_run(run_dir: str, warn) -> list[dict]:
    """Return raw slice rows from one run directory."""
    csv_path = os.path.join(run_dir, 'per_slice_metrics.csv')
    if not os.path.isfile(csv_path):
        warn(f'no per_slice_metrics.csv in {run_dir} — skipped (incomplete run?)')
        return []

    seed = seed_of(run_dir)
    if seed is None:
        warn(f'cannot determine seed for {run_dir} — skipped')
        return []

    rows = []
    with open(csv_path, newline='') as fh:
        for r in csv.DictReader(fh):
            method = r['method']
            if method not in METHOD_MAP:
                warn(f'unknown method {method!r} in {run_dir} — skipped')
                continue
            backbone, loss = METHOD_MAP[method]
            rows.append({'seed': seed, 'backbone': backbone, 'loss': loss,
                         'subject_id': r['subject'], 'raw': r,
                         'run_dir': run_dir})
    return rows


def as_float(text: str) -> float:
    """'' and 'nan' both mean undefined."""
    if text is None or text.strip() == '':
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def main():
    args = parse_args()
    out_path = args.out or os.path.join(args.sweep_dir, 'tidy_results.csv')

    warnings: list[str] = []
    def warn(msg):
        warnings.append(msg)
        print(f'  WARN: {msg}', file=sys.stderr)

    run_dirs = []
    if os.path.isdir(args.sweep_dir):
        run_dirs += sorted(
            os.path.join(args.sweep_dir, d)
            for d in os.listdir(args.sweep_dir)
            if os.path.isdir(os.path.join(args.sweep_dir, d)) and d != 'logs')
    else:
        warn(f'sweep_dir {args.sweep_dir} does not exist')
    run_dirs += [d.rstrip('/') for d in args.extra]

    if not run_dirs:
        sys.exit('No run directories found. Has the sweep been run?')

    print(f'Reading {len(run_dirs)} run directories...')
    raw_rows = []
    for d in run_dirs:
        got = read_run(d, warn)
        if got:
            print(f'  {os.path.basename(d):<32} {len(got):>6} slice rows  '
                  f'seed={got[0]["seed"]}')
        raw_rows += got

    if not raw_rows:
        sys.exit('No usable rows. Nothing written.')

    # ── slice -> subject ─────────────────────────────────────────────────────
    # key: (seed, backbone, loss, subject_id, metric) -> list of slice values
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for row in raw_rows:
        base = (row['seed'], row['backbone'], row['loss'], row['subject_id'])
        for metric in METRICS:
            buckets[base + (metric,)].append(as_float(row['raw'].get(metric, '')))

    tidy = []
    n_all_nan = defaultdict(int)
    for key, vals in buckets.items():
        if args.level == 'slice':
            for v in vals:
                if not math.isnan(v):
                    tidy.append(key + (v,))
            continue
        defined = [v for v in vals if not math.isnan(v)]
        if not defined:
            n_all_nan[key[4]] += 1      # region absent in every slice
            continue
        tidy.append(key + (sum(defined) / len(defined),))

    tidy.sort(key=lambda t: (t[0], t[1], t[2], t[3], t[4]))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['seed', 'backbone', 'loss', 'subject_id', 'metric', 'value'])
        for t in tidy:
            w.writerow([t[0], t[1], t[2], t[3], t[4], f'{t[5]:.6g}'])

    # ── summary ──────────────────────────────────────────────────────────────
    seeds = sorted({t[0] for t in tidy})
    configs = sorted({(t[1], t[2]) for t in tidy})
    subjects = sorted({t[3] for t in tidy})

    print(f'\nWrote {len(tidy)} rows -> {out_path}')
    print(f'  level      : {args.level}')
    print(f'  seeds      : {seeds}  (n={len(seeds)})')
    print(f'  subjects   : {len(subjects)}')
    print('  configs    :')
    for backbone, loss in configs:
        got = sorted({t[0] for t in tidy if t[1] == backbone and t[2] == loss})
        flag = '' if got == seeds else f'   <-- INCOMPLETE, has only {got}'
        print(f'      {backbone:<8} {loss:<16} seeds={got}{flag}')
    if n_all_nan:
        print('  subjects dropped (region absent in every slice):')
        for metric, n in sorted(n_all_nan.items()):
            print(f'      {metric:<10} {n}')
    if warnings:
        print(f'\n  {len(warnings)} warning(s) above.')

    print('\nNext:')
    print(f'  python3 analyze_multiseed.py --tidy {out_path} --margin <your_margin>')

    if args.strict and warnings:
        sys.exit(1)


if __name__ == '__main__':
    main()
