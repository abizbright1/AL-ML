#!/usr/bin/env python3
"""
compare_all_losses.py — one table, every loss, ranked
======================================================

Reads a finished run and prints the comparison the study exists to make:
every training objective, side by side, on the metrics that matter, with the
no-restoration floor and the clean-image ceiling as anchors.

Those two anchors are the point. A Dice of 0.55 means nothing on its own; it
means a great deal once you know that doing nothing scores 0.24 and the best
achievable is 0.66.

Usage
-----
    python3 compare_all_losses.py --run results/comparison
    python3 compare_all_losses.py --run results/comparison --metric dice_tc
    python3 compare_all_losses.py --run results/comparison --csv table.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics as st
import sys
from collections import defaultdict


def load_rows(run_dir: str):
    files = sorted(glob.glob(os.path.join(run_dir, '*', 'per_slice_metrics.csv')))
    if not files:
        f = os.path.join(run_dir, 'per_slice_metrics.csv')
        files = [f] if os.path.isfile(f) else []
    if not files:
        sys.exit(f'No per_slice_metrics.csv under {run_dir}. Has the run finished?')
    rows = []
    for path in files:
        seed = os.path.basename(os.path.dirname(path))
        with open(path, newline='') as fh:
            for r in csv.DictReader(fh):
                r['_run'] = seed
                rows.append(r)
    return rows, files


def num(x):
    try:
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return None


def subject_means(rows, metric):
    """Average slices within a subject -- slices are not independent."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = num(r.get(metric))
        if v is not None:
            acc[r['method']][(r['_run'], r['subject'])].append(v)
    return {m: [sum(v) / len(v) for v in subj.values()] for m, subj in acc.items()}


SHORT = {
    'Noisy input (no restoration) [FLOOR]': 'FLOOR  no restoration',
    'Clean image (oracle) [CEILING]': 'CEILING  clean image',
}


def label(method: str, keep_backbone: bool = False) -> str:
    """Readable arm name. The backbone prefix is kept only when the run
    contains more than one, so a single-backbone study reads cleanly while a
    two-backbone study never shows the same label twice."""
    if method in SHORT:
        return SHORT[method]
    bb = 'Uformer' if method.startswith('Uformer') else (
         'SwinIR' if method.startswith('SwinIR') else '')
    out = (method.replace('Uformer + ', '').replace('SwinIR + ', '')
            .replace('Uformer-lite (L1)', 'L1').replace('SwinIR-lite (L1)', 'L1')
            .replace(' (current, nested)', '').replace(' [ABLATION]', ' (abl)')
            .replace(' (no pathology) [CONTROL]', '')
            .replace('[Zhao 2017]', '').replace('[Johnson 2016]', '')
            .replace('[Yang 2017]', '').replace('[BraSyn 2023]', '')
            .replace('[ROIRecNet-inspired]', '').replace('[LIDnet-inspired]', '')
            .replace('[DAGAN-inspired]', '').strip())
    if keep_backbone and bb:
        out = f'{bb[:4]}:{out}'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='results/comparison')
    ap.add_argument('--metric', default='dice_et', help='ranking metric')
    ap.add_argument('--csv', default=None, help='also write the table here')
    args = ap.parse_args()

    rows, files = load_rows(args.run)
    runs = sorted({r['_run'] for r in rows})

    METRICS = ['dice_et', 'dice_tc', 'dice_wt', 'psnr', 'psnr_et', 'ssim']
    data = {m: subject_means(rows, m) for m in METRICS}
    methods = sorted(data[args.metric], key=lambda m: -st.mean(data[args.metric][m]))
    _bbs = {('Uformer' if m.startswith('Uformer') else
             'SwinIR' if m.startswith('SwinIR') else '') for m in methods}
    MULTI = len(_bbs - {''}) > 1

    floor = ceiling = None
    for m in methods:
        if 'FLOOR' in m:
            floor = st.mean(data[args.metric][m])
        if 'CEILING' in m:
            ceiling = st.mean(data[args.metric][m])

    bar = '=' * 104
    print(bar)
    print(f'  ALL OBJECTIVES COMPARED   —   ranked by {args.metric}')
    print(bar)
    print(f'  run(s)   : {", ".join(runs)}   ({len(files)} file(s))')
    print(f'  methods  : {len(methods)}')
    if floor is not None and ceiling is not None:
        print(f'  anchors  : floor {floor:.4f}   ceiling {ceiling:.4f}   '
              f'achievable gap {ceiling - floor:.4f}')
    print()

    hdr = (f'  {"#":>2} {"objective":<30} {"dice_ET":>9} {"+/-":>7} '
           f'{"dice_TC":>8} {"PSNR":>7} {"PSNR_ET":>8} {"%gap":>7}')
    print(hdr)
    print('  ' + '-' * 100)

    table = []
    for i, m in enumerate(methods, 1):
        v = data[args.metric][m]
        mean = st.mean(v)
        sd = st.stdev(v) if len(v) > 1 else float('nan')
        def g(metric):
            vals = data[metric].get(m)
            return st.mean(vals) if vals else float('nan')
        pct = (100 * (mean - floor) / (ceiling - floor)
               if floor is not None and ceiling is not None and ceiling != floor
               else float('nan'))
        name = label(m, MULTI)
        mark = ''
        if 'PathologyLoss' in m and 'abl' not in name and 'ABLATION' not in m:
            mark = '  <<< THE METHOD'
        elif 'FLOOR' in m or 'CEILING' in m:
            mark = '  <<< anchor'
        print(f'  {i:>2} {name:<30} {mean:9.4f} {sd:7.4f} '
              f'{g("dice_tc"):8.4f} {g("psnr"):7.2f} {g("psnr_et"):8.2f} '
              f'{pct:6.1f}%{mark}')
        table.append({'rank': i, 'objective': name, 'n_subjects': len(v),
                      args.metric: round(mean, 5), 'sd': round(sd, 5),
                      'dice_tc': round(g('dice_tc'), 5),
                      'dice_wt': round(g('dice_wt'), 5),
                      'psnr': round(g('psnr'), 3),
                      'psnr_et': round(g('psnr_et'), 3),
                      'ssim': round(g('ssim'), 5),
                      'pct_of_gap': round(pct, 2)})

    print('  ' + '-' * 100)
    print('  %gap = share of the floor-to-ceiling gap this objective closes.')
    print('         Whole-image PSNR is dominated by non-tumour voxels;')
    print('         PSNR_ET measures fidelity where the claim actually lives.')
    print()

    # ── the contrasts that decide the paper ─────────────────────────────────
    def find(*keys):
        for m in methods:
            if all(k in m for k in keys):
                return m
        return None

    pl = find('PathologyLoss', 'current')
    pairs = [
        ('Does restoration help at all?', find('FLOOR'), find('lite (L1)') or find('-lite')),
        ('Is the gain just SSIM + cross-modal?', find('lite (L1)') or find('-lite'), find('matched base')),
        ('The pathology term, isolated', find('matched base'), pl),
        ('Does flat tumour weighting help?', find('matched base'), find('FlatROI')),
        ('Nesting + 3:2:1 over flat weighting', find('FlatROI'), pl),
        ('Versus the closest prior art', find('masked L1'), pl),
        ('Does the clinical ordering matter?', find('Equal weights'), pl),
    ]
    print('  DECISIVE CONTRASTS')
    print('  ' + '-' * 100)
    for question, a, b in pairs:
        if not a or not b:
            print(f'  {question:<40} -- arm not in this run')
            continue
        va, vb = data[args.metric][a], data[args.metric][b]
        common = min(len(va), len(vb))
        d = st.mean(vb) - st.mean(va)
        print(f'  {question:<40} {label(a, MULTI):<20} -> {label(b, MULTI):<20} '
              f'{d:+8.4f}')
    print()
    print('  Paired tests, CIs and equivalence testing:')
    print(f'    python3 analyze_multiseed.py --tidy <tidy.csv> --margin 0.05 \\')
    print(f'        --baseline MatchedBase --treatment PathologyLoss')
    print(bar)

    if args.csv:
        with open(args.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
            w.writeheader()
            w.writerows(table)
        print(f'\n  table written to {args.csv}')


if __name__ == '__main__':
    main()
