#!/usr/bin/env python3
"""
verify_pathologyloss.py — does the benchmarked loss equal the published one?
============================================================================

Run this BEFORE any benchmark run. It answers one question:

    Is the PathologyLoss in comparison_losses.py the same object as the
    PathologyLoss that produced the manuscript results?

If it is not, the benchmark measures something the paper does not describe,
and every comparison built on it is void.

What it compares
----------------
  A. AS RUN      -- what SwinIRPathologyTrainer / UformerPathologyTrainer
                    actually minimise:
                        PPMAELoss(lambda1=1.0, lambda2=0.5,
                                  mode='clinical_risk', ssim_weight=0.5)
                    = L1 + 0.5*SSIM + 1.0*L_clinical_risk + 0.5*L_crossmodal

  B. AS WRITTEN  -- Equation (3) of the manuscript:
                        L1 + sum_r w_r ||M_r (X^-X)||_1 / |M_r|
                    over nested ET/TC/WT with w = 1, 2, 3.

It decomposes A term by term so the size of each discrepancy is visible, and
it prints the clinical-risk weights R_r that the as-run loss actually applies
-- the numbers that decide whether "w_ET = 3" is a fair description of the
experiment.

Usage
-----
    python3 verify_pathologyloss.py
    python3 verify_pathologyloss.py --device mps --trials 5

Exit status is 0 whatever the outcome; this is a measurement, not a gate.
The decision about which object the paper benchmarks is yours.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pp_mae'))

import torch
import torch.nn.functional as F


def synthetic_batch(B=2, C=4, H=96, W=96, device='cpu', seed=0):
    """A batch with realistic BraTS label geometry: ET inside TC inside WT."""
    g = torch.Generator().manual_seed(seed)
    target = torch.rand(B, C, H, W, generator=g)
    pred = (target + 0.08 * torch.randn(B, C, H, W, generator=g)).clamp(0, 1)
    seg = torch.zeros(B, 1, H, W, dtype=torch.long)
    seg[:, :, 20:60, 20:60] = 2        # oedema      -> WT only
    seg[:, :, 32:50, 32:50] = 1        # necrotic    -> WT, TC
    seg[:, :, 38:46, 38:46] = 3        # enhancing   -> WT, TC, ET
    return (pred.to(device).requires_grad_(True),
            target.to(device), seg.to(device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--trials', type=int, default=3)
    ap.add_argument('--tol', type=float, default=1e-5,
                    help='Relative agreement required to call the two identical')
    args = ap.parse_args()
    dev = args.device

    from losses import PPMAELoss
    from comparison_losses import RegionWeightedL1, region_masks

    bar = '=' * 74
    print(bar)
    print('  PathologyLoss regression check')
    print(bar)
    print(f'  device {dev}   trials {args.trials}')
    print()

    as_run = PPMAELoss(lambda1=1.0, lambda2=0.5,
                       mode='clinical_risk', ssim_weight=0.5).to(dev)
    as_written = RegionWeightedL1(mode='nested')     # w = WT 1, TC 2, ET 3

    rows = []
    for t in range(args.trials):
        pred, target, seg = synthetic_batch(device=dev, seed=t)

        with torch.no_grad():
            d = as_run(pred, target, seg)
            total_run = d['total'].item()
            l_global = d['global'].item()
            l_path_run = d['pathology'].item()
            l_xm = d['crossmodal'].item()
            R = {k: v for k, v in d.items() if str(k).startswith('R_')}

            l1_only = F.l1_loss(pred, target).item()
            l_ssim_part = l_global - l1_only          # GlobalReconLoss = L1 + 0.5*SSIM
            l_path_written = as_written(pred, target, seg).item()
            total_written = l1_only + l_path_written

        rows.append((total_run, total_written, l1_only, l_ssim_part,
                     l_path_run, l_path_written, l_xm, R))

    # ── term-by-term ────────────────────────────────────────────────────────
    print('  TERM-BY-TERM, trial 0')
    print('  ' + '-' * 70)
    (tr, tw, l1, ss, pr, pw, xm, R) = rows[0]
    print(f'    {"L1 (shared by both)":<34} {l1:12.6f}')
    print(f'    {"+ 0.5 * SSIM   [as-run only]":<34} {ss:12.6f}'
          f'   {"<-- absent from Eq.(3)" if abs(ss) > 1e-9 else ""}')
    print(f'    {"+ 1.0 * pathology (as-run)":<34} {pr:12.6f}')
    print(f'    {"+ 1.0 * pathology (Eq.3)":<34} {pw:12.6f}')
    print(f'    {"+ 0.5 * cross-modal [as-run]":<34} {0.5*xm:12.6f}'
          f'   {"<-- absent from Eq.(3)" if abs(xm) > 1e-9 else "   (inert)"}')
    print('  ' + '-' * 70)
    print(f'    {"TOTAL as run":<34} {tr:12.6f}')
    print(f'    {"TOTAL as written (Eq.3)":<34} {tw:12.6f}')
    gap = tr - tw
    rel = abs(gap) / max(abs(tw), 1e-12)
    print(f'    {"difference":<34} {gap:+12.6f}   ({100*rel:.1f}% of Eq.3)')
    print()

    # ── the clinical-risk weights actually applied ─────────────────────────
    print('  WEIGHTS THE AS-RUN LOSS ACTUALLY APPLIES')
    print('  ' + '-' * 70)
    if R:
        for t, (*_rest, Rt) in enumerate(rows):
            vals = '  '.join(f'{k}={float(v):.3f}' for k, v in sorted(Rt.items()))
            print(f'    trial {t}:  {vals}')
        print(f'    manuscript states:  R_WT=1.000  R_TC=2.000  R_ET=3.000')
        drift = max(abs(float(v) - {'R_WT': 1., 'R_TC': 2., 'R_ET': 3.}[k])
                    for *_x, Rt in rows for k, v in Rt.items())
        print(f'    largest deviation from the stated prior: {drift:.3f}')
        if drift < 0.05:
            print('    -> close to the prior; a footnote can cover it.')
        else:
            print('    -> NOT the stated weights. Equation (3) does not describe')
            print('       the experiment, and this needs saying in the paper.')
    else:
        print('    (no R_* returned; mode may not be clinical_risk)')
    print()

    # ── across-trial stability ─────────────────────────────────────────────
    print('  ACROSS TRIALS')
    print('  ' + '-' * 70)
    print(f'    {"trial":>6} {"as run":>12} {"as written":>12} {"rel. diff":>12}')
    for t, (tr, tw, *_r) in enumerate(rows):
        print(f'    {t:>6} {tr:12.6f} {tw:12.6f} '
              f'{abs(tr-tw)/max(abs(tw),1e-12):11.1%}')
    print()

    # ── verdict ────────────────────────────────────────────────────────────
    worst = max(abs(tr - tw) / max(abs(tw), 1e-12) for tr, tw, *_r in rows)
    print(bar)
    if worst <= args.tol:
        print('  VERDICT: AGREE. comparison_losses.py reproduces the published')
        print('           objective. Benchmark against it directly.')
    else:
        print(f'  VERDICT: DISAGREE by {100*worst:.1f}%.')
        print()
        print('  The published numbers come from the AS-RUN objective; the')
        print('  manuscript describes the AS-WRITTEN one. They are different')
        print('  objects, so you must choose before benchmarking:')
        print()
        print('    (a) Benchmark AS RUN  -- competitors face the objective that')
        print('        produced Tables I-VII. Nothing is re-run, but the paper')
        print('        must describe the SSIM term, the cross-modal term and')
        print('        the network-emitted weights.')
        print()
        print('    (b) Benchmark AS WRITTEN -- competitors face Equation (3),')
        print('        and the headline PathologyLoss numbers are regenerated')
        print('        so the paper and the experiment agree. Costs one rerun.')
        print()
        print('  (a) is cheaper. (b) is what a reviewer reproducing Equation (3)')
        print('  would get. Do not mix them: a table where the primary arm is')
        print('  as-run and the text defines it as-written is the failure mode')
        print('  this script exists to prevent.')
    print(bar)


if __name__ == '__main__':
    main()
