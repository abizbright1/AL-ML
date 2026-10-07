#!/usr/bin/env python3
"""
diagnose_objective.py — what the as-run objective actually does
===============================================================

Three measurements the loss value alone cannot give you.

  A. INPUT-DEPENDENCE OF THE REGION WEIGHTS
     ClinicalRiskScore reads seven image features -- the three region volume
     fractions, a ratio, and three heterogeneity terms -- so R_WT/R_TC/R_ET
     vary with the IMAGE, not only with the module's initialisation. An
     earlier check held the input fixed and varied the init, which measured
     the wrong axis. This varies lesion geometry and intensity statistics at
     fixed init, and reports the spread of the realised weight ratio.

  B. GRADIENT INFLUENCE OF EACH TERM
     A term with a small scalar value can still steer training. What matters
     is the norm of its gradient with respect to the prediction, relative to
     the other terms. Reported as a share of total gradient magnitude.

  C. HISTORICAL vs CURRENT CROSS-MODAL TERM
     Before commit 6635296 the term pooled each modality to 1x1 before taking
     a cosine, which makes the cosine identically 1 and the loss identically
     0 with exactly zero gradient. After the fix it is non-zero. Both are
     evaluated so the two code states can be told apart.

Usage
-----
    python3 diagnose_objective.py                      # synthetic geometries
    python3 diagnose_objective.py --data ~/BraTS2021   # real batches (preferred)
    python3 diagnose_objective.py --n 40 --device mps

With --data the weights are measured on real patients, which is what the
manuscript needs to quote. Without it, the synthetic sweep spans a wider
range of geometries than BraTS contains and is a stress test, not a
substitute.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pp_mae'))

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
#  Synthetic geometries spanning plausible lesion configurations
# ═══════════════════════════════════════════════════════════════════════════

def make_case(wt_frac, tc_of_wt, et_of_tc, contrast, noise, H=96, W=96,
              C=4, B=2, device='cpu', gen=None):
    """A batch whose lesion geometry and intensity statistics are controlled."""
    side = max(2, int(round(math.sqrt(wt_frac * H * W))))
    tc = max(1, int(round(side * math.sqrt(tc_of_wt))))
    et = max(1, int(round(tc * math.sqrt(et_of_tc))))
    c = H // 2

    seg = torch.zeros(B, 1, H, W, dtype=torch.long)
    def box(n): return slice(max(0, c - n // 2), min(H, c - n // 2 + n))
    seg[:, :, box(side), box(side)] = 2
    seg[:, :, box(tc),   box(tc)]   = 1
    seg[:, :, box(et),   box(et)]   = 3

    target = 0.35 + noise * torch.randn(B, C, H, W, generator=gen)
    target[:, :, box(side), box(side)] += contrast          # lesion is brighter
    target = target.clamp(0, 1)
    pred = (target + 0.08 * torch.randn(B, C, H, W, generator=gen)).clamp(0, 1)
    return (pred.to(device).requires_grad_(True), target.to(device), seg.to(device))


def real_batches(data_dir, n, device):
    """Yield real BraTS batches if the dataset is reachable."""
    from brats_loader import BraTSDataset
    ds = BraTSDataset(os.path.expanduser(data_dir), max_subjects=20, cache=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
    out = []
    for i, b in enumerate(dl):
        if i >= n:
            break
        out.append((b['noisy'].to(device).requires_grad_(True),
                    b['target'].to(device), b['seg'].to(device)))
    return out


# ═══════════════════════════════════════════════════════════════════════════

def grad_norm(term, pred):
    """L2 norm of d(term)/d(pred). Zero if the term has no gradient path."""
    if not term.requires_grad:
        return 0.0
    g = torch.autograd.grad(term, pred, retain_graph=True, allow_unused=True)[0]
    return 0.0 if g is None else float(g.norm())


class HistoricalCrossModal(torch.nn.Module):
    """The pre-6635296 implementation: pool each modality to 1x1, then cosine.

    A (B,1) vector has cosine similarity 1 with any other (B,1) vector of the
    same sign, so the term is identically 0 and its gradient is identically
    zero. Reproduced here so the two code states can be distinguished.
    """
    PAIRS = [(1, 2), (2, 3)]

    def forward(self, pred, target):
        tot = torch.zeros((), device=pred.device, dtype=pred.dtype)
        for i, j in self.PAIRS:
            pi = F.adaptive_avg_pool2d(pred[:, i:i + 1], (1, 1)).flatten(1)
            pj = F.adaptive_avg_pool2d(pred[:, j:j + 1], (1, 1)).flatten(1)
            tot = tot + (1 - F.cosine_similarity(pi, pj, dim=1)).mean()
        return tot / len(self.PAIRS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=None, help='BraTS root (preferred)')
    ap.add_argument('--n', type=int, default=24, help='number of cases')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    dev = args.device

    from losses import PPMAELoss, CrossModalConsistencyLoss, GlobalReconLoss
    from comparison_losses import RegionWeightedL1

    torch.manual_seed(args.seed)
    loss = PPMAELoss(lambda1=1.0, lambda2=0.5,
                     mode='clinical_risk', ssim_weight=0.5).to(dev)
    xm_now = CrossModalConsistencyLoss().to(dev)
    xm_old = HistoricalCrossModal().to(dev)
    path_eq3 = RegionWeightedL1(mode='nested')

    # ── build cases ─────────────────────────────────────────────────────────
    if args.data:
        cases = real_batches(args.data, args.n, dev)
        source = f'real BraTS batches from {args.data}'
    else:
        g = torch.Generator().manual_seed(args.seed)
        grid = []
        for wt in (0.01, 0.04, 0.10, 0.20):
            for tc in (0.2, 0.5, 0.8):
                for et in (0.1, 0.5):
                    grid.append((wt, tc, et))
        cases = [make_case(wt, tc, et, contrast=0.25, noise=0.10,
                           device=dev, gen=g) for wt, tc, et in grid[:args.n]]
        source = f'{len(cases)} synthetic geometries (stress test, not BraTS)'

    bar = '=' * 76
    print(bar)
    print('  OBJECTIVE DIAGNOSTICS')
    print(bar)
    print(f'  source : {source}')
    print(f'  device : {dev}   weighting-net init seed : {args.seed} (FIXED)')
    print()

    # ── A. input-dependence of the region weights ───────────────────────────
    Rs, geom = [], []
    for pred, target, seg in cases:
        with torch.no_grad():
            d = loss(pred, target, seg)
        Rs.append((float(d['R_WT']), float(d['R_TC']), float(d['R_ET'])))
        n = int((seg > 0).sum()), int(((seg == 1) | (seg == 3)).sum()), int((seg == 3).sum())
        geom.append(n)

    import statistics as st
    print('  A. REGION WEIGHTS ACROSS INPUTS  (initialisation held fixed)')
    print('  ' + '-' * 72)
    print(f'     {"":8} {"min":>8} {"mean":>8} {"max":>8} {"SD":>8} {"range":>9}   stated')
    for j, nm in enumerate(['R_WT', 'R_TC', 'R_ET']):
        v = [r[j] for r in Rs]
        sd = st.stdev(v) if len(v) > 1 else 0.0
        print(f'     {nm:<8} {min(v):8.3f} {st.mean(v):8.3f} {max(v):8.3f} '
              f'{sd:8.3f} {max(v)-min(v):9.3f}   {[1,2,3][j]}')
    r_et = [r[2] / max(r[0], 1e-9) for r in Rs]
    r_tc = [r[1] / max(r[0], 1e-9) for r in Rs]
    print(f'\n     realised ratio ET:TC:WT  '
          f'{st.mean(r_et):.2f} : {st.mean(r_tc):.2f} : 1.00'
          f'   (range of ET:WT  {min(r_et):.2f}-{max(r_et):.2f})')
    worst = max(abs(st.mean([r[j] for r in Rs]) - [1, 2, 3][j]) for j in range(3))
    spread = max(max(r[j] for r in Rs) - min(r[j] for r in Rs) for j in range(3))
    print(f'     largest mean deviation from the prior : {worst:.3f}')
    print(f'     largest spread ACROSS INPUTS          : {spread:.3f}')
    if spread < 0.10:
        print('     -> weights are effectively constant; Eq.(3) describes them.')
    elif spread < 0.50:
        print('     -> mildly input-dependent; quote the measured range, not 3:2:1.')
    else:
        print('     -> STRONGLY input-dependent. Eq.(3) does NOT describe the')
        print('        weighting, and the manuscript must say what it does.')
    print()

    # ── B. gradient influence of each term ──────────────────────────────────
    print('  B. GRADIENT INFLUENCE  (||d term / d pred||, share of total)')
    print('  ' + '-' * 72)
    acc = {k: [] for k in ('L1', '0.5*SSIM', 'pathology', '0.5*xm_now', '0.5*xm_old')}
    for pred, target, seg in cases[:8]:
        g_recon = GlobalReconLoss(ssim_weight=0.5, channel=pred.shape[1]).to(dev)
        l1 = F.l1_loss(pred, target)
        ssim_part = g_recon(pred, target) - l1
        d = loss(pred, target, seg)
        acc['L1'].append(grad_norm(l1, pred))
        acc['0.5*SSIM'].append(grad_norm(ssim_part, pred))
        acc['pathology'].append(grad_norm(d['pathology'], pred))
        acc['0.5*xm_now'].append(grad_norm(0.5 * xm_now(pred, target), pred))
        acc['0.5*xm_old'].append(grad_norm(0.5 * xm_old(pred, target), pred))
    live = ['L1', '0.5*SSIM', 'pathology', '0.5*xm_now']
    tot = sum(st.mean(acc[k]) for k in live)
    print(f'     {"term":<14} {"grad norm":>12} {"share":>9}')
    for k in live + ['0.5*xm_old']:
        m = st.mean(acc[k])
        sh = f'{100*m/tot:8.2f}%' if k in live else '   (hist)'
        print(f'     {k:<14} {m:12.6f} {sh}')
    xm_share = 100 * st.mean(acc['0.5*xm_now']) / tot
    print()
    print(f'     cross-modal, current implementation : {xm_share:.2f}% of gradient')
    print(f'     cross-modal, historical (pre-6635296): '
          f'{st.mean(acc["0.5*xm_old"]):.3g} — exactly zero by construction')
    if xm_share < 0.5:
        print('     -> negligible either way; the 6635296 fix changed little.')
    else:
        print('     -> NOT negligible. Runs before and after 6635296 were')
        print('        trained on materially different objectives.')
    print()

    # ── C. how far the as-run objective sits from Eq.(3) ───────────────────
    print('  C. AS-RUN vs EQUATION (3)')
    print('  ' + '-' * 72)
    rel = []
    for pred, target, seg in cases[:8]:
        with torch.no_grad():
            tr = float(loss(pred, target, seg)['total'])
            tw = float(F.l1_loss(pred, target)) + float(path_eq3(pred, target, seg))
        rel.append(abs(tr - tw) / max(abs(tw), 1e-12))
    print(f'     relative difference  mean {st.mean(rel):.1%}   '
          f'min {min(rel):.1%}   max {max(rel):.1%}')
    print()
    print(bar)
    print('  The treatment arm carries L1 + 0.5*SSIM + pathology + 0.5*cross-modal.')
    print('  A pure-L1 control differs from it in FOUR ways, so that contrast does')
    print('  not isolate the pathology term. Use the matched base control.')
    print(bar)


if __name__ == '__main__':
    main()
