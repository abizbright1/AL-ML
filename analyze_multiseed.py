#!/usr/bin/env python3
"""
analyze_multiseed.py — variance, mixed-effects, equivalence and effect size
===========================================================================

Consumes the tidy CSV from ``aggregate_seeds.py`` and answers the two
reviewer criticisms that a single seed-42 run cannot answer:

  ISSUE A  "results come from one run, so the reported gain could be seed
            noise"
            -> per-seed effect, SD across seeds, cluster-bootstrap 95% CI

  ISSUE B  "a non-significant difference is reported as no difference"
            -> two one-sided tests (TOST) against an explicitly declared
               equivalence margin, so 'no meaningful difference' becomes a
               positive claim rather than the absence of one

Outputs, in order:

  (a) per-seed, per-backbone mean paired effect  (PathologyLoss - L1),
      the SD across seeds, and a percentile bootstrap 95% CI that resamples
      *subjects*, not slices
  (b) a mixed-effects model with subject and seed as random effects
      (statsmodels).  If statsmodels is unavailable the script falls back to a
      run-level paired test on the five per-seed mean deltas and says so
      loudly — the fallback is valid but far less powerful, and the printed
      report marks which one produced the numbers
  (c) TOST equivalence test.  --margin is REQUIRED and has no default: the
      smallest effect still worth reporting is a clinical judgement, not a
      statistical one, and a silent default would invent that judgement
  (d) matched-pairs rank-biserial correlation, the effect size that belongs
      with a Wilcoxon signed-rank test
  (e) a paste-ready manuscript block

Requires numpy and scipy.  statsmodels is optional (see (b)).

Usage
-----
    python3 analyze_multiseed.py --tidy results/sweep/tidy_results.csv \\
                                 --metric dice_et --margin 0.02

    # equivalence of the two backbones' *gains* rather than loss-vs-loss
    python3 analyze_multiseed.py --tidy ... --margin 0.02 --contrast backbone

On choosing --margin
--------------------
It is the largest difference you are willing to call "practically nothing",
in the metric's own units, decided BEFORE looking at the result.  For Dice_ET,
0.02 is a defensible starting point (roughly the inter-rater floor reported
for enhancing tumour), but it is your claim to defend, not this script's.
State the margin and its justification in the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

BASELINE = 'L1'
TREATMENT = 'PathologyLoss'


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Multi-seed variance, mixed-effects, TOST and effect size.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tidy', default='results/sweep/tidy_results.csv',
                   help='Tidy CSV from aggregate_seeds.py')
    p.add_argument('--metric', default='dice_et',
                   help='Metric column to analyse (default: dice_et)')
    p.add_argument('--margin', type=float, required=True,
                   help='REQUIRED. Equivalence margin in the metric\'s own '
                        'units, chosen a priori. No default is supplied on '
                        'purpose: it is a clinical judgement.')
    p.add_argument('--contrast', choices=['loss', 'backbone'], default='loss',
                   help="'loss' (default): PathologyLoss vs L1 within each "
                        "backbone.  'backbone': the PathologyLoss gain on "
                        "SwinIR vs on Uformer, i.e. does the benefit transfer?")
    p.add_argument('--alpha', type=float, default=0.05,
                   help='Significance level for both NHST and TOST (0.05)')
    p.add_argument('--n_boot', type=int, default=10000,
                   help='Bootstrap resamples (default 10000)')
    p.add_argument('--boot_seed', type=int, default=0,
                   help='Seed for the bootstrap RNG, so the CI is reproducible')
    p.add_argument('--out', default=None,
                   help='Also write the report to this text file')
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  Loading
# ═══════════════════════════════════════════════════════════════════════════

def load(tidy_path: str, metric: str):
    """-> {(seed, backbone, loss): {subject_id: value}}"""
    if not os.path.isfile(tidy_path):
        sys.exit(f'Tidy CSV not found: {tidy_path}\n'
                 f'Run aggregate_seeds.py first.')
    table: dict[tuple, dict[str, float]] = defaultdict(dict)
    seen_metrics = set()
    with open(tidy_path, newline='') as fh:
        for r in csv.DictReader(fh):
            seen_metrics.add(r['metric'])
            if r['metric'] != metric:
                continue
            try:
                v = float(r['value'])
            except ValueError:
                continue
            if math.isnan(v):
                continue
            key = (int(r['seed']), r['backbone'], r['loss'])
            table[key][r['subject_id']] = v
    if not table:
        sys.exit(f'No rows for metric {metric!r}. '
                 f'Available: {sorted(seen_metrics)}')
    return table


def paired_deltas(table, backbone, seed):
    """Per-subject (TREATMENT - BASELINE) for one backbone at one seed.

    Only subjects present in BOTH arms are used; an unpaired subject carries no
    information about the difference and including it would bias the mean.
    """
    a = table.get((seed, backbone, TREATMENT), {})
    b = table.get((seed, backbone, BASELINE), {})
    subs = sorted(set(a) & set(b))
    return subs, np.array([a[s] - b[s] for s in subs], dtype=float)


def transfer_deltas(table, backbones, seed):
    """Per-subject difference of gains between two backbones, at one seed.

    (gain on backbones[0]) - (gain on backbones[1]).  This is the quantity the
    'does the benefit transfer across architectures?' claim rests on, and it is
    the one where equivalence — not a non-significant p — is the right tool:
    showing the two gains are the same within a margin is a positive finding.
    """
    b0, b1 = backbones
    s0, d0 = paired_deltas(table, b0, seed)
    s1, d1 = paired_deltas(table, b1, seed)
    m0, m1 = dict(zip(s0, d0)), dict(zip(s1, d1))
    subs = sorted(set(m0) & set(m1))
    return subs, np.array([m0[s] - m1[s] for s in subs], dtype=float)


# ═══════════════════════════════════════════════════════════════════════════
#  (a) variance across seeds
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_ci(deltas: np.ndarray, n_boot: int, rng, alpha=0.05):
    """Percentile bootstrap CI for the mean, resampling subjects (clusters).

    Resampling subjects rather than slices is the point: slices within a
    subject are correlated, so a slice-level bootstrap would understate the
    interval exactly the way slice-level p-values understate a p-value.
    """
    if len(deltas) < 2:
        return math.nan, math.nan
    idx = rng.integers(0, len(deltas), size=(n_boot, len(deltas)))
    means = deltas[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def rank_biserial(deltas: np.ndarray):
    """Matched-pairs rank-biserial correlation, r = (W+ - W-) / (W+ + W-).

    Ranges -1..+1.  It is the effect size that belongs with Wilcoxon: it says
    what fraction of the signed rank mass favours the treatment, and unlike
    Cohen's d it makes no normality assumption.  Zero pairs are dropped, as in
    the 'wilcox' zero-handling method.
    """
    d = deltas[deltas != 0]
    if d.size == 0:
        return 0.0, 0
    ranks = stats.rankdata(np.abs(d))
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    total = w_pos + w_neg
    return float((w_pos - w_neg) / total) if total else 0.0, int(d.size)


def describe_rb(r):
    a = abs(r)
    return ('negligible' if a < 0.1 else 'small' if a < 0.3
            else 'medium' if a < 0.5 else 'large')


# ═══════════════════════════════════════════════════════════════════════════
#  (b) mixed-effects model
# ═══════════════════════════════════════════════════════════════════════════

def mixed_effects(table, backbone, seeds, out):
    """value ~ loss, random intercepts for subject and for seed.

    This is the right model for the design: each subject is measured under
    both losses at every seed, so subject and seed are crossed random effects
    and the loss coefficient is estimated within-subject.  Returns a dict, or
    None if statsmodels is missing (the caller then uses the fallback).
    """
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError as e:
        out(f'    statsmodels/pandas unavailable ({e.name}); using fallback.')
        return None

    rows = []
    for seed in seeds:
        for loss in (BASELINE, TREATMENT):
            for subj, v in table.get((seed, backbone, loss), {}).items():
                rows.append({'value': v, 'loss': loss,
                             'subject': subj, 'seed': str(seed)})
    if len(rows) < 8:
        out('    too few observations for a mixed model; using fallback.')
        return None

    df = pd.DataFrame(rows)
    df['loss'] = pd.Categorical(df['loss'], categories=[BASELINE, TREATMENT])
    try:
        md = smf.mixedlm('value ~ loss', df, groups=df['subject'],
                         vc_formula={'seed': '0 + C(seed)'})
        fit = md.fit(reml=True, method='lbfgs')
    except Exception as e:                                   # convergence etc.
        out(f'    mixed model failed to fit ({type(e).__name__}: {e}); '
            f'using fallback.')
        return None

    term = next((t for t in fit.params.index if t.startswith('loss')), None)
    if term is None:
        out('    loss term absent from the fit; using fallback.')
        return None
    ci = fit.conf_int().loc[term]
    return {'coef': float(fit.params[term]), 'se': float(fit.bse[term]),
            'p': float(fit.pvalues[term]),
            'ci': (float(ci[0]), float(ci[1])),
            'n_obs': int(fit.nobs), 'converged': bool(fit.converged)}


def run_level_fallback(per_seed_means: np.ndarray):
    """Paired test on the k per-seed mean deltas — the documented fallback.

    Each seed contributes exactly one number, so this respects the run as the
    unit of replication and cannot pseudo-replicate.  The cost is power: with
    k=5 a two-sided Wilcoxon signed-rank test cannot reach p < 0.05 at all
    (its minimum attainable p is 0.0625), so the t-statistic is reported
    alongside and the limitation is stated in the output.
    """
    k = len(per_seed_means)
    res = {'k': k, 'mean': float(per_seed_means.mean()) if k else math.nan,
           'sd': float(per_seed_means.std(ddof=1)) if k > 1 else math.nan,
           't_p': math.nan, 'w_p': math.nan, 'w_floor': k < 6}
    if k > 1:
        res['t_p'] = float(stats.ttest_1samp(per_seed_means, 0.0).pvalue)
        try:
            res['w_p'] = float(
                stats.wilcoxon(per_seed_means, alternative='two-sided').pvalue)
        except ValueError:
            pass
    return res


# ═══════════════════════════════════════════════════════════════════════════
#  (c) TOST
# ═══════════════════════════════════════════════════════════════════════════

def tost(deltas: np.ndarray, margin: float, alpha=0.05):
    """Two one-sided t-tests for equivalence of paired differences.

    H01: delta <= -margin      H02: delta >= +margin
    Rejecting both concludes equivalence: the true effect lies inside
    (-margin, +margin).  Equivalently, the (1-2*alpha) CI — the 90% CI at
    alpha=0.05 — sits entirely inside the margins; that CI is returned so the
    reader can check the conclusion by eye.

    Note the asymmetry a reviewer will look for: failing to reject in an
    ordinary test is NOT evidence of no effect, whereas rejecting both of
    these IS.
    """
    n = len(deltas)
    if n < 2:
        return None
    mean = float(deltas.mean())
    sd = float(deltas.std(ddof=1))
    se = sd / math.sqrt(n)
    df = n - 1
    if se == 0:
        p_lo = p_hi = 0.0 if abs(mean) < margin else 1.0
        return {'n': n, 'mean': mean, 'se': 0.0, 'p_lower': p_lo,
                'p_upper': p_hi, 'p_tost': max(p_lo, p_hi),
                'ci90': (mean, mean), 'equivalent': abs(mean) < margin,
                'margin': margin}

    # H01: delta <= -margin   (upper-tailed)
    p_lower = float(stats.t.sf((mean + margin) / se, df))
    # H02: delta >= +margin   (lower-tailed)
    p_upper = float(stats.t.cdf((mean - margin) / se, df))
    t_crit = stats.t.ppf(1 - alpha, df)
    ci = (mean - t_crit * se, mean + t_crit * se)
    p_tost = max(p_lower, p_upper)
    return {'n': n, 'mean': mean, 'se': se, 'p_lower': p_lower,
            'p_upper': p_upper, 'p_tost': p_tost,
            'ci90': (float(ci[0]), float(ci[1])),
            'equivalent': p_tost < alpha, 'margin': margin}


def verdict(tost_res, p_nhst, margin, alpha, metric):
    """Plain-English reading of the NHST x TOST 2x2."""
    if tost_res is None:
        return 'Not enough paired observations to test equivalence.'
    sig = (not math.isnan(p_nhst)) and p_nhst < alpha
    equ = tost_res['equivalent']
    lo, hi = tost_res['ci90']
    band = f'90% CI [{lo:+.4f}, {hi:+.4f}], margin +/-{margin:g}'
    if sig and not equ:
        return (f'DIFFERENT AND NON-TRIVIAL. The effect is statistically '
                f'detectable and cannot be ruled larger than the {margin:g} '
                f'{metric} margin. Report it as a real difference. {band}.')
    if sig and equ:
        return (f'STATISTICALLY DETECTABLE BUT PRACTICALLY EQUIVALENT. The '
                f'effect is real but smaller than {margin:g} {metric}. Do not '
                f'present it as a clinically meaningful gain. {band}.')
    if (not sig) and equ:
        return (f'EQUIVALENT. Absence of effect is positively demonstrated, '
                f'not merely unproven: the true difference is inside '
                f'+/-{margin:g} {metric}. This is the claim to make instead of '
                f'"no significant difference". {band}.')
    return (f'INCONCLUSIVE. The study can neither detect a difference nor rule '
            f'out one as large as {margin:g} {metric} — an underpowered result, '
            f'which is NOT evidence of equivalence. Say so, or add seeds or '
            f'subjects. {band}.')


# ═══════════════════════════════════════════════════════════════════════════
#  Report
# ═══════════════════════════════════════════════════════════════════════════

def fmt(x, nd=4):
    return 'n/a' if x is None or (isinstance(x, float) and math.isnan(x)) \
        else f'{x:.{nd}f}'


def pfmt(p, nd=3):
    """Format a p-value without ever rounding it to a misleading '0.000'."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return 'n/a'
    return f'{p:.{nd}f}' if p >= 10 ** -nd else f'< {10 ** -nd:g}'


def peq(p, nd=3):
    """'= 0.202' or '< 0.001' — so prose never reads 'p = < 0.001'."""
    t = pfmt(p, nd)
    return t if t.startswith('<') else f'= {t}'


def main():
    args = parse_args()
    if args.margin <= 0:
        sys.exit('--margin must be positive.')

    lines: list[str] = []
    def out(s=''):
        print(s)
        lines.append(s)

    table = load(args.tidy, args.metric)
    seeds = sorted({k[0] for k in table})
    backbones = sorted({k[1] for k in table
                        if k[2] in (BASELINE, TREATMENT)})
    rng = np.random.default_rng(args.boot_seed)

    # Each group is (label, seed -> (subjects, deltas), backbone-for-MixedLM).
    # The third element is None where no single-factor mixed model applies, in
    # which case the run-level fallback is used by construction, not by failure.
    if args.contrast == 'backbone':
        if len(backbones) != 2:
            sys.exit(f'--contrast backbone needs exactly two backbones; '
                     f'found {backbones}')
        bb = tuple(backbones)
        groups = [(f'{bb[0]} gain - {bb[1]} gain',
                   lambda seed, bb=bb: transfer_deltas(table, bb, seed), None)]
    else:
        groups = [(b, lambda seed, b=b: paired_deltas(table, b, seed), b)
                  for b in backbones]

    bar = '=' * 76
    out(bar)
    out(f'  MULTI-SEED ANALYSIS — {args.metric}')
    out(bar)
    out(f'  tidy file : {args.tidy}')
    out(f'  contrast  : {TREATMENT} - {BASELINE}'
        if args.contrast == 'loss' else
        f'  contrast  : PathologyLoss gain, SwinIR vs Uformer')
    out(f'  seeds     : {seeds}  (k={len(seeds)})')
    out(f'  backbones : {backbones}')
    out(f'  alpha     : {args.alpha}    equivalence margin: +/-{args.margin:g}')
    if len(seeds) < 3:
        out('  !! Fewer than 3 seeds: SD and the run-level test are not')
        out('     meaningful. Finish the sweep before quoting these numbers.')
    out()

    summary: dict[str, dict] = {}

    # ── (a) per-seed effects ─────────────────────────────────────────────────
    out('-' * 76)
    out('(a) PER-SEED EFFECT, SD ACROSS SEEDS, BOOTSTRAP 95% CI')
    out('-' * 76)
    out('    CIs resample subjects (clusters), 95% percentile bootstrap,')
    out(f'    {args.n_boot} resamples, bootstrap RNG seed {args.boot_seed}.')
    out()

    per_seed_means: dict[str, np.ndarray] = {}
    pooled: dict[str, tuple[list, np.ndarray]] = {}

    for label, dfun, _mm in groups:
        out(f'  {label}')
        out(f'    {"seed":>6} {"n_subj":>7} {"mean d":>10} {"SD(subj)":>10} '
            f'{"95% CI (bootstrap)":>26}')
        means = []
        for seed in seeds:
            subs, d = dfun(seed)
            if d.size == 0:
                out(f'    {seed:>6} {"—":>7}   (no paired subjects)')
                continue
            lo, hi = bootstrap_ci(d, args.n_boot, rng, 1 - 0.95)
            sd = d.std(ddof=1) if d.size > 1 else math.nan
            out(f'    {seed:>6} {d.size:>7} {d.mean():>+10.4f} {fmt(sd):>10} '
                f'{f"[{lo:+.4f}, {hi:+.4f}]":>26}')
            means.append(d.mean())
        means = np.array(means, dtype=float)
        per_seed_means[label] = means

        # Average each subject's delta over seeds -> one delta per subject.
        by_subject: dict[str, list[float]] = defaultdict(list)
        for seed in seeds:
            subs, d = dfun(seed)
            for s, v in zip(subs, d):
                by_subject[s].append(v)
        subs = sorted(by_subject)
        pooled_d = np.array([np.mean(by_subject[s]) for s in subs])
        pooled[label] = (subs, pooled_d)

        if means.size:
            sd_seed = means.std(ddof=1) if means.size > 1 else math.nan
            out(f'    {"":>6} {"":>7} {"-" * 10}')
            out(f'    across seeds: mean {means.mean():+.4f}   '
                f'SD {fmt(sd_seed)}   '
                f'range [{means.min():+.4f}, {means.max():+.4f}]')
            if means.size > 1 and not math.isnan(sd_seed) and sd_seed > 0:
                ratio = abs(means.mean()) / sd_seed
                out(f'    effect / seed-SD = {ratio:.2f}'
                    + ('   <-- effect is within seed noise'
                       if ratio < 1 else '   (effect exceeds seed noise)'))
            if means.size > 1 and (means.min() > 0) != (means.max() > 0):
                out('    !! The effect CHANGES SIGN across seeds. Any claim '
                    'of a consistent')
                out('       direction is not supported by these runs.')
        out()

    # ── (b) mixed-effects ────────────────────────────────────────────────────
    out('-' * 76)
    out('(b) MIXED-EFFECTS MODEL  (random intercepts: subject, seed)')
    out('-' * 76)
    out('    value ~ loss + (1|subject) + (1|seed)')
    out()
    for label, dfun, mm_backbone in groups:
        out(f'  {label}')
        if mm_backbone is None:
            out('    a single-factor mixed model does not apply to a '
                'difference of gains;')
            out('    the run-level test below is the primary inference.')
            me = None
        else:
            me = mixed_effects(table, mm_backbone, seeds, out)
        if me:
            lo, hi = me['ci']
            out(f'    method    : statsmodels MixedLM (REML)'
                f'{"" if me["converged"] else "   !! DID NOT CONVERGE"}')
            out(f'    n_obs     : {me["n_obs"]}')
            out(f'    coef      : {me["coef"]:+.4f}  (SE {me["se"]:.4f})')
            out(f'    95% CI    : [{lo:+.4f}, {hi:+.4f}]')
            out(f'    p         : {me["p"]:.4g}')
            summary.setdefault(label, {})['model'] = 'mixed'
            summary[label]['p'] = me['p']
            summary[label]['coef'] = me['coef']
        else:
            fb = run_level_fallback(per_seed_means[label])
            out('    method    : FALLBACK — run-level paired test on the '
                f'{fb["k"]} per-seed mean deltas')
            out('                (valid but low-powered; the seed is the unit '
                'of replication)')
            out(f'    mean      : {fmt(fb["mean"])}   SD {fmt(fb["sd"])}')
            out(f'    t-test p  : {pfmt(fb["t_p"], 4)}')
            out(f'    Wilcoxon p: {fmt(fb["w_p"], 4)}'
                + ('   (k<6: min attainable two-sided p is 0.0625, so this '
                   'test CANNOT reach 0.05)' if fb['w_floor'] else ''))
            summary.setdefault(label, {})['model'] = 'run-level fallback'
            summary[label]['p'] = fb['t_p']
            summary[label]['coef'] = fb['mean']
        out()

    # ── (c) TOST + (d) rank-biserial ─────────────────────────────────────────
    out('-' * 76)
    out(f'(c) EQUIVALENCE (TOST)  and  (d) MATCHED-PAIRS EFFECT SIZE')
    out('-' * 76)
    out('    Both computed on per-subject deltas averaged over seeds, so each')
    out('    subject contributes exactly one paired observation.')
    out()
    for label, _dfun, _mm in groups:
        subs, d = pooled[label]
        out(f'  {label}   (n = {len(subs)} subjects)')
        if d.size < 2:
            out('    too few paired subjects.')
            out()
            continue

        try:
            p_nhst = float(stats.wilcoxon(d, alternative='two-sided').pvalue)
        except ValueError:
            p_nhst = math.nan
        r_rb, n_nonzero = rank_biserial(d)
        t = tost(d, args.margin, args.alpha)

        out(f'    mean delta       : {d.mean():+.4f}   '
            f'median {np.median(d):+.4f}')
        out(f'    Wilcoxon p       : {pfmt(p_nhst, 4)}  '
            f'(two-sided, subject-level)')
        out(f'    rank-biserial r  : {r_rb:+.3f}  ({describe_rb(r_rb)}, '
            f'{n_nonzero} non-zero pairs)')
        out(f'    TOST p_lower     : {t["p_lower"]:.4g}   '
            f'(H0: delta <= {-args.margin:g})')
        out(f'    TOST p_upper     : {t["p_upper"]:.4g}   '
            f'(H0: delta >= {+args.margin:g})')
        out(f'    TOST p (max)     : {t["p_tost"]:.4g}   '
            f'-> {"equivalent" if t["equivalent"] else "NOT equivalent"} '
            f'at alpha={args.alpha}')
        out(f'    90% CI           : [{t["ci90"][0]:+.4f}, '
            f'{t["ci90"][1]:+.4f}]')
        out(f'    VERDICT          : {verdict(t, p_nhst, args.margin, args.alpha, args.metric)}')
        out()

        summary.setdefault(label, {}).update(
            {'n': len(subs), 'mean': float(d.mean()), 'p_nhst': p_nhst,
             'r_rb': r_rb, 'tost': t})

    # ── (e) manuscript block ─────────────────────────────────────────────────
    out(bar)
    out('(e) PASTE-READY MANUSCRIPT TEXT')
    out(bar)
    out()
    k = len(seeds)
    out('  --- Methods ---')
    out(f'  Every configuration was trained {k} times with independent random')
    out(f'  seeds ({", ".join(str(s) for s in seeds)}); the seed controls weight')
    out('  initialisation, the subject-level train/validation split and batch')
    out('  ordering. All randomness is seeded through a single entry point '
        '(set_all_seeds),')
    out('  including each DataLoader generator. Runs on Apple MPS are seeded '
        'but not')
    out('  bit-deterministic, as the Metal backend offers no determinism '
        'guarantee;')
    out('  seeding therefore makes runs comparable rather than identical.')
    out('  Metrics are averaged within a subject before any test, so the '
        'subject,')
    out('  not the slice, is the unit of analysis. Effects are paired '
        'differences')
    out(f'  ({TREATMENT} minus {BASELINE}) on the same subjects. Uncertainty '
        'is a')
    out(f'  percentile bootstrap over subjects ({args.n_boot} resamples). '
        'Inference uses')
    model_used = {v.get('model') for v in summary.values()}
    if model_used == {'mixed'}:
        out('  a linear mixed-effects model with random intercepts for subject '
            'and seed.')
    elif 'mixed' in model_used:
        out('  a linear mixed-effects model with random intercepts for subject '
            'and seed,')
        out('  with a run-level paired test on per-seed means where that model '
            'did not fit.')
    else:
        out('  a run-level paired test on the per-seed mean differences, the '
            'seed being')
        out('  the unit of replication.')
    out(f'  Equivalence was assessed by two one-sided tests against an a-priori')
    out(f'  margin of {args.margin:g} {args.metric}, so that the absence of an '
        'effect is')
    out('  reported as a positive finding rather than as a failure to reject.')
    out()
    out('  --- Results ---')
    for label, _dfun, _mm in groups:
        s = summary.get(label)
        if not s or ('tost' in s and s['tost'] is None):
            continue
        m = per_seed_means.get(label, np.array([]))
        sd = m.std(ddof=1) if m.size > 1 else math.nan
        t = s.get('tost')
        lead = (f'  On {label}, {TREATMENT} changed {args.metric} by '
                f'{s["mean"]:+.3f}') if args.contrast == 'loss' else (
                f'  The {TREATMENT} gain in {args.metric} differed between '
                f'backbones ({label}) by {s["mean"]:+.3f}')
        out(lead)
        out(f'  (mean over {s["n"]} subjects; across-seed SD {fmt(sd, 3)}, '
            f'k={k} seeds;')
        if t:
            out(f'  90% CI [{t["ci90"][0]:+.3f}, {t["ci90"][1]:+.3f}]; '
                f'Wilcoxon p {peq(s["p_nhst"])},')
            out(f'  rank-biserial r = {s["r_rb"]:+.2f}; TOST against '
                f'+/-{args.margin:g}: p {peq(t["p_tost"])},')
            out(f'  {"equivalence supported" if t["equivalent"] else "equivalence not supported"}).')
        out()
    out('  --- Limitation to state ---')
    inconclusive = [b for b, s in summary.items()
                    if s.get('tost') and not s['tost']['equivalent']
                    and not (s.get('p_nhst', 1) < args.alpha)]
    if inconclusive:
        out(f'  For {", ".join(inconclusive)}, the data neither detect a '
            'difference nor')
        out(f'  exclude one of {args.margin:g} {args.metric}. This is reported '
            'as inconclusive;')
        out('  it is not evidence that the two objectives perform alike.')
    else:
        out('  Every contrast reached a decision (difference or equivalence) '
            'at the')
        out(f'  stated margin; no contrast is left inconclusive.')
    out()
    out(bar)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        print(f'\nReport written to {args.out}')


if __name__ == '__main__':
    main()
