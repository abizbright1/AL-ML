"""
Multi-level evaluation framework for PP-MAE.

Covers all metrics in the study protocol:
    Image Quality    : PSNR, SSIM, NRMSE
    Structural       : ICC, Bland-Altman, IEI (interchangeability index)
    Segmentation     : DSC, HD95  (WT / TC / ET)
    Grading          : AUROC  (WHO grade, IDH status)
    Qualitative      : Fleiss' kappa for inter-reader agreement
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Sequence


# ---------------------------------------------------------------------------
# Image quality
# ---------------------------------------------------------------------------

def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(data_range ** 2 / mse)


def nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Normalised root mean squared error (normalised by target RMS)."""
    rms_target = np.sqrt(np.mean(target ** 2))
    return np.sqrt(np.mean((pred - target) ** 2)) / (rms_target + 1e-8)


def ssim_numpy(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Simple 2-D SSIM without external dependencies."""
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_p, mu_t = pred.mean(), target.mean()
    sig_p  = pred.var()
    sig_t  = target.var()
    sig_pt = np.mean((pred - mu_p) * (target - mu_t))
    num = (2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)
    den = (mu_p ** 2 + mu_t ** 2 + C1) * (sig_p + sig_t + C2)
    return float(num / (den + 1e-8))


def compute_image_quality_metrics(
    pred: np.ndarray,   # (C, H, W) or (H, W)
    target: np.ndarray,
) -> dict:
    results = {}
    results["psnr"]  = psnr(pred, target)
    results["ssim"]  = ssim_numpy(pred, target)
    results["nrmse"] = nrmse(pred, target)
    return results


# ---------------------------------------------------------------------------
# Structural integrity — ICC and Bland-Altman
# ---------------------------------------------------------------------------

def icc(y1: np.ndarray, y2: np.ndarray, model: str = "ICC(2,1)") -> float:
    """
    Two-way mixed ICC (consistency).  y1, y2: 1-D arrays of measurements.
    Returns ICC value in [0, 1].
    """
    n = len(y1)
    data = np.column_stack([y1, y2])
    grand_mean = data.mean()
    row_means  = data.mean(axis=1)
    col_means  = data.mean(axis=0)

    SS_rows = 2 * np.sum((row_means - grand_mean) ** 2)
    SS_cols = n * np.sum((col_means - grand_mean) ** 2)
    SS_err  = np.sum((data - row_means[:, None] - col_means[None, :] + grand_mean) ** 2)

    MS_rows = SS_rows / (n - 1)
    MS_err  = SS_err  / ((n - 1) * (2 - 1))

    return float((MS_rows - MS_err) / (MS_rows + MS_err + 1e-8))


def bland_altman(
    m1: np.ndarray,
    m2: np.ndarray,
) -> dict:
    """
    Bland-Altman analysis for two measurement arrays.
    Returns mean difference (bias), 95% LoA, and proportional bias flag.
    """
    diff  = m1 - m2
    mean  = (m1 + m2) / 2
    bias  = diff.mean()
    sd    = diff.std(ddof=1)
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    # Proportional bias: Pearson r between mean and difference
    corr = np.corrcoef(mean, diff)[0, 1]
    return {
        "bias":        float(bias),
        "sd":          float(sd),
        "loa_upper":   float(loa_upper),
        "loa_lower":   float(loa_lower),
        "prop_bias_r": float(corr),        # |r| > 0.3 suggests proportional bias
    }


def interchangeability_index(
    m1: np.ndarray,
    m2: np.ndarray,
    acceptable_diff: float,
) -> float:
    """
    IEI (Fujita et al. 2025): fraction of pairs within acceptable difference.
    acceptable_diff is expressed in the same units as the measurements.
    """
    return float(np.mean(np.abs(m1 - m2) <= acceptable_diff))


# ---------------------------------------------------------------------------
# Segmentation — DSC and HD95  (BraTS subregions)
# ---------------------------------------------------------------------------

def dice_score(pred_bin: np.ndarray, target_bin: np.ndarray) -> float:
    intersection = (pred_bin & target_bin).sum()
    union        = pred_bin.sum() + target_bin.sum()
    if union == 0:
        return 1.0  # both empty → perfect
    return float(2 * intersection / union)


def hausdorff95(pred_bin: np.ndarray, target_bin: np.ndarray) -> float:
    """
    95th-percentile bidirectional Hausdorff distance.
    Requires scipy; falls back to inf if not available.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return float("inf")

    if pred_bin.sum() == 0 or target_bin.sum() == 0:
        return float("inf")

    dist_pred   = distance_transform_edt(~pred_bin)
    dist_target = distance_transform_edt(~target_bin)

    hd_pt = dist_target[pred_bin].ravel()
    hd_tp = dist_pred[target_bin].ravel()
    combined = np.concatenate([hd_pt, hd_tp])
    return float(np.percentile(combined, 95))


def segmentation_metrics(
    pred_seg: np.ndarray,    # integer label map (BraTS convention)
    target_seg: np.ndarray,
) -> dict:
    """Compute DSC and HD95 for WT, TC, ET subregions."""
    regions = {
        "WT": lambda s: s > 0,
        "TC": lambda s: (s == 1) | (s == 3),
        "ET": lambda s: s == 3,
    }
    results = {}
    for name, fn in regions.items():
        p_mask = fn(pred_seg)
        t_mask = fn(target_seg)
        results[f"DSC_{name}"]  = dice_score(p_mask, t_mask)
        results[f"HD95_{name}"] = hausdorff95(p_mask, t_mask)
    return results


# ---------------------------------------------------------------------------
# Grading — AUROC
# ---------------------------------------------------------------------------

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Binary AUROC via trapezoidal rule.  scores: continuous predictions in [0,1].
    labels: binary (0/1).
    """
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except ImportError:
        # Manual trapezoid implementation
        thresholds = np.sort(np.unique(scores))[::-1]
        tprs, fprs = [0.0], [0.0]
        pos = labels.sum()
        neg = len(labels) - pos
        for thr in thresholds:
            pred = (scores >= thr).astype(int)
            tp = ((pred == 1) & (labels == 1)).sum()
            fp = ((pred == 1) & (labels == 0)).sum()
            tprs.append(tp / (pos + 1e-8))
            fprs.append(fp / (neg + 1e-8))
        tprs.append(1.0)
        fprs.append(1.0)
        return float(np.trapezoid(tprs, fprs))


# ---------------------------------------------------------------------------
# Inter-reader agreement — Fleiss' kappa
# ---------------------------------------------------------------------------

def fleiss_kappa(ratings: np.ndarray) -> float:
    """
    Compute Fleiss' kappa for multiple raters.

    Args:
        ratings: (N_subjects, N_categories) count matrix — each row sums to
                 the number of raters assigning that rating to that subject.

    Returns:
        kappa in [-1, 1].
    """
    N, k = ratings.shape
    n = ratings[0].sum()              # raters per subject

    p_j = ratings.sum(axis=0) / (N * n)   # category marginals
    P_i = ((ratings ** 2).sum(axis=1) - n) / (n * (n - 1))
    P_bar  = P_i.mean()
    Pe_bar = (p_j ** 2).sum()

    if abs(1 - Pe_bar) < 1e-10:
        return 1.0
    return float((P_bar - Pe_bar) / (1 - Pe_bar))


# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

def evaluate_full(
    pred_imgs: list[np.ndarray],       # list of (C, H, W) denoised images
    target_imgs: list[np.ndarray],     # list of (C, H, W) ground truth images
    pred_segs: list[np.ndarray],       # predicted segmentation label maps
    target_segs: list[np.ndarray],     # ground truth segmentation label maps
    grade_scores: np.ndarray | None = None,  # model probability outputs for grading
    grade_labels: np.ndarray | None = None,  # 0/1 binary labels
    idh_scores:   np.ndarray | None = None,
    idh_labels:   np.ndarray | None = None,
) -> dict:
    """Aggregate all metrics across a dataset split."""

    # Image quality
    iq_keys = ["psnr", "ssim", "nrmse"]
    iq_accumulator = {k: [] for k in iq_keys}
    for pred, tgt in zip(pred_imgs, target_imgs):
        m = compute_image_quality_metrics(pred, tgt)
        for k in iq_keys:
            iq_accumulator[k].append(m[k])

    # Segmentation
    seg_keys = [f"{m}_{r}" for m in ["DSC", "HD95"] for r in ["WT", "TC", "ET"]]
    seg_accumulator = {k: [] for k in seg_keys}
    for p_seg, t_seg in zip(pred_segs, target_segs):
        m = segmentation_metrics(p_seg, t_seg)
        for k in seg_keys:
            seg_accumulator[k].append(m[k])

    results = {}
    for k, vals in {**iq_accumulator, **seg_accumulator}.items():
        results[k] = float(np.nanmean(vals))

    # Grading
    if grade_scores is not None and grade_labels is not None:
        results["AUROC_grade"] = auroc(grade_scores, grade_labels)
    if idh_scores is not None and idh_labels is not None:
        results["AUROC_IDH"] = auroc(idh_scores, idh_labels)

    return results
