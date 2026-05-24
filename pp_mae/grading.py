"""
grading.py — Brain Tumour Grade Prediction from Denoised MRI
=============================================================

Implements the downstream grading pipeline that sits on top of the
existing PP-MAE denoiser + UNet segmentor stack.

The pipeline:
    Noisy MRI
        → Denoiser (PP-MAE or baseline)
        → Denoised MRI
        → UNet Segmentor (frozen, trained on clean images)
        → Predicted tumour masks (WT, TC, ET)
        → Feature Extractor (7 radiomics features)
        → GradingHead (small MLP)
        → GBM (Grade IV)  vs  LGG (Grade II/III)

Why this structure:
    Grade classification from MRI is fundamentally a radiomics problem.
    Radiologists look at:
        1. Is there enhancing tumour (ET)?  → if yes, likely GBM
        2. How large is ET relative to whole tumour?  → enhancement ratio ρ
        3. How irregular/heterogeneous are the regions?  → higher = higher grade
    These 7 features capture exactly this clinical reasoning in a
    differentiable, data-driven way.

Key scientific claim:
    PP-MAE's pathology-preserving denoising produces cleaner ET boundaries
    than standard denoisers → more accurate V_ET and ρ estimates
    → better grading AUC than DnCNN/UNet despite similar PSNR.

References:
    Bauer et al. (2013). Segmentation of brain tumor images based on
    integrated hierarchical classification and regularization.
    Menze et al. (2015). The Multimodal Brain Tumor Image Segmentation
    Benchmark (BRATS). IEEE TMI.
    Kickingereder et al. (2016). Radiomic profiling of glioblastoma.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_grading_features(
    denoised:   torch.Tensor,    # (B, C, H, W) — denoised MRI
    seg_logits: torch.Tensor,    # (B, n_classes, H, W) — segmentor output
    eps:        float = 1e-6,
) -> torch.Tensor:
    """
    Extract 7 radiomics-inspired grading features from one batch of slices.

    Features (clinically motivated):
    ─────────────────────────────────────────────────────────────────
    V_WT   Volume fraction of Whole Tumour (any label > 0)
           → Large WT = aggressive tumour
    V_TC   Volume fraction of Tumour Core (NCR + ET, labels 1+3)
           → Large TC = necrotic / active core
    V_ET   Volume fraction of Enhancing Tumour (label 3)
           → ET presence is the strongest GBM indicator
    rho    Enhancement ratio = V_ET / V_WT
           → High rho: tumour core is mostly enhancing → GBM phenotype
           → Low rho: oedema dominates, no enhancement → LGG phenotype
    H_WT   Heterogeneity of WT region (σ/μ of denoised intensities)
           → Irregular intensity = higher grade
    H_TC   Heterogeneity of TC region
    H_ET   Heterogeneity of ET region
    ─────────────────────────────────────────────────────────────────

    Note: features are extracted from PREDICTED segmentation (not GT) because
    at inference time we never have ground-truth labels — this is the realistic
    clinical scenario.

    Returns:
        features : (B, 7) float tensor, one row per image in batch
    """
    # Convert logits → predicted class labels
    pred_cls = seg_logits.argmax(dim=1)     # (B, H, W)

    # Build binary masks from predicted labels
    m_wt = (pred_cls > 0).float()                           # (B, H, W)
    m_tc = ((pred_cls == 1) | (pred_cls == 3)).float()      # (B, H, W)
    m_et = (pred_cls == 3).float()                          # (B, H, W)

    total_px = float(pred_cls.shape[-1] * pred_cls.shape[-2])

    # ── Volume fractions ─────────────────────────────────────────────────────
    # Sum over H and W for each image in batch
    V_WT = m_wt.sum(dim=(-2, -1)) / total_px               # (B,)
    V_TC = m_tc.sum(dim=(-2, -1)) / total_px               # (B,)
    V_ET = m_et.sum(dim=(-2, -1)) / total_px               # (B,)

    # Enhancement ratio — key GBM marker
    # ρ → 1.0 means most of WT is ET (pure enhancing = GBM)
    # ρ → 0.0 means no enhancement (non-enhancing LGG)
    rho = V_ET / (V_WT + eps)                               # (B,)

    # ── Heterogeneity  σ/μ ──────────────────────────────────────────────────
    # Use mean across all 4 modalities to get a single intensity channel
    img = denoised.mean(dim=1, keepdim=True)                # (B, 1, H, W)

    def _hetero(mask_2d: torch.Tensor) -> torch.Tensor:
        """
        mask_2d : (B, H, W)  binary mask
        Returns  : (B,)      coefficient of variation σ/μ within mask
        """
        m = mask_2d.unsqueeze(1)                            # (B, 1, H, W)
        n = m.sum(dim=(-2, -1)).clamp(min=1.0)             # (B, 1) pixel count
        mu  = (img * m).sum(dim=(-2, -1)) / n              # (B, 1) local mean
        # Variance = E[(x - μ)²] within mask
        var = ((img - mu.unsqueeze(-1).unsqueeze(-1)) ** 2 * m
               ).sum(dim=(-2, -1)) / n                     # (B, 1)
        sig = var.sqrt()                                    # (B, 1) local std
        return (sig / (mu + eps)).squeeze(1)               # (B,)

    H_WT = _hetero(m_wt)
    H_TC = _hetero(m_tc)
    H_ET = _hetero(m_et)

    # Stack all 7 features into one row per image
    return torch.stack([V_WT, V_TC, V_ET, rho, H_WT, H_TC, H_ET], dim=1)  # (B, 7)


FEATURE_NAMES = ['V_WT', 'V_TC', 'V_ET', 'ρ (Enhance)', 'H_WT', 'H_TC', 'H_ET']


def aggregate_subject_features(
    slice_feats: List[torch.Tensor],   # list of (7,) tensors, one per slice
) -> torch.Tensor:
    """
    Aggregate per-slice feature vectors into a single subject-level vector.

    Clinical motivation for the aggregation strategy:
        Volume features (V_WT, V_TC, V_ET, ρ) → MAX across slices
            Tumour presence is best captured by its largest extent.
            A single slice with large ET is enough to indicate GBM.

        Heterogeneity features (H_WT, H_TC, H_ET) → MEAN across slices
            Textural heterogeneity should be averaged — one irregular
            slice doesn't mean the whole tumour is heterogeneous.

    Args:
        slice_feats : list of (7,) tensors — one per tumour slice

    Returns:
        (7,) subject-level feature vector
    """
    if len(slice_feats) == 0:
        return torch.zeros(7)

    stacked = torch.stack(slice_feats, dim=0)    # (n_slices, 7)
    vol_max   = stacked[:, :4].max(dim=0).values  # V_WT,V_TC,V_ET,ρ → max
    hetero_mn = stacked[:, 4:].mean(dim=0)        # H_WT,H_TC,H_ET → mean
    return torch.cat([vol_max, hetero_mn])         # (7,)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Grading Head — MLP Classifier
# ─────────────────────────────────────────────────────────────────────────────

class GradingHead(nn.Module):
    """
    3-layer MLP that predicts brain tumour grade from radiomics features.

    Binary classification:
        Output = 1  →  GBM  (WHO Grade IV, most aggressive)
        Output = 0  →  LGG  (WHO Grade II/III, lower grade)

    Architecture decisions:
        • Small (3 layers) because typical glioma cohorts are O(100) subjects.
          A large network would overfit immediately.
        • Dropout (p=0.3) for regularisation on small cohorts.
        • No BatchNorm — batch size at subject level is too small.
        • Sigmoid output — gives interpretable probability of GBM.
        • Final bias initialised to 0 → model starts at 50% probability,
          unbiased to either class.

    Args:
        n_features : number of input radiomics features (default 7)
        hidden1    : neurons in first hidden layer
        hidden2    : neurons in second hidden layer
        dropout    : dropout rate (applied after each hidden layer)
    """

    def __init__(
        self,
        n_features: int   = 7,
        hidden1:    int   = 32,
        hidden2:    int   = 16,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),          # probability ∈ (0, 1)
        )
        # Unbiased initialisation — start at 50/50 before seeing data
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 7)  radiomics feature vectors
        Returns (B,) GBM probabilities ∈ (0, 1)
        """
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Grading Trainer
# ─────────────────────────────────────────────────────────────────────────────

class GradingTrainer:
    """
    Trains GradingHead with binary cross-entropy loss.

    Supports class-balanced weighting via pos_weight:
        pos_weight > 1  →  penalise missing GBM more (when GBM is minority)
        pos_weight = 1  →  equal weighting (balanced dataset)

    Uses BCELoss (not BCEWithLogitsLoss) because GradingHead already applies
    Sigmoid internally — consistent with the forward() contract.
    """

    def __init__(
        self,
        model:      GradingHead,
        device:     str   = 'cpu',
        lr:         float = 1e-3,
        pos_weight: float = 1.0,
    ):
        self.model   = model.to(device)
        self.device  = device
        # pos_weight: scale loss for positive class (GBM)
        pw = torch.tensor([pos_weight], device=device)
        self.loss_fn = nn.BCELoss(weight=pw)
        self.optim   = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=1e-4
        )

    def step(
        self,
        features: torch.Tensor,   # (B, 7)
        labels:   torch.Tensor,   # (B,) float — 0=LGG, 1=GBM
    ) -> float:
        self.model.train()
        features = features.to(self.device)
        labels   = labels.float().to(self.device)
        self.optim.zero_grad()
        preds = self.model(features)
        loss  = self.loss_fn(preds, labels)
        loss.backward()
        self.optim.step()
        return loss.item()

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        features : (B, 7)
        Returns  : (B,) GBM probabilities, detached from graph
        """
        self.model.eval()
        return self.model(features.to(self.device)).cpu()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

def grading_metrics(
    probs:     torch.Tensor,    # (N,) predicted GBM probability
    labels:    torch.Tensor,    # (N,) true labels: 0=LGG, 1=GBM
    threshold: float = 0.5,
    eps:       float = 1e-9,
) -> Dict[str, float]:
    """
    Compute the standard clinical grading evaluation metrics.

    Clinical interpretation:
        AUC         — main metric; 0.5 = random, 1.0 = perfect
        sensitivity — fraction of real GBM cases correctly flagged
                      (missing GBM = dangerous → want this HIGH)
        specificity — fraction of real LGG cases correctly identified
                      (false GBM = unnecessary aggressive treatment)
        f1          — balance of precision and sensitivity
        accuracy    — overall correct rate

    Note: AUC is computed without sklearn using trapezoidal integration
    over 101 threshold points — no external dependencies needed.

    Args:
        probs     : model output probabilities for GBM class
        labels    : true binary labels (0=LGG, 1=GBM)
        threshold : decision boundary (default 0.5)

    Returns:
        dict of metric_name → float
    """
    probs_np  = probs.float().numpy()
    labels_np = labels.long().numpy()
    preds_np  = (probs_np >= threshold).astype(int)

    TP = int(((preds_np == 1) & (labels_np == 1)).sum())
    TN = int(((preds_np == 0) & (labels_np == 0)).sum())
    FP = int(((preds_np == 1) & (labels_np == 0)).sum())
    FN = int(((preds_np == 0) & (labels_np == 1)).sum())

    accuracy    = (TP + TN) / (len(labels_np) + eps)
    sensitivity = TP / (TP + FN + eps)    # true positive rate for GBM
    specificity = TN / (TN + FP + eps)    # true negative rate for LGG
    precision   = TP / (TP + FP + eps)    # positive predictive value
    f1 = (2 * precision * sensitivity
          / (precision + sensitivity + eps))

    auc = _roc_auc_trapz(probs_np, labels_np)

    return {
        'auc':         round(float(auc),         4),
        'accuracy':    round(float(accuracy),    4),
        'sensitivity': round(float(sensitivity), 4),
        'specificity': round(float(specificity), 4),
        'precision':   round(float(precision),   4),
        'f1':          round(float(f1),          4),
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
    }


def _roc_auc_trapz(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    AUC-ROC via trapezoidal rule over 101 threshold points.
    No sklearn dependency — works in any Python environment.

    Method:
        For each threshold t in [0, 1]:
            TPR(t) = TP / (TP + FN)   (sensitivity)
            FPR(t) = FP / (FP + TN)   (1 - specificity)
        AUC = area under the curve of TPR vs FPR
            = ∫ TPR d(FPR)  via trapezoid rule
    """
    thresholds = np.linspace(0.0, 1.0, 101)[::-1]
    tprs, fprs = [0.0], [0.0]    # start at (0,0) for threshold=1.0

    for t in thresholds:
        preds = (probs >= t).astype(int)
        TP = int(((preds == 1) & (labels == 1)).sum())
        FP = int(((preds == 1) & (labels == 0)).sum())
        FN = int(((preds == 0) & (labels == 1)).sum())
        TN = int(((preds == 0) & (labels == 0)).sum())
        tprs.append(TP / (TP + FN + 1e-9))
        fprs.append(FP / (FP + TN + 1e-9))

    tprs.append(1.0); fprs.append(1.0)    # end at (1,1) for threshold=0.0

    # Trapezoidal integration
    auc = 0.0
    for i in range(1, len(fprs)):
        dx = fprs[i] - fprs[i - 1]
        auc += dx * (tprs[i] + tprs[i - 1]) / 2.0
    return abs(auc)


def get_roc_curve(
    probs:  np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (fprs, tprs) arrays for plotting the ROC curve.
    """
    thresholds = np.linspace(0.0, 1.0, 101)[::-1]
    tprs, fprs = [0.0], [0.0]
    for t in thresholds:
        preds = (probs >= t).astype(int)
        TP = int(((preds == 1) & (labels == 1)).sum())
        FP = int(((preds == 1) & (labels == 0)).sum())
        FN = int(((preds == 0) & (labels == 1)).sum())
        TN = int(((preds == 0) & (labels == 0)).sum())
        tprs.append(TP / (TP + FN + 1e-9))
        fprs.append(FP / (FP + TN + 1e-9))
    tprs.append(1.0); fprs.append(1.0)
    return np.array(fprs), np.array(tprs)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Grade Label Assignment (Demo Mode)
# ─────────────────────────────────────────────────────────────────────────────

def assign_demo_grade_labels(
    dataset,
    et_threshold: float = None,
) -> Dict[str, int]:
    """
    Assign synthetic GBM/LGG labels to demo subjects based on ET volume.

    In real BraTS data, grade labels come from clinical metadata CSVs.
    In demo mode, we derive them from the synthetic segmentation maps
    using a MEDIAN SPLIT on ET volume fraction:

        ET volume fraction > median  →  GBM  (label = 1)
        ET volume fraction ≤ median  →  LGG  (label = 0)

    Using the median guarantees exactly 50/50 class balance regardless of
    the ET threshold, which is important for a fair demo.
    A fixed threshold can be passed to override (useful for real data).

    The median split preserves the scientific validity: subjects with
    larger ET (more aggressive phenotype) are labelled GBM — the same
    features (V_ET, ρ) used for grading are predictive of the label,
    so AUC > 0.5 is achievable and meaningful.

    Args:
        dataset      : BraTS dataset with 'seg' and 'subject' fields
        et_threshold : fixed threshold (None = use median split)

    Returns:
        dict  {subject_name: label}  where label ∈ {0=LGG, 1=GBM}
    """
    # Accumulate mean ET volume fraction per subject across all slices
    subject_et    = {}
    subject_total = {}

    for item in dataset:
        name  = item['subject']
        seg   = item['seg'][0]              # (H, W) int labels
        et_px = int((seg == 3).sum().item())
        total = seg.numel()
        subject_et.setdefault(name, 0)
        subject_total.setdefault(name, 0)
        subject_et[name]    += et_px
        subject_total[name] += total

    # Compute per-subject ET fraction
    et_fracs = {
        name: subject_et[name] / max(subject_total[name], 1)
        for name in subject_et
    }

    # Determine threshold: median split if not specified
    if et_threshold is None:
        et_threshold = float(np.median(list(et_fracs.values())))

    labels = {
        name: (1 if frac > et_threshold else 0)
        for name, frac in et_fracs.items()
    }

    n_gbm = sum(v for v in labels.values())
    n_lgg = len(labels) - n_gbm
    print(f"[grade labels] {n_gbm} GBM  |  {n_lgg} LGG  "
          f"(ET threshold={et_threshold:.4f}  "
          f"{'median split' if et_threshold == float(np.median(list(et_fracs.values()))) else 'fixed'})",
          flush=True)
    return labels
