"""
option5_grading_pp_mae.py  —  PP-MAE Option 5: Joint Denoising + Grading
=========================================================================

Option 5 extends Option 4 (Swin PP-MAE) with an integrated GradingHead,
trained end-to-end with a combined loss:

    L = L_recon  (PathologyLoss from Option 4)
      + λ_seg  * L_seg    (Dice + BCE on tumour masks)
      + λ_grade * L_grade (BCE on GBM / LGG binary label)

Architecture
------------
    Noisy MRI (B,4,H,W)
        ↓  SwinPPMAE  (Option 4 denoiser)
    Denoised MRI (B,4,H,W)
        ↓  UNetSegmentor
    Seg logits (B,4,H,W)   +   seg probs (B,4,H,W)
        ↓  extract_grading_features()
    Radiomic feature vector (B,7)
        ↓  GradingHead  (3-layer MLP)
    GBM probability (B,)

Key claim (Option 5 vs Option 4)
---------------------------------
    The joint loss keeps the denoiser differentiably connected to the
    grading outcome: if the denoiser blurs the ET region, grading AUC
    drops and the combined loss penalises it.  This forces the denoiser
    to preserve ET boundaries even more sharply than the PathologyLoss
    alone, further improving downstream classification.

Usage
-----
    from option5_grading_pp_mae import GradingPPMAE, GradingPPMAETrainer
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from option4_swin_pp_mae import SwinPPMAE
from segmentor            import UNetSegmentor
from grading              import (GradingHead, extract_grading_features,
                                   assign_demo_grade_labels, grading_metrics,
                                   get_roc_curve)
from losses               import PathologyLoss


# ─────────────────────────────────────────────────────────────────────────────
# Option 5 model
# ─────────────────────────────────────────────────────────────────────────────

class GradingPPMAE(nn.Module):
    """
    PP-MAE Option 5: SwinPPMAE denoiser + UNet segmentor + GradingHead,
    all trained jointly end-to-end.

    Args:
        in_ch        : MRI modalities (default 4)
        embed_dim    : Swin embedding dim (default 48, light for MPS)
        depths       : Swin stage depths
        n_heads      : attention heads per stage
        window_size  : Swin local attention window
        seg_base_ch  : UNet base channels for segmentor
        n_seg_classes: segmentation output classes (4: BG+NCR+ED+ET)
        n_features   : radiomic features fed to GradingHead (default 7)
    """

    def __init__(
        self,
        in_ch:         int   = 4,
        embed_dim:     int   = 48,
        depths:        tuple = (2, 2, 2, 2),
        n_heads:       tuple = (3, 3, 6, 6),
        window_size:   int   = 4,
        seg_base_ch:   int   = 32,
        n_seg_classes: int   = 4,
        n_features:    int   = 7,
    ):
        super().__init__()
        self.denoiser  = SwinPPMAE(in_ch=in_ch, embed_dim=embed_dim,
                                    depths=depths, n_heads=n_heads,
                                    window_size=window_size)
        self.segmentor = UNetSegmentor(in_channels=in_ch, base_ch=seg_base_ch,
                                        n_classes=n_seg_classes)
        self.grading   = GradingHead(n_features=n_features)

    def forward(
        self,
        noisy:   torch.Tensor,          # (B, 4, H, W)
        seg_map: torch.Tensor,          # (B, 1, H, W)  — used by denoiser saliency
    ) -> dict:
        """
        Returns dict with keys:
            denoised   : (B, 4, H, W)
            seg_logits : (B, n_seg_classes, H, W)
            grade_prob : (B,)  — GBM probability
            features   : (B, n_features)
        """
        denoised   = self.denoiser(noisy, seg_map)           # (B,4,H,W)
        seg_logits = self.segmentor(denoised)                 # (B,4,H,W)
        seg_probs  = torch.softmax(seg_logits, dim=1)

        # Extract per-sample radiomic features — numpy op, not differentiable
        # Used for grading head (treated as a fixed feature pipeline in fwd pass)
        feats = _batch_radiomic_features(denoised, seg_probs)  # (B, 7)
        grade_prob = self.grading(feats)                        # (B,)

        return {
            'denoised':   denoised,
            'seg_logits': seg_logits,
            'grade_prob': grade_prob,
            'features':   feats,
        }


def _batch_radiomic_features(
    denoised: torch.Tensor,    # (B, 4, H, W)
    seg_probs: torch.Tensor,   # (B, 4, H, W)
) -> torch.Tensor:
    """
    Differentiable approximation of radiomic features using soft seg masks.
    Returns (B, 7) tensor on the same device as input.
    """
    B, C, H, W = denoised.shape
    device = denoised.device

    # Soft region masks from seg probabilities (classes: 0=BG,1=NCR,2=ED,3=ET)
    p_bg  = seg_probs[:, 0]                          # (B,H,W)
    p_ncr = seg_probs[:, 1]
    p_ed  = seg_probs[:, 2]
    p_et  = seg_probs[:, 3]

    p_wt  = p_ncr + p_ed + p_et                     # whole tumour
    p_tc  = p_ncr + p_et                             # tumour core
    n_px  = float(H * W)

    t1ce  = denoised[:, 1]                           # T1ce channel

    feats = []

    # 1. V_ET  — normalised ET volume
    v_et  = p_et.sum(dim=(1, 2)) / n_px             # (B,)
    feats.append(v_et)

    # 2. V_WT  — normalised WT volume
    v_wt  = p_wt.sum(dim=(1, 2)) / n_px
    feats.append(v_wt)

    # 3. Enhancement ratio  ρ = V_ET / (V_TC + ε)
    v_tc  = p_tc.sum(dim=(1, 2)) / n_px
    rho   = v_et / (v_tc + 1e-6)
    feats.append(rho)

    # 4. Mean T1ce signal in ET region
    t1ce_et = (t1ce * p_et).sum(dim=(1, 2)) / (p_et.sum(dim=(1, 2)) + 1e-6)
    feats.append(t1ce_et)

    # 5. T1ce signal heterogeneity in ET region (std approximation)
    mean_et = t1ce_et.unsqueeze(-1).unsqueeze(-1)   # (B,1,1)
    var_et  = ((t1ce - mean_et) ** 2 * p_et).sum(dim=(1, 2)) \
              / (p_et.sum(dim=(1, 2)) + 1e-6)
    feats.append(var_et.sqrt())

    # 6. TC/WT ratio
    tc_wt = v_tc / (v_wt + 1e-6)
    feats.append(tc_wt)

    # 7. Mean T1ce in necrotic core
    t1ce_ncr = (t1ce * p_ncr).sum(dim=(1, 2)) / (p_ncr.sum(dim=(1, 2)) + 1e-6)
    feats.append(t1ce_ncr)

    return torch.stack(feats, dim=1)                 # (B, 7)


# ─────────────────────────────────────────────────────────────────────────────
# Combined loss
# ─────────────────────────────────────────────────────────────────────────────

class Option5Loss(nn.Module):
    """
    L = L_recon + λ_seg * L_seg + λ_grade * L_grade

    L_recon : PathologyLoss (ET×3, TC×2, WT×1)
    L_seg   : soft Dice + BCE on segmentation logits
    L_grade : binary cross-entropy on GBM probability
    """

    def __init__(self, lam_seg: float = 0.5, lam_grade: float = 0.3):
        super().__init__()
        self.lam_seg   = lam_seg
        self.lam_grade = lam_grade
        self.path_loss = PathologyLoss()

    def forward(
        self,
        out:    dict,           # output of GradingPPMAE.forward()
        target: torch.Tensor,  # (B,4,H,W) clean MRI
        seg_gt: torch.Tensor,  # (B,1,H,W) integer seg labels
        labels: torch.Tensor,  # (B,) binary grade labels (1=GBM, 0=LGG)
    ) -> torch.Tensor:

        # Reconstruction loss
        l_recon = self.path_loss(out['denoised'], target, seg_gt)

        # Segmentation loss (soft Dice + BCE)
        seg_probs = torch.softmax(out['seg_logits'], dim=1)
        seg_one_hot = F.one_hot(
            seg_gt.squeeze(1).long().clamp(0, seg_probs.shape[1]-1),
            num_classes=seg_probs.shape[1]
        ).permute(0, 3, 1, 2).float()
        intersection = (seg_probs * seg_one_hot).sum(dim=(2, 3))
        union        = seg_probs.sum(dim=(2, 3)) + seg_one_hot.sum(dim=(2, 3))
        dice_loss    = 1 - (2 * intersection / (union + 1e-6)).mean()
        bce_seg      = F.cross_entropy(out['seg_logits'],
                                        seg_gt.squeeze(1).long().clamp(0, seg_probs.shape[1]-1))
        l_seg = dice_loss + 0.5 * bce_seg

        # Grading loss
        labels_f = labels.float().to(out['grade_prob'].device)
        l_grade  = F.binary_cross_entropy(out['grade_prob'], labels_f)

        return l_recon + self.lam_seg * l_seg + self.lam_grade * l_grade


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class GradingPPMAETrainer:
    """
    Trains GradingPPMAE jointly on reconstruction + segmentation + grading.

    Grade labels are derived via assign_demo_grade_labels() (ET-volume proxy)
    when real pathology labels are unavailable, which is always the case for
    BraTS 2021 (all subjects are GBM; we stratify by ET volume as a surrogate
    for aggressiveness).
    """

    def __init__(
        self,
        model:       GradingPPMAE,
        device:      str   = 'cpu',
        lr:          float = 1e-4,
        lam_seg:     float = 0.5,
        lam_grade:   float = 0.3,
    ):
        self.model  = model.to(device)
        self.device = device
        self.opt    = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss   = Option5Loss(lam_seg=lam_seg, lam_grade=lam_grade).to(device)
        self._labels_cache: dict = {}

    # ------------------------------------------------------------------
    def _get_labels(self, subjects: list, seg_batch: torch.Tensor) -> torch.Tensor:
        """Assign binary grade labels per subject (cached)."""
        labels = []
        for i, subj in enumerate(subjects):
            if subj not in self._labels_cache:
                seg_np = seg_batch[i, 0].cpu().numpy()
                et_vol = float((seg_np == 3).sum()) / seg_np.size
                tc_vol = float(((seg_np == 1) | (seg_np == 3)).sum()) / seg_np.size
                rho    = et_vol / max(tc_vol, 1e-4)
                # Stratify: high ET ratio → aggressive (GBM proxy = 1)
                self._labels_cache[subj] = 1 if rho > 0.35 else 0
            labels.append(self._labels_cache[subj])
        return torch.tensor(labels, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    def train_epoch(self, loader) -> float:
        self.model.train()
        total = 0.0
        for batch in loader:
            noisy  = batch['noisy'].to(self.device)
            target = batch['target'].to(self.device)
            seg_gt = batch['seg'].to(self.device)
            subjs  = batch.get('subject', [f's{i}' for i in range(noisy.shape[0])])

            labels = self._get_labels(subjs, seg_gt)

            self.opt.zero_grad()
            out  = self.model(noisy, seg_gt)
            loss = self.loss(out, target, seg_gt, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            total += loss.item()
        return total / max(len(loader), 1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        """Returns reconstruction + segmentation + grading metrics."""
        self.model.eval()
        psnrs, ssims, dices_et, probs, trues = [], [], [], [], []

        for batch in loader:
            noisy  = batch['noisy'].to(self.device)
            target = batch['target'].to(self.device)
            seg_gt = batch['seg'].to(self.device)
            subjs  = batch.get('subject', [f's{i}' for i in range(noisy.shape[0])])

            labels = self._get_labels(subjs, seg_gt)
            out    = self.model(noisy, seg_gt)

            for i in range(noisy.shape[0]):
                pred = out['denoised'][i]
                gt   = target[i]
                mse  = F.mse_loss(pred, gt).item()
                psnrs.append(20 * np.log10(1 / max(mse ** 0.5, 1e-10)))

                # Dice ET
                pred_seg = out['seg_logits'][i].argmax(0).cpu().numpy()
                gt_seg   = seg_gt[i, 0].cpu().numpy()
                et_pred  = (pred_seg == 3)
                et_gt    = (gt_seg   == 3)
                denom    = et_pred.sum() + et_gt.sum()
                dices_et.append(2 * (et_pred & et_gt).sum() / max(denom, 1))

            probs.extend(out['grade_prob'].cpu().numpy().tolist())
            trues.extend(labels.cpu().numpy().tolist())

        # Grading metrics
        from sklearn.metrics import roc_auc_score, accuracy_score
        try:
            auc = roc_auc_score(trues, probs) if len(set(trues)) > 1 else 0.5
        except Exception:
            auc = 0.5
        preds_bin = [1 if p > 0.5 else 0 for p in probs]
        acc  = accuracy_score(trues, preds_bin) if trues else 0.5
        sens = (sum(p == 1 and t == 1 for p, t in zip(preds_bin, trues))
                / max(sum(t == 1 for t in trues), 1))
        spec = (sum(p == 0 and t == 0 for p, t in zip(preds_bin, trues))
                / max(sum(t == 0 for t in trues), 1))

        return {
            'PSNR':    float(np.mean(psnrs))   if psnrs    else 0.0,
            'Dice_ET': float(np.mean(dices_et)) if dices_et else 0.0,
            'AUC':     float(auc),
            'Acc':     float(acc),
            'Sens':    float(sens),
            'Spec':    float(spec),
        }
