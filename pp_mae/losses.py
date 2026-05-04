"""
Shared loss functions for all PP-MAE options.

Composite loss:
    L_total = L_global + λ1 * L_pathology + λ2 * L_crossmodal

    L_global      — pixel-wise reconstruction over the whole volume (L1 + SSIM)
    L_pathology   — weighted reconstruction inside tumour subregion masks
    L_crossmodal  — cosine/feature consistency between modality pairs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Structural Similarity (SSIM) — differentiable single-scale version
# ---------------------------------------------------------------------------

class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, channel: int = 1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self._create_window(window_size, channel)

    @staticmethod
    def _gaussian(window_size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        _1d = self._gaussian(window_size).unsqueeze(1)
        _2d = _1d.mm(_1d.t()).unsqueeze(0).unsqueeze(0)
        return _2d.expand(channel, 1, window_size, window_size).contiguous()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred.shape
        window = self.window.to(pred.device)

        mu1 = F.conv2d(pred,   window, padding=self.window_size // 2, groups=C)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=C)

        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred,     window, padding=self.window_size // 2, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=C) - mu2_sq
        sigma12   = F.conv2d(pred * target,   window, padding=self.window_size // 2, groups=C) - mu1_mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()


# ---------------------------------------------------------------------------
# Global reconstruction loss  (L1 + SSIM)
# ---------------------------------------------------------------------------

class GlobalReconLoss(nn.Module):
    def __init__(self, ssim_weight: float = 0.5, channel: int = 4):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss(channel=channel)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1(pred, target) + self.ssim_weight * self.ssim(pred, target)


# ---------------------------------------------------------------------------
# Pathology-preserving regional loss
#
# Tumour subregions (BraTS convention):
#   label 1 → necrotic core (NCR)
#   label 2 → peritumoral oedema (ED)
#   label 3 → enhancing tumour (ET)
#
# Composed regions:
#   Whole Tumour (WT)  = 1 + 2 + 3
#   Tumour Core  (TC)  = 1 + 3
#   Enhancing    (ET)  = 3
#
# Each region is weighted independently; ET receives the highest weight
# because it is the most clinically critical and noise-sensitive subregion.
# ---------------------------------------------------------------------------

class PathologyLoss(nn.Module):
    """
    Computes a weighted reconstruction loss inside tumour subregion masks.

    Args:
        region_weights: dict mapping region name → scalar weight.
            Higher weight → more penalty for errors in that region.
        base_loss: 'l1' or 'mse'
    """

    REGION_WEIGHTS_DEFAULT = {
        "WT": 1.0,   # whole tumour
        "TC": 2.0,   # tumour core
        "ET": 3.0,   # enhancing tumour (highest clinical priority)
    }

    def __init__(
        self,
        region_weights: dict | None = None,
        base_loss: str = "l1",
    ):
        super().__init__()
        self.region_weights = region_weights or self.REGION_WEIGHTS_DEFAULT
        self.loss_fn = nn.L1Loss(reduction="none") if base_loss == "l1" else nn.MSELoss(reduction="none")

    @staticmethod
    def _build_masks(seg_map: torch.Tensor) -> dict:
        """
        seg_map: (B, 1, H, W) integer label map.
        Returns binary masks for WT, TC, ET.
        """
        wt = (seg_map > 0).float()
        tc = ((seg_map == 1) | (seg_map == 3)).float()
        et = (seg_map == 3).float()
        return {"WT": wt, "TC": tc, "ET": et}

    def forward(
        self,
        pred: torch.Tensor,       # (B, C, H, W)
        target: torch.Tensor,     # (B, C, H, W)
        seg_map: torch.Tensor,    # (B, 1, H, W)  integer labels
    ) -> torch.Tensor:
        masks = self._build_masks(seg_map)
        pixel_loss = self.loss_fn(pred, target)  # (B, C, H, W)

        total = torch.zeros(1, device=pred.device)
        for region, weight in self.region_weights.items():
            mask = masks[region]                  # (B, 1, H, W)
            n_pixels = mask.sum().clamp(min=1.0)
            region_loss = (pixel_loss * mask).sum() / (n_pixels * pred.shape[1])
            total = total + weight * region_loss

        return total


# ---------------------------------------------------------------------------
# Cross-modal consistency loss
#
# Enforces that the latent representations of paired modalities remain
# coherent after reconstruction.  We use cosine similarity on spatially
# pooled feature maps so the loss is resolution-agnostic.
#
# Canonical complementary pairs for glioma:
#   (T1Wce, T2W)   — enhancing core ↔ oedema
#   (T2W,   FLAIR) — oedema ↔ infiltrative margin
# ---------------------------------------------------------------------------

class CrossModalConsistencyLoss(nn.Module):
    """
    Args:
        modality_pairs: list of (index_a, index_b) into the channel dim of
                        the reconstructed tensor.  Default follows the glioma
                        imaging protocol: T1W=0, T1Wce=1, T2W=2, FLAIR=3.
    """

    DEFAULT_PAIRS = [(1, 2), (2, 3)]   # (T1Wce↔T2W), (T2W↔FLAIR)

    def __init__(self, modality_pairs: list | None = None):
        super().__init__()
        self.pairs = modality_pairs or self.DEFAULT_PAIRS

    def forward(
        self,
        pred_features: torch.Tensor,    # (B, C, H, W) — multi-channel recon
        target_features: torch.Tensor,  # (B, C, H, W) — ground truth
    ) -> torch.Tensor:
        loss = torch.zeros(1, device=pred_features.device)
        for i, j in self.pairs:
            pred_i   = F.adaptive_avg_pool2d(pred_features[:, i:i+1],   (1, 1)).flatten(1)
            pred_j   = F.adaptive_avg_pool2d(pred_features[:, j:j+1],   (1, 1)).flatten(1)
            target_i = F.adaptive_avg_pool2d(target_features[:, i:i+1], (1, 1)).flatten(1)
            target_j = F.adaptive_avg_pool2d(target_features[:, j:j+1], (1, 1)).flatten(1)

            # The relative similarity pattern between modalities should be
            # consistent between prediction and target.
            pred_sim   = F.cosine_similarity(pred_i,   pred_j,   dim=1)
            target_sim = F.cosine_similarity(target_i, target_j, dim=1)
            loss = loss + F.mse_loss(pred_sim, target_sim)
        return loss / max(len(self.pairs), 1)


# ---------------------------------------------------------------------------
# Composite PP-MAE loss   L_total = L_global + λ1·L_path + λ2·L_crossmodal
# ---------------------------------------------------------------------------

class PPMAELoss(nn.Module):
    """
    Args:
        lambda1: weight for pathology-preserving loss
        lambda2: weight for cross-modal consistency loss
        ssim_weight: SSIM fraction inside the global reconstruction loss
        n_modalities: number of input modalities (default 4 for glioma protocol)
    """

    def __init__(
        self,
        lambda1: float = 1.0,
        lambda2: float = 0.5,
        ssim_weight: float = 0.5,
        n_modalities: int = 4,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.global_loss   = GlobalReconLoss(ssim_weight=ssim_weight, channel=n_modalities)
        self.pathology_loss = PathologyLoss()
        self.crossmodal_loss = CrossModalConsistencyLoss()

    def forward(
        self,
        pred: torch.Tensor,       # (B, C, H, W)
        target: torch.Tensor,     # (B, C, H, W)
        seg_map: torch.Tensor,    # (B, 1, H, W) integer tumour labels
    ) -> dict:
        l_global   = self.global_loss(pred, target)
        l_path     = self.pathology_loss(pred, target, seg_map)
        l_crossmod = self.crossmodal_loss(pred, target)

        l_total = l_global + self.lambda1 * l_path + self.lambda2 * l_crossmod
        return {
            "total":     l_total,
            "global":    l_global,
            "pathology": l_path,
            "crossmodal": l_crossmod,
        }
