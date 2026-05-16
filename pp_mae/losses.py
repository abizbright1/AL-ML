"""
Shared loss functions for all PP-MAE options.

Composite loss:
    L_total = L_global + λ1 * L_pathology + λ2 * L_crossmodal

    L_global      — pixel-wise reconstruction over the whole volume (L1 + SSIM)
    L_pathology   — weighted reconstruction inside tumour subregion masks
                    supports BOTH fixed and adaptive (learned) region weights
    L_crossmodal  — cosine/feature consistency between modality pairs

Adaptive Pathology Weight Learning (Section A):
    Instead of fixed w_ET=3, w_TC=2, w_WT=1, the model learns:

        w_r = softplus( C_r  +  MLP([F_r, U_r]) )

    where:
        F_r  — Feature severity:   statistics of target image in region r
        U_r  — Uncertainty:        prediction variance within region r
        C_r  — Clinical prior:     learnable parameter, init from domain knowledge

    The residual design (C_r + MLP) means training starts from the clinical
    prior and learns data-driven corrections rather than from scratch.

    Gradient note: U_r is computed with pred.detach() to prevent the model
    from gaming the loss by reducing within-region variance instead of
    improving reconstruction accuracy.
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
# Fixed-weight pathology loss (original formulation)
# ---------------------------------------------------------------------------

class PathologyLoss(nn.Module):
    """
    Weighted reconstruction loss inside tumour subregion masks.

    BraTS label convention:
        0 → background
        1 → necrotic core (NCR)
        2 → peritumoral oedema (ED)
        3 → enhancing tumour (ET)

    Composed regions:
        WT (Whole Tumour)  = labels 1+2+3   weight 1.0
        TC (Tumour Core)   = labels 1+3      weight 2.0
        ET (Enhancing)     = label  3        weight 3.0
    """

    REGION_WEIGHTS_DEFAULT = {"WT": 1.0, "TC": 2.0, "ET": 3.0}

    def __init__(self, region_weights: dict | None = None, base_loss: str = "l1"):
        super().__init__()
        self.region_weights = region_weights or self.REGION_WEIGHTS_DEFAULT
        self.loss_fn = nn.L1Loss(reduction="none") if base_loss == "l1" else nn.MSELoss(reduction="none")

    @staticmethod
    def _build_masks(seg_map: torch.Tensor) -> dict:
        wt = (seg_map > 0).float()
        tc = ((seg_map == 1) | (seg_map == 3)).float()
        et = (seg_map == 3).float()
        return {"WT": wt, "TC": tc, "ET": et}

    def forward(self, pred: torch.Tensor, target: torch.Tensor, seg_map: torch.Tensor) -> torch.Tensor:
        masks    = self._build_masks(seg_map)
        pix_loss = self.loss_fn(pred, target)
        total    = torch.zeros(1, device=pred.device)
        for region, weight in self.region_weights.items():
            mask     = masks[region]
            n        = mask.sum().clamp(min=1.0)
            total    = total + weight * (pix_loss * mask).sum() / (n * pred.shape[1])
        return total


# ---------------------------------------------------------------------------
# Adaptive pathology weight learning  w_r = φ(F_r, U_r, C_r)
# ---------------------------------------------------------------------------

class AdaptivePathologyLoss(nn.Module):
    """
    Learns region-specific loss weights dynamically each forward pass.

    Weight formula (per region r ∈ {WT, TC, ET}):

        w_r = softplus( C_r  +  MLP([F_r, U_r]) )

        F_r  = severity_net( [μ_target_r, σ_target_r] )
               Measures how pathologically extreme region r looks in the
               target image.  Higher intensity / higher contrast → higher F_r.

        U_r  = var( pred.detach() in region r )
               Prediction uncertainty: high variance within a tumour subregion
               means the model is unsure about the fine structure there.
               Gradient is stopped to prevent the model gaming the loss by
               artificially homogenising predictions.

        C_r  = exp(log_C_r),  log_C_r learnable
               Clinical aggressiveness prior, initialised from domain knowledge
               (WT=1.0, TC=2.0, ET=3.0) and learned from data.

    The residual structure means the network starts at the clinical prior
    and learns corrections — not from random initialisation.

    Args:
        n_modalities:  number of MRI channels (default 4)
        hidden:        width of MLP layers
        base_loss:     'l1' or 'mse' for per-pixel loss
    """

    REGIONS          = ["WT", "TC", "ET"]
    CLINICAL_PRIORS  = [1.0, 2.0, 3.0]   # initialised from domain knowledge

    def __init__(self, n_modalities: int = 4, hidden: int = 32, base_loss: str = "l1"):
        super().__init__()

        # C_r: learnable clinical aggressiveness (in log-space → always positive)
        self.log_C = nn.Parameter(
            torch.log(torch.tensor(self.CLINICAL_PRIORS, dtype=torch.float32))
        )

        # F_r network: [μ_ch1..4, σ_ch1..4] → severity scalar (positive)
        self.severity_net = nn.Sequential(
            nn.Linear(2 * n_modalities, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Softplus(),
        )

        # Adjustment MLP: [F_r, U_r] → Δw  (can be positive or negative)
        self.adjustment_net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        self.loss_fn = nn.L1Loss(reduction="none") if base_loss == "l1" else nn.MSELoss(reduction="none")
        self.n_modalities = n_modalities

    @staticmethod
    def _build_masks(seg_map: torch.Tensor) -> dict:
        return {
            "WT": (seg_map > 0).float(),
            "TC": ((seg_map == 1) | (seg_map == 3)).float(),
            "ET": (seg_map == 3).float(),
        }

    @staticmethod
    def _region_feature_stats(target: torch.Tensor, mask: torch.Tensor):
        """
        Compute mean and std of target pixel values within mask region.

        F_r captures image appearance: brighter / more heterogeneous regions
        are assigned higher severity scores by the severity_net.

        Returns:
            mean_r, std_r — each shape (C,), averaged over batch dimension.
        """
        n      = mask.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)   # (B,1,1,1)
        mean_r = (target * mask).sum(dim=(2, 3), keepdim=True) / n    # (B,C,1,1)
        diff   = (target - mean_r) * mask
        std_r  = (diff ** 2).sum(dim=(2, 3), keepdim=True).sqrt() / n.sqrt()
        return mean_r.mean(0).squeeze(), std_r.mean(0).squeeze()       # (C,), (C,)

    @staticmethod
    def _region_uncertainty(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Prediction variance within region r — used as uncertainty proxy U_r.

        Computed on pred.detach() so gradients do NOT flow back through U_r.
        This blocks the shortcut where the model could reduce U_r (by making
        predictions more uniform within the region) rather than reducing the
        actual reconstruction error.

        Returns a scalar tensor.
        """
        p      = pred.detach()
        n      = mask.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
        mean_r = (p * mask).sum(dim=(2, 3), keepdim=True) / n
        var_r  = ((p - mean_r) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)
        return var_r

    def forward(
        self,
        pred:    torch.Tensor,    # (B, C, H, W)
        target:  torch.Tensor,    # (B, C, H, W)
        seg_map: torch.Tensor,    # (B, 1, H, W) integer tumour labels
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss  — scalar tensor (has gradients)
            weight_dict — {region: float} for logging/analysis
        """
        masks    = self._build_masks(seg_map)
        pix_loss = self.loss_fn(pred, target)              # (B, C, H, W)
        C        = self.log_C.exp()                        # (3,) positive clinical priors

        total       = torch.zeros(1, device=pred.device)
        weight_dict = {}

        for i, region in enumerate(self.REGIONS):
            mask = masks[region]                           # (B, 1, H, W)

            # --- F_r: feature severity from target image statistics ----------
            t_mean, t_std = self._region_feature_stats(target, mask)  # (C,)
            stats = torch.cat([t_mean, t_std]).unsqueeze(0)            # (1, 2C)
            f_r   = self.severity_net(stats)                           # (1, 1)

            # --- U_r: prediction uncertainty (gradient stopped) -------------
            u_scalar = self._region_uncertainty(pred, mask)            # scalar
            u_r = u_scalar.unsqueeze(0).unsqueeze(0)                   # (1, 1)

            # --- w_r = softplus(C_r + MLP([F_r, U_r])) ---------------------
            combined   = torch.cat([f_r, u_r], dim=-1)                # (1, 2)
            adjustment = self.adjustment_net(combined)                 # (1, 1)
            w_r        = F.softplus(C[i] + adjustment.squeeze())      # scalar

            # --- Weighted region reconstruction loss ------------------------
            n_pixels    = mask.sum().clamp(min=1.0)
            region_loss = (pix_loss * mask).sum() / (n_pixels * pred.shape[1])
            total       = total + w_r * region_loss

            weight_dict[region] = w_r.item()

        return total, weight_dict

    def weight_summary(self) -> str:
        """Human-readable current learned weights (for logging)."""
        C = self.log_C.exp().tolist()
        return " | ".join(
            f"{r}: C={c:.3f}" for r, c in zip(self.REGIONS, C)
        )


# ---------------------------------------------------------------------------
# Cross-modal consistency loss
# ---------------------------------------------------------------------------

class CrossModalConsistencyLoss(nn.Module):
    """
    Cosine similarity consistency between complementary modality pairs.

    Canonical pairs for glioma protocol (T1W=0, T1Wce=1, T2W=2, FLAIR=3):
        (T1Wce, T2W)   — enhancing core ↔ oedema boundary
        (T2W,   FLAIR) — oedema ↔ infiltrative margin
    """

    DEFAULT_PAIRS = [(1, 2), (2, 3)]

    def __init__(self, modality_pairs: list | None = None):
        super().__init__()
        self.pairs = modality_pairs or self.DEFAULT_PAIRS

    def forward(self, pred_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros(1, device=pred_features.device)
        for i, j in self.pairs:
            pred_i   = F.adaptive_avg_pool2d(pred_features[:, i:i+1],   (1, 1)).flatten(1)
            pred_j   = F.adaptive_avg_pool2d(pred_features[:, j:j+1],   (1, 1)).flatten(1)
            target_i = F.adaptive_avg_pool2d(target_features[:, i:i+1], (1, 1)).flatten(1)
            target_j = F.adaptive_avg_pool2d(target_features[:, j:j+1], (1, 1)).flatten(1)
            pred_sim   = F.cosine_similarity(pred_i,   pred_j,   dim=1)
            target_sim = F.cosine_similarity(target_i, target_j, dim=1)
            loss = loss + F.mse_loss(pred_sim, target_sim)
        return loss / max(len(self.pairs), 1)


# ---------------------------------------------------------------------------
# Composite PP-MAE loss   L_total = L_global + λ1·L_path + λ2·L_crossmodal
# ---------------------------------------------------------------------------

class PPMAELoss(nn.Module):
    """
    Composite loss supporting both fixed and adaptive pathology weights.

    Args:
        lambda1:   weight for pathology loss term
        lambda2:   weight for cross-modal consistency term
        ssim_weight: SSIM fraction inside global reconstruction loss
        n_modalities: number of MRI channels (default 4)
        adaptive:  if True, use AdaptivePathologyLoss (learned weights);
                   if False, use PathologyLoss (fixed weights w_ET=3 etc.)
    """

    def __init__(
        self,
        lambda1:      float = 1.0,
        lambda2:      float = 0.5,
        ssim_weight:  float = 0.5,
        n_modalities: int   = 4,
        adaptive:     bool  = False,
    ):
        super().__init__()
        self.lambda1  = lambda1
        self.lambda2  = lambda2
        self.adaptive = adaptive

        self.global_loss     = GlobalReconLoss(ssim_weight=ssim_weight, channel=n_modalities)
        self.pathology_loss  = (AdaptivePathologyLoss(n_modalities=n_modalities)
                                if adaptive else PathologyLoss())
        self.crossmodal_loss = CrossModalConsistencyLoss()

    def forward(
        self,
        pred:    torch.Tensor,    # (B, C, H, W)
        target:  torch.Tensor,    # (B, C, H, W)
        seg_map: torch.Tensor,    # (B, 1, H, W) integer tumour labels
    ) -> dict:
        l_global   = self.global_loss(pred, target)
        l_crossmod = self.crossmodal_loss(pred, target)

        if self.adaptive:
            l_path, weight_dict = self.pathology_loss(pred, target, seg_map)
        else:
            l_path      = self.pathology_loss(pred, target, seg_map)
            weight_dict = {}

        l_total = l_global + self.lambda1 * l_path + self.lambda2 * l_crossmod

        result = {
            "total":      l_total,
            "global":     l_global,
            "pathology":  l_path,
            "crossmodal": l_crossmod,
        }
        # Log the learned weights when adaptive mode is on
        result.update({f"w_{k}": torch.tensor(v) for k, v in weight_dict.items()})
        return result
