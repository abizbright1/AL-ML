from typing import Optional
"""
Shared loss functions for all PP-MAE options.

Composite loss:
    L_total = L_global + λ1 * L_pathology + λ2 * L_crossmodal

Three pathology loss modes (set via PPMAELoss(mode=...)):

    'fixed'          — classic fixed weights  w_ET=3, w_TC=2, w_WT=1
    'adaptive'       — learned weights  w_r = softplus(C_r + MLP([F_r, U_r]))
    'clinical_risk'  — patient risk scores  L_path = Σ_r R_r · L_r
    'combined'       — clinical risk × adaptive  w_r = R_r · φ(F_r, U_r, C_r)

─────────────────────────────────────────────────────────────────────────────
A. Adaptive Pathology Weight Learning

    w_r = softplus( C_r  +  MLP([F_r, U_r]) )

    F_r  — Feature severity:   statistics of target image in region r
    U_r  — Uncertainty:        prediction variance in region r (pred.detach())
    C_r  — Clinical prior:     learnable parameter, init [WT=1, TC=2, ET=3]

    Gradient note: U_r uses pred.detach() to prevent the model from reducing
    within-region variance (instead of reconstruction error) to lower U_r.

─────────────────────────────────────────────────────────────────────────────
B. Formal Clinical Risk Optimization

    L_pathology = Σ_r  R_r · L_r

    R_r = ψ(V_ET, V_TC, V_WT, ρ, H_WT, H_TC, H_ET, [b_grade, b_IDH, b_MGMT, b_age])

    Image-derived risk features (no extra data needed):
        V_r  — normalised volume fraction of region r
        ρ    — enhancement ratio  V_ET / V_WT  (aggressive phenotype indicator)
        H_r  — heterogeneity  σ_r / μ_r  (intra-tumour heterogeneity)

    Optional clinical biomarkers b:
        b_grade  ∈ {0,1}  — WHO grade  (IV → 1)
        b_IDH   ∈ {0,1}   — IDH status (wildtype → 1, higher risk)
        b_MGMT  ∈ {0,1}   — MGMT methylation (unmethylated → 1)
        b_age   ∈ [0,1]   — patient age, normalised (older → 1)

    Biological rationale for risk scores:
        Large ET volume + high enhancement ratio + wildtype IDH
        → aggressive GBM phenotype → R_ET >> R_TC >> R_WT
        → the optimizer focuses denoising quality where it matters most.

    Network initialised so ψ(·) ≈ [1, 2, 3] at zero input, matching the
    clinical prior, and learns patient-specific deviations from data.

─────────────────────────────────────────────────────────────────────────────
C. Combined mode

    w_r = R_r · φ(F_r, U_r, C_r)

    R_r captures patient-level risk (who is this person?).
    φ(·)  captures scan-level difficulty (how hard is this image?).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SSIM Loss
# ---------------------------------------------------------------------------

class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, channel: int = 1):
        super().__init__()
        self.window_size = window_size
        self.channel     = channel
        self.window      = self._create_window(window_size, channel)

    @staticmethod
    def _gaussian(window_size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        _1d = self._gaussian(window_size).unsqueeze(1)
        _2d = _1d.mm(_1d.t()).unsqueeze(0).unsqueeze(0)
        return _2d.expand(channel, 1, window_size, window_size).contiguous()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred.shape
        win = self.window.to(pred.device)
        pad = self.window_size // 2
        mu1 = F.conv2d(pred,   win, padding=pad, groups=C)
        mu2 = F.conv2d(target, win, padding=pad, groups=C)
        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
        s1 = F.conv2d(pred*pred,     win, padding=pad, groups=C) - mu1_sq
        s2 = F.conv2d(target*target, win, padding=pad, groups=C) - mu2_sq
        s12= F.conv2d(pred*target,   win, padding=pad, groups=C) - mu1_mu2
        C1, C2 = 0.01**2, 0.03**2
        ssim = ((2*mu1_mu2+C1)*(2*s12+C2)) / ((mu1_sq+mu2_sq+C1)*(s1+s2+C2))
        return 1.0 - ssim.mean()


# ---------------------------------------------------------------------------
# Global reconstruction loss  (L1 + SSIM)
# ---------------------------------------------------------------------------

class GlobalReconLoss(nn.Module):
    def __init__(self, ssim_weight: float = 0.5, channel: int = 4):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.l1   = nn.L1Loss()
        self.ssim = SSIMLoss(channel=channel)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1(pred, target) + self.ssim_weight * self.ssim(pred, target)


# ---------------------------------------------------------------------------
# Shared mask builder
# ---------------------------------------------------------------------------

def build_region_masks(seg_map: torch.Tensor) -> dict:
    """
    seg_map: (B, 1, H, W) integer BraTS labels.
    Returns float masks for WT, TC, ET.
    """
    return {
        "WT": (seg_map > 0).float(),
        "TC": ((seg_map == 1) | (seg_map == 3)).float(),
        "ET": (seg_map == 3).float(),
    }


# ---------------------------------------------------------------------------
# A. Fixed-weight pathology loss
# ---------------------------------------------------------------------------

class PathologyLoss(nn.Module):
    """Fixed weights w_ET=3, w_TC=2, w_WT=1 (clinical prior, not learned)."""

    REGION_WEIGHTS = {"WT": 1.0, "TC": 2.0, "ET": 3.0}

    def __init__(self, region_weights: Optional[dict] = None, base_loss: str = "l1"):
        super().__init__()
        self.region_weights = region_weights or self.REGION_WEIGHTS
        self.loss_fn = nn.L1Loss(reduction="none") if base_loss == "l1" \
                       else nn.MSELoss(reduction="none")

    def forward(self, pred, target, seg_map) -> torch.Tensor:
        masks    = build_region_masks(seg_map)
        pix_loss = self.loss_fn(pred, target)
        total    = torch.zeros(1, device=pred.device)
        for region, w in self.region_weights.items():
            mask  = masks[region]
            n     = mask.sum().clamp(min=1.0)
            total = total + w * (pix_loss * mask).sum() / (n * pred.shape[1])
        return total


# ---------------------------------------------------------------------------
# A. Adaptive pathology weight learning
# ---------------------------------------------------------------------------

class AdaptivePathologyLoss(nn.Module):
    """
    Learns region weights dynamically:  w_r = softplus(C_r + MLP([F_r, U_r]))

    Can be used standalone or via PPMAELoss(mode='adaptive').
    Also exposes compute_weights() for use inside ClinicalRiskPathologyLoss
    combined mode.
    """

    REGIONS         = ["WT", "TC", "ET"]
    CLINICAL_PRIORS = [1.0, 2.0, 3.0]

    def __init__(self, n_modalities: int = 4, hidden: int = 32, base_loss: str = "l1"):
        super().__init__()
        # C_r in log-space → always positive, init from clinical knowledge
        self.log_C = nn.Parameter(
            torch.log(torch.tensor(self.CLINICAL_PRIORS, dtype=torch.float32))
        )
        # F_r: severity from target image statistics
        self.severity_net = nn.Sequential(
            nn.Linear(2 * n_modalities, hidden), nn.GELU(),
            nn.Linear(hidden, 1), nn.Softplus(),
        )
        # Δw: data-driven adjustment (can be positive or negative)
        self.adjustment_net = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.loss_fn      = nn.L1Loss(reduction="none") if base_loss == "l1" \
                            else nn.MSELoss(reduction="none")
        self.n_modalities = n_modalities

    # ------------------------------------------------------------------
    @staticmethod
    def _feature_stats(target: torch.Tensor, mask: torch.Tensor):
        """Mean and std of target pixel values within mask. Returns (C,), (C,)."""
        n      = mask.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
        mean_r = (target * mask).sum(dim=(2, 3), keepdim=True) / n
        std_r  = ((target - mean_r)**2 * mask).sum(dim=(2, 3), keepdim=True).sqrt() / n.sqrt()
        return mean_r.mean(0).squeeze(), std_r.mean(0).squeeze()

    @staticmethod
    def _uncertainty(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Prediction variance within region (gradient STOPPED via detach)."""
        p      = pred.detach()
        n      = mask.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
        mean_r = (p * mask).sum(dim=(2, 3), keepdim=True) / n
        return ((p - mean_r)**2 * mask).sum() / mask.sum().clamp(min=1.0)

    # ------------------------------------------------------------------
    def compute_weights(self, pred: torch.Tensor, target: torch.Tensor,
                        masks: dict) -> dict:
        """
        Returns {region: w_r scalar tensor} without computing the loss.
        Called by ClinicalRiskPathologyLoss in combined mode.
        """
        C       = self.log_C.exp()
        weights = {}
        for i, region in enumerate(self.REGIONS):
            mask      = masks[region]
            t_mean, t_std = self._feature_stats(target, mask)
            f_r       = self.severity_net(torch.cat([t_mean, t_std]).unsqueeze(0))   # (1,1)
            u_r       = self._uncertainty(pred, mask).unsqueeze(0).unsqueeze(0)       # (1,1)
            adj       = self.adjustment_net(torch.cat([f_r, u_r], dim=-1))            # (1,1)
            weights[region] = F.softplus(C[i] + adj.squeeze())
        return weights

    def forward(self, pred, target, seg_map) -> tuple[torch.Tensor, dict]:
        masks    = build_region_masks(seg_map)
        pix_loss = self.loss_fn(pred, target)
        weights  = self.compute_weights(pred, target, masks)
        total    = torch.zeros(1, device=pred.device)
        for region, w_r in weights.items():
            mask  = masks[region]
            n     = mask.sum().clamp(min=1.0)
            L_r   = (pix_loss * mask).sum() / (n * pred.shape[1])
            total = total + w_r * L_r
        weight_dict = {r: w.item() for r, w in weights.items()}
        return total, weight_dict

    def weight_summary(self) -> str:
        C = self.log_C.exp().tolist()
        return " | ".join(f"{r}: C={c:.3f}" for r, c in zip(self.REGIONS, C))


# ---------------------------------------------------------------------------
# B. Clinical risk score network  ψ(image features, biomarkers) → R_r
# ---------------------------------------------------------------------------

class ClinicalRiskScore(nn.Module):
    """
    Estimates patient-specific clinical risk score per tumour region.

    Image-derived features (7 scalars, computed every forward pass):
        V_ET, V_TC, V_WT  — normalised volume fractions
        rho               — enhancement ratio  V_ET / V_WT
        H_WT, H_TC, H_ET  — heterogeneity (coefficient of variation σ/μ)

    Optional clinical biomarkers b (4 scalars):
        b_grade  ∈ {0,1}  WHO grade  (1 = high grade)
        b_IDH   ∈ {0,1}   IDH status (1 = wildtype → higher risk)
        b_MGMT  ∈ {0,1}   MGMT methylation (1 = unmethylated → poor response)
        b_age   ∈ [0,1]   normalised age

    Output:
        R ∈ (0,∞)^3  one score per region [R_WT, R_TC, R_ET]

    Initialised so R ≈ [1, 2, 3] for zero input (matching the clinical prior).
    The network then learns how each risk feature modulates this baseline.

    Biological interpretation of learned R_r:
        Large ET + high ρ + wildtype IDH → R_ET ↑↑
        → optimizer penalises ET reconstruction errors more heavily
        → model learns to preserve enhancing tumour boundaries for high-risk patients
    """

    N_IMAGE_FEATURES = 7
    N_BIOMARKERS     = 4

    def __init__(self, hidden: int = 64, use_biomarkers: bool = False):
        super().__init__()
        self.use_biomarkers = use_biomarkers
        n_in = self.N_IMAGE_FEATURES + (self.N_BIOMARKERS if use_biomarkers else 0)

        self.risk_net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 3),   # one score per region
            nn.Softplus(),               # R_r > 0 always
        )

        # Initialise last linear bias so Softplus output ≈ [1, 2, 3]
        # Softplus(x) = log(1+exp(x));  softplus(log(exp(c)-1)) = c
        with torch.no_grad():
            init_bias = torch.log(torch.exp(torch.tensor([1., 2., 3.])) - 1.)
            self.risk_net[-2].bias.copy_(init_bias)

    # ------------------------------------------------------------------
    @staticmethod
    def _volume_fraction(mask: torch.Tensor, total: int) -> torch.Tensor:
        return mask.sum() / max(total, 1)

    @staticmethod
    def _heterogeneity(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Coefficient of variation  σ/μ  within mask."""
        n    = mask.sum().clamp(min=1.0)
        mu   = (x * mask).sum() / (n * x.shape[1])
        sig  = ((x - mu)**2 * mask).sum().sqrt() / n.sqrt()
        return sig / mu.clamp(min=1e-6)

    def _extract_features(self, target: torch.Tensor,
                          masks: dict) -> torch.Tensor:
        """
        Compute the 7 image-derived risk features.
        All features are scalar tensors on the same device as target.
        """
        total = target.shape[-1] * target.shape[-2]

        V_WT = self._volume_fraction(masks["WT"], total)
        V_TC = self._volume_fraction(masks["TC"], total)
        V_ET = self._volume_fraction(masks["ET"], total)
        rho  = V_ET / V_WT.clamp(min=1e-6)          # enhancement ratio

        H_WT = self._heterogeneity(target, masks["WT"])
        H_TC = self._heterogeneity(target, masks["TC"])
        H_ET = self._heterogeneity(target, masks["ET"])

        return torch.stack([V_WT, V_TC, V_ET, rho, H_WT, H_TC, H_ET]).unsqueeze(0)  # (1,7)

    # ------------------------------------------------------------------
    def forward(self, target: torch.Tensor, masks: dict,
                biomarkers: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            target:     (B, C, H, W) clean reference image
            masks:      dict of region masks from build_region_masks()
            biomarkers: (B, 4) float tensor [grade, IDH, MGMT, age_norm]
                        or None (network zero-pads)

        Returns:
            R: (3,) tensor  [R_WT, R_TC, R_ET]
        """
        features = self._extract_features(target, masks)             # (1, 7)

        if self.use_biomarkers:
            if biomarkers is not None:
                b = biomarkers.float().mean(0, keepdim=True)         # (1, 4)
            else:
                b = torch.zeros(1, self.N_BIOMARKERS, device=target.device)
            features = torch.cat([features, b], dim=-1)              # (1, 11)

        return self.risk_net(features).squeeze(0)                    # (3,)

    def risk_summary(self, target: torch.Tensor, masks: dict,
                     biomarkers: Optional[torch.Tensor] = None) -> str:
        with torch.no_grad():
            R = self.forward(target, masks, biomarkers).tolist()
        regions = ["WT", "TC", "ET"]
        return " | ".join(f"{r}: R={v:.3f}" for r, v in zip(regions, R))


# ---------------------------------------------------------------------------
# B. Clinical Risk Pathology Loss  L_path = Σ_r R_r · L_r
# ---------------------------------------------------------------------------

class ClinicalRiskPathologyLoss(nn.Module):
    """
    Formal Clinical Risk Optimization:

        L_pathology = Σ_r  R_r · L_r

    Optimization is tied directly to estimated patient risk.
    High-risk regions receive higher loss weights automatically.

    Combined mode (combine_with_adaptive=True):

        w_r = R_r · φ(F_r, U_r, C_r)

        R_r captures patient-level risk  (population context)
        φ(·) captures scan-level difficulty  (this specific image)

    Args:
        n_modalities:          number of MRI channels
        hidden:                width of MLP layers in risk network
        base_loss:             'l1' or 'mse'
        use_biomarkers:        if True, risk net accepts clinical metadata
        combine_with_adaptive: if True, multiply R_r by adaptive weights φ(·)
    """

    REGIONS = ["WT", "TC", "ET"]

    def __init__(
        self,
        n_modalities:          int  = 4,
        hidden:                int  = 64,
        base_loss:             str  = "l1",
        use_biomarkers:        bool = False,
        combine_with_adaptive: bool = False,
    ):
        super().__init__()
        self.combine_with_adaptive = combine_with_adaptive
        self.risk_scorer = ClinicalRiskScore(hidden=hidden,
                                             use_biomarkers=use_biomarkers)
        if combine_with_adaptive:
            self.adaptive   = AdaptivePathologyLoss(n_modalities=n_modalities,
                                                    base_loss=base_loss)
        self.loss_fn = nn.L1Loss(reduction="none") if base_loss == "l1" \
                       else nn.MSELoss(reduction="none")

    def forward(
        self,
        pred:        torch.Tensor,               # (B, C, H, W)
        target:      torch.Tensor,               # (B, C, H, W)
        seg_map:     torch.Tensor,               # (B, 1, H, W)
        biomarkers:  Optional[torch.Tensor] = None, # (B, 4) or None
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss — scalar with gradients
            info_dict  — {region: R_r float} and optionally {region: w_r float}
        """
        masks    = build_region_masks(seg_map)
        pix_loss = self.loss_fn(pred, target)

        # Compute clinical risk scores R_r ∈ (0,∞) per region
        R = self.risk_scorer(target, masks, biomarkers)   # (3,)

        # Optionally compute adaptive weights φ(F_r, U_r, C_r)
        adaptive_w = None
        if self.combine_with_adaptive:
            adaptive_w = self.adaptive.compute_weights(pred, target, masks)

        total    = torch.zeros(1, device=pred.device)
        info     = {}

        for i, region in enumerate(self.REGIONS):
            mask = masks[region]
            n    = mask.sum().clamp(min=1.0)
            L_r  = (pix_loss * mask).sum() / (n * pred.shape[1])
            R_r  = R[i]

            if adaptive_w is not None:
                # Combined: w_r = R_r · φ(F_r, U_r, C_r)
                w_total      = R_r * adaptive_w[region]
                total        = total + w_total * L_r
                info[f"R_{region}"] = R_r.item()
                info[f"w_{region}"] = adaptive_w[region].item()
                info[f"combined_{region}"] = w_total.item()
            else:
                total              = total + R_r * L_r
                info[f"R_{region}"] = R_r.item()

        return total, info


# ---------------------------------------------------------------------------
# Cross-modal consistency loss
# ---------------------------------------------------------------------------

class CrossModalConsistencyLoss(nn.Module):
    """Cosine similarity consistency between complementary modality pairs."""

    DEFAULT_PAIRS = [(1, 2), (2, 3)]   # T1Wce↔T2W, T2W↔FLAIR

    def __init__(self, modality_pairs: Optional[list] = None):
        super().__init__()
        self.pairs = modality_pairs or self.DEFAULT_PAIRS

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros(1, device=pred.device)
        for i, j in self.pairs:
            pi = F.adaptive_avg_pool2d(pred[:,   i:i+1], (1,1)).flatten(1)
            pj = F.adaptive_avg_pool2d(pred[:,   j:j+1], (1,1)).flatten(1)
            ti = F.adaptive_avg_pool2d(target[:, i:i+1], (1,1)).flatten(1)
            tj = F.adaptive_avg_pool2d(target[:, j:j+1], (1,1)).flatten(1)
            loss = loss + F.mse_loss(F.cosine_similarity(pi, pj, dim=1),
                                     F.cosine_similarity(ti, tj, dim=1))
        return loss / max(len(self.pairs), 1)


# ---------------------------------------------------------------------------
# Composite PP-MAE loss  (unified entry point)
# ---------------------------------------------------------------------------

class PPMAELoss(nn.Module):
    """
    Composite loss:  L_total = L_global + λ1·L_pathology + λ2·L_crossmodal

    Args:
        mode: pathology loss strategy
            'fixed'         — fixed weights w_ET=3, w_TC=2, w_WT=1
            'adaptive'      — learned w_r = softplus(C_r + MLP([F_r, U_r]))
            'clinical_risk' — patient risk  L_path = Σ_r R_r · L_r
            'combined'      — risk × adaptive  w_r = R_r · φ(F_r, U_r, C_r)

        use_biomarkers: (clinical_risk / combined only)
            pass biomarkers tensor to forward() for personalized risk scoring

        lambda1, lambda2: relative weights of pathology and cross-modal terms
    """

    VALID_MODES = ("fixed", "adaptive", "clinical_risk", "combined")

    def __init__(
        self,
        lambda1:        float = 1.0,
        lambda2:        float = 0.5,
        ssim_weight:    float = 0.5,
        n_modalities:   int   = 4,
        mode:           str   = "fixed",
        use_biomarkers: bool  = False,
    ):
        super().__init__()
        assert mode in self.VALID_MODES, \
            f"mode must be one of {self.VALID_MODES}, got '{mode}'"

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mode    = mode

        self.global_loss     = GlobalReconLoss(ssim_weight=ssim_weight,
                                               channel=n_modalities)
        self.crossmodal_loss = CrossModalConsistencyLoss()

        if mode == "fixed":
            self.pathology_loss = PathologyLoss()
        elif mode == "adaptive":
            self.pathology_loss = AdaptivePathologyLoss(n_modalities=n_modalities)
        elif mode == "clinical_risk":
            self.pathology_loss = ClinicalRiskPathologyLoss(
                n_modalities=n_modalities, use_biomarkers=use_biomarkers)
        elif mode == "combined":
            self.pathology_loss = ClinicalRiskPathologyLoss(
                n_modalities=n_modalities, use_biomarkers=use_biomarkers,
                combine_with_adaptive=True)

    def forward(
        self,
        pred:       torch.Tensor,               # (B, C, H, W)
        target:     torch.Tensor,               # (B, C, H, W)
        seg_map:    torch.Tensor,               # (B, 1, H, W)
        biomarkers: Optional[torch.Tensor] = None, # (B, 4) optional clinical data
    ) -> dict:
        l_global   = self.global_loss(pred, target)
        l_crossmod = self.crossmodal_loss(pred, target)

        if self.mode == "fixed":
            l_path = self.pathology_loss(pred, target, seg_map)
            extra  = {}
        else:
            if self.mode == "adaptive":
                l_path, extra = self.pathology_loss(pred, target, seg_map)
            else:
                l_path, extra = self.pathology_loss(pred, target, seg_map,
                                                     biomarkers=biomarkers)

        l_total = l_global + self.lambda1 * l_path + self.lambda2 * l_crossmod

        result = {
            "total":      l_total,
            "global":     l_global,
            "pathology":  l_path,
            "crossmodal": l_crossmod,
        }
        result.update({k: torch.tensor(v) for k, v in extra.items()})
        return result
