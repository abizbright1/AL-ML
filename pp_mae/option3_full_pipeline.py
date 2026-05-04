"""
Option 3 — End-to-End PP-MAE + Downstream Pipeline
====================================================

WHY THIS OPTION:
    The primary clinical validation endpoint in the study protocol is not
    image quality (PSNR / SSIM) but downstream performance: does denoising
    improve tumour segmentation (DSC, HD95) and grading (AUROC)?  Option 3
    implements a fully differentiable pipeline that back-propagates gradients
    from both the denoising loss AND the downstream task losses simultaneously.
    This joint training ensures that the denoiser is optimised to preserve
    information that is genuinely useful for segmentation and grading, not
    just pixel-level fidelity.

ARCHITECTURE:
    ┌──────────────┐      ┌─────────────────────────────┐
    │  Noisy MRI   │ ───► │  PP-MAE Denoiser (Option 1  │
    │  (B, 4, H,W) │      │  backbone for efficiency)    │
    └──────────────┘      └──────────────┬──────────────┘
                                         │ Denoised MRI (B, 4, H, W)
                          ┌──────────────┴──────────────┐
                          │                              │
                    ┌─────▼──────┐               ┌──────▼──────┐
                    │ Seg Head   │               │ Grade Head  │
                    │ 4-class    │               │ Binary MLP  │
                    │ pixel-wise │               │ WHO grade   │
                    │ (nnU-Net   │               │ IDH status  │
                    │  inspired) │               └─────────────┘
                    └────────────┘

LOSS:
    L_total = L_denoising + α·L_segmentation + β·L_grading

    L_denoising    = PPMAELoss (global + pathology + crossmodal)
    L_segmentation = weighted cross-entropy + Dice  (tumour subregions)
    L_grading      = binary cross-entropy  (WHO grade / IDH)

PHD TIP:
    Train in two stages:
      Stage 1: Pre-train denoiser only (Options 1 or 2) until convergence.
      Stage 2: Freeze denoiser encoder, fine-tune denoiser + downstream heads
               jointly with a small denoiser LR (1e-5) and larger head LR (1e-4).
    This prevents the segmentation gradient from corrupting learned
    denoising representations early in training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import PPMAELoss
from option1_cnn_pp_mae import CNNPPMAE, SaliencyMasking


# ---------------------------------------------------------------------------
# Segmentation head — nnU-Net-inspired lightweight decoder
# ---------------------------------------------------------------------------

class SegHead(nn.Module):
    """
    Pixel-wise segmentation head producing 4 class logits:
        0 = background
        1 = necrotic core  (NCR)
        2 = peritumoral oedema (ED)
        3 = enhancing tumour  (ET)

    Input: denoised multimodal feature map (B, 4, H, W)
    Architecture: lightweight 3-layer CNN with skip connections
    """

    def __init__(self, in_ch: int = 4, n_classes: int = 4, hidden: int = 64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, 1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(hidden // 2, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.conv2(self.conv1(x))
        return self.out(self.conv3(f))   # (B, n_classes, H, W) — raw logits


# ---------------------------------------------------------------------------
# Grading head — global average pool + MLP binary classifier
# ---------------------------------------------------------------------------

class GradingHead(nn.Module):
    """
    Binary classifier for:
        - WHO grade (low-grade vs high-grade)
        - IDH mutation status (wildtype vs mutant)

    Can be used as two separate instances, one per task.

    Input: denoised image (B, 4, H, W) — uses global average pooling
    """

    def __init__(self, in_ch: int = 4, hidden: int = 128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1),   # logit for binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pool(x)).squeeze(1)   # (B,)


# ---------------------------------------------------------------------------
# Segmentation loss — weighted CE + Dice (class-balanced)
# ---------------------------------------------------------------------------

class SegmentationLoss(nn.Module):
    """
    Combines weighted cross-entropy and Dice loss.

    Class weights are set to upweight tumour classes relative to background
    because background dominates MRI slices.

    Args:
        class_weights: per-class CE weight [BG, NCR, ED, ET]
        dice_weight:   contribution of Dice loss relative to CE
    """

    DEFAULT_WEIGHTS = torch.tensor([0.1, 1.0, 1.0, 2.0])  # ET highest weight

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        dice_weight:   float = 0.5,
    ):
        super().__init__()
        weights = class_weights if class_weights is not None else self.DEFAULT_WEIGHTS
        self.register_buffer("class_weights", weights)
        self.dice_weight = dice_weight

    def _dice_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs  = F.softmax(logits, dim=1)          # (B, C, H, W)
        n_cls  = logits.shape[1]
        target_oh = F.one_hot(target.squeeze(1), n_cls).permute(0, 3, 1, 2).float()
        intersection = (probs * target_oh).sum(dim=(0, 2, 3))
        union        = (probs + target_oh).sum(dim=(0, 2, 3))
        dice_per_cls = 1 - 2 * intersection / (union + 1e-6)
        return dice_per_cls.mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, target.squeeze(1).long(),
                               weight=self.class_weights.to(logits.device))
        dice = self._dice_loss(logits, target)
        return ce + self.dice_weight * dice


# ---------------------------------------------------------------------------
# Full end-to-end pipeline
# ---------------------------------------------------------------------------

class PPMAEPipeline(nn.Module):
    """
    Option 3: Jointly trained denoising + segmentation + grading pipeline.

    Args:
        denoiser_kwargs: passed to CNNPPMAE
        freeze_encoder:  if True, the denoiser's encoder weights are frozen —
                         use in Stage 2 training after denoiser pre-training
    """

    def __init__(
        self,
        denoiser_kwargs: dict | None = None,
        freeze_encoder:  bool = False,
    ):
        super().__init__()
        dkw = denoiser_kwargs or {"in_channels": 4, "base_ch": 64, "depth": 4}
        self.denoiser   = CNNPPMAE(**dkw)
        self.seg_head   = SegHead(in_ch=4, n_classes=4)
        self.grade_head = GradingHead(in_ch=4)   # WHO grade
        self.idh_head   = GradingHead(in_ch=4)   # IDH status

        if freeze_encoder:
            for p in self.denoiser.encoders.parameters():
                p.requires_grad_(False)

    def forward(
        self,
        noisy:   torch.Tensor,   # (B, 4, H, W)
        seg_map: torch.Tensor,   # (B, 1, H, W) for saliency masking
    ) -> dict:
        denoised   = self.denoiser(noisy, seg_map)   # (B, 4, H, W)
        seg_logits = self.seg_head(denoised)          # (B, 4, H, W)
        grade_logit = self.grade_head(denoised)       # (B,)
        idh_logit   = self.idh_head(denoised)         # (B,)

        return {
            "denoised":    denoised,
            "seg_logits":  seg_logits,
            "grade_logit": grade_logit,
            "idh_logit":   idh_logit,
        }


# ---------------------------------------------------------------------------
# Multi-task composite loss
# ---------------------------------------------------------------------------

class PipelineLoss(nn.Module):
    """
    L_total = L_denoise + α·L_seg + β·L_grade + β·L_idh

    Args:
        alpha: weight for segmentation loss
        beta:  weight for each grading task
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        super().__init__()
        self.alpha       = alpha
        self.beta        = beta
        self.denoise_loss = PPMAELoss()
        self.seg_loss    = SegmentationLoss()
        self.bce         = nn.BCEWithLogitsLoss()

    def forward(
        self,
        outputs: dict,
        target:  torch.Tensor,        # (B, 4, H, W) clean MRI
        seg_map: torch.Tensor,        # (B, 1, H, W) integer labels
        grade_labels: torch.Tensor,   # (B,) binary 0/1
        idh_labels:   torch.Tensor,   # (B,) binary 0/1
    ) -> dict:
        l_den = self.denoise_loss(outputs["denoised"], target, seg_map)["total"]
        l_seg = self.seg_loss(outputs["seg_logits"], seg_map)
        l_grd = self.bce(outputs["grade_logit"], grade_labels.float())
        l_idh = self.bce(outputs["idh_logit"],   idh_labels.float())

        l_total = l_den + self.alpha * l_seg + self.beta * (l_grd + l_idh)
        return {
            "total":   l_total,
            "denoise": l_den,
            "seg":     l_seg,
            "grade":   l_grd,
            "idh":     l_idh,
        }


# ---------------------------------------------------------------------------
# Two-stage training manager
# ---------------------------------------------------------------------------

class PipelineTrainer:
    """
    Manages the two-stage training procedure.

    Stage 1: denoiser only  (call stage1_step)
    Stage 2: full pipeline  (call stage2_step)
    """

    def __init__(
        self,
        pipeline: PPMAEPipeline,
        device: str = "cuda",
    ):
        self.pipeline = pipeline.to(device)
        self.device   = device

        # Stage 1: denoiser-only optimizer
        self.optim_stage1 = torch.optim.AdamW(
            pipeline.denoiser.parameters(), lr=1e-4, weight_decay=1e-5
        )
        # Stage 2: full pipeline — lower LR for denoiser, higher for heads
        self.optim_stage2 = torch.optim.AdamW([
            {"params": pipeline.denoiser.parameters(), "lr": 1e-5},
            {"params": pipeline.seg_head.parameters(), "lr": 1e-4},
            {"params": pipeline.grade_head.parameters(), "lr": 1e-4},
            {"params": pipeline.idh_head.parameters(), "lr": 1e-4},
        ], weight_decay=1e-5)

        self.denoise_loss  = PPMAELoss()
        self.pipeline_loss = PipelineLoss()

    def stage1_step(self, batch: dict) -> dict:
        """Denoiser pre-training step (no downstream heads)."""
        self.pipeline.train()
        noisy  = batch["noisy"].to(self.device)
        target = batch["target"].to(self.device)
        seg    = batch["seg"].to(self.device)

        self.optim_stage1.zero_grad()
        denoised = self.pipeline.denoiser(noisy, seg)
        losses   = self.denoise_loss(denoised, target, seg)
        losses["total"].backward()
        nn.utils.clip_grad_norm_(self.pipeline.denoiser.parameters(), 1.0)
        self.optim_stage1.step()

        return {k: v.item() for k, v in losses.items()}

    def stage2_step(self, batch: dict) -> dict:
        """Joint fine-tuning step (denoiser + all heads)."""
        self.pipeline.train()
        noisy         = batch["noisy"].to(self.device)
        target        = batch["target"].to(self.device)
        seg           = batch["seg"].to(self.device)
        grade_labels  = batch["grade"].to(self.device)
        idh_labels    = batch["idh"].to(self.device)

        self.optim_stage2.zero_grad()
        outputs = self.pipeline(noisy, seg)
        losses  = self.pipeline_loss(outputs, target, seg, grade_labels, idh_labels)
        losses["total"].backward()
        nn.utils.clip_grad_norm_(self.pipeline.parameters(), 1.0)
        self.optim_stage2.step()

        return {k: v.item() for k, v in losses.items()}

    def save_checkpoint(self, path: str, stage: int, epoch: int):
        torch.save({
            "stage": stage, "epoch": epoch,
            "model_state": self.pipeline.state_dict(),
            "optim1_state": self.optim_stage1.state_dict(),
            "optim2_state": self.optim_stage2.state_dict(),
        }, path)


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    pipeline: PPMAEPipeline,
    noisy:    torch.Tensor,
    seg_map:  torch.Tensor,
    device:   str = "cuda",
) -> dict:
    """Return denoised image + segmentation + grade / IDH probabilities."""
    pipeline.eval()
    noisy   = noisy.to(device)
    seg_map = seg_map.to(device)
    outputs = pipeline(noisy, seg_map)
    return {
        "denoised":     outputs["denoised"].cpu(),
        "seg_pred":     outputs["seg_logits"].argmax(dim=1).cpu(),
        "grade_prob":   torch.sigmoid(outputs["grade_logit"]).cpu(),
        "idh_prob":     torch.sigmoid(outputs["idh_logit"]).cpu(),
    }


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = PPMAEPipeline({"in_channels": 4, "base_ch": 32, "depth": 3})
    trainer  = PipelineTrainer(pipeline, device=device)

    B, C, H, W = 2, 4, 128, 128
    batch = {
        "noisy":  torch.rand(B, C, H, W),
        "target": torch.rand(B, C, H, W),
        "seg":    torch.randint(0, 4, (B, 1, H, W)),
        "grade":  torch.randint(0, 2, (B,)),
        "idh":    torch.randint(0, 2, (B,)),
    }

    print("Option 3 — Stage 1 (denoiser pre-training)")
    m1 = trainer.stage1_step(batch)
    for k, v in m1.items():
        print(f"  {k:12s}: {v:.4f}")

    print("\nOption 3 — Stage 2 (joint fine-tuning)")
    m2 = trainer.stage2_step(batch)
    for k, v in m2.items():
        print(f"  {k:12s}: {v:.4f}")
