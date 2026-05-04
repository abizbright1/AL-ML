"""
Option 1 — CNN-based PP-MAE  (U-Net backbone with saliency masking)
====================================================================

WHY THIS OPTION:
    Lowest compute cost. Fastest to converge. Ideal for initial ablation
    studies to validate loss function design before scaling to ViT/Swin
    architectures. Runs on a single GPU (≥ 16 GB VRAM).

ARCHITECTURE OVERVIEW:
    Input : noisy multimodal MRI  (B, 4, H, W)
    ↓ Saliency mask generation from segmentation map
    ↓ Token-level soft masking  → encoder-compatible input
    ↓ U-Net encoder  (ResNet-style blocks, skip connections)
    ↓ Bottleneck  (context aggregation)
    ↓ U-Net decoder  (upsampling + skip fusion)
    Output: denoised multimodal MRI  (B, 4, H, W)

    Loss = L_global + λ1·L_pathology + λ2·L_crossmodal

    Downstream heads (optional, Option 3 extends this):
        Segmentation head — 4-class pixel-wise classifier
        Grading head      — binary MLP (grade / IDH)

PHD TIP:
    Treat this as your "workhorse" baseline. Run full ablations here
    (no pathology loss, no cross-modal loss, both) before implementing
    Options 2–4. This proves your loss design matters independently of
    the architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import PPMAELoss


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pad: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    """Residual block with two conv layers and an identity shortcut."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBnRelu(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.conv2(self.conv1(x)))


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_res: int = 2):
        super().__init__()
        self.conv_in = ConvBnRelu(in_ch, out_ch)
        self.res     = nn.Sequential(*[ResBlock(out_ch) for _ in range(n_res)])
        self.pool    = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.res(self.conv_in(x))
        return self.pool(x), x    # (downsampled, skip)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = ConvBnRelu(in_ch // 2 + skip_ch, out_ch)
        self.res  = ResBlock(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(self.conv(x))


# ---------------------------------------------------------------------------
# Saliency-guided soft masking
#
# Rather than hard binary masking (which loses tumour context), we apply a
# soft attention weight derived from the segmentation map.  Tumour regions
# receive weight > 1 so the encoder attends more to them, while background
# is attenuated (weight < 1).  This avoids the oversmoothing risk flagged
# by Choi et al. (2025).
# ---------------------------------------------------------------------------

class SaliencyMasking(nn.Module):
    """
    Computes per-pixel soft attention weights from a segmentation map.

    Args:
        tumour_weight:     weight applied to tumour pixels (> 1 amplifies)
        background_weight: weight applied to non-tumour pixels (< 1 attenuates)
        blur_sigma:        Gaussian blur to smooth the boundary — avoids
                           hard edges that destabilise gradient flow.
    """

    def __init__(
        self,
        tumour_weight:     float = 2.0,
        background_weight: float = 0.5,
        blur_sigma:        float = 2.0,
    ):
        super().__init__()
        self.tw = tumour_weight
        self.bw = background_weight

        # Fixed Gaussian kernel for boundary smoothing
        k = 7
        coords = torch.arange(k, dtype=torch.float32) - k // 2
        g = torch.exp(-coords ** 2 / (2 * blur_sigma ** 2))
        g = g / g.sum()
        kernel = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel)

    def forward(self, seg_map: torch.Tensor) -> torch.Tensor:
        """
        seg_map: (B, 1, H, W) integer label map.
        Returns soft weight map (B, 1, H, W) in [background_weight, tumour_weight].
        """
        tumour_bin = (seg_map > 0).float()

        # Smooth the binary mask to soften boundaries
        smoothed = F.conv2d(
            tumour_bin,
            self.kernel,
            padding=self.kernel.shape[-1] // 2,
        )
        smoothed = smoothed.clamp(0, 1)

        weight = self.bw + (self.tw - self.bw) * smoothed
        return weight


# ---------------------------------------------------------------------------
# CNN-based PP-MAE
# ---------------------------------------------------------------------------

class CNNPPMAE(nn.Module):
    """
    Option 1: U-Net PP-MAE with saliency-guided masking.

    Args:
        in_channels:  number of MRI modalities (default 4)
        base_ch:      base channel count — controls model capacity
        depth:        number of encoder/decoder stages
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_ch:     int = 64,
        depth:       int = 4,
    ):
        super().__init__()
        self.saliency = SaliencyMasking()

        # Encoder
        chs = [in_channels] + [base_ch * (2 ** i) for i in range(depth)]
        self.encoders = nn.ModuleList([
            EncoderBlock(chs[i], chs[i + 1]) for i in range(depth)
        ])

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBnRelu(chs[-1], chs[-1] * 2),
            ResBlock(chs[-1] * 2),
            ResBlock(chs[-1] * 2),
            ConvBnRelu(chs[-1] * 2, chs[-1]),
        )

        # Decoder
        # dec_chs[i] matches both the input channels (from prior decoder stage or
        # bottleneck) and the skip channels (from the symmetric encoder stage).
        dec_chs = list(reversed(chs[1:]))    # [ch_depth, ..., ch_1]
        self.decoders = nn.ModuleList([
            DecoderBlock(dec_chs[i], dec_chs[i],
                         dec_chs[i + 1] if i + 1 < len(dec_chs) else dec_chs[-1])
            for i in range(depth - 1)
        ] + [
            DecoderBlock(dec_chs[-1], chs[1], chs[1])
        ])

        self.head = nn.Conv2d(chs[1], in_channels, 1)

    def forward(
        self,
        x: torch.Tensor,          # (B, 4, H, W) noisy input
        seg_map: torch.Tensor,    # (B, 1, H, W) tumour labels
    ) -> torch.Tensor:
        # Saliency-weighted input — tumour regions amplified before encoding
        weight = self.saliency(seg_map)
        x = x * weight

        skips, out = [], x
        for enc in self.encoders:
            out, skip = enc(out)
            skips.append(skip)

        out = self.bottleneck(out)

        for dec, skip in zip(self.decoders, reversed(skips)):
            out = dec(out, skip)

        return torch.sigmoid(self.head(out))   # (B, 4, H, W) in [0, 1]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PPMAETrainer:
    """
    Minimal training loop for Option 1.

    Usage:
        trainer = PPMAETrainer(model, optimizer, device="cuda")
        for batch in dataloader:
            metrics = trainer.step(batch)
    """

    def __init__(
        self,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        device:    str = "cuda",
        lambda1:   float = 1.0,
        lambda2:   float = 0.5,
    ):
        self.model  = model.to(device)
        self.optim  = optimizer
        self.device = device
        self.loss_fn = PPMAELoss(lambda1=lambda1, lambda2=lambda2)

    @torch.no_grad()
    def validate(self, batch: dict) -> dict:
        self.model.eval()
        noisy  = batch["noisy"].to(self.device)
        target = batch["target"].to(self.device)
        seg    = batch["seg"].to(self.device)
        pred   = self.model(noisy, seg)
        return self.loss_fn(pred, target, seg)

    def step(self, batch: dict) -> dict:
        self.model.train()
        noisy  = batch["noisy"].to(self.device)
        target = batch["target"].to(self.device)
        seg    = batch["seg"].to(self.device)

        self.optim.zero_grad()
        pred = self.model(noisy, seg)
        losses = self.loss_fn(pred, target, seg)
        losses["total"].backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optim.step()

        return {k: v.item() for k, v in losses.items()}


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = CNNPPMAE(in_channels=4, base_ch=32, depth=3).to(device)

    B, C, H, W = 2, 4, 128, 128
    noisy  = torch.rand(B, C, H, W, device=device)
    target = torch.rand(B, C, H, W, device=device)
    seg    = torch.randint(0, 4, (B, 1, H, W), device=device)

    optim   = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    trainer = PPMAETrainer(model, optim, device=device)
    metrics = trainer.step({"noisy": noisy, "target": target, "seg": seg})

    print("Option 1 — CNN PP-MAE")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
