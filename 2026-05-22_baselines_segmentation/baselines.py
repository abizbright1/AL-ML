"""
baselines.py — Standard denoising baselines for comparison with PP-MAE
=======================================================================

Implements 4 reference models used in the MRI denoising literature:

  1. DnCNN        (Zhang et al., 2017, TIP)          — residual CNN, σ-blind
  2. StandardUNet (Ronneberger et al., 2015)          — U-Net + plain L1 loss
  3. Noise2Noise  (Lehtinen et al., 2018, ICML)       — self-supervised, no clean targets
  4. REDNet       (Mao et al., 2016, NeurIPS)         — symmetric residual en/decoder

All trainers follow the same interface:
    trainer = BaselineTrainer(model, device='cpu')
    metrics = trainer.step(batch)   # batch has keys 'noisy', 'target', 'seg'
    metrics  → dict with 'total' key

Tumour-region metrics:
    dice_score(pred, seg)  → dict {wt, tc, et}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_region_masks(seg: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    seg : (B, 1, H, W) long tensor  [0=BG, 1=NCR, 2=ED, 3=ET]
    Returns float masks for WT (1+2+3), TC (1+3), ET (3)
    """
    wt = (seg > 0).float()
    tc = ((seg == 1) | (seg == 3)).float()
    et = (seg == 3).float()
    return {'wt': wt, 'tc': tc, 'et': et}


def dice_score(pred: torch.Tensor, seg: torch.Tensor, eps: float = 1e-6) -> Dict[str, float]:
    """
    Compute Dice scores for WT / TC / ET tumour regions.

    pred : (B, C, H, W)  — predicted image (values 0-1)
    seg  : (B, 1, H, W)  — integer segmentation map

    Strategy: binarise the predicted tumour channel mean by Otsu-like threshold
    (mean > 0.5) and compare against GT tumour masks.
    """
    masks = build_region_masks(seg)
    scores = {}
    # Use mean of all channels as a tumour "likelihood" map
    pred_mean = pred.mean(dim=1, keepdim=True)  # (B,1,H,W)
    pred_bin  = (pred_mean > 0.5).float()
    for name, gt_mask in masks.items():
        inter = (pred_bin * gt_mask).sum()
        union = pred_bin.sum() + gt_mask.sum()
        scores[name] = (2. * inter / (union + eps)).item()
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DnCNN  (Zhang et al., 2017)
# ─────────────────────────────────────────────────────────────────────────────

class DnCNN(nn.Module):
    """
    Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising
    Zhang et al., IEEE TIP 2017.

    Architecture: Conv → [BN+ReLU+Conv] × (depth-2) → Conv
    Output: noise residual  →  clean = noisy − residual
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 4,
                 num_layers: int = 17, features: int = 64):
        super().__init__()
        layers = [nn.Conv2d(in_channels, features, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [nn.Conv2d(features, features, 3, padding=1),
                       nn.BatchNorm2d(features),
                       nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(features, out_channels, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.net(x)
        return (x - residual).clamp(0., 1.)


class DnCNNTrainer:
    def __init__(self, model: DnCNN, device: str = 'cuda', lr: float = 1e-3):
        self.model  = model.to(device)
        self.device = device
        self.optim  = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def step(self, batch: dict) -> dict:
        self.model.train()
        x   = batch['noisy'].to(self.device)
        tgt = batch['target'].to(self.device)
        self.optim.zero_grad()
        pred = self.model(x)
        loss = self.loss_fn(pred, tgt)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        return {'total': loss.item()}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Standard U-Net  (Ronneberger et al., 2015) — plain L1
# ─────────────────────────────────────────────────────────────────────────────

class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class StandardUNet(nn.Module):
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    Ronneberger et al., MICCAI 2015.

    Trained here with plain L1 (MAE) loss — no pathology awareness.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 4, base_ch: int = 32):
        super().__init__()
        b = base_ch
        self.enc1 = _DoubleConv(in_channels, b)
        self.enc2 = _DoubleConv(b,  b*2)
        self.enc3 = _DoubleConv(b*2, b*4)
        self.bot  = _DoubleConv(b*4, b*8)
        self.up3  = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(b*8, b*4, kernel_size=3, padding=1))
        self.dec3 = _DoubleConv(b*8, b*4)
        self.up2  = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(b*4, b*2, kernel_size=3, padding=1))
        self.dec2 = _DoubleConv(b*4, b*2)
        self.up1  = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(b*2, b, kernel_size=3, padding=1))
        self.dec1 = _DoubleConv(b*2, b)
        self.out  = nn.Conv2d(b, out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bot(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1).clamp(0., 1.)


class StandardUNetTrainer:
    def __init__(self, model: StandardUNet, device: str = 'cuda', lr: float = 1e-4):
        self.model  = model.to(device)
        self.device = device
        self.optim  = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.L1Loss()

    def step(self, batch: dict) -> dict:
        self.model.train()
        x   = batch['noisy'].to(self.device)
        tgt = batch['target'].to(self.device)
        self.optim.zero_grad()
        pred = self.model(x)
        loss = self.loss_fn(pred, tgt)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        return {'total': loss.item()}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Noise2Noise  (Lehtinen et al., 2018)
# ─────────────────────────────────────────────────────────────────────────────

class Noise2Noise(nn.Module):
    """
    Noise2Noise: Learning Image Restoration without Clean Data
    Lehtinen et al., ICML 2018.

    Architecture: identical to DnCNN backbone.
    Training: predicts a second noisy realisation of the same scene — no clean target needed.
    At inference, output = denoised prediction from a single noisy input.

    Here we simulate a second noisy view by adding independent noise to the input.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 4,
                 num_layers: int = 17, features: int = 64):
        super().__init__()
        layers = [nn.Conv2d(in_channels, features, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [nn.Conv2d(features, features, 3, padding=1),
                       nn.BatchNorm2d(features),
                       nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(features, out_channels, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.net(x)
        return (x - residual).clamp(0., 1.)


class Noise2NoiseTrainer:
    """
    Trains Noise2Noise: input view_1 → predict view_2 (another noisy realisation).
    Loss is L2 between prediction and view_2 (NOT the clean target).
    At eval time, standard forward pass acts as denoiser.
    """
    def __init__(self, model: Noise2Noise, device: str = 'cuda',
                 lr: float = 1e-3, noise_sigma: float = 0.08):
        self.model  = model.to(device)
        self.device = device
        self.sigma  = noise_sigma
        self.optim  = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def step(self, batch: dict) -> dict:
        self.model.train()
        x = batch['noisy'].to(self.device)
        # Simulate second noisy observation (same clean scene, independent noise)
        view2 = (x + self.sigma * torch.randn_like(x)).clamp(0., 1.)
        self.optim.zero_grad()
        pred = self.model(x)
        loss = self.loss_fn(pred, view2)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        return {'total': loss.item()}


# ─────────────────────────────────────────────────────────────────────────────
# 4.  REDNet  (Mao et al., 2016)
# ─────────────────────────────────────────────────────────────────────────────

class _ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, **kw):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, **kw),
                                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)


class _DeconvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, **kw):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, **kw),
                                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)


class REDNet(nn.Module):
    """
    Beyond a Gaussian Denoiser → REDNet-30 variant.
    Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks
    with Symmetric Skip Connections. Mao et al., NeurIPS 2016.

    Symmetric skip connections every 2 layers between encoder and decoder.
    num_layers: depth of the encoder (= depth of the decoder).
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 4,
                 num_layers: int = 5, features: int = 64):
        super().__init__()
        self.num_layers = num_layers

        # Encoder
        self.encoders = nn.ModuleList()
        self.encoders.append(_ConvBnRelu(in_channels, features, kernel_size=3, padding=1))
        for _ in range(num_layers - 1):
            self.encoders.append(_ConvBnRelu(features, features, kernel_size=3, padding=1))

        # Decoder (symmetric)
        self.decoders = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.decoders.append(_DeconvBnRelu(features, features, kernel_size=3, padding=1))
        self.decoders.append(nn.Conv2d(features, out_channels, kernel_size=3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_feats = []
        h = x
        for enc in self.encoders:
            h = enc(h)
            enc_feats.append(h)

        for i, dec in enumerate(self.decoders):
            h = dec(h)
            # Symmetric skip connection: add encoder feature from mirror layer
            mirror_idx = self.num_layers - 2 - i
            if mirror_idx >= 0 and h.shape == enc_feats[mirror_idx].shape:
                h = h + enc_feats[mirror_idx]
                h = F.relu(h, inplace=True)

        return h.clamp(0., 1.)


class REDNetTrainer:
    def __init__(self, model: REDNet, device: str = 'cuda', lr: float = 1e-4):
        self.model  = model.to(device)
        self.device = device
        self.optim  = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def step(self, batch: dict) -> dict:
        self.model.train()
        x   = batch['noisy'].to(self.device)
        tgt = batch['target'].to(self.device)
        self.optim.zero_grad()
        pred = self.model(x)
        loss = self.loss_fn(pred, tgt)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        return {'total': loss.item()}


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def make_all_baselines(in_channels: int = 4, device: str = 'cpu'):
    """
    Returns a dict of (model, trainer) pairs for all 4 baselines.
    """
    m1 = DnCNN(in_channels)
    m2 = StandardUNet(in_channels)
    m3 = Noise2Noise(in_channels)
    m4 = REDNet(in_channels)
    return {
        'DnCNN':       (m1, DnCNNTrainer(m1, device=device)),
        'UNet-L1':     (m2, StandardUNetTrainer(m2, device=device)),
        'Noise2Noise': (m3, Noise2NoiseTrainer(m3, device=device)),
        'REDNet':      (m4, REDNetTrainer(m4, device=device)),
    }
