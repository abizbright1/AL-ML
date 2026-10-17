"""
literature_losses.py — published restoration objectives, as baselines
======================================================================

Loss functions taken from the papers supplied, implemented against the same
backbone so the objective is the only thing that differs between arms.

Sources, with the exact forms the papers specify:

  Zhao H, Gallo O, Frosio I, Kautz J. "Loss Functions for Image Restoration
  with Neural Networks." IEEE Trans. Computational Imaging, 2017
  (arXiv:1511.08861v3).
      L2, SSIM, MS-SSIM, and their Mix:
          L_Mix = a * L_MS-SSIM + (1 - a) * G_sigmaM . L_l1        (Eq. 13)
      with a = 0.84 (Sec. III-D). The Gaussian G_sigmaM multiplies the L1 map
      POINT-WISE -- the paper is explicit that this matters, because MS-SSIM
      propagates error at pixel q by its contribution to the central pixel,
      so the l1 term must be weighted the same way. Implementations that drop
      the Gaussian are not running Zhao's Mix.

  Johnson J, Alahi A, Fei-Fei L. "Perceptual Losses for Real-Time Style
  Transfer and Super-Resolution." ECCV 2016 (arXiv:1603.08155v1).
      Feature-reconstruction loss on a pretrained VGG.

  Yang G, Yu S, Dong H, et al. "DAGAN: Deep De-Aliasing Generative
  Adversarial Networks for Fast Compressed Sensing MRI Reconstruction."
  IEEE Trans. Medical Imaging, 2017 (DOI 10.1109/TMI.2017.2785879).
      L_TOTAL = a*L_iMSE + b*L_fMSE + g*L_VGG + L_GEN                (Eq. 13)
      with a = 15, b = 0.1, g = 0.0025 (Sec. on training settings).
      L_fMSE is MSE between the FOURIER TRANSFORMS of output and target.

IMPORTANT SCOPE NOTE on DAGAN. It solves compressed-sensing MRI
RECONSTRUCTION from under-sampled k-space; this study performs image-domain
DENOISING of already-reconstructed BraTS volumes. There is no measured
k-space here, so DAGAN's data-consistency setting does not transfer. What
does transfer is its OBJECTIVE: pixel MSE + Fourier-domain MSE + VGG, which
is computable from any image pair. The adversarial term is omitted because
this pipeline has no discriminator. The arm is therefore
"DAGAN-style (non-adversarial)" and must never be presented as DAGAN.

Every loss below is spatially UNIFORM: a tumour pixel is treated exactly like
a ventricle pixel. That is the point. They occupy the conventional-strength
tier of the benchmark, showing whether PathologyLoss gains anything over
strong standard objectives rather than only over plain L1.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
#  Structural losses (Zhao et al.)
# ═══════════════════════════════════════════════════════════════════════════

def _gaussian_window(ws: int, sigma: float, channel: int) -> torch.Tensor:
    coords = torch.arange(ws, dtype=torch.float32) - ws // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w2 = g.unsqueeze(1).mm(g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    return w2.expand(channel, 1, ws, ws).contiguous()


def _ssim_maps(pred, target, win, pad, C):
    """Return (luminance*contrast*structure map, contrast*structure map)."""
    mu1 = F.conv2d(pred, win, padding=pad, groups=C)
    mu2 = F.conv2d(target, win, padding=pad, groups=C)
    mu1s, mu2s, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = F.conv2d(pred * pred, win, padding=pad, groups=C) - mu1s
    s2 = F.conv2d(target * target, win, padding=pad, groups=C) - mu2s
    s12 = F.conv2d(pred * target, win, padding=pad, groups=C) - mu12
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    cs = (2 * s12 + C2) / (s1 + s2 + C2)
    ssim = ((2 * mu12 + C1) / (mu1s + mu2s + C1)) * cs
    return ssim, cs


class MSSSIMLoss(nn.Module):
    """1 - MS-SSIM over a dyadic pyramid (Zhao et al., Eq. 12).

    Scale count adapts to the patch size. The 11x11 window needs at least
    ~11 px per side, so a 96x96 patch supports 3 scales (96 -> 48 -> 24) and
    not the 5 used on larger natural images. Forcing 5 would convolve an
    11x11 window over a 6x6 image, which is meaningless. The realised number
    of scales is exposed as `self.levels` and should be stated in the paper.
    """

    FULL_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)

    def __init__(self, channel: int = 4, window_size: int = 11,
                 patch_size: int = 96, max_levels: int = 5):
        super().__init__()
        self.ws, self.channel = window_size, channel
        lv = 1
        s = patch_size
        while lv < max_levels and s // 2 >= window_size:
            s //= 2
            lv += 1
        self.levels = lv
        w = torch.tensor(self.FULL_WEIGHTS[:lv])
        self.register_buffer('weights', w / w.sum())
        self.register_buffer('win', _gaussian_window(window_size, 1.5, channel))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pad = self.ws // 2
        vals = []
        x, y = pred, target
        for i in range(self.levels):
            ssim, cs = _ssim_maps(x, y, self.win.to(x.device), pad, self.channel)
            vals.append((cs.mean() if i < self.levels - 1 else ssim.mean()).clamp(min=1e-6))
            if i < self.levels - 1:
                x = F.avg_pool2d(x, 2)
                y = F.avg_pool2d(y, 2)
        w = self.weights.to(pred.device)
        msssim = torch.prod(torch.stack(vals) ** w)
        return 1.0 - msssim


class MixLoss(nn.Module):
    """Zhao et al. Eq. 13:  a * L_MS-SSIM + (1 - a) * G_sigmaM . L_l1,  a = 0.84.

    The Gaussian multiplies the per-pixel L1 map before averaging. Dropping it
    gives a different objective that the paper does not evaluate.
    """

    def __init__(self, channel: int = 4, alpha: float = 0.84,
                 sigma_m: float = 8.0, patch_size: int = 96):
        super().__init__()
        self.alpha = alpha
        self.msssim = MSSSIMLoss(channel=channel, patch_size=patch_size)
        ws = int(2 * round(3 * sigma_m) + 1)                 # ~3 sigma support
        self.register_buffer('gwin', _gaussian_window(ws, sigma_m, channel))
        self.pad = ws // 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_ms = self.msssim(pred, target)
        l1_map = (pred - target).abs()
        g = self.gwin.to(pred.device)
        l1_w = F.conv2d(l1_map, g, padding=self.pad, groups=pred.shape[1])
        return self.alpha * l_ms + (1.0 - self.alpha) * l1_w.mean()


# ═══════════════════════════════════════════════════════════════════════════
#  Frequency-domain loss (DAGAN's L_fMSE)
# ═══════════════════════════════════════════════════════════════════════════

class FourierMSELoss(nn.Module):
    """MSE between the Fourier transforms of output and target.

    Needs no k-space: the transform is taken of the images themselves, so it
    applies to image-domain denoising even though DAGAN used it for k-space
    reconstruction. Tumour margins are high-frequency content, which makes
    this the one spatially-uniform loss with a plausible route to preserving
    them without a mask -- and therefore a genuine threat to the
    PathologyLoss claim rather than a token baseline.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fp = torch.fft.fft2(pred.float(), norm='ortho')
        ft = torch.fft.fft2(target.float(), norm='ortho')
        return ((fp - ft).abs() ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════════
#  Perceptual loss (Johnson et al.; VGG as used by DAGAN)
# ═══════════════════════════════════════════════════════════════════════════

class VGGPerceptualLoss(nn.Module):
    """Feature-reconstruction loss on a pretrained VGG.

    Two caveats that belong in the paper, not in a footnote nobody reads.

    1. VGG is trained on 3-channel ImageNet photographs. BraTS has four
       co-registered MRI modalities, none of them RGB. Channels are mapped to
       pseudo-RGB and passed through in groups; this is the standard
       workaround and it is still a domain mismatch. Several low-dose CT
       papers abandoned VGG perceptual loss for exactly this reason.
    2. The weights download on first use. If the machine is offline, pass
       `weights=None` to get a randomly initialised VGG -- which is the
       "untrained perceptual loss" variant reported to beat L1 on MR
       line-like structures, so it is a legitimate arm in its own right, but
       it is a DIFFERENT arm and must be labelled as such.
    """

    def __init__(self, layer: int = 16, weights: Optional[str] = 'DEFAULT',
                 resize_to: int = 0):
        super().__init__()
        from torchvision.models import vgg16
        net = vgg16(weights=weights)
        self.features = nn.Sequential(*list(net.features.children())[:layer]).eval()
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.pretrained = weights is not None
        self.resize_to = resize_to
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
        elif C == 3:
            pass
        else:                                   # 4 modalities -> pseudo-RGB triples
            x = torch.cat([x[:, i:i + 3] if i + 3 <= C
                           else x[:, -3:] for i in range(0, C, 3)], dim=0)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        if self.resize_to:
            x = F.interpolate(x, size=(self.resize_to, self.resize_to),
                              mode='bilinear', align_corners=False)
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fp = self.features(self._prep(pred))
        with torch.no_grad():
            ft = self.features(self._prep(target))
        return F.mse_loss(fp, ft)


# ═══════════════════════════════════════════════════════════════════════════
#  Composites
# ═══════════════════════════════════════════════════════════════════════════

class DAGANStyleLoss(nn.Module):
    """a*L_iMSE + b*L_fMSE + g*L_VGG, with DAGAN's published weights.

    a = 15, b = 0.1, g = 0.0025. The adversarial term L_GEN is omitted --
    this pipeline has no discriminator. Report as "DAGAN-style
    (non-adversarial)", never as DAGAN.
    """

    def __init__(self, channel: int = 4, a: float = 15.0, b: float = 0.1,
                 g: float = 0.0025, vgg_weights: Optional[str] = 'DEFAULT'):
        super().__init__()
        self.a, self.b, self.g = a, b, g
        self.fourier = FourierMSELoss()
        self.vgg = VGGPerceptualLoss(weights=vgg_weights) if g else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = self.a * F.mse_loss(pred, target) + self.b * self.fourier(pred, target)
        if self.vgg is not None:
            total = total + self.g * self.vgg(pred, target)
        return total


class HighFreqWeightedLoss(nn.Module):
    """Fourier MSE with a radial high-pass weight.

    WHY THIS EXISTS. Plain Fourier MSE under an orthonormal transform is
    IDENTICAL to image-domain MSE -- Parseval's theorem, since the transform
    is unitary. DAGAN's L_fMSE is informative there because it is matched
    against MEASURED k-space, which is a different quantity from the image.
    Transplanted into image-domain denoising it is redundant with L_iMSE and
    tests nothing.

    Weighting the spectrum breaks that equivalence and asks the question the
    redundant version cannot: enhancing-tumour margins are high-frequency
    content, so can a spatially UNIFORM objective preserve them by
    emphasising high frequencies, with no mask at all? If it can, it is a
    serious competitor to PathologyLoss on far simpler machinery.

    w(f) = (|f| / f_nyquist) ** p,  p = 1 by default.
    """

    def __init__(self, power: float = 1.0):
        super().__init__()
        self.power = power
        self._cache = {}

    def _weight(self, H, W, device, dtype):
        key = (H, W, device, dtype)
        if key not in self._cache:
            fy = torch.fft.fftfreq(H, device=device).view(-1, 1)
            fx = torch.fft.fftfreq(W, device=device).view(1, -1)
            r = (fy ** 2 + fx ** 2).sqrt()
            self._cache[key] = (r / r.max()).pow(self.power).to(dtype)
        return self._cache[key]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fp = torch.fft.fft2(pred.float(), norm='ortho')
        ft = torch.fft.fft2(target.float(), norm='ortho')
        w = self._weight(pred.shape[-2], pred.shape[-1], pred.device, torch.float32)
        return (w * (fp - ft).abs() ** 2).mean()


LITERATURE_LOSSES = {
    'l2':        lambda ch, ps: (lambda p, t: F.mse_loss(p, t)),
    'msssim':    lambda ch, ps: MSSSIMLoss(channel=ch, patch_size=ps),
    'mix':       lambda ch, ps: MixLoss(channel=ch, patch_size=ps),
    'fourier':   lambda ch, ps: FourierMSELoss(),      # == l2, see docstring
    'highfreq':  lambda ch, ps: HighFreqWeightedLoss(power=1.0),
    'vgg':       lambda ch, ps: VGGPerceptualLoss(),
    'vgg-untrained': lambda ch, ps: VGGPerceptualLoss(weights=None),
    'dagan':     lambda ch, ps: DAGANStyleLoss(channel=ch),
    'dagan-novgg': lambda ch, ps: DAGANStyleLoss(channel=ch, g=0.0),
}


def _self_test() -> None:
    torch.manual_seed(0)
    B, C, H = 2, 4, 96
    pred = torch.rand(B, C, H, H, requires_grad=True)
    target = torch.rand(B, C, H, H)
    print(f'  patch {H}x{H}, {C} channels')
    ms = MSSSIMLoss(channel=C, patch_size=H)
    print(f'  MS-SSIM pyramid levels resolved to {ms.levels} '
          f'(5 would need a {11 * 2 ** 4}px patch)')
    for name in ('l2', 'msssim', 'mix', 'fourier', 'highfreq', 'dagan-novgg'):
        fn = LITERATURE_LOSSES[name](C, H)
        v = fn(pred, target)
        assert v.ndim == 0 and torch.isfinite(v), name
        v.backward(retain_graph=True)
        print(f'  {name:<14} ok   value={v.item():.6f}')
    a = LITERATURE_LOSSES['l2'](C, H)(pred, target)
    b = LITERATURE_LOSSES['fourier'](C, H)(pred, target)
    print(f'\n  Parseval check: l2={a.item():.8f}  fourier={b.item():.8f}'
          f'  identical={torch.allclose(a, b, atol=1e-6)}')
    print('  -> plain Fourier MSE is redundant with l2; use highfreq instead.')
    print('\n  VGG arms skipped here (weights download on first use).')
    print('  all checks passed')


if __name__ == '__main__':
    _self_test()
