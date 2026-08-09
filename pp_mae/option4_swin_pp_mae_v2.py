"""
option4_swin_pp_mae_v2.py — PP-MAE with CMA and SAR actually implemented
=========================================================================

A COPY of option4_swin_pp_mae.py with the two contributions fixed.
The original file is untouched, so you can A/B the two directly.

What was wrong in v1
--------------------
1. CMA: the CrossModalAttention *class* was correct, but the caller did
       tokens, _ = self.cross_modal[i](tokens, tokens)
   passing the same tensor as both arguments -> self-attention. The
   modality indices (a_idx, b_idx) were unpacked and discarded.

   Root cause: patch_embed was Conv2d(4, D, 4, stride=4), which collapses
   all four modalities into ONE token stream at the first layer. After
   that there is nothing left to attend *between*.

2. SAR: SaliencyFeatureReweighter took the GROUND-TRUTH mask as an input
   and looked it up in nn.Embedding. Nothing was predicted, and the model
   required ground truth at inference. Its gate was a Sigmoid, so features
   were ATTENUATED (x0..1) where Eq 4.5 specifies AMPLIFICATION (x1..3).

What v2 does instead
--------------------
1. MultiModalPatchEmbed keeps four separate token streams, so
   CrossModalAttention receives genuinely different modalities:
       T1ce <-> T2      (enhancement vs. fluid)
       T2   <-> FLAIR   (fluid vs. oedema)

2. SaliencyPredictor estimates S from the encoder's own features:
       S = sigmoid(conv(feat))            in [0, 1]
       F~ = F * (1 + lambda_s * S)        gain in [1, 3] with lambda_s = 2
   Ground truth enters ONLY through an auxiliary soft-Dice loss during
   training. seg_map is optional at inference -> mask-free, as promised
   in proposal section 6.3.

Usage
-----
    from option4_swin_pp_mae_v2 import SwinPPMAEv2, SwinPPMAEv2Trainer

    model = SwinPPMAEv2(in_ch=4, embed_dim=48, depths=(2,2,2,2),
                        n_heads=(3,3,6,6), window_size=4, lambda_s=2.0)
    recon, saliency = model(noisy)          # NO mask needed

Self-test
---------
    python3 pp_mae/option4_swin_pp_mae_v2.py
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse everything that was already correct in v1.
from option4_swin_pp_mae import (
    _SafeLayerNorm,
    _MPSMHA,
    CrossModalAttention,     # this class was fine; only its caller was broken
    SwinStage,
    SwinDecoderStage,
)
from losses import PPMAELoss


# ---------------------------------------------------------------------------
# 1. Per-modality patch embedding  —  makes cross-modal attention possible
# ---------------------------------------------------------------------------

class MultiModalPatchEmbed(nn.Module):
    """Embed each MRI modality into its OWN token stream.

    v1 used a single Conv2d(4, D, 4, stride=4), fusing all modalities
    immediately. That is why cross-modal attention could not work: there
    was only one stream left.

    Here each modality gets its own Conv2d(1, D, 4, stride=4), so four
    streams survive into the attention stage.

    Shapes (H = W = 96, patch = 4, D = embed_dim):
        input   (B, 4, 96, 96)
        output  list of 4 tensors, each (B, 576, D)      576 = 24 * 24
                plus (Ph, Pw) = (24, 24)
    """

    def __init__(self, in_ch: int = 4, embed_dim: int = 48, patch: int = 4):
        super().__init__()
        self.in_ch = in_ch
        self.embeds = nn.ModuleList([
            nn.Conv2d(1, embed_dim, patch, stride=patch, bias=False)
            for _ in range(in_ch)
        ])
        self.norm = _SafeLayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], Tuple[int, int]]:
        tokens: List[torch.Tensor] = []
        Ph = Pw = None
        for m in range(self.in_ch):
            # x[:, m:m+1] keeps the channel dim -> (B, 1, H, W)
            t = self.embeds[m](x[:, m:m + 1])           # (B, D, Ph, Pw)
            _, D, Ph, Pw = t.shape
            # (B, D, Ph, Pw) -> (B, Ph, Pw, D) -> (B, N, D)
            t = t.permute(0, 2, 3, 1).contiguous().reshape(x.shape[0], Ph * Pw, D)
            tokens.append(self.norm(t))
        return tokens, (Ph, Pw)


# ---------------------------------------------------------------------------
# 2. Saliency-Aware Reweighting  —  Equation 4.5, finally
# ---------------------------------------------------------------------------

class SaliencyPredictor(nn.Module):
    """Predicts a tumour-saliency map from features, then amplifies.

        S  = sigmoid(conv_stack(F))          S in [0, 1]
        F~ = F * (1 + lambda_s * S)          gain in [1, 1 + lambda_s]

    With lambda_s = 2.0 the gain spans [1, 3]: background features pass
    through untouched (gain 1), confident-tumour features are tripled.

    Note what is NOT here: any seg argument. S comes from the features
    alone, which is what makes inference mask-free.
    """

    def __init__(self, channels: int, lambda_s: float = 2.0):
        super().__init__()
        self.lambda_s = lambda_s
        hidden = max(channels // 4, 8)
        self.saliency = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),                      # bounds S to [0, 1]
        )
        # Start near S ~ 0.5 so early training neither saturates nor dies.
        nn.init.zeros_(self.saliency[2].bias)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """feat: (B, H, W, C)  ->  (modulated (B, H, W, C), S (B, 1, H, W))"""
        # Conv2d wants channels first.
        f = feat.permute(0, 3, 1, 2).contiguous()      # (B, C, H, W)
        S = self.saliency(f)                            # (B, 1, H, W) in [0,1]
        gain = 1.0 + self.lambda_s * S                  # (B, 1, H, W) in [1,3]
        out = f * gain                                  # broadcasts over C
        return out.permute(0, 2, 3, 1).contiguous(), S


class SARLoss(nn.Module):
    """Soft-Dice between predicted saliency and the true tumour mask.

    Dice rather than BCE because enhancing tumour is ~1-3% of pixels: a
    network can score ~97% on BCE by predicting "no tumour anywhere" and
    learning nothing useful. Dice normalises by region size, so it cannot
    be won that way.

    Used at TRAINING time only. Inference never sees a mask.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, S: torch.Tensor, tumour_mask: torch.Tensor) -> torch.Tensor:
        """S: (B, 1, h, w) predicted.  tumour_mask: (B, 1, H, W) binary."""
        m = F.interpolate(tumour_mask.float(), size=S.shape[-2:], mode="nearest")
        s_flat = S.reshape(S.shape[0], -1)
        m_flat = m.reshape(m.shape[0], -1)
        inter = (s_flat * m_flat).sum(dim=1)
        union = s_flat.sum(dim=1) + m_flat.sum(dim=1)
        dice = (2.0 * inter + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


# ---------------------------------------------------------------------------
# 3. The model
# ---------------------------------------------------------------------------

class SwinPPMAEv2(nn.Module):
    """PP-MAE with working cross-modal attention and predicted saliency.

    forward(x, seg_map=None) -> (reconstruction, [S_stage0, S_stage1, ...])

    seg_map is accepted for API compatibility with v1 but IGNORED in the
    forward pass. Ground truth is used only by the trainer's SARLoss.
    """

    # Clinically paired modalities. Index order: T1=0, T1ce=1, T2=2, FLAIR=3
    #   (1, 2)  T1ce <-> T2     enhancement vs. fluid signal
    #   (2, 3)  T2   <-> FLAIR  fluid vs. oedema extent
    MODAL_PAIRS = [(1, 2), (2, 3)]

    def __init__(
        self,
        in_ch: int = 4,
        embed_dim: int = 48,
        depths: Tuple[int, ...] = (2, 2, 2, 2),
        n_heads: Tuple[int, ...] = (3, 3, 6, 6),
        window_size: int = 4,
        lambda_s: float = 2.0,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.lambda_s = lambda_s

        # --- per-modality embedding (replaces the single fused Conv2d) ----
        self.patch_embed = MultiModalPatchEmbed(in_ch, embed_dim, patch=4)

        # --- cross-modal attention, one module per clinical pair ----------
        self.cross_modal = nn.ModuleList([
            CrossModalAttention(embed_dim) for _ in self.MODAL_PAIRS
        ])

        # --- fuse the four streams back into one ---------------------------
        # Concatenate then project. Concat+Linear rather than summing so the
        # network can weight modalities differently instead of being forced
        # to treat them as interchangeable.
        self.fuse = nn.Linear(in_ch * embed_dim, embed_dim)
        self.patch_norm = _SafeLayerNorm(embed_dim)

        # --- encoder (unchanged from v1) -----------------------------------
        dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        self.enc_stages = nn.ModuleList()
        for i, (d, h) in enumerate(zip(depths, n_heads)):
            out_dim = dims[i + 1] if i + 1 < len(dims) else dims[i]
            self.enc_stages.append(SwinStage(dims[i], out_dim, h, window_size, d))

        # --- saliency predictors, one per encoder stage ---------------------
        self.saliency = nn.ModuleList([
            SaliencyPredictor(dims[i], lambda_s) for i in range(len(depths))
        ])

        # --- decoder (unchanged from v1) ------------------------------------
        rev_dims = list(reversed(dims))
        self.dec_stages = nn.ModuleList()
        for i in range(len(depths) - 1):
            self.dec_stages.append(
                SwinDecoderStage(in_ch=rev_dims[i],
                                 skip_ch=rev_dims[i + 1],
                                 out_ch=rev_dims[i + 1])
            )

        # Upsample + Conv2d, not ConvTranspose2d — the latter crashes MPS
        # backward (the bug you traced with anomaly mode).
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="nearest"),
            nn.Conv2d(rev_dims[-1], rev_dims[-1], kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(rev_dims[-1], in_ch, kernel_size=1),
        )

    # ------------------------------------------------------------------
    def _cross_modal_fuse(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        """Apply attention across clinical pairs, then merge the streams.

        THIS is the line that was broken in v1:
            v1:  self.cross_modal[i](tokens, tokens)     <- same tensor twice
            v2:  self.cross_modal[i](t[a], t[b])         <- two modalities
        """
        t = list(tokens)                                  # shallow copy
        for i, (a, b) in enumerate(self.MODAL_PAIRS):
            t[a], t[b] = self.cross_modal[i](t[a], t[b])
        fused = torch.cat(t, dim=-1)                      # (B, N, 4*D)
        return self.fuse(fused)                           # (B, N, D)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                       # (B, 4, H, W)
        seg_map: Optional[torch.Tensor] = None,  # accepted, deliberately unused
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B = x.shape[0]

        # 1. embed each modality separately        4 x (B, N, D)
        tokens, (Ph, Pw) = self.patch_embed(x)

        # 2. cross-modal attention, then fuse      (B, N, D)
        flat = self._cross_modal_fuse(tokens)
        feat = self.patch_norm(flat).reshape(B, Ph, Pw, -1)

        # 3. encoder, amplifying predicted tumour regions at every stage
        skips: List[torch.Tensor] = []
        saliency_maps: List[torch.Tensor] = []
        for stage, sal in zip(self.enc_stages, self.saliency):
            feat, S = sal(feat)                 # predicts S, applies Eq 4.5
            saliency_maps.append(S)
            feat, skip = stage(feat)
            skips.append(skip)

        # 4. decoder
        dec = feat
        for dec_stage, skip in zip(self.dec_stages, reversed(skips[:-1])):
            dec = dec_stage(dec.permute(0, 3, 1, 2).contiguous(), skip)

        out = dec.permute(0, 3, 1, 2).contiguous()
        out = self.final_up(out)
        return torch.sigmoid(out), saliency_maps


# ---------------------------------------------------------------------------
# 4. Trainer  —  adds the auxiliary saliency supervision
# ---------------------------------------------------------------------------

class SwinPPMAEv2Trainer:
    """L = L_recon + lambda_sar * mean_l SoftDice(S_l, tumour_mask)

    lambda_sar = 0.1 by default: enough to steer the saliency branch,
    small enough that reconstruction stays the primary objective.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
        lambda1: float = 1.0,
        lambda2: float = 0.5,
        lambda_sar: float = 0.1,
        lr: float = 1e-4,
        et_label: int = 3,
    ):
        self.model = model.to(device)
        self.optim = optimizer or torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.05)
        self.device = device
        self.loss_fn = PPMAELoss(lambda1=lambda1, lambda2=lambda2)
        self.sar_loss = SARLoss().to(device)
        self.lambda_sar = lambda_sar
        self.et_label = et_label

    def _tumour_mask(self, seg: torch.Tensor) -> torch.Tensor:
        """Whole tumour (any non-zero label) as the saliency target.

        WT rather than ET alone: ET is ~1-3% of pixels, too sparse to
        supervise a coarse feature map at stage 3 (which is 3x3 after
        three downsamplings). Try ET here as an ablation and compare TBR.
        """
        return (seg > 0).float()

    def step(self, batch: dict) -> dict:
        self.model.train()
        noisy = batch["noisy"].to(self.device)
        target = batch["target"].to(self.device)
        seg = batch["seg"].to(self.device)

        self.optim.zero_grad()
        pred, saliency_maps = self.model(noisy)      # no mask at forward time
        losses = self.loss_fn(pred, target, seg)

        mask = self._tumour_mask(seg)
        l_sar = sum(self.sar_loss(S, mask) for S in saliency_maps) / max(
            len(saliency_maps), 1)

        total = losses["total"] + self.lambda_sar * l_sar
        total.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()

        out = {k: v.item() for k, v in losses.items()}
        out["sar"] = float(l_sar.item())
        out["total"] = float(total.item())
        return out


# ---------------------------------------------------------------------------
# 5. Diagnostics  —  proof that each module does what it claims
# ---------------------------------------------------------------------------

@torch.no_grad()
def saliency_metrics(S: torch.Tensor, tumour_mask: torch.Tensor,
                     threshold: float = 0.5) -> dict:
    """TBR, localisation Dice, saturation fraction.

    TBR (tumour-to-background ratio) is the one that matters:
        > 2.0   saliency concentrates on the lesion
        ~ 1.0   tumour-blind: it learned some other gating
    Saturation > 0.5 means S ~ 1 everywhere, i.e. SAR collapsed into a
    constant 3x multiplier — a learning-rate change in disguise.
    """
    m = F.interpolate(tumour_mask.float(), size=S.shape[-2:], mode="nearest")
    inside, outside = S[m > 0.5], S[m <= 0.5]
    tbr = float(inside.mean() / (outside.mean() + 1e-6)) if inside.numel() else float("nan")

    pred = (S > threshold).float()
    inter = (pred * m).sum()
    union = pred.sum() + m.sum()
    loc_dice = float(2.0 * inter / (union + 1e-6)) if union > 0 else float("nan")

    return {"tbr": tbr, "loc_dice": loc_dice,
            "frac_saturated": float((S > 0.9).float().mean())}


def _self_test() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print(f"device: {device}\n")

    # --- SAR: shapes and Eq 4.5 gain range ----------------------------------
    sar = SaliencyPredictor(channels=48, lambda_s=2.0).to(device)
    feat = torch.randn(2, 24, 24, 48, device=device)
    out, S = sar(feat)
    gain = 1.0 + 2.0 * S
    print("SAR")
    print(f"  feat in  {tuple(feat.shape)}  ->  out {tuple(out.shape)}")
    print(f"  S {tuple(S.shape)}  range [{S.min():.3f}, {S.max():.3f}]")
    print(f"  gain range [{gain.min():.3f}, {gain.max():.3f}]   (Eq 4.5 wants [1, 3])")
    assert out.shape == feat.shape
    assert 0.0 <= S.min() and S.max() <= 1.0
    assert 1.0 <= gain.min() and gain.max() <= 3.0
    print("  PASS\n")

    # --- CMA: does the output actually depend on modality B? ----------------
    cma = CrossModalAttention(48).to(device)
    ta = torch.randn(2, 576, 48, device=device)
    tb = torch.randn(2, 576, 48, device=device)
    a1, _ = cma(ta, tb)
    a2, _ = cma(ta, torch.randn_like(tb))
    print("CMA")
    print(f"  max |out(a,b1) - out(a,b2)| = {(a1 - a2).abs().max():.6f}")
    assert not torch.allclose(a1, a2), "CMA is ignoring modality B — still self-attention"
    print("  PASS — output depends on the second modality\n")

    # --- full model, mask-free forward --------------------------------------
    model = SwinPPMAEv2(in_ch=4, embed_dim=48, depths=(2, 2, 2, 2),
                        n_heads=(3, 3, 6, 6), window_size=4).to(device)
    x = torch.rand(2, 4, 96, 96, device=device)
    recon, sal_maps = model(x)                      # NO seg_map passed
    print("SwinPPMAEv2")
    print(f"  input  {tuple(x.shape)}  ->  recon {tuple(recon.shape)}")
    print(f"  saliency maps: {[tuple(s.shape) for s in sal_maps]}")
    assert recon.shape == x.shape
    print("  PASS — forward runs without a ground-truth mask\n")

    # --- one training step ---------------------------------------------------
    seg = torch.randint(0, 4, (2, 1, 96, 96), device=device)
    trainer = SwinPPMAEv2Trainer(model, device=device)
    metrics = trainer.step({"noisy": x, "target": torch.rand_like(x), "seg": seg})
    print("training step")
    for k, v in metrics.items():
        print(f"  {k:10s} {v:.4f}")

    print("\ndiagnostics on untrained saliency (expect TBR ~ 1.0 before training)")
    for k, v in saliency_metrics(sal_maps[0], (seg > 0).float()).items():
        print(f"  {k:16s} {v:.4f}")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    _self_test()
