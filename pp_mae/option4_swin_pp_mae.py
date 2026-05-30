from typing import Optional
"""
Option 4 — Swin Transformer PP-MAE  (hierarchical shifted-window attention)
============================================================================

WHY THIS OPTION:
    Tumour subregions span vastly different spatial scales — from the
    millimetre-scale enhancing core to the centimetre-scale peritumoral
    oedema.  Swin Transformers compute attention within local windows at
    multiple resolution stages, providing efficient multi-scale feature
    extraction without the quadratic cost of global ViT attention.  The
    hierarchical design also makes features directly usable by U-Net-style
    decoders, aligning naturally with the BraTS segmentation literature
    (SwinUNETR, nnU-Net).

    Use this option when:
      • GPU VRAM is limited but you need more receptive field than Option 1
      • You want to compare against SwinUNETR as the segmentation backbone
      • You need multi-scale cross-modal fusion for T1/T2/FLAIR consistency

ARCHITECTURE:
    Multi-channel input (B, 4, H, W)
    ↓ Channel-wise cross-modal attention  (fuses T1W↔T1Wce, T2W↔FLAIR pairs)
    ↓ Shared-weight Swin Transformer encoder  (4 stages, downscaling)
    ↓ Saliency-aware feature reweighting  (tumour-region amplification)
    ↓ Lightweight symmetric decoder  (upsampling + skip connections)
    Output: (B, 4, H, W) denoised image

PHD TIP:
    The Swin backbone can be initialised from a pretrained SwinUNETR
    checkpoint trained on BraTS.  This gives you a strong prior on
    tumour anatomy before you even begin denoising training — and
    significantly reduces the labelled data requirement.
    Pretrained weights: https://github.com/Project-MONAI/research-contributions
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import PPMAELoss


# ---------------------------------------------------------------------------
# MPS-safe LayerNorm — forces contiguous memory before every forward call.
# PyTorch's built-in LayerNorm backward uses .view() internally, which crashes
# on Apple MPS when the input tensor is non-contiguous (e.g. after permute/roll).
# ---------------------------------------------------------------------------

class _SafeLayerNorm(nn.LayerNorm):
    """
    LayerNorm computed manually with elementwise ops.

    PyTorch's built-in nn.LayerNorm calls the fused C++ kernel
    `native_layer_norm`, whose *backward* kernel internally runs `.view()`
    on the incoming gradient.  On Apple MPS that gradient is frequently
    non-contiguous (it flows back from permute / roll / window ops), so the
    backward crashes with:
        "view size is not compatible … Use .reshape() instead."

    Simply calling `x.contiguous()` in forward does NOT help — it only
    affects the forward input, not the gradient that arrives during
    backprop.  The reliable fix is to avoid the fused kernel altogether and
    compose LayerNorm from mean / var / elementwise ops, all of which have
    MPS-safe backward kernels.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-len(self.normalized_shape), 0))
        x = x.contiguous()
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias
        return x_norm


# ---------------------------------------------------------------------------
# MPS-compatible multi-head attention
# ---------------------------------------------------------------------------

class _MPSMHA(nn.Module):
    """
    Drop-in replacement for nn.MultiheadAttention(batch_first=True).

    PyTorch's built-in MultiheadAttention uses internal .view() calls in its
    C++ backward kernel that fail on Apple Silicon MPS with:
      "RuntimeError: view size is not compatible … Use .reshape() instead."

    This implementation uses F.scaled_dot_product_attention and only
    .reshape() / .permute(), which are fully MPS-compatible.
    """

    def __init__(self, embed_dim: int, num_heads: int, batch_first: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads

        self.q_proj   = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj   = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj   = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        query:            torch.Tensor,
        key:              torch.Tensor,
        value:            torch.Tensor,
        attn_mask:        Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights:     bool = True,
    ):
        B, S, E = query.shape
        T = key.shape[1]
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(query).reshape(B, S, H, D).permute(0, 2, 1, 3)  # (B,H,S,D)
        k = self.k_proj(key  ).reshape(B, T, H, D).permute(0, 2, 1, 3)  # (B,H,T,D)
        v = self.v_proj(value).reshape(B, T, H, D).permute(0, 2, 1, 3)  # (B,H,T,D)

        # Manual scaled dot-product attention — avoids MPS backward issues
        # with F.scaled_dot_product_attention on Python 3.9 / older PyTorch builds
        scale = D ** -0.5
        attn  = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,H,S,T)
        if attn_mask is not None:
            attn = attn + attn_mask
        attn  = F.softmax(attn, dim=-1)
        out   = torch.matmul(attn, v)                          # (B,H,S,D)

        out = out.permute(0, 2, 1, 3).reshape(B, S, E)
        out = self.out_proj(out)
        return out, None   # mirrors nn.MultiheadAttention return signature


# ---------------------------------------------------------------------------
# Cross-modal attention  (fuses complementary modality pairs)
# ---------------------------------------------------------------------------

class CrossModalAttention(nn.Module):
    """
    Bilateral cross-attention between two modality channels.

    For each complementary pair (e.g., T1Wce ↔ T2W), the query from one
    modality attends to keys/values from the other.  The output enriches
    each modality with information from its complement, enforcing the
    cross-modal consistency principle of the PP-MAE.

    Args:
        ch:      number of feature channels per modality
        n_heads: attention heads
    """

    def __init__(self, ch: int, n_heads: int = 4):
        super().__init__()
        self.to_qkv_a = nn.Linear(ch, ch * 3, bias=False)
        self.to_qkv_b = nn.Linear(ch, ch * 3, bias=False)
        self.attn_a   = _MPSMHA(ch, n_heads)
        self.attn_b   = _MPSMHA(ch, n_heads)
        self.norm_a   = _SafeLayerNorm(ch)
        self.norm_b   = _SafeLayerNorm(ch)

    def forward(
        self,
        feat_a: torch.Tensor,   # (B, N, ch)
        feat_b: torch.Tensor,   # (B, N, ch)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # a attends to b
        out_a, _ = self.attn_a(feat_a, feat_b, feat_b)
        feat_a   = self.norm_a(feat_a + out_a)

        # b attends to a
        out_b, _ = self.attn_b(feat_b, feat_a, feat_a)
        feat_b   = self.norm_b(feat_b + out_b)

        return feat_a, feat_b


# ---------------------------------------------------------------------------
# Swin window partition utilities
# ---------------------------------------------------------------------------

def window_partition(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, tuple]:
    """
    (B, H, W, C) → (B*n_windows, ws, ws, C).
    Pads H and W to the nearest multiple of window_size when necessary.
    Returns (windows, (H_pad, W_pad)) for use in window_reverse.
    """
    B, H, W, C = x.shape
    H_pad = (window_size - H % window_size) % window_size
    W_pad = (window_size - W % window_size) % window_size
    if H_pad or W_pad:
        x = F.pad(x, (0, 0, 0, W_pad, 0, H_pad))
    Hp, Wp = H + H_pad, W + W_pad
    x = x.reshape(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.reshape(-1, window_size, window_size, C), (H, W)


def window_reverse(windows: torch.Tensor, window_size: int, orig_hw: tuple) -> torch.Tensor:
    """(B*n_windows, ws, ws, C) → (B, H_orig, W_orig, C)"""
    H_orig, W_orig = orig_hw
    Hp = math.ceil(H_orig / window_size) * window_size
    Wp = math.ceil(W_orig / window_size) * window_size
    n_windows_h, n_windows_w = Hp // window_size, Wp // window_size
    B = windows.shape[0] // (n_windows_h * n_windows_w)
    x = windows.reshape(B, n_windows_h, n_windows_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.reshape(B, Hp, Wp, -1)
    return x[:, :H_orig, :W_orig, :].contiguous()


# ---------------------------------------------------------------------------
# Swin Transformer Block (simplified — single-stage, no relative position bias)
# ---------------------------------------------------------------------------

class SwinBlock(nn.Module):
    """
    Swin Transformer block with window attention and optional cyclic shift.

    Args:
        dim:         channel dimension
        n_heads:     number of attention heads
        window_size: local attention window size
        shift_size:  shift for SW-MSA (0 = W-MSA, window_size//2 = SW-MSA)
        mlp_ratio:   FFN hidden expansion ratio
    """

    def __init__(
        self,
        dim:         int,
        n_heads:     int,
        window_size: int = 7,
        shift_size:  int = 0,
        mlp_ratio:   float = 4.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size  = shift_size

        self.norm1 = _SafeLayerNorm(dim)
        self.attn  = _MPSMHA(dim, n_heads)
        self.norm2 = _SafeLayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        ws = min(self.window_size, H, W)

        residual = x
        x = self.norm1(x)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)).contiguous()

        windows, orig_hw = window_partition(x, ws)   # (B*nW, ws, ws, C)
        nW = windows.shape[0]
        tokens = windows.reshape(nW, ws * ws, C)

        tokens, _ = self.attn(tokens, tokens, tokens)
        windows = tokens.reshape(nW, ws, ws, C).contiguous()
        x = window_reverse(windows, ws, orig_hw)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)).contiguous()

        x = residual + x
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Swin Encoder Stage
# ---------------------------------------------------------------------------

class SwinStage(nn.Module):
    """
    One encoder stage: 2 consecutive Swin blocks (W-MSA + SW-MSA pair)
    followed by patch merging (downsampling).
    """

    def __init__(
        self,
        dim:         int,
        out_dim:     int,
        n_heads:     int,
        window_size: int = 7,
        n_blocks:    int = 2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(
                dim, n_heads, window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2
            )
            for i in range(n_blocks)
        ])
        # Patch merging: concatenate 2×2 neighbours → linear projection
        self.downsample = nn.Sequential(
            _SafeLayerNorm(dim * 4),
            nn.Linear(dim * 4, out_dim, bias=False),
        ) if out_dim != dim else nn.Identity()
        self.do_downsample = out_dim != dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for blk in self.blocks:
            x = blk(x)
        skip = x

        if self.do_downsample:
            B, H, W, C = x.shape
            # Pad to even H, W
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
            _, Hp, Wp, _ = x.shape
            x0 = x[:, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, :]
            x3 = x[:, 1::2, 1::2, :]
            x  = torch.cat([x0, x1, x2, x3], dim=-1)   # (B, H/2, W/2, 4C)
            x  = self.downsample(x)

        return x, skip   # (downsampled, skip at original resolution)


# ---------------------------------------------------------------------------
# Tumour-saliency feature reweighting
# ---------------------------------------------------------------------------

class SaliencyFeatureReweighter(nn.Module):
    """
    Channel-wise attention conditioned on segmentation map features.
    Amplifies feature channels that are most relevant to tumour subregions
    without discarding background context.
    """

    def __init__(self, channels: int, n_classes: int = 4):
        super().__init__()
        self.seg_embed = nn.Embedding(n_classes, channels)
        self.gate = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        """
        feat: (B, H, W, C)
        seg:  (B, 1, H, W) integer labels  → downsampled to feat resolution
        """
        B, H, W, C = feat.shape
        # Downsample seg to feature resolution
        seg_ds = F.interpolate(seg.float(), size=(H, W), mode="nearest").long()
        seg_ds = seg_ds.squeeze(1)                   # (B, H, W)
        seg_emb = self.seg_embed(seg_ds)             # (B, H, W, C)
        gate = self.gate(seg_emb)                    # (B, H, W, C)
        return feat * gate


# ---------------------------------------------------------------------------
# Swin PP-MAE decoder stage
# ---------------------------------------------------------------------------

class SwinDecoderStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up    = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.merge = nn.Linear(in_ch // 2 + skip_ch, out_ch)
        self.norm  = _SafeLayerNorm(out_ch)
        self.swin  = SwinBlock(out_ch, max(1, out_ch // 32))

    def forward(
        self,
        x:    torch.Tensor,   # (B, C, H, W)
        skip: torch.Tensor,   # (B, H*2, W*2, skip_ch)
    ) -> torch.Tensor:
        x    = self.up(x)                                  # (B, C//2, H*2, W*2)
        x    = x.permute(0, 2, 3, 1).contiguous()         # → (B, H*2, W*2, C//2)
        x    = torch.cat([x, skip], dim=-1).contiguous()  # (B, H*2, W*2, C//2+skip_ch)
        x    = self.norm(self.merge(x))
        x    = self.swin(x)
        return x                              # (B, H*2, W*2, out_ch)


# ---------------------------------------------------------------------------
# Full Swin PP-MAE
# ---------------------------------------------------------------------------

class SwinPPMAE(nn.Module):
    """
    Option 4: Swin Transformer PP-MAE with cross-modal attention and
    saliency-conditioned feature reweighting.

    Args:
        in_ch:       number of MRI modalities (default 4)
        embed_dim:   initial embedding dimension
        depths:      number of Swin blocks per stage
        n_heads:     attention heads per stage
        window_size: local attention window size
    """

    # Complementary modality pairs (index into in_ch)
    # T1W=0, T1Wce=1, T2W=2, FLAIR=3
    MODAL_PAIRS = [(1, 2), (2, 3)]

    def __init__(
        self,
        in_ch:       int = 4,
        embed_dim:   int = 96,
        depths:      tuple[int, ...] = (2, 2, 6, 2),
        n_heads:     tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 7,
    ):
        super().__init__()
        self.in_ch = in_ch

        # Patch embedding  (4×4 conv, following Swin-T)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_ch, embed_dim, 4, stride=4, bias=False),
            _SafeLayerNorm([embed_dim, 1, 1]),    # dummy to store shape; applied below
        )
        # Rewrite as proper LN over channels
        self.patch_embed = nn.Conv2d(in_ch, embed_dim, 4, stride=4, bias=False)
        self.patch_norm  = _SafeLayerNorm(embed_dim)

        # Cross-modal attention (applied at patch-token level before encoder)
        self.cross_modal = nn.ModuleList([
            CrossModalAttention(embed_dim) for _ in self.MODAL_PAIRS
        ])

        # Encoder stages
        dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        self.enc_stages = nn.ModuleList()
        for i, (d, h) in enumerate(zip(depths, n_heads)):
            out_dim = dims[i + 1] if i + 1 < len(dims) else dims[i]
            self.enc_stages.append(
                SwinStage(dims[i], out_dim, h, window_size, d)
            )

        # Saliency reweighters (one per encoder stage)
        self.reweighters = nn.ModuleList([
            SaliencyFeatureReweighter(dims[i]) for i in range(len(depths))
        ])

        # Decoder stages
        rev_dims = list(reversed(dims))
        self.dec_stages = nn.ModuleList()
        for i in range(len(depths) - 1):
            self.dec_stages.append(
                SwinDecoderStage(
                    in_ch    = rev_dims[i],
                    skip_ch  = rev_dims[i + 1],
                    out_ch   = rev_dims[i + 1],
                )
            )

        # Final upsampling × 4 to match input resolution
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(rev_dims[-1], rev_dims[-1], 4, stride=4),
            nn.GELU(),
            nn.Conv2d(rev_dims[-1], in_ch, 1),
        )

        self.n_stages = len(depths)

    # ------------------------------------------------------------------
    def _apply_cross_modal(
        self, tokens: torch.Tensor, B: int, H: int, W: int
    ) -> torch.Tensor:
        """tokens: (B, H*W, C); apply cross-modal attention to paired channels."""
        # For simplicity, run cross-modal on the full token sequence
        # (in practice, apply per-modality sub-embedding)
        for i, (a_idx, b_idx) in enumerate(self.MODAL_PAIRS):
            # Treat token sequence as a monolithic unit; share weights across pairs
            tokens, _ = self.cross_modal[i](tokens, tokens)
        return tokens

    # ------------------------------------------------------------------
    def forward(
        self,
        x:       torch.Tensor,   # (B, 4, H, W)
        seg_map: torch.Tensor,   # (B, 1, H, W)
    ) -> torch.Tensor:
        B, C, H, W = x.shape

        # Patch embedding
        tokens = self.patch_embed(x)        # (B, embed_dim, H//4, W//4)
        _, E, Ph, Pw = tokens.shape
        tokens = tokens.permute(0, 2, 3, 1).contiguous()   # (B, Ph, Pw, E)
        tokens = self.patch_norm(tokens)

        # Cross-modal attention on patch tokens
        flat = tokens.reshape(B, Ph * Pw, E)
        flat = self._apply_cross_modal(flat, B, Ph, Pw)
        tokens = flat.reshape(B, Ph, Pw, E)

        # Encoder with saliency reweighting
        skips = []
        feat  = tokens
        for stage, reweighter in zip(self.enc_stages, self.reweighters):
            feat = reweighter(feat, F.interpolate(
                seg_map.float(), size=(feat.shape[1], feat.shape[2]), mode="nearest"
            ).long())
            feat, skip = stage(feat)
            skips.append(skip)

        # Decoder
        dec = feat
        for dec_stage, skip in zip(self.dec_stages, reversed(skips[:-1])):
            dec_spatial = dec.permute(0, 3, 1, 2).contiguous()   # → (B, C, H, W)
            dec = dec_stage(dec_spatial, skip)

        # Final spatial upsampling
        out = dec.permute(0, 3, 1, 2).contiguous()  # (B, C, H//4, W//4)
        out = self.final_up(out)                     # (B, 4, H, W)
        return torch.sigmoid(out)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SwinPPMAETrainer:
    def __init__(
        self,
        model:     nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device:    str   = "cuda",
        lambda1:   float = 1.0,
        lambda2:   float = 0.5,
        lr:        float = 1e-4,
    ):
        self.model   = model.to(device)
        self.optim   = optimizer if optimizer is not None else \
                       torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        self.device  = device
        self.loss_fn = PPMAELoss(lambda1=lambda1, lambda2=lambda2)

    def step(self, batch: dict) -> dict:
        self.model.train()
        noisy  = batch["noisy"].to(self.device)
        target = batch["target"].to(self.device)
        seg    = batch["seg"].to(self.device)

        self.optim.zero_grad()
        pred   = self.model(noisy, seg)
        losses = self.loss_fn(pred, target, seg)
        losses["total"].backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()

        return {k: v.item() for k, v in losses.items()}


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SwinPPMAE(
        in_ch=4,
        embed_dim=48,          # half of Swin-T for rapid testing
        depths=(2, 2, 2, 2),
        n_heads=(3, 3, 6, 6),
        window_size=4,
    ).to(device)

    B, C, H, W = 2, 4, 128, 128
    noisy  = torch.rand(B, C, H, W, device=device)
    target = torch.rand(B, C, H, W, device=device)
    seg    = torch.randint(0, 4, (B, 1, H, W), device=device)

    optim   = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    trainer = SwinPPMAETrainer(model, optim, device=device)
    metrics = trainer.step({"noisy": noisy, "target": target, "seg": seg})

    print("Option 4 — Swin PP-MAE")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total:,}")
