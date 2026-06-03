"""
Option 2 — Vision Transformer PP-MAE  (3D volumetric, ViT backbone)
====================================================================

WHY THIS OPTION:
    The reference MAE paper (He et al., 2021) proves that masking 75% of
    patches during pre-training forces the encoder to learn meaningful global
    representations rather than exploiting local texture shortcuts.  For MRI
    this is ideal: the model must reconstruct masked brain tissue using
    long-range context, which is exactly what is needed for tumour-aware
    denoising.

    This is the RECOMMENDED production-grade option for the PP-MAE as
    described in the study protocol.  It directly implements the saliency-
    guided masking strategy with a ViT encoder and lightweight decoder.

ARCHITECTURE:
    1. 3D Patch Embedding  → flatten (4, D, H, W) into sequence of tokens
    2. Saliency-guided masking  → retain tumour tokens, randomly mask rest
    3. ViT Transformer Encoder  → self-attention over visible tokens
    4. MAE Decoder (shallow ViT) → reconstruct all tokens from visible subset
    5. Unpatchify → (B, 4, D, H, W) denoised volume

MASKING STRATEGY:
    mask_ratio is applied to non-tumour tokens.
    Tumour tokens (saliency > threshold) are always kept visible.
    This prioritises pathological regions during representation learning
    consistent with the PP-MAE design principle (Section 5.1).

PHD TIP:
    Pre-train the encoder on unlabelled MRI data first (self-supervised),
    then fine-tune with the pathology-aware loss on labelled glioma data.
    The two-stage approach substantially reduces the labelled data requirement
    — crucial given typical glioma cohort sizes (n = 100–300).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import PPMAELoss


# ---------------------------------------------------------------------------
# MPS-safe LayerNorm — forces .contiguous() before forward to prevent the
# Apple MPS backward ".view() size not compatible" crash on non-contiguous
# tensors produced by permute/roll in transformer blocks.
# ---------------------------------------------------------------------------

class _SafeLayerNorm(nn.LayerNorm):
    """LayerNorm via elementwise ops — avoids the fused C++ backward's .view() crash on MPS."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-len(self.normalized_shape), 0))
        x = x.contiguous()
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias
        return x_norm


class _ContiguousFunc(torch.autograd.Function):
    """Contiguity barrier — makes tensor contiguous in BOTH forward and backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.contiguous()

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        return grad.contiguous()


def _c(x: torch.Tensor) -> torch.Tensor:
    return _ContiguousFunc.apply(x)




# ---------------------------------------------------------------------------
# 3D Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbed3D(nn.Module):
    """
    Splits a 5-D volume into non-overlapping 3D patches and projects
    each patch to an embedding vector.

    Args:
        vol_size:   (D, H, W) of input volume
        patch_size: cubic patch size (same along all axes)
        in_chans:   number of MRI modalities
        embed_dim:  dimensionality of token embedding
    """

    def __init__(
        self,
        vol_size:   tuple[int, int, int] = (96, 96, 96),
        patch_size: int = 16,
        in_chans:   int = 4,
        embed_dim:  int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size  = tuple(s // patch_size for s in vol_size)
        self.n_patches  = math.prod(self.grid_size)

        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, D, H, W) → (B, n_patches, embed_dim)"""
        x = self.proj(x)                 # (B, embed_dim, Gd, Gh, Gw)
        B, E, Gd, Gh, Gw = x.shape
        return x.flatten(2).transpose(1, 2)   # (B, n_patches, E)


# ---------------------------------------------------------------------------
# Saliency-guided masking
# ---------------------------------------------------------------------------

def build_saliency_mask(
    seg_map:    torch.Tensor,   # (B, 1, D, H, W) integer labels
    patch_size: int,
    mask_ratio: float = 0.75,
    tumour_retain: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        ids_keep   : (B, n_vis) indices of visible tokens
        ids_masked : (B, n_mask) indices of masked tokens
        ids_restore: (B, n_patches) argsort to restore original order
    """
    B = seg_map.shape[0]
    # Pool segmentation to patch grid (any tumour voxel in patch → tumour patch)
    seg_float  = (seg_map > 0).float()
    patch_pool = F.avg_pool3d(seg_float, patch_size, stride=patch_size)  # (B,1,Gd,Gh,Gw)
    is_tumour  = (patch_pool > 0).flatten(1)   # (B, n_patches)

    n_patches = is_tumour.shape[1]
    n_keep    = int(n_patches * (1 - mask_ratio))

    ids_keep_list, ids_masked_list, ids_restore_list = [], [], []

    for b in range(B):
        tumour_idx = is_tumour[b].nonzero(as_tuple=False).squeeze(1)
        bg_idx     = (~is_tumour[b].bool()).nonzero(as_tuple=False).squeeze(1)

        if tumour_retain:
            # Always keep all tumour tokens
            n_bg_keep = max(n_keep - len(tumour_idx), 0)
            perm      = torch.randperm(len(bg_idx), device=seg_map.device)
            bg_keep   = bg_idx[perm[:n_bg_keep]]
            bg_mask   = bg_idx[perm[n_bg_keep:]]
            keep   = torch.cat([tumour_idx, bg_keep])
            masked = bg_mask
        else:
            perm   = torch.randperm(n_patches, device=seg_map.device)
            keep   = perm[:n_keep]
            masked = perm[n_keep:]

        # Sort so attention can be applied without position confusion
        keep,   _ = torch.sort(keep)
        masked, _ = torch.sort(masked)

        restore = torch.argsort(torch.cat([keep, masked]))

        ids_keep_list.append(keep)
        ids_masked_list.append(masked)
        ids_restore_list.append(restore)

    ids_keep    = torch.stack(ids_keep_list)
    ids_masked  = torch.stack(ids_masked_list)
    ids_restore = torch.stack(ids_restore_list)
    return ids_keep, ids_masked, ids_restore


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = _SafeLayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x), self.norm(x), self.norm(x))[0]


class FFN(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.net  = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm = _SafeLayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads)
        self.ffn  = FFN(embed_dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.attn(x))


# ---------------------------------------------------------------------------
# ViT PP-MAE
# ---------------------------------------------------------------------------

class ViTPPMAE(nn.Module):
    """
    Option 2: Vision Transformer PP-MAE for volumetric glioma MRI.

    Args:
        vol_size:      (D, H, W) — expected patch dimensions
        patch_size:    cubic patch size (default 16)
        in_chans:      number of MRI modalities (default 4)
        embed_dim:     encoder embedding dim
        depth:         number of encoder Transformer blocks
        n_heads:       attention heads (embed_dim must be divisible by n_heads)
        decoder_dim:   decoder embedding dim (smaller than encoder, following MAE)
        decoder_depth: decoder Transformer blocks
        mask_ratio:    fraction of non-tumour tokens to mask
        mlp_ratio:     MLP hidden dim expansion ratio
    """

    def __init__(
        self,
        vol_size:      tuple[int, int, int] = (96, 96, 96),
        patch_size:    int   = 16,
        in_chans:      int   = 4,
        embed_dim:     int   = 384,
        depth:         int   = 12,
        n_heads:       int   = 6,
        decoder_dim:   int   = 192,
        decoder_depth: int   = 4,
        mask_ratio:    float = 0.75,
        mlp_ratio:     float = 4.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # Patch embedding + positional encoding
        self.patch_embed = PatchEmbed3D(vol_size, patch_size, in_chans, embed_dim)
        n_patches = self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Mask token (learnable placeholder for masked positions in decoder)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Encoder
        self.encoder_norm   = _SafeLayerNorm(embed_dim)
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio) for _ in range(depth)
        ])

        # Encoder → decoder projection
        self.enc_to_dec = nn.Linear(embed_dim, decoder_dim, bias=True)

        # Decoder positional encoding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, n_patches, decoder_dim))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        # Decoder
        self.decoder_norm   = _SafeLayerNorm(decoder_dim)
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, max(1, decoder_dim // 64), mlp_ratio)
            for _ in range(decoder_depth)
        ])

        # Prediction head: token → patch pixels
        patch_dim = in_chans * (patch_size ** 3)
        self.pred_head = nn.Linear(decoder_dim, patch_dim)

        self.in_chans    = in_chans
        self.n_patches   = n_patches
        self.vol_size    = vol_size

    # ------------------------------------------------------------------
    def encode(
        self,
        x:       torch.Tensor,    # (B, 4, D, H, W)
        seg_map: torch.Tensor,    # (B, 1, D, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.patch_embed(x) + self.pos_embed   # (B, N, E)

        ids_keep, ids_masked, ids_restore = build_saliency_mask(
            seg_map, self.patch_size, self.mask_ratio
        )

        # Select only visible tokens for encoder
        B = tokens.shape[0]
        tokens_vis = tokens[
            torch.arange(B, device=tokens.device).unsqueeze(1),
            ids_keep
        ]

        for blk in self.encoder_blocks:
            tokens_vis = blk(tokens_vis)
        tokens_vis = self.encoder_norm(tokens_vis)

        return tokens_vis, ids_keep, ids_restore

    # ------------------------------------------------------------------
    def decode(
        self,
        tokens_vis:  torch.Tensor,   # (B, n_vis, E)
        ids_restore: torch.Tensor,   # (B, N)
    ) -> torch.Tensor:
        B, n_vis, _ = tokens_vis.shape
        tokens_vis  = self.enc_to_dec(tokens_vis)   # project to decoder_dim

        n_mask = self.n_patches - n_vis
        mask_tokens = self.mask_token.expand(B, n_mask, -1)

        # Restore original sequence order
        full_seq = torch.cat([tokens_vis, mask_tokens], dim=1)
        full_seq = full_seq[
            torch.arange(B, device=full_seq.device).unsqueeze(1),
            torch.argsort(ids_restore, dim=1)
        ]
        full_seq = full_seq + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            full_seq = blk(full_seq)
        full_seq = self.decoder_norm(full_seq)

        return self.pred_head(full_seq)   # (B, N, patch_dim)

    # ------------------------------------------------------------------
    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, N, patch_dim) → (B, C, D, H, W)"""
        P  = self.patch_size
        C  = self.in_chans
        D, H, W = self.vol_size
        Gd, Gh, Gw = D // P, H // P, W // P

        tokens = _c(tokens.reshape(-1, Gd, Gh, Gw, C, P, P, P)).permute(0, 4, 1, 5, 2, 6, 3, 7)
        return _c(tokens).reshape(-1, C, D, H, W)

    # ------------------------------------------------------------------
    def forward(
        self,
        x:       torch.Tensor,   # (B, 4, D, H, W)
        seg_map: torch.Tensor,   # (B, 1, D, H, W)
    ) -> torch.Tensor:
        tokens_vis, _, ids_restore = self.encode(x, seg_map)
        pred_tokens = self.decode(tokens_vis, ids_restore)
        return torch.sigmoid(self.unpatchify(pred_tokens))


# ---------------------------------------------------------------------------
# Trainer with learning-rate warmup (critical for ViT stability)
# ---------------------------------------------------------------------------

class ViTPPMAETrainer:
    def __init__(
        self,
        model:        nn.Module,
        optimizer:    Optional[torch.optim.Optimizer] = None,
        device:       str   = "cuda",
        lambda1:      float = 1.0,
        lambda2:      float = 0.5,
        warmup_steps: int   = 1000,
        lr:           float = 1e-4,
    ):
        self.model     = model.to(device)
        self.optim     = optimizer if optimizer is not None else \
                         torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        self.device    = device
        self.loss_fn   = PPMAELoss(lambda1=lambda1, lambda2=lambda2)
        self.warmup    = warmup_steps
        self._step     = 0

    def _lr_scale(self) -> float:
        """Linear warmup followed by cosine decay placeholder."""
        if self._step < self.warmup:
            return self._step / max(1, self.warmup)
        return 1.0

    @staticmethod
    def _to_2d(t: torch.Tensor) -> torch.Tensor:
        """Merge batch and depth dims so (B, C, D, H, W) → (B*D, C, H, W)."""
        if t.dim() == 5:
            B, C, D, H, W = t.shape
            return _c(t.permute(0, 2, 1, 3, 4)).reshape(B * D, C, H, W)
        return t

    def step(self, batch: dict) -> dict:
        self.model.train()
        self._step += 1

        for g in self.optim.param_groups:
            g["lr"] = g.get("base_lr", g["lr"]) * self._lr_scale()

        x      = batch["noisy"].to(self.device)
        target = batch["target"].to(self.device)
        seg    = batch["seg"].to(self.device)

        # Ensure 5D for volumetric model
        if x.dim() == 4:
            x, target, seg = x.unsqueeze(2), target.unsqueeze(2), seg.unsqueeze(2)

        self.optim.zero_grad()
        pred = self.model(x, seg)

        # Flatten D into batch for 2D-compatible loss functions
        pred_2d   = self._to_2d(pred)
        target_2d = self._to_2d(target)
        seg_2d    = self._to_2d(seg)

        losses = self.loss_fn(pred_2d, target_2d, seg_2d)
        losses["total"].backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optim.step()

        return {k: v.item() for k, v in losses.items()}


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ViTPPMAE(
        vol_size=(32, 32, 32),   # small for testing
        patch_size=8,
        in_chans=4,
        embed_dim=192,
        depth=4,
        n_heads=3,
        decoder_dim=96,
        decoder_depth=2,
    ).to(device)

    B = 2
    x      = torch.rand(B, 4, 32, 32, 32, device=device)
    target = torch.rand(B, 4, 32, 32, 32, device=device)
    seg    = torch.randint(0, 4, (B, 1, 32, 32, 32), device=device)

    optim   = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    trainer = ViTPPMAETrainer(model, optim, device=device)
    metrics = trainer.step({"noisy": x, "target": target, "seg": seg})

    print("Option 2 — ViT PP-MAE")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total:,}")
