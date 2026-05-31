"""
option_baselines.py — Architecture-matched baselines for PP-MAE Options 2-4
============================================================================

Implements baseline models that mirror the architectural families of the four
PP-MAE options, for fair ablation comparison:

  A. ViT / MAE family  (Option 2 baselines)
       VanillaMAE2D   — 2D ViT with random masking, plain L1 loss
       ViTPPMAE2D     — 2D ViT with saliency masking, PPMAELoss
       SparKCNN       — CNN MAE with sparse (masked) input  (ICLR 2023 style)

  B. Multi-task family  (Option 3 baselines)
       _SimpleUNet    — standard CNN UNet, no pathology loss
       MultiTaskUNet  — denoiser + seg head, joint L1 + CE training
       TransUNetLite  — CNN encoder + ViT bottleneck + CNN decoder

  C. Swin family  (Option 4 baselines)
       SwinIRLite     — Swin blocks for image restoration, no pathology loss
       UformerLite    — U-Net shaped network with window attention at each scale

All models work with 2D slices: (B, 4, H, W)
All trainers share the interface: step(batch) -> dict with 'total' key
"""

from typing import Optional, Dict, List, Tuple

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
    """Contiguity barrier — makes tensor contiguous in BOTH forward and backward.
    Place between reshape() and permute()/transpose() so backward gradient is
    made contiguous before reshape's backward calls .view() on MPS."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.contiguous()

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        return grad.contiguous()


def _c(x: torch.Tensor) -> torch.Tensor:
    return _ContiguousFunc.apply(x)




# ---------------------------------------------------------------------------
# MPS-compatible multi-head attention (shared across all baseline models)
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

        q = _c(self.q_proj(query).reshape(B, S, H, D)).permute(0, 2, 1, 3)  # (B,H,S,D)
        k = _c(self.k_proj(key  ).reshape(B, T, H, D)).permute(0, 2, 1, 3)  # (B,H,T,D)
        v = _c(self.v_proj(value).reshape(B, T, H, D)).permute(0, 2, 1, 3)  # (B,H,T,D)

        scale = D ** -0.5
        attn  = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn  = F.softmax(attn, dim=-1)
        out   = torch.matmul(attn, v)
        out = _c(out.permute(0, 2, 1, 3)).reshape(B, S, E)
        out = self.out_proj(out)
        return out, None   # mirrors nn.MultiheadAttention return signature


# =============================================================================
# A.  ViT / MAE FAMILY  (Option 2 baselines)
# =============================================================================

# ---------------------------------------------------------------------------
# Shared ViT building blocks
# ---------------------------------------------------------------------------

class PatchEmbed2D(nn.Module):
    """
    Split image into non-overlapping patches and project each to embed_dim.

    Uses a single Conv2d with kernel=patch_size, stride=patch_size, which is
    mathematically equivalent to splitting the image into patches and applying
    a linear projection — the standard ViT patch embedding.
    """

    def __init__(
        self,
        img_size:   int = 96,
        patch_size: int = 8,
        in_chans:   int = 4,
        embed_dim:  int = 128,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches  = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)  →  (B, n_patches, embed_dim)"""
        x = self.proj(x)                         # (B, D, H//p, W//p)
        x = x.flatten(2).transpose(1, 2)         # (B, n_patches, D)
        return x


class _ViTBlock(nn.Module):
    """
    Pre-LN Transformer block  (Pre-LayerNorm, following ViT-MAE convention).

    Structure:
        y  = x + MHA(LN(x))
        y' = y + FFN(LN(y))
    """

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = _SafeLayerNorm(dim)
        self.attn  = _MPSMHA(dim, n_heads)
        self.norm2 = _SafeLayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        a, _ = self.attn(n, n, n)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# VanillaMAE2D — random masking, no pathology loss
# ---------------------------------------------------------------------------

class VanillaMAE2D(nn.Module):
    """
    2D ViT Masked Autoencoder with RANDOM masking.

    Follows the MAE paper (He et al., CVPR 2022) but applied to 2D MRI slices.
    No segmentation / pathology loss — used as a direct ablation for ViTPPMAE2D.

    Args:
        img_size:     spatial size of input patch (assumed square)
        patch:        patch size  (patch × patch pixels per token)
        embed:        encoder embedding dimension
        depth:        number of encoder transformer blocks
        n_heads:      attention heads in encoder
        decoder_dim:  decoder embedding dimension
        decoder_depth: number of decoder transformer blocks
        in_chans:     number of MRI channels  (default 4)
        mask_ratio:   fraction of patches to mask during training
    """

    def __init__(
        self,
        img_size:     int = 96,
        patch:        int = 8,
        embed:        int = 128,
        depth:        int = 4,
        n_heads:      int = 4,
        decoder_dim:  int = 64,
        decoder_depth: int = 2,
        in_chans:     int = 4,
        mask_ratio:   float = 0.75,
    ):
        super().__init__()
        self.patch      = patch
        self.in_chans   = in_chans
        self.mask_ratio = mask_ratio
        self.img_size   = img_size

        # Encoder
        self.patch_embed = PatchEmbed2D(img_size, patch, in_chans, embed)
        n_patches        = self.patch_embed.n_patches
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches, embed))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.encoder = nn.Sequential(*[
            _ViTBlock(embed, n_heads) for _ in range(depth)
        ])
        self.enc_norm = _SafeLayerNorm(embed)

        # Decoder
        self.mask_token   = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.enc_to_dec   = nn.Linear(embed, decoder_dim, bias=True)
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, n_patches, decoder_dim))
        nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)

        self.decoder = nn.Sequential(*[
            _ViTBlock(decoder_dim, max(1, decoder_dim // 32))
            for _ in range(decoder_depth)
        ])
        self.dec_norm  = _SafeLayerNorm(decoder_dim)
        self.pred_head = nn.Linear(decoder_dim, in_chans * patch * patch)

    def _random_mask(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask (1 - keep_ratio) patches.

        Returns:
            visible:   (B, n_keep, D) — encoder input
            ids_keep:  (B, n_keep)    — indices of visible patches
            ids_mask:  (B, n_mask)    — indices of masked patches
        """
        B, N, D = x.shape
        n_keep   = max(1, int(N * (1.0 - self.mask_ratio)))
        noise    = torch.rand(B, N, device=x.device)
        ids_sort = torch.argsort(noise, dim=1)
        ids_keep = ids_sort[:, :n_keep]
        ids_mask = ids_sort[:, n_keep:]
        visible  = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        return visible, ids_keep, ids_mask

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: (B, n_patches, in_chans * patch^2)  →  (B, in_chans, H, W)
        """
        p = self.patch
        h = w = self.img_size // p
        B = patches.shape[0]
        patches = _c(patches.reshape(B, h, w, self.in_chans, p, p)).permute(0, 3, 1, 4, 2, 5)
        return _c(patches).reshape(B, self.in_chans, h * p, w * p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_chans, H, W)
        Returns: (B, in_chans, H, W) — sigmoid reconstruction
        """
        B, C, H, W = x.shape
        tokens = self.patch_embed(x)         # (B, N, embed)
        tokens = tokens + self.pos_embed

        if self.training and self.mask_ratio > 0.0:
            visible, ids_keep, ids_mask = self._random_mask(tokens)
        else:
            # At inference: encode all patches
            visible   = tokens
            N         = tokens.shape[1]
            ids_keep  = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
            ids_mask  = torch.zeros(B, 0, dtype=torch.long, device=x.device)

        # Encode visible patches
        enc_out = self.encoder(visible)
        enc_out = self.enc_norm(enc_out)

        # Project to decoder dim
        dec_tokens = self.enc_to_dec(enc_out)   # (B, n_keep, decoder_dim)

        # Reconstruct full sequence with mask tokens
        N_full = self.pos_embed.shape[1]
        N_keep = ids_keep.shape[1]
        N_mask = N_full - N_keep

        full_tokens = torch.zeros(B, N_full, dec_tokens.shape[-1], device=x.device)
        full_tokens.scatter_(
            1,
            ids_keep.unsqueeze(-1).expand(-1, -1, dec_tokens.shape[-1]),
            dec_tokens,
        )
        if N_mask > 0:
            mask_tok = self.mask_token.expand(B, N_mask, -1)
            full_tokens.scatter_(
                1,
                ids_mask.unsqueeze(-1).expand(-1, -1, dec_tokens.shape[-1]),
                mask_tok,
            )

        full_tokens = full_tokens + self.dec_pos_embed
        dec_out     = self.decoder(full_tokens)
        dec_out     = self.dec_norm(dec_out)

        patches = self.pred_head(dec_out)       # (B, N, C*p*p)
        return torch.sigmoid(self._unpatchify(patches))


# ---------------------------------------------------------------------------
# ViTPPMAE2D — saliency masking, PPMAELoss
# ---------------------------------------------------------------------------

class ViTPPMAE2D(nn.Module):
    """
    2D ViT PP-MAE with SALIENCY masking.

    Tumour patches (where the pooled segmentation > 0) are always kept
    visible to the encoder — the model is forced to reconstruct them from
    the decoder, which is trained with PPMAELoss for pathology awareness.

    Args: same as VanillaMAE2D
    """

    def __init__(
        self,
        img_size:      int = 96,
        patch:         int = 8,
        embed:         int = 128,
        depth:         int = 4,
        n_heads:       int = 4,
        decoder_dim:   int = 64,
        decoder_depth: int = 2,
        in_chans:      int = 4,
        mask_ratio:    float = 0.75,
    ):
        super().__init__()
        self.patch      = patch
        self.in_chans   = in_chans
        self.mask_ratio = mask_ratio
        self.img_size   = img_size

        # Encoder
        self.patch_embed = PatchEmbed2D(img_size, patch, in_chans, embed)
        n_patches        = self.patch_embed.n_patches
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches, embed))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.encoder  = nn.Sequential(*[_ViTBlock(embed, n_heads) for _ in range(depth)])
        self.enc_norm = _SafeLayerNorm(embed)

        # Decoder
        self.mask_token    = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.enc_to_dec    = nn.Linear(embed, decoder_dim, bias=True)
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, n_patches, decoder_dim))
        nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)

        self.decoder   = nn.Sequential(*[
            _ViTBlock(decoder_dim, max(1, decoder_dim // 32))
            for _ in range(decoder_depth)
        ])
        self.dec_norm  = _SafeLayerNorm(decoder_dim)
        self.pred_head = nn.Linear(decoder_dim, in_chans * patch * patch)

    def _saliency_mask(
        self,
        x:   torch.Tensor,   # (B, N, D)
        seg: torch.Tensor,   # (B, 1, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Saliency masking: tumour patches always visible, random mask on rest.

        Returns:
            visible:  (B, n_keep, D)
            ids_keep: (B, n_keep)
            ids_mask: (B, n_mask)
        """
        B, N, D = x.shape
        p = self.patch
        h = w = self.img_size // p

        # Pool segmentation to patch grid
        seg_pool = F.max_pool2d(seg.float(), kernel_size=p, stride=p)   # (B,1,h,w)
        tumour   = (seg_pool > 0).reshape(B, N)   # (B, N) bool

        ids_keep_list = []
        ids_mask_list = []
        n_keep_list   = []

        for b in range(B):
            tumour_idx     = tumour[b].nonzero(as_tuple=False).squeeze(1)   # tumour patches
            non_tumour_idx = (~tumour[b]).nonzero(as_tuple=False).squeeze(1)

            n_non          = non_tumour_idx.shape[0]
            n_tumour       = tumour_idx.shape[0]
            n_keep_non     = max(0, int(n_non * (1.0 - self.mask_ratio)))

            # Shuffle non-tumour patches
            perm           = torch.randperm(n_non, device=x.device)
            keep_non       = non_tumour_idx[perm[:n_keep_non]]
            mask_non       = non_tumour_idx[perm[n_keep_non:]]

            keep_b = torch.cat([tumour_idx, keep_non], dim=0)
            mask_b = mask_non

            ids_keep_list.append(keep_b)
            ids_mask_list.append(mask_b)
            n_keep_list.append(keep_b.shape[0])

        # Pad to equal length within batch
        max_keep = max(n_keep_list)
        max_mask = max(m.shape[0] for m in ids_mask_list)

        padded_keep = torch.zeros(B, max_keep, dtype=torch.long, device=x.device)
        padded_mask = torch.zeros(B, max_mask, dtype=torch.long, device=x.device)

        for b in range(B):
            k = ids_keep_list[b].shape[0]
            m = ids_mask_list[b].shape[0]
            padded_keep[b, :k] = ids_keep_list[b]
            if k < max_keep:
                padded_keep[b, k:] = ids_keep_list[b][0]  # pad with first
            if m > 0:
                padded_mask[b, :m] = ids_mask_list[b]
            if m < max_mask and m > 0:
                padded_mask[b, m:] = ids_mask_list[b][0]
            elif m == 0 and max_mask > 0:
                padded_mask[b, :] = ids_keep_list[b][0]

        visible = torch.gather(x, 1, padded_keep.unsqueeze(-1).expand(-1, -1, D))
        return visible, padded_keep, padded_mask

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        p = self.patch
        h = w = self.img_size // p
        B = patches.shape[0]
        patches = _c(patches.reshape(B, h, w, self.in_chans, p, p)).permute(0, 3, 1, 4, 2, 5)
        return _c(patches).reshape(B, self.in_chans, h * p, w * p)

    def forward(
        self,
        x:   torch.Tensor,             # (B, in_chans, H, W)
        seg: Optional[torch.Tensor] = None,  # (B, 1, H, W) tumour labels
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = self.patch_embed(x) + self.pos_embed

        if self.training and seg is not None and self.mask_ratio > 0.0:
            visible, ids_keep, ids_mask = self._saliency_mask(tokens, seg)
        else:
            visible   = tokens
            N         = tokens.shape[1]
            ids_keep  = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
            ids_mask  = torch.zeros(B, 0, dtype=torch.long, device=x.device)

        enc_out    = self.enc_norm(self.encoder(visible))
        dec_tokens = self.enc_to_dec(enc_out)

        N_full = self.pos_embed.shape[1]
        N_keep = ids_keep.shape[1]
        N_mask = N_full - N_keep

        full_tokens = torch.zeros(B, N_full, dec_tokens.shape[-1], device=x.device)
        full_tokens.scatter_(
            1,
            ids_keep.unsqueeze(-1).expand(-1, -1, dec_tokens.shape[-1]),
            dec_tokens,
        )
        if N_mask > 0 and ids_mask.shape[1] > 0:
            mask_tok = self.mask_token.expand(B, ids_mask.shape[1], -1)
            full_tokens.scatter_(
                1,
                ids_mask.unsqueeze(-1).expand(-1, -1, dec_tokens.shape[-1]),
                mask_tok,
            )

        full_tokens = full_tokens + self.dec_pos_embed
        dec_out     = self.dec_norm(self.decoder(full_tokens))
        patches     = self.pred_head(dec_out)
        return torch.sigmoid(self._unpatchify(patches))


# ---------------------------------------------------------------------------
# SparKCNN — CNN MAE with sparse masked input  (ICLR 2023 style)
# ---------------------------------------------------------------------------

class _SparKEncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        return self.pool(feat), feat   # (downsampled, skip)


class _SparKDecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        # Upsample halves channels; then cat with skip
        self.up_ch = in_ch // 2
        self.up    = nn.ConvTranspose2d(in_ch, self.up_ch, kernel_size=2, stride=2)
        self.conv  = nn.Sequential(
            nn.Conv2d(self.up_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle potential size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SparKCNN(nn.Module):
    """
    CNN MAE with sparse (randomly masked) input — SparK style (ICLR 2023).

    At training: randomly zero out 75% of non-overlapping patch_size=8 blocks
    in the input before passing through a UNet backbone, then reconstruct the
    full clean image with L1 loss.

    At inference: no masking applied — full image passed through UNet.

    Args:
        in_channels: number of MRI channels  (default 4)
        base_ch:     base channel width       (default 32)
        patch_size:  block size for masking   (default 8)
        mask_ratio:  fraction of patches to zero  (default 0.75)
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_ch:     int = 32,
        patch_size:  int = 8,
        mask_ratio:  float = 0.75,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        b = base_ch
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )
        # Encoder
        self.enc1 = _SparKEncoderBlock(b,    b * 2)
        self.enc2 = _SparKEncoderBlock(b * 2, b * 4)
        self.enc3 = _SparKEncoderBlock(b * 4, b * 8)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(b * 8, b * 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(b * 8),
            nn.ReLU(inplace=True),
        )

        # Decoder:
        #  dec3: in=b*8 (bottleneck) → up to b*4, cat sk3 (b*8) → conv → b*4
        #  dec2: in=b*4              → up to b*2, cat sk2 (b*4) → conv → b*2
        #  dec1: in=b*2              → up to b,   cat sk1 (b*2) → conv → b
        self.dec3 = _SparKDecoderBlock(b * 8, b * 8, b * 4)
        self.dec2 = _SparKDecoderBlock(b * 4, b * 4, b * 2)
        self.dec1 = _SparKDecoderBlock(b * 2, b * 2, b)

        self.head = nn.Sequential(
            nn.Conv2d(b, in_channels, 1),
            nn.Sigmoid(),
        )

    def _apply_patch_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Zero out mask_ratio fraction of non-overlapping patches."""
        B, C, H, W = x.shape
        p = self.patch_size
        ph = H // p
        pw = W // p
        n_patches = ph * pw
        n_mask    = int(n_patches * self.mask_ratio)

        masked = x.clone()
        for b in range(B):
            ids   = torch.randperm(n_patches, device=x.device)[:n_mask]
            rows  = ids // pw
            cols  = ids % pw
            for r, c in zip(rows.tolist(), cols.tolist()):
                masked[b, :, r * p:(r + 1) * p, c * p:(c + 1) * p] = 0.0
        return masked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, H, W)  →  (B, in_channels, H, W)"""
        if self.training:
            x = self._apply_patch_mask(x)

        s    = self.stem(x)
        e1, sk1 = self.enc1(s)
        e2, sk2 = self.enc2(e1)
        e3, sk3 = self.enc3(e2)
        bot  = self.bottleneck(e3)
        d3   = self.dec3(bot, sk3)
        d2   = self.dec2(d3, sk2)
        d1   = self.dec1(d2, sk1)
        return self.head(d1)


# ---------------------------------------------------------------------------
# ViT-family Trainers
# ---------------------------------------------------------------------------

class VanillaMAETrainer:
    """Trainer for VanillaMAE2D — plain L1 reconstruction loss."""

    def __init__(
        self,
        model:  VanillaMAE2D,
        device: str   = 'cuda',
        lr:     float = 1e-4,
    ):
        self.model  = model.to(device)
        self.device = device
        self.optim  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    def step(self, batch: Dict) -> Dict[str, float]:
        self.model.train()
        noisy  = batch['noisy'].to(self.device)
        target = batch['target'].to(self.device)

        self.optim.zero_grad()
        pred = self.model(noisy)
        loss = F.l1_loss(pred, target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        return {'total': loss.item()}

    @torch.no_grad()
    def predict(self, noisy: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(noisy.to(self.device)).cpu()


class ViTPPMAE2DTrainer:
    """Trainer for ViTPPMAE2D — saliency masking + PPMAELoss."""

    def __init__(
        self,
        model:   ViTPPMAE2D,
        device:  str   = 'cuda',
        lr:      float = 1e-4,
        lambda1: float = 1.0,
        lambda2: float = 0.5,
    ):
        self.model   = model.to(device)
        self.device  = device
        self.loss_fn = PPMAELoss(lambda1=lambda1, lambda2=lambda2).to(device)
        all_params   = list(model.parameters()) + list(self.loss_fn.parameters())
        self.optim   = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-5)

    def step(self, batch: Dict) -> Dict[str, float]:
        self.model.train()
        noisy  = batch['noisy'].to(self.device)
        target = batch['target'].to(self.device)
        seg    = batch['seg'].to(self.device)

        self.optim.zero_grad()
        pred   = self.model(noisy, seg)
        losses = self.loss_fn(pred, target, seg)
        losses['total'].backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def predict(self, noisy: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(noisy.to(self.device)).cpu()


class SparKCNNTrainer:
    """Trainer for SparKCNN — masked input L1 reconstruction."""

    def __init__(
        self,
        model:  SparKCNN,
        device: str   = 'cuda',
        lr:     float = 1e-4,
    ):
        self.model  = model.to(device)
        self.device = device
        self.optim  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    def step(self, batch: Dict) -> Dict[str, float]:
        self.model.train()
        noisy  = batch['noisy'].to(self.device)
        target = batch['target'].to(self.device)

        self.optim.zero_grad()
        pred = self.model(noisy)   # masking applied inside model during training
        loss = F.l1_loss(pred, target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        return {'total': loss.item()}

    @torch.no_grad()
    def predict(self, noisy: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(noisy.to(self.device)).cpu()  # no masking at eval


# =============================================================================
# B.  MULTI-TASK FAMILY  (Option 3 baselines)
# =============================================================================

# ---------------------------------------------------------------------------
# _SimpleUNet — plain CNN UNet, no pathology loss
# ---------------------------------------------------------------------------

class _EncBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        return self.pool(feat), feat   # (down, skip)


class _DecBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class _SimpleUNet(nn.Module):
    """
    Standard CNN UNet — no pathology loss, no saliency masking.

    Args:
        in_channels: number of MRI channels
        base_ch:     base channel width
        depth:       number of encoder/decoder stages  (max 3)
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_ch:     int = 16,
        depth:       int = 3,
    ):
        super().__init__()
        self.depth = depth

        chs = [in_channels] + [base_ch * (2 ** i) for i in range(depth)]
        self.encoders = nn.ModuleList([
            _EncBlock(chs[i], chs[i + 1]) for i in range(depth)
        ])

        bot_ch = chs[-1] * 2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(chs[-1], bot_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(bot_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(bot_ch, chs[-1], 3, padding=1, bias=False),
            nn.BatchNorm2d(chs[-1]),
            nn.ReLU(inplace=True),
        )

        dec_chs = list(reversed(chs[1:]))   # [deepest, ..., shallowest]
        self.decoders = nn.ModuleList([
            _DecBlock(dec_chs[i], dec_chs[i], dec_chs[i + 1] if i + 1 < len(dec_chs) else dec_chs[-1])
            for i in range(depth - 1)
        ] + [
            _DecBlock(dec_chs[-1], chs[1], chs[1])
        ])

        self.head = nn.Sequential(
            nn.Conv2d(chs[1], in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        out   = x
        for enc in self.encoders:
            out, skip = enc(out)
            skips.append(skip)

        out = self.bottleneck(out)

        for dec, skip in zip(self.decoders, reversed(skips)):
            out = dec(out, skip)

        return self.head(out)


# ---------------------------------------------------------------------------
# MultiTaskUNet — denoiser + segmentation head
# ---------------------------------------------------------------------------

class MultiTaskUNet(nn.Module):
    """
    Multi-task UNet: shared UNet denoiser + 4-class segmentation head.

    The denoiser reconstructs the clean MRI; the segmentation head predicts
    tumour sub-region labels from the denoised output.

    Args:
        in_ch:   number of MRI channels
        base_ch: base channel width for the denoiser UNet
        depth:   number of encoder/decoder stages
    """

    def __init__(
        self,
        in_ch:   int = 4,
        base_ch: int = 16,
        depth:   int = 3,
    ):
        super().__init__()
        self.denoiser = _SimpleUNet(in_channels=in_ch, base_ch=base_ch, depth=depth)
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, 1),   # 4-class logits  [BG, NCR, ED, ET]
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: (B, in_ch, H, W)  →  {'denoised': tensor, 'seg_logits': tensor}"""
        denoised   = self.denoiser(x)
        seg_logits = self.seg_head(denoised)
        return {'denoised': denoised, 'seg_logits': seg_logits}


class MultiTaskUNetTrainer:
    """
    Trainer for MultiTaskUNet.
    Loss = L1(denoised, target) + 0.5 * CrossEntropy(seg_logits, seg_gt)
    """

    def __init__(
        self,
        model:  MultiTaskUNet,
        device: str   = 'cuda',
        lr:     float = 1e-4,
    ):
        self.model   = model.to(device)
        self.device  = device
        self.optim   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.seg_loss = nn.CrossEntropyLoss()

    def step(self, batch: Dict) -> Dict[str, float]:
        self.model.train()
        noisy  = batch['noisy'].to(self.device)
        target = batch['target'].to(self.device)
        seg    = batch['seg'].to(self.device)

        seg_gt = seg[:, 0].long()   # (B, H, W) integer labels

        self.optim.zero_grad()
        out    = self.model(noisy)
        l_den  = F.l1_loss(out['denoised'], target)
        l_seg  = self.seg_loss(out['seg_logits'], seg_gt)
        loss   = l_den + 0.5 * l_seg
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        return {
            'total':   loss.item(),
            'denoise': l_den.item(),
            'seg':     l_seg.item(),
        }

    @torch.no_grad()
    def predict(self, noisy: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(noisy.to(self.device))['denoised'].cpu()


# ---------------------------------------------------------------------------
# TransUNetLite — CNN encoder + ViT bottleneck + CNN decoder
# ---------------------------------------------------------------------------

class TransUNetLite(nn.Module):
    """
    Lightweight TransUNet: CNN encoder → ViT bottleneck → CNN decoder.

    Architecture:
        CNN encoder:
            4 → 32ch  (3x3 conv + BN + ReLU)
            maxpool
            32 → 64ch  (3x3 conv + BN + ReLU)
            maxpool
            64 → 128ch  (bottleneck conv)

        ViT bottleneck:
            Flatten spatial (H//4, W//4) to N tokens
            4 × _ViTBlock(128, n_heads=4)
            Reshape back to (B, 128, H//4, W//4)

        CNN decoder:
            Upsample + cat(64ch skip) → conv → 64ch
            Upsample + cat(32ch skip) → conv → 32ch
            Conv → 4ch + sigmoid

    Args:
        in_ch:   number of input MRI channels
        n_heads: attention heads in ViT bottleneck
    """

    def __init__(self, in_ch: int = 4, n_heads: int = 4):
        super().__init__()
        # CNN Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # ViT bottleneck
        self.vit_norm = _SafeLayerNorm(128)
        self.vit_blocks = nn.Sequential(*[
            _ViTBlock(128, n_heads) for _ in range(4)
        ])

        # CNN Decoder
        self.up3  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up2  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv2d(32, in_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_ch, H, W)  →  (B, in_ch, H, W)"""
        # Encoder
        s1 = self.enc1(x)                    # (B, 32, H, W)
        e2 = self.enc2(self.pool1(s1))       # (B, 64, H//2, W//2)
        s2 = e2
        e3 = self.enc3(self.pool2(e2))       # (B, 128, H//4, W//4)

        # ViT bottleneck
        B, C, H4, W4 = e3.shape
        tokens = e3.flatten(2).transpose(1, 2)      # (B, H4*W4, 128)
        tokens = self.vit_norm(tokens)
        tokens = self.vit_blocks(tokens)            # (B, N, 128)
        bottleneck = _c(tokens.transpose(1, 2)).reshape(B, C, H4, W4)

        # Decoder
        d3 = self.up3(bottleneck)
        if d3.shape[2:] != s2.shape[2:]:
            d3 = F.interpolate(d3, size=s2.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, s2], dim=1))  # (B, 64, H//2, W//2)

        d2 = self.up2(d3)
        if d2.shape[2:] != s1.shape[2:]:
            d2 = F.interpolate(d2, size=s1.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, s1], dim=1))  # (B, 32, H, W)

        return self.head(d2)


class TransUNetTrainer:
    """Trainer for TransUNetLite — plain L1 loss."""

    def __init__(
        self,
        model:  TransUNetLite,
        device: str   = 'cuda',
        lr:     float = 1e-4,
    ):
        self.model  = model.to(device)
        self.device = device
        self.optim  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    def step(self, batch: Dict) -> Dict[str, float]:
        self.model.train()
        noisy  = batch['noisy'].to(self.device)
        target = batch['target'].to(self.device)

        self.optim.zero_grad()
        pred = self.model(noisy)
        loss = F.l1_loss(pred, target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        return {'total': loss.item()}

    @torch.no_grad()
    def predict(self, noisy: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(noisy.to(self.device)).cpu()


# =============================================================================
# C.  SWIN FAMILY  (Option 4 baselines)
# =============================================================================

# ---------------------------------------------------------------------------
# _WindowAttnBlock — simplified Swin window attention (no shift)
# ---------------------------------------------------------------------------

class _WindowAttnBlock(nn.Module):
    """
    Simplified Swin Transformer block with non-overlapping window attention.

    For simplicity, no cyclic shift is applied (which is the main SWIN
    distinction). Padding is applied when H or W is not divisible by window_size.

    Input/output: (B, H, W, C)  — spatial-first format.

    Args:
        dim:         channel dimension
        n_heads:     number of attention heads
        window_size: local attention window size
    """

    def __init__(self, dim: int, n_heads: int, window_size: int = 4):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.norm1 = _SafeLayerNorm(dim)
        self.attn  = _MPSMHA(dim, n_heads)
        self.norm2 = _SafeLayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def _partition_windows(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, int, int, int, int]:
        """
        Partition (B, H, W, C) into windows of size window_size.
        Returns: (n_windows*B, window_size*window_size, C), H_pad, W_pad, H, W
        """
        B, H, W, C = x.shape
        ws = self.window_size
        # Pad so H, W are divisible by ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        x = _c(x.reshape(B, Hp // ws, ws, Wp // ws, ws, C)).permute(0, 1, 3, 2, 4, 5)
        windows = _c(x).reshape(-1, ws * ws, C)          # (B*nH*nW, ws^2, C)
        return windows, Hp, Wp, H, W

    def _unpartition_windows(
        self,
        windows: torch.Tensor,
        Hp: int, Wp: int,
        H: int, W: int,
        B: int,
    ) -> torch.Tensor:
        """Reconstruct (B, H, W, C) from windows."""
        ws = self.window_size
        x = _c(windows.reshape(B, Hp // ws, Wp // ws, ws, ws, -1)).permute(0, 1, 3, 2, 4, 5)
        x = _c(x).reshape(B, Hp, Wp, -1)
        return x[:, :H, :W, :].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C)  →  (B, H, W, C)"""
        B, H, W, C = x.shape

        # Window attention
        shortcut = x
        x_norm   = self.norm1(x)
        windows, Hp, Wp, H_orig, W_orig = self._partition_windows(x_norm)

        attn_out, _ = self.attn(windows, windows, windows)
        attn_out    = self._unpartition_windows(attn_out, Hp, Wp, H, W, B)
        x           = shortcut + attn_out

        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# SwinIRLite — Swin blocks for image restoration
# ---------------------------------------------------------------------------

class SwinIRLite(nn.Module):
    """
    Lightweight SwinIR — Swin Transformer blocks for image restoration.

    No pathology loss and no saliency masking — used as a direct ablation
    for SwinPPMAE.

    Args:
        in_ch:       number of MRI channels (default 4)
        dim:         channel dimension for Swin blocks
        n_blocks:    number of _WindowAttnBlocks to stack
        window_size: local attention window size
    """

    def __init__(
        self,
        in_ch:       int = 4,
        dim:         int = 64,
        n_blocks:    int = 4,
        window_size: int = 4,
    ):
        super().__init__()
        # Stem: project to dim channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dim, 3, padding=1),
            nn.GELU(),
        )

        # Swin blocks — operate on (B, H, W, dim)
        self.blocks = nn.ModuleList([
            _WindowAttnBlock(dim, n_heads=4, window_size=window_size)
            for _ in range(n_blocks)
        ])

        # Residual connection from stem
        self.norm = _SafeLayerNorm(dim)

        # Head: project back to in_ch
        self.head = nn.Sequential(
            nn.Conv2d(dim, in_ch, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_ch, H, W)  →  (B, in_ch, H, W)"""
        feat = self.stem(x)                          # (B, dim, H, W)
        # Convert to spatial-first for Swin blocks
        s    = feat.permute(0, 2, 3, 1)             # (B, H, W, dim)

        for blk in self.blocks:
            s = blk(s)

        s    = self.norm(s)
        s    = s.permute(0, 3, 1, 2)               # (B, dim, H, W)
        # Global residual
        s    = s + feat
        return self.head(s)


class SwinIRTrainer:
    """Trainer for SwinIRLite — plain L1 loss."""

    def __init__(
        self,
        model:  SwinIRLite,
        device: str   = 'cuda',
        lr:     float = 1e-4,
    ):
        self.model  = model.to(device)
        self.device = device
        self.optim  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    def step(self, batch: Dict) -> Dict[str, float]:
        self.model.train()
        noisy  = batch['noisy'].to(self.device)
        target = batch['target'].to(self.device)

        self.optim.zero_grad()
        pred = self.model(noisy)
        loss = F.l1_loss(pred, target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        return {'total': loss.item()}

    @torch.no_grad()
    def predict(self, noisy: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(noisy.to(self.device)).cpu()


class SwinIRPathologyTrainer:
    """
    Trainer for SwinIRLite using the pathology-preserving composite loss
    (global + PathologyLoss + cross-modal) INSTEAD of plain L1.

    This is the controlled-comparison partner of SwinIRTrainer: the network
    architecture is byte-for-byte identical, only the loss function changes.
    Comparing the two isolates the effect of pathology-preserving training on
    a Swin backbone — exactly the Round-3 design (MultiTask-UNet vs PP-MAE
    Option 3) but applied to the Swin family.
    """

    def __init__(
        self,
        model:  SwinIRLite,
        device: str   = 'cuda',
        lr:     float = 1e-4,
        mode:   str   = 'clinical_risk',
        lambda1: float = 1.0,
        lambda2: float = 0.5,
    ):
        from losses import PPMAELoss
        self.model   = model.to(device)
        self.device  = device
        self.optim   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.loss_fn = PPMAELoss(lambda1=lambda1, lambda2=lambda2, mode=mode)

    def step(self, batch: Dict) -> Dict[str, float]:
        self.model.train()
        noisy  = batch['noisy'].to(self.device)
        target = batch['target'].to(self.device)
        seg    = batch['seg'].to(self.device)

        self.optim.zero_grad()
        pred   = self.model(noisy)                 # same SwinIR forward, no seg needed
        losses = self.loss_fn(pred, target, seg)   # pathology-preserving loss
        losses['total'].backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        return {k: (v.item() if torch.is_tensor(v) else v) for k, v in losses.items()}

    @torch.no_grad()
    def predict(self, noisy: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(noisy.to(self.device)).cpu()


# ---------------------------------------------------------------------------
# UformerLite — U-Net shaped network with window attention at each scale
# ---------------------------------------------------------------------------

class UformerLite(nn.Module):
    """
    Lightweight Uformer: U-Net shaped network with Swin window attention at
    each scale (encoder, bottleneck, decoder).

    Architecture:
        Encoder:
            enc1: Conv(in_ch → dim, 3,1,1) → GELU → WindowAttn(dim) → skip1
            enc2: Conv(dim → dim*2, 3,2,1) → GELU → WindowAttn(dim*2) → skip2
            enc3: Conv(dim*2 → dim*4, 3,2,1) → GELU → WindowAttn(dim*4) → skip3

        Bottleneck:
            2 × WindowAttn(dim*4)

        Decoder:
            dec3: upsample(skip3) → cat(skip2) → conv(dim*4+dim*2→dim*2) → WindowAttn(dim*2)
            dec2: upsample → cat(skip1) → conv(dim*2+dim→dim) → WindowAttn(dim)
            head: Conv(dim → in_ch, 1) + sigmoid

    Args:
        in_ch:       number of MRI channels (default 4)
        dim:         base channel dimension (default 32)
        window_size: local attention window size (default 4)
    """

    def __init__(
        self,
        in_ch:       int = 4,
        dim:         int = 32,
        window_size: int = 4,
    ):
        super().__init__()
        ws = window_size

        # Encoder stage 1  — full resolution
        self.enc1_conv = nn.Sequential(
            nn.Conv2d(in_ch, dim, 3, 1, 1),
            nn.GELU(),
        )
        self.enc1_attn = _WindowAttnBlock(dim, n_heads=max(1, dim // 8), window_size=ws)

        # Encoder stage 2  — stride-2 down
        self.enc2_conv = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 3, 2, 1),
            nn.GELU(),
        )
        self.enc2_attn = _WindowAttnBlock(dim * 2, n_heads=max(1, (dim * 2) // 8), window_size=ws)

        # Encoder stage 3  — stride-2 down again
        self.enc3_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, 3, 2, 1),
            nn.GELU(),
        )
        self.enc3_attn = _WindowAttnBlock(dim * 4, n_heads=max(1, (dim * 4) // 8), window_size=ws)

        # Bottleneck
        self.bot1 = _WindowAttnBlock(dim * 4, n_heads=max(1, (dim * 4) // 8), window_size=ws)
        self.bot2 = _WindowAttnBlock(dim * 4, n_heads=max(1, (dim * 4) // 8), window_size=ws)

        # Decoder stage 3  — upsample × 2, merge with skip2 (dim*2)
        self.up3       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(dim * 4 + dim * 2, dim * 2, 3, 1, 1),
            nn.GELU(),
        )
        self.dec3_attn = _WindowAttnBlock(dim * 2, n_heads=max(1, (dim * 2) // 8), window_size=ws)

        # Decoder stage 2  — upsample × 2, merge with skip1 (dim)
        self.up2       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(dim * 2 + dim, dim, 3, 1, 1),
            nn.GELU(),
        )
        self.dec2_attn = _WindowAttnBlock(dim, n_heads=max(1, dim // 8), window_size=ws)

        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(dim, in_ch, 1),
            nn.Sigmoid(),
        )

    def _to_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, H, W, C)"""
        return x.permute(0, 2, 3, 1)

    def _from_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, W, C) → (B, C, H, W)"""
        return x.permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_ch, H, W)  →  (B, in_ch, H, W)"""
        # Encoder
        s1 = self.enc1_conv(x)                             # (B, dim, H, W)
        s1 = self._from_spatial(self.enc1_attn(self._to_spatial(s1)))   # skip1

        s2 = self.enc2_conv(s1)                            # (B, dim*2, H//2, W//2)
        s2 = self._from_spatial(self.enc2_attn(self._to_spatial(s2)))   # skip2

        s3 = self.enc3_conv(s2)                            # (B, dim*4, H//4, W//4)
        s3 = self._from_spatial(self.enc3_attn(self._to_spatial(s3)))   # skip3

        # Bottleneck
        bot = self._to_spatial(s3)
        bot = self.bot1(bot)
        bot = self.bot2(bot)
        bot = self._from_spatial(bot)                      # (B, dim*4, H//4, W//4)

        # Decoder stage 3
        d3 = self.up3(bot)
        if d3.shape[2:] != s2.shape[2:]:
            d3 = F.interpolate(d3, size=s2.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3_conv(torch.cat([d3, s2], dim=1))   # (B, dim*2, H//2, W//2)
        d3 = self._from_spatial(self.dec3_attn(self._to_spatial(d3)))

        # Decoder stage 2
        d2 = self.up2(d3)
        if d2.shape[2:] != s1.shape[2:]:
            d2 = F.interpolate(d2, size=s1.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2_conv(torch.cat([d2, s1], dim=1))   # (B, dim, H, W)
        d2 = self._from_spatial(self.dec2_attn(self._to_spatial(d2)))

        return self.head(d2)


class UformerTrainer:
    """Trainer for UformerLite — plain L1 loss."""

    def __init__(
        self,
        model:  UformerLite,
        device: str   = 'cuda',
        lr:     float = 1e-4,
    ):
        self.model  = model.to(device)
        self.device = device
        self.optim  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    def step(self, batch: Dict) -> Dict[str, float]:
        self.model.train()
        noisy  = batch['noisy'].to(self.device)
        target = batch['target'].to(self.device)

        self.optim.zero_grad()
        pred = self.model(noisy)
        loss = F.l1_loss(pred, target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        return {'total': loss.item()}

    @torch.no_grad()
    def predict(self, noisy: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(noisy.to(self.device)).cpu()


class UformerPathologyTrainer:
    """
    Trainer for UformerLite using the pathology-preserving composite loss
    (global + PathologyLoss + cross-modal) instead of plain L1.

    Architecture is byte-for-byte identical to UformerTrainer — only the
    loss function changes.  Paired with UformerTrainer as a controlled
    experiment: same Uformer backbone, L1 vs PathologyLoss.
    """

    def __init__(
        self,
        model:   UformerLite,
        device:  str   = 'cuda',
        lr:      float = 1e-4,
        mode:    str   = 'clinical_risk',
        lambda1: float = 1.0,
        lambda2: float = 0.5,
    ):
        from losses import PPMAELoss
        self.model   = model.to(device)
        self.device  = device
        self.optim   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.loss_fn = PPMAELoss(lambda1=lambda1, lambda2=lambda2, mode=mode)

    def step(self, batch: Dict) -> Dict[str, float]:
        self.model.train()
        noisy  = batch['noisy'].to(self.device)
        target = batch['target'].to(self.device)
        seg    = batch['seg'].to(self.device)

        self.optim.zero_grad()
        pred   = self.model(noisy)
        losses = self.loss_fn(pred, target, seg)
        losses['total'].backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        return {k: (v.item() if torch.is_tensor(v) else v) for k, v in losses.items()}

    @torch.no_grad()
    def predict(self, noisy: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(noisy.to(self.device)).cpu()


# =============================================================================
# D.  EXISTING PUBLISHED WORK BASELINES  (Option 3 comparison)
# =============================================================================
#
# Three published architectures rephrased as 2D slice pipelines to give a
# fair "does PP-MAE beat prior art?" comparison for the pipeline round.
#
#   1. UNETRLite      —  UNETR (Hatamizadeh et al., WACV 2022)
#                        ViT encoder + CNN decoder; joint L1 + CE
#   2. SwinUNETRLite  —  SwinUNETR (Hatamizadeh et al., CVPR 2022)
#                        Swin encoder + U-Net decoder; joint L1 + CE
#   3. SeqPipeline    —  Standard clinical workflow: DnCNN first then a
#                        separately-trained segmentor; no joint training,
#                        no pathology loss, no feedback loop
#
# ALL use standard L1 + equal-weight CrossEntropy — the ONLY difference
# from PP-MAE Pipeline is the missing PathologyLoss.
# =============================================================================


# ---------------------------------------------------------------------------
# 1.  UNETRLite  (UNETR, Hatamizadeh et al., WACV 2022)
# ---------------------------------------------------------------------------

class UNETRLite(nn.Module):
    """
    Lightweight 2D adaptation of UNETR (WACV 2022).

    UNETR uses a pure ViT encoder (no CNN inductive bias) and routes skip
    connections from intermediate transformer layers into a CNN decoder.
    For 2D slices we use a 4-layer ViT and 4-level CNN decoder.

    Key difference from PP-MAE Pipeline:
      - Same ViT + CNN structure but trained with plain L1 + CrossEntropy
      - No PathologyLoss, no saliency masking, no cross-modal loss
      - Represents "ViT architecture, standard loss" ablation
    """

    def __init__(
        self,
        in_ch:      int = 4,
        img_size:   int = 96,
        patch_size: int = 16,
        embed_dim:  int = 192,
        n_heads:    int = 4,
        n_layers:   int = 4,
        base_ch:    int = 32,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.n_grid     = img_size // patch_size   # number of patches per dim
        n_patches       = self.n_grid ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_ch, embed_dim, patch_size, stride=patch_size)
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ViT body — 4 transformer layers; save outputs at layers 1,2,3,4
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads,
                dim_feedforward=embed_dim * 4,
                batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        ])

        G = self.n_grid
        # UNETR-style skip projectors: reshape token → spatial feature map
        self.skip_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, base_ch * (2 ** i)),
                nn.GELU(),
            )
            for i in range(n_layers)
        ])

        # CNN decoder  (4 levels, upsampling × patch_size total)
        #   level 0: (B, base_ch*1,  G,  G)  → (B, 32, G, G)
        #   level 1: (B, base_ch*2,  G,  G)  with skip from layer 0
        #   level 2: (B, base_ch*4,  G,  G)  ...
        #   level 3: (B, base_ch*8,  G,  G)
        # then upsample to full image
        ch = [base_ch * (2 ** i) for i in range(n_layers)]  # [32,64,128,256]
        self.dec_ups   = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        for i in range(n_layers - 1, 0, -1):
            # merge current level + skip from previous
            merge_ch = ch[i] + ch[i - 1]
            self.dec_ups.append(nn.Identity())          # no spatial upsampling inside ViT stage
            self.dec_convs.append(nn.Sequential(
                nn.Conv2d(merge_ch, ch[i - 1], 3, padding=1),
                nn.InstanceNorm2d(ch[i - 1]), nn.GELU(),
            ))

        # Final upsample to image resolution
        up_factor = patch_size
        layers_list: List[nn.Module] = []
        c = ch[0]
        while up_factor > 1:
            layers_list += [
                nn.ConvTranspose2d(c, c // 2, 2, stride=2),
                nn.GELU(),
            ]
            c = c // 2
            up_factor //= 2
        self.final_up = nn.Sequential(*layers_list)

        self.denoiser_head = nn.Conv2d(c, in_ch, 1)
        self.seg_head      = nn.Conv2d(c, 4,     1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = x.shape
        G = self.n_grid

        # Patch embed → tokens
        tokens = self.patch_embed(x)                 # (B, embed_dim, G, G)
        tokens = tokens.flatten(2).permute(0, 2, 1)  # (B, G², embed_dim)
        tokens = tokens + self.pos_embed

        # Run transformer layers, collect intermediate features
        skip_maps = []
        for i, layer in enumerate(self.layers):
            tokens = layer(tokens)
            # Project and reshape to spatial map
            feat = self.skip_proj[i](tokens)          # (B, G², ch[i])
            ch_i = feat.shape[-1]
            feat = _c(feat.permute(0, 2, 1)).reshape(B, ch_i, G, G)
            skip_maps.append(feat)

        # Decode by merging skip connections top-down
        x_dec = skip_maps[-1]
        for i, (up, conv) in enumerate(zip(self.dec_ups, self.dec_convs)):
            skip = skip_maps[-(i + 2)]
            x_dec = torch.cat([x_dec, skip], dim=1)
            x_dec = conv(x_dec)

        x_dec = self.final_up(x_dec)   # (B, small_ch, H, W)
        return {
            'denoised':   self.denoiser_head(x_dec),
            'seg_logits': self.seg_head(x_dec),
        }


class UNETRLiteTrainer:
    """
    Trainer for UNETRLite.
    Loss = L1(denoised, target) + 0.5 * CrossEntropy(seg, seg_gt)
    Standard multi-task loss — no PathologyLoss.
    """

    def __init__(
        self,
        model:  UNETRLite,
        device: str   = 'cuda',
        lr:     float = 1e-4,
    ):
        self.model    = model.to(device)
        self.device   = device
        self.optim    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.seg_loss = nn.CrossEntropyLoss()

    def step(self, batch: Dict) -> Dict[str, float]:
        self.model.train()
        noisy  = batch['noisy'].to(self.device)
        target = batch['target'].to(self.device)
        seg_gt = batch['seg'][:, 0].long().to(self.device)

        self.optim.zero_grad()
        out    = self.model(noisy)
        l_den  = F.l1_loss(out['denoised'], target)
        l_seg  = self.seg_loss(out['seg_logits'], seg_gt)
        loss   = l_den + 0.5 * l_seg
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        return {'total': loss.item(), 'den': l_den.item(), 'seg': l_seg.item()}

    @torch.no_grad()
    def predict(self, noisy: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(noisy.to(self.device))['denoised'].cpu()


# ---------------------------------------------------------------------------
# 2.  SwinUNETRLite  (SwinUNETR, Hatamizadeh et al., CVPR 2022)
# ---------------------------------------------------------------------------

class SwinUNETRLite(nn.Module):
    """
    Lightweight 2D adaptation of SwinUNETR (CVPR 2022).

    SwinUNETR is the gold-standard transformer baseline for BraTS segmentation
    (winner/near-winner of several BraTS challenges).  It uses a Swin Transformer
    encoder with hierarchical windows and a CNN U-Net decoder with skip
    connections.

    This lite 2D version mirrors the SwinUNETR design at reduced scale for
    our 2D slice experiments:
      - 3 Swin encoder stages (no downsampling inside ViT, patch merging between)
      - Symmetric CNN decoder with skip connections
      - Joint L1 denoising + CrossEntropy segmentation head

    Key difference from PP-MAE Swin (Option 4) + Pipeline:
      - No PathologyLoss, no saliency reweighting, no cross-modal attention
      - Represents "SwinUNETR architecture, standard loss"
    """

    def __init__(
        self,
        in_ch:       int = 4,
        base_ch:     int = 48,
        window_size: int = 7,
    ):
        super().__init__()

        # Stem: CNN projection to base_ch feature map
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.InstanceNorm2d(base_ch), nn.GELU(),
        )

        # Encoder stages: Swin blocks + patch-merging downsampling
        # Stage 0: base_ch  → base_ch       (no downsample, skip here)
        # Stage 1: base_ch  → base_ch*2     (downsample ×2)
        # Stage 2: base_ch*2→ base_ch*4     (downsample ×2)
        dims = [base_ch, base_ch * 2, base_ch * 4]
        self.enc0 = nn.Sequential(
            _WindowAttnBlock(dims[0], max(1, dims[0] // 16), window_size),
            _WindowAttnBlock(dims[0], max(1, dims[0] // 16), window_size),
        )
        self.down01 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], 2, stride=2),
            nn.InstanceNorm2d(dims[1]), nn.GELU(),
        )
        self.enc1 = nn.Sequential(
            _WindowAttnBlock(dims[1], max(1, dims[1] // 16), window_size),
            _WindowAttnBlock(dims[1], max(1, dims[1] // 16), window_size),
        )
        self.down12 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], 2, stride=2),
            nn.InstanceNorm2d(dims[2]), nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            _WindowAttnBlock(dims[2], max(1, dims[2] // 16), window_size),
            _WindowAttnBlock(dims[2], max(1, dims[2] // 16), window_size),
        )

        # Decoder with skip connections (SwinUNETR style)
        # Note: Conv2d merges → (B,C,H,W); _WindowAttnBlock takes (B,H,W,C).
        # We store them separately and apply them with explicit format swaps in forward().
        self.up21      = nn.ConvTranspose2d(dims[2], dims[1], 2, stride=2)
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(dims[1] * 2, dims[1], 3, padding=1),
            nn.InstanceNorm2d(dims[1]), nn.GELU(),
        )
        self.dec1_attn = _WindowAttnBlock(dims[1], max(1, dims[1] // 16), window_size)

        self.up10      = nn.ConvTranspose2d(dims[1], dims[0], 2, stride=2)
        self.dec0_conv = nn.Sequential(
            nn.Conv2d(dims[0] * 2, dims[0], 3, padding=1),
            nn.InstanceNorm2d(dims[0]), nn.GELU(),
        )
        self.dec0_attn = _WindowAttnBlock(dims[0], max(1, dims[0] // 16), window_size)

        self.denoiser_head = nn.Conv2d(dims[0], in_ch, 1)
        self.seg_head      = nn.Conv2d(dims[0], 4, 1)

    @staticmethod
    def _to_sp(x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, H, W, C)."""
        return x.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _fr_sp(x: torch.Tensor) -> torch.Tensor:
        """(B, H, W, C) → (B, C, H, W)."""
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        s0 = self.stem(x)                                        # (B, d0, H, W)

        # Encoder — convert to spatial format for window attention
        e0 = self._fr_sp(self.enc0(self._to_sp(s0)))            # (B, d0, H, W)
        e1 = self._fr_sp(self.enc1(self._to_sp(self.down01(e0))))
        e2 = self._fr_sp(self.enc2(self._to_sp(self.down12(e1))))

        # Decoder — Conv2d merge then window attention with explicit format swap
        d1 = self.dec1_conv(torch.cat([self.up21(e2), e1], dim=1))  # (B, d1, H/2, W/2)
        d1 = self._fr_sp(self.dec1_attn(self._to_sp(d1)))

        d0 = self.dec0_conv(torch.cat([self.up10(d1), e0], dim=1))  # (B, d0, H, W)
        d0 = self._fr_sp(self.dec0_attn(self._to_sp(d0)))

        return {
            'denoised':   self.denoiser_head(d0),
            'seg_logits': self.seg_head(d0),
        }


class SwinUNETRLiteTrainer:
    """
    Trainer for SwinUNETRLite.
    Loss = L1(denoised, target) + 0.5 * CrossEntropy(seg, seg_gt)
    Standard multi-task loss — no PathologyLoss.
    """

    def __init__(
        self,
        model:  SwinUNETRLite,
        device: str   = 'cuda',
        lr:     float = 1e-4,
    ):
        self.model    = model.to(device)
        self.device   = device
        self.optim    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.seg_loss = nn.CrossEntropyLoss()

    def step(self, batch: Dict) -> Dict[str, float]:
        self.model.train()
        noisy  = batch['noisy'].to(self.device)
        target = batch['target'].to(self.device)
        seg_gt = batch['seg'][:, 0].long().to(self.device)

        self.optim.zero_grad()
        out    = self.model(noisy)
        l_den  = F.l1_loss(out['denoised'], target)
        l_seg  = self.seg_loss(out['seg_logits'], seg_gt)
        loss   = l_den + 0.5 * l_seg
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        return {'total': loss.item(), 'den': l_den.item(), 'seg': l_seg.item()}

    @torch.no_grad()
    def predict(self, noisy: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(noisy.to(self.device))['denoised'].cpu()


# ---------------------------------------------------------------------------
# 3.  SeqPipeline  (Standard clinical workflow — sequential, no joint training)
# ---------------------------------------------------------------------------

class _SeqDnCNN(nn.Module):
    """DnCNN-style denoiser used inside SeqPipeline."""

    def __init__(self, in_ch: int = 4, n_layers: int = 15, ch: int = 64):
        super().__init__()
        layers: List[nn.Module] = [nn.Conv2d(in_ch, ch, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(n_layers - 2):
            layers += [nn.Conv2d(ch, ch, 3, padding=1),
                       nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(ch, in_ch, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.net(x)   # residual denoising


class _SeqSegNet(nn.Module):
    """Small U-Net segmentor used inside SeqPipeline."""

    def __init__(self, in_ch: int = 4, base: int = 32):
        super().__init__()
        def _cb(i, o): return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.InstanceNorm2d(o), nn.GELU(),
            nn.Conv2d(o, o, 3, padding=1), nn.InstanceNorm2d(o), nn.GELU(),
        )
        self.e1 = _cb(in_ch, base)
        self.e2 = _cb(base,  base * 2)
        self.e3 = _cb(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2  = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2   = _cb(base * 4, base * 2)
        self.up1  = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1   = _cb(base * 2, base)
        self.head = nn.Conv2d(base, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        b  = self.e3(self.pool(s2))
        d2 = self.d2(torch.cat([self.up2(b), s2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), s1], dim=1))
        return self.head(d1)


class SeqPipeline(nn.Module):
    """
    Sequential Pipeline — standard clinical workflow (no joint training).

    Represents the naive approach used in practice before end-to-end learning:
      1. Train a DnCNN denoiser independently (L1 loss only)
      2. Take its output, train a segmentor independently (CrossEntropy only)
      3. At inference: denoised = denoiser(noisy), seg = segmentor(denoised)

    Key differences from PP-MAE Pipeline:
      • No joint gradient flow — denoiser never sees segmentation signal
      • No PathologyLoss — denoiser optimises pixel fidelity only
      • No saliency masking — treats all regions equally
      • No cross-modal consistency loss

    This is the 'decoupled baseline' that answers:
      'Is joint training + pathology loss necessary, or is sequential good enough?'
    """

    def __init__(self, in_ch: int = 4):
        super().__init__()
        self.denoiser   = _SeqDnCNN(in_ch)
        self.segmentor  = _SeqSegNet(in_ch)

    def forward_denoise(self, noisy: torch.Tensor) -> torch.Tensor:
        return self.denoiser(noisy)

    def forward_seg(self, clean: torch.Tensor) -> torch.Tensor:
        return self.segmentor(clean)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        denoised   = self.denoiser(x)
        seg_logits = self.segmentor(denoised.detach())   # DETACHED — no joint gradient
        return {'denoised': denoised, 'seg_logits': seg_logits}


class SeqPipelineTrainer:
    """
    Trainer for SeqPipeline.

    Phase 1 (denoiser only): L1 loss, segmentor frozen.
    Phase 2 (segmentor only): CrossEntropy loss, denoiser frozen.

    In practice we interleave them: odd batches → denoiser step,
    even batches → segmentor step.  This matches the 'train separately'
    paradigm from the clinical literature.
    """

    def __init__(
        self,
        model:  SeqPipeline,
        device: str   = 'cuda',
        lr:     float = 1e-4,
    ):
        self.model    = model.to(device)
        self.device   = device
        self.optim_d  = torch.optim.AdamW(
            model.denoiser.parameters(),  lr=lr, weight_decay=1e-5)
        self.optim_s  = torch.optim.AdamW(
            model.segmentor.parameters(), lr=lr, weight_decay=1e-5)
        self.seg_loss = nn.CrossEntropyLoss()
        self._step_counter = 0

    def step(self, batch: Dict) -> Dict[str, float]:
        self.model.train()
        noisy  = batch['noisy'].to(self.device)
        target = batch['target'].to(self.device)
        seg_gt = batch['seg'][:, 0].long().to(self.device)

        self._step_counter += 1
        if self._step_counter % 2 == 1:
            # ---- Denoiser step (segmentor frozen) ----
            self.optim_d.zero_grad()
            denoised = self.model.denoiser(noisy)
            loss     = F.l1_loss(denoised, target)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.denoiser.parameters(), 1.0)
            self.optim_d.step()
            return {'total': loss.item(), 'den': loss.item(), 'seg': 0.0}
        else:
            # ---- Segmentor step (denoiser frozen) ----
            self.optim_s.zero_grad()
            with torch.no_grad():
                denoised = self.model.denoiser(noisy)
            seg_logits = self.model.segmentor(denoised)
            loss       = self.seg_loss(seg_logits, seg_gt)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.segmentor.parameters(), 1.0)
            self.optim_s.step()
            return {'total': loss.item(), 'den': 0.0, 'seg': loss.item()}

    @torch.no_grad()
    def predict(self, noisy: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model.denoiser(noisy.to(self.device)).cpu()


# =============================================================================
# Quick sanity check
# =============================================================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    B, C, H, W = 2, 4, 96, 96
    noisy  = torch.rand(B, C, H, W).to(device)
    seg    = torch.randint(0, 4, (B, 1, H, W)).to(device)
    target = torch.rand(B, C, H, W).to(device)
    batch  = {'noisy': noisy, 'target': target, 'seg': seg}

    print("\n-- ViT family --")
    vm = VanillaMAE2D(img_size=H, patch=8, embed=64, depth=2, n_heads=4,
                      decoder_dim=32, decoder_depth=1).to(device)
    t  = VanillaMAETrainer(vm, device=device)
    print("VanillaMAE2D step:", t.step(batch))

    vp = ViTPPMAE2D(img_size=H, patch=8, embed=64, depth=2, n_heads=4,
                    decoder_dim=32, decoder_depth=1).to(device)
    tp = ViTPPMAE2DTrainer(vp, device=device)
    print("ViTPPMAE2D step:", tp.step(batch))

    sk = SparKCNN(in_channels=C, base_ch=16).to(device)
    ts = SparKCNNTrainer(sk, device=device)
    print("SparKCNN step:", ts.step(batch))

    print("\n-- Multi-task family --")
    mu = MultiTaskUNet(in_ch=C, base_ch=8, depth=2).to(device)
    tm = MultiTaskUNetTrainer(mu, device=device)
    print("MultiTaskUNet step:", tm.step(batch))

    tu = TransUNetLite(in_ch=C).to(device)
    tt = TransUNetTrainer(tu, device=device)
    print("TransUNetLite step:", tt.step(batch))

    print("\n-- Swin family --")
    si = SwinIRLite(in_ch=C, dim=32, n_blocks=2, window_size=4).to(device)
    ts2 = SwinIRTrainer(si, device=device)
    print("SwinIRLite step:", ts2.step(batch))

    uf = UformerLite(in_ch=C, dim=16, window_size=4).to(device)
    tu2 = UformerTrainer(uf, device=device)
    print("UformerLite step:", tu2.step(batch))

    print("\nAll sanity checks passed.")
