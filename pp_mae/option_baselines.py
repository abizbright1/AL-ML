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
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
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
        self.enc_norm = nn.LayerNorm(embed)

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
        self.dec_norm  = nn.LayerNorm(decoder_dim)
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
        patches = patches.reshape(B, h, w, self.in_chans, p, p)
        patches = patches.permute(0, 3, 1, 4, 2, 5)   # (B, C, h, p, w, p)
        return patches.reshape(B, self.in_chans, h * p, w * p)

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
        self.enc_norm = nn.LayerNorm(embed)

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
        self.dec_norm  = nn.LayerNorm(decoder_dim)
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
        patches = patches.reshape(B, h, w, self.in_chans, p, p)
        patches = patches.permute(0, 3, 1, 4, 2, 5)
        return patches.reshape(B, self.in_chans, h * p, w * p)

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
        self.vit_norm = nn.LayerNorm(128)
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
        bottleneck = tokens.transpose(1, 2).reshape(B, C, H4, W4)

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
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
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

        x = x.reshape(B, Hp // ws, ws, Wp // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()   # (B, nH, nW, ws, ws, C)
        windows = x.reshape(-1, ws * ws, C)              # (B*nH*nW, ws^2, C)
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
        x  = windows.reshape(B, Hp // ws, Wp // ws, ws, ws, -1)
        x  = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, Hp, Wp, -1)
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
        self.norm = nn.LayerNorm(dim)

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
