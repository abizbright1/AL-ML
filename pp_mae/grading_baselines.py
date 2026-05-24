"""
grading_baselines.py — Literature Grading Baselines (2023–2026)
===============================================================
Three state-of-the-art brain tumour grading models for comparison
against the PP-MAE grading pipeline.

1. RadioTransformer  (Bhalerao et al., TMI 2022 / updated TMI 2024)
   ─────────────────────────────────────────────────────────────────
   Our 7 radiomics features → learnable token embeddings →
   multi-head self-attention → MLP head → GBM probability.

   Why it matters:  Standard MLP (our GradingHead) treats all 7 features
   independently.  RadioTransformer learns INTERACTIONS between features
   (e.g., high V_ET AND high H_ET together → much stronger GBM signal
   than either alone).  Self-attention captures these joint patterns.

2. CBAM-ResNet  (Woo et al., ECCV 2018 / TMI 2023 brain tumour adaptation)
   ─────────────────────────────────────────────────────────────────────────
   Trains END-TO-END on denoised 4-channel MRI slices.
   Dual attention: channel attention (which modality matters?) +
   spatial attention (which location matters?).

   Why it matters:  Our GradingHead + RadioTransformer both rely on our
   hand-crafted 7 features.  CBAM-ResNet learns its own features directly
   from pixels.  This tests whether learned features beat hand-crafted ones.

3. DINOv2 Linear Probe  (Oquab et al., ICCV 2023 / MICCAI 2024 use)
   ──────────────────────────────────────────────────────────────────
   Frozen DINOv2 ViT-S/14 (self-supervised, 142M images) → 384-dim
   features from denoised MRI slices → subject-level aggregation →
   logistic regression / 1-layer MLP.

   Why it matters:  DINOv2 is the current best feature extractor for
   transfer learning without labels.  If it beats our domain-specific
   approach on BraTS, that suggests more pre-training data matters more
   than domain knowledge.

   Fallback:  If DINOv2 is not available (no internet / first run),
   automatically uses LightViT — a small ViT we implement from scratch
   that represents the same architecture family.

All three are slice-level models (except RadioTransformer which is
subject-level like GradingHead) and share a common SliceGradingTrainer.

Interface
─────────
    # Subject-level (RadioTransformer):
    trainer = RadioTransformerTrainer(model, device='cpu')
    trainer.step(features_7dim, labels)   # same as GradingTrainer

    # Slice-level (CBAM-ResNet, DINOv2Probe):
    trainer = SliceGradingTrainer(model, device='cpu')
    trainer.step(mri_slices, slice_labels)
    probs   = trainer.predict_subject(slice_dataset, subject_name)
"""

import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ═════════════════════════════════════════════════════════════════════════════
# 1.  RadioTransformer
#     Bhalerao et al. "RadioTransformer: A Cascaded Global-Focal Transformer
#     for Visual Attention-Guided Disease Classification."
#     TMI 2022 / adapted for glioma grading TMI 2024.
# ═════════════════════════════════════════════════════════════════════════════

class FeatureTokenizer(nn.Module):
    """
    Projects each scalar radiomics feature to a d_model-dim embedding.

    Each of the 7 features (V_WT, V_TC, V_ET, ρ, H_WT, H_TC, H_ET) becomes
    one token in a sequence of length 7.  A learnable positional encoding
    is added so the Transformer knows which feature each token represents.

    Why not just embed the whole 7-vector?
        Embedding features individually lets multi-head attention compute
        pairwise interactions between any two features (e.g., V_ET ↔ ρ).
        A single linear projection loses this structure.
    """
    def __init__(self, n_features: int = 7, d_model: int = 64):
        super().__init__()
        # One linear projection per feature: scalar → d_model
        self.proj = nn.Linear(1, d_model)
        # Learned positional encodings: one per feature position
        self.pos  = nn.Embedding(n_features, d_model)
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, n_features)
        Returns : (B, n_features, d_model)  — sequence of feature tokens
        """
        B = x.shape[0]
        # Project each feature to d_model (unsqueeze → (B,7,1) → (B,7,d))
        tokens = self.proj(x.unsqueeze(-1))                # (B, 7, d_model)
        # Add positional encoding so attention knows feature identity
        pos_ids = torch.arange(self.n_features, device=x.device)
        tokens  = tokens + self.pos(pos_ids).unsqueeze(0)  # (B, 7, d_model)
        return tokens


class RadioTransformer(nn.Module):
    """
    Self-attention over radiomics feature tokens → GBM probability.

    Architecture (per Bhalerao et al. adapted for 7-feature input):
        Input:  (B, 7) radiomics features
        ↓  FeatureTokenizer  →  (B, 7, d_model)
        ↓  TransformerEncoder (n_heads, d_model, n_layers)
        ↓  [CLS token] global pooling via mean
        ↓  LayerNorm → Linear → Sigmoid
        Output: (B,) GBM probabilities

    Key hyperparameters:
        d_model   : token embedding size (64 — small for 7-feature input)
        n_heads   : attention heads (4 — each attends to 16-dim subspace)
        n_layers  : transformer depth (2 — sufficient for 7 tokens)
        dropout   : regularisation (0.1 — small model, small dataset)
    """

    def __init__(
        self,
        n_features: int   = 7,
        d_model:    int   = 64,
        n_heads:    int   = 4,
        n_layers:   int   = 2,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = n_heads,
            dim_feedforward= d_model * 4,
            dropout        = dropout,
            batch_first    = True,    # (B, seq, d) layout
            norm_first     = True,    # Pre-LN: more stable for small models
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 7) → (B,) GBM probabilities"""
        tokens = self.tokenizer(x)              # (B, 7, d_model)
        enc    = self.transformer(tokens)       # (B, 7, d_model)
        pooled = enc.mean(dim=1)               # (B, d_model) — mean over tokens
        pooled = self.norm(pooled)
        return self.head(pooled).squeeze(-1)    # (B,)


class RadioTransformerTrainer:
    """
    Subject-level trainer for RadioTransformer.
    Identical interface to GradingTrainer — drop-in replacement.
    """
    def __init__(
        self,
        model:      RadioTransformer,
        device:     str   = 'cpu',
        lr:         float = 1e-3,
        pos_weight: float = 1.0,
    ):
        self.model   = model.to(device)
        self.device  = device
        pw = torch.tensor([pos_weight], device=device)
        self.loss_fn = nn.BCELoss(weight=pw)
        self.optim   = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=1e-4)

    def step(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        self.model.train()
        features = features.to(self.device)
        labels   = labels.float().to(self.device)
        self.optim.zero_grad()
        preds = self.model(features)
        loss  = self.loss_fn(preds, labels)
        loss.backward()
        self.optim.step()
        return loss.item()

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(features.to(self.device)).cpu()


# ═════════════════════════════════════════════════════════════════════════════
# 2.  CBAM-ResNet
#     Woo et al. "CBAM: Convolutional Block Attention Module." ECCV 2018.
#     Brain tumour grading adaptation: TMI 2023.
# ═════════════════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    """
    Channel attention: WHAT features to emphasise.

    For 4-channel MRI:
        "Should I pay more attention to T1CE (contrast) or FLAIR (oedema)
         for this particular patient?"

    Method (Woo et al.):
        Global Average Pool + Global Max Pool → shared MLP → sigmoid weights
        Using both average AND max avoids losing either fine details (max)
        or overall statistics (average).
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        avg = F.adaptive_avg_pool2d(x, 1)   # (B, C, 1, 1)
        mx  = F.adaptive_max_pool2d(x, 1)   # (B, C, 1, 1)
        # Both paths share the same MLP (tied weights — reduces parameters)
        w   = torch.sigmoid(
            self.mlp(avg).unsqueeze(-1).unsqueeze(-1) +
            self.mlp(mx).unsqueeze(-1).unsqueeze(-1)
        )                                    # (B, C, 1, 1)
        return x * w                         # channel-scaled feature map


class SpatialAttention(nn.Module):
    """
    Spatial attention: WHERE to look.

    "Focus on the tumour region, not the skull."

    Method (Woo et al.):
        Channel-wise avg + max → 2-channel map → 7×7 conv → sigmoid weights
        7×7 kernel chosen to capture a large spatial context.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise statistics → spatial descriptor
        avg = x.mean(dim=1, keepdim=True)    # (B, 1, H, W)
        mx  = x.max(dim=1, keepdim=True).values
        desc = torch.cat([avg, mx], dim=1)   # (B, 2, H, W)
        w    = torch.sigmoid(self.conv(desc))# (B, 1, H, W)
        return x * w                         # spatially scaled feature map


class CBAM(nn.Module):
    """
    CBAM: Channel Attention → Spatial Attention (sequential application).

    Sequential is better than parallel because channel attention first
    selects WHAT features are relevant, then spatial attention decides
    WHERE they are relevant.
    """
    def __init__(self, channels: int, reduction: int = 4, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


class _CBAMBlock(nn.Module):
    """Residual block with CBAM attention."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch))
        self.cbam   = CBAM(out_ch)
        # Shortcut: match dimensions if stride or channels changed
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch)
        ) if stride != 1 or in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        out = self.cbam(out)                  # apply dual attention
        return self.relu(out + self.shortcut(x))


class CBAMResNet(nn.Module):
    """
    Lightweight ResNet with CBAM dual-attention for glioma grading.

    4 stages of residual blocks (stride-2 downsampling at each).
    Significantly smaller than ResNet-18 (~2M params vs 11M) — appropriate
    for BraTS cohort sizes (100–500 subjects).

    Input:  (B, 4, H, W)  denoised 4-channel MRI slice
    Output: (B,)          GBM probability

    Trains SLICE-LEVEL: each slice gets the subject's grade label.
    Inference: aggregate slice predictions to subject level.
    """
    def __init__(self, in_channels: int = 4, base_ch: int = 32):
        super().__init__()
        b = base_ch
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, b, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1))
        self.layer1 = _CBAMBlock(b,    b,    stride=1)
        self.layer2 = _CBAMBlock(b,    b*2,  stride=2)
        self.layer3 = _CBAMBlock(b*2,  b*4,  stride=2)
        self.layer4 = _CBAMBlock(b*4,  b*8,  stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(b*8, b*2),
            nn.ReLU(),
            nn.Linear(b*2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.head(x).squeeze(-1)   # (B,)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  DINOv2 Linear Probe
#     Oquab et al. "DINOv2: Learning Robust Visual Features without Supervision"
#     ICCV 2023 / TMLR 2024.  Applied to brain tumour grading: MICCAI 2024.
# ═════════════════════════════════════════════════════════════════════════════

class LightViT(nn.Module):
    """
    Lightweight Vision Transformer — fallback when DINOv2 is unavailable.

    Represents the same ViT architecture family as DINOv2 but
    implemented from scratch with no pre-training.  Useful for:
        • Demo mode (no internet)
        • Ablation: pre-training vs random init

    Architecture: 4×4 patch embedding → 6-layer ViT → [CLS] → head
    Input:  (B, 1, H, W)  single-channel grayscale MRI slice
    Output: (B, embed_dim)  feature vector
    """
    def __init__(self, img_size: int = 96, patch: int = 8,
                 embed: int = 128, depth: int = 4, heads: int = 4):
        super().__init__()
        n_patches   = (img_size // patch) ** 2
        self.patch  = nn.Conv2d(1, embed, patch, stride=patch)
        self.cls    = nn.Parameter(torch.zeros(1, 1, embed))
        self.pos    = nn.Parameter(torch.zeros(1, n_patches + 1, embed))
        enc_layer   = nn.TransformerEncoderLayer(
            embed, heads, embed * 4, 0.1, batch_first=True, norm_first=True)
        self.enc    = nn.TransformerEncoder(enc_layer, depth)
        self.norm   = nn.LayerNorm(embed)
        self.embed_dim = embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) → (B, embed_dim) CLS token"""
        B  = x.shape[0]
        x  = self.patch(x).flatten(2).transpose(1, 2)  # (B, N, embed)
        cls = self.cls.expand(B, -1, -1)
        x  = torch.cat([cls, x], dim=1) + self.pos     # (B, N+1, embed)
        x  = self.norm(self.enc(x))
        return x[:, 0]                                  # (B, embed) CLS token


class DINOv2Probe(nn.Module):
    """
    DINOv2 (ViT-S/14) frozen feature extractor + trainable linear probe.

    Pipeline:
        Denoised MRI (4 channels) → grayscale (mean across channels) →
        resize to 224×224 (DINOv2 input size) →
        frozen DINOv2 ViT-S/14 → 384-dim feature vector →
        subject-level mean aggregation →
        trainable 2-layer MLP → GBM probability

    Why grayscale?
        DINOv2 was pre-trained on 3-channel RGB images.  Converting 4-channel
        MRI to 1-channel grayscale and repeating 3× is the standard
        adaptation used in MICCAI 2024 glioma grading papers.

    Fallback:
        If DINOv2 unavailable (no internet / first run), uses LightViT —
        same interface but randomly initialised small ViT.  Clearly labelled
        as fallback in output.
    """

    DINO_MODEL  = 'facebookresearch/dinov2'
    DINO_NAME   = 'dinov2_vits14'
    DINO_DIM    = 384   # ViT-S/14 output dimension

    def __init__(self, img_size: int = 96, use_dino: bool = True,
                 device: str = 'cpu'):
        super().__init__()
        self.img_size   = img_size
        self.using_dino = False
        self.embed_dim  = self.DINO_DIM

        if use_dino:
            try:
                self.backbone = torch.hub.load(
                    self.DINO_MODEL, self.DINO_NAME, verbose=False)
                self.backbone.eval()
                for p in self.backbone.parameters():
                    p.requires_grad_(False)
                self.using_dino = True
                print("[DINOv2Probe] ✅ DINOv2 ViT-S/14 loaded from torch.hub",
                      flush=True)
            except Exception as e:
                warnings.warn(
                    f"[DINOv2Probe] ⚠️ DINOv2 unavailable ({e}). "
                    f"Using LightViT fallback.")

        if not self.using_dino:
            self.backbone  = LightViT(img_size=img_size)
            self.embed_dim = self.backbone.embed_dim
            print("[DINOv2Probe] 📌 Using LightViT fallback (no pre-training)",
                  flush=True)

        # Trainable linear probe head (small MLP)
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 4, H, W) MRI slices
        Returns: (B, embed_dim) feature vectors
        """
        # Convert 4-channel MRI → 1-channel grayscale → repeat to 3 channels
        gray = x.mean(dim=1, keepdim=True)        # (B, 1, H, W)

        if self.using_dino:
            # DINOv2 expects 3-channel 224×224
            rgb  = gray.repeat(1, 3, 1, 1)        # (B, 3, H, W)
            rgb  = F.interpolate(rgb, (224, 224),
                                 mode='bilinear', align_corners=False)
            with torch.no_grad():
                feats = self.backbone(rgb)         # (B, 384)
        else:
            feats = self.backbone(gray)            # (B, embed_dim)

        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 4, H, W) → (B,) GBM probabilities"""
        feats = self.extract_features(x)
        return self.head(feats).squeeze(-1)        # (B,)


# ═════════════════════════════════════════════════════════════════════════════
# Shared Slice-Level Trainer (for CBAM-ResNet and DINOv2Probe)
# ═════════════════════════════════════════════════════════════════════════════

class SliceGradingTrainer:
    """
    Trainer for slice-level grading models (CBAM-ResNet, DINOv2Probe).

    Key design decisions:
    ─────────────────────
    1. LABEL PROPAGATION: each slice receives its subject's grade label.
       So a GBM patient's 155 axial slices all have label=1.
       This is standard practice in the BraTS grading literature.

    2. SUBJECT-LEVEL INFERENCE: at test time, predict on all slices of a
       subject, then aggregate with MAX (most-likely-GBM slice wins).
       MAX is preferred over MEAN because a single clearly-enhancing slice
       is sufficient evidence for GBM — consistent with clinical practice.

    3. POS_WEIGHT: BCELoss class weighting to handle label imbalance
       (more slices per subject doesn't mean more GBM subjects).

    Args:
        model      : CBAMResNet or DINOv2Probe
        device     : 'cpu' or 'cuda'
        lr         : learning rate (default 1e-4 for CNN, 1e-3 for probe)
        pos_weight : scale loss for positive (GBM) class
    """

    def __init__(
        self,
        model:      nn.Module,
        device:     str   = 'cpu',
        lr:         float = 1e-4,
        pos_weight: float = 1.0,
    ):
        self.model   = model.to(device)
        self.device  = device
        pw = torch.tensor([pos_weight], device=device)
        self.loss_fn = nn.BCELoss(weight=pw)
        self.optim   = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=1e-4)

    def step(
        self,
        slices: torch.Tensor,   # (B, 4, H, W)  denoised MRI slices
        labels: torch.Tensor,   # (B,)  subject-level labels propagated to slice
    ) -> float:
        self.model.train()
        slices = slices.to(self.device)
        labels = labels.float().to(self.device)
        self.optim.zero_grad()
        preds = self.model(slices)
        loss  = self.loss_fn(preds, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        return loss.item()

    @torch.no_grad()
    def predict_slice(self, slices: torch.Tensor) -> torch.Tensor:
        """(B, 4, H, W) → (B,) slice-level GBM probabilities"""
        self.model.eval()
        return self.model(slices.to(self.device)).cpu()


# ═════════════════════════════════════════════════════════════════════════════
# Slice-level dataset builder and subject-level aggregation
# ═════════════════════════════════════════════════════════════════════════════

def build_slice_dataset(
    loader,            # DataLoader from BraTS dataset
    model_denoiser,    # denoiser (or None for noisy baseline)
    grade_labels: Dict[str, int],
    subjects:     set,
    device:       str = 'cpu',
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Collect ALL denoised slices from the given subjects with their
    corresponding subject-level grade labels.

    Returns:
        all_slices  : (N_slices, 4, H, W)  denoised MRI slices
        all_labels  : (N_slices,)           grade labels (0/1) per slice
        all_subjects: [str] * N_slices      subject name per slice
    """
    slices_list, labels_list, subj_list = [], [], []

    if model_denoiser is not None:
        model_denoiser.eval()

    with torch.no_grad():
        for b in loader:
            noisy   = b['noisy']
            seg_gt  = b['seg']
            names   = b['subject']

            # Filter to requested subjects
            keep = [i for i, n in enumerate(names) if n in subjects]
            if not keep:
                continue

            noisy   = noisy[keep]
            seg_gt  = seg_gt[keep]
            names   = [names[i] for i in keep]

            # Denoise
            if model_denoiser is None:
                denoised = noisy
            else:
                try:
                    denoised = model_denoiser(
                        noisy.to(device), seg_gt.to(device)).cpu()
                except TypeError:
                    denoised = model_denoiser(noisy.to(device)).cpu()

            for i, subj in enumerate(names):
                label = grade_labels.get(subj, 0)
                slices_list.append(denoised[i])
                labels_list.append(label)
                subj_list.append(subj)

    if not slices_list:
        return torch.zeros(1, 4, 96, 96), torch.zeros(1), ['unknown']

    return (torch.stack(slices_list, dim=0),
            torch.tensor(labels_list).long(),
            subj_list)


def aggregate_to_subject(
    slice_probs:    torch.Tensor,   # (N_slices,)
    subject_names:  List[str],      # subject per slice
    grade_labels:   Dict[str, int],
    subjects:       set,
    mode:           str = 'max',    # 'max' or 'mean'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregate slice-level predictions → subject-level predictions.

    mode='max' : one highly-enhancing slice is sufficient for GBM
    mode='mean': average evidence across all slices

    Returns:
        probs  : (N_subjects,)  subject-level GBM probabilities
        labels : (N_subjects,)  true grade labels
    """
    subj_probs  = defaultdict(list)
    for prob, name in zip(slice_probs.tolist(), subject_names):
        if name in subjects:
            subj_probs[name].append(prob)

    probs_out, labels_out = [], []
    for subj in sorted(subjects):
        if subj not in subj_probs:
            continue
        ps = subj_probs[subj]
        agg = max(ps) if mode == 'max' else float(sum(ps) / len(ps))
        probs_out.append(agg)
        labels_out.append(grade_labels.get(subj, 0))

    if not probs_out:
        return torch.zeros(1), torch.zeros(1)

    return torch.tensor(probs_out), torch.tensor(labels_out).long()
