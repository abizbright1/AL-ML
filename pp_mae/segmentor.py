"""
segmentor.py — Standalone U-Net segmentation model
====================================================
Used to evaluate DOWNSTREAM segmentation quality after denoising.

The pipeline:
  1. Train Segmentor on CLEAN images (ground-truth MRI → seg labels)
  2. Freeze Segmentor weights
  3. Pass denoised outputs from each denoising model through the Segmentor
  4. Compare resulting Dice / IoU / Accuracy vs. ground-truth labels

This is the standard way to show whether denoising helps downstream tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


# ── U-Net Segmentor ───────────────────────────────────────────────────────────

class _DC(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True),
            nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True),
        )
    def forward(self, x): return self.net(x)


class UNetSegmentor(nn.Module):
    """
    4-class U-Net: output logits shape (B, n_classes, H, W)
    Labels: 0=BG, 1=NCR, 2=ED, 3=ET  (BraTS convention)
    """
    def __init__(self, in_channels: int = 4, n_classes: int = 4, base_ch: int = 32):
        super().__init__()
        b = base_ch
        self.e1 = _DC(in_channels, b)
        self.e2 = _DC(b,   b*2)
        self.e3 = _DC(b*2, b*4)
        self.bt = _DC(b*4, b*8)
        self.u3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(b*8, b*4, kernel_size=3, padding=1))
        self.d3 = _DC(b*8, b*4)
        self.u2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(b*4, b*2, kernel_size=3, padding=1))
        self.d2 = _DC(b*4, b*2)
        self.u1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(b*2, b, kernel_size=3, padding=1))
        self.d1 = _DC(b*2, b)
        self.out = nn.Conv2d(b, n_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b  = self.bt(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(b),  e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)   # (B, n_classes, H, W) logits


# ── Training ──────────────────────────────────────────────────────────────────

class SegTrainer:
    def __init__(self, model: UNetSegmentor, device: str = 'cpu', lr: float = 1e-3):
        self.model  = model.to(device)
        self.device = device
        self.optim  = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def step(self, images: torch.Tensor, labels: torch.Tensor) -> float:
        """
        images : (B, C, H, W)  float32, clean MRI
        labels : (B, H, W)     int64   seg map
        """
        self.model.train()
        images = images.to(self.device)
        labels = labels.to(self.device)
        self.optim.zero_grad()
        logits = self.model(images)
        loss   = self.loss_fn(logits, labels)
        loss.backward()
        self.optim.step()
        return loss.item()


# ── Evaluation metrics ────────────────────────────────────────────────────────

def seg_metrics(pred_logits: torch.Tensor,
                gt_labels:   torch.Tensor,
                eps:         float = 1e-6) -> Dict[str, float]:
    """
    pred_logits : (B, n_classes, H, W) — raw output from UNetSegmentor
    gt_labels   : (B, H, W)            — integer ground-truth labels

    Returns dict with:
      pixel_acc          — overall pixel accuracy
      mean_iou           — mean IoU across non-background classes
      dice_wt / dice_tc / dice_et  — BraTS region Dice scores
      iou_wt  / iou_tc  / iou_et   — BraTS region IoU scores
    """
    pred_cls = pred_logits.argmax(dim=1)   # (B, H, W)

    # ── pixel accuracy ────────────────────────────────────────────────────────
    pixel_acc = (pred_cls == gt_labels).float().mean().item()

    # ── per-class IoU (classes 1,2,3 — skip background) ─────────────────────
    ious = []
    for c in [1, 2, 3]:
        p = (pred_cls == c).float()
        g = (gt_labels == c).float()
        inter = (p * g).sum()
        union = p.sum() + g.sum() - inter
        ious.append((inter / (union + eps)).item())
    mean_iou = float(sum(ious) / len(ious))

    # ── BraTS region masks ───────────────────────────────────────────────────
    def region_dice(pred_mask, gt_mask):
        inter = (pred_mask * gt_mask).sum()
        denom = pred_mask.sum() + gt_mask.sum()
        return (2 * inter / (denom + eps)).item()

    def region_iou(pred_mask, gt_mask):
        inter = (pred_mask * gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum() - inter
        return (inter / (union + eps)).item()

    gt_l = gt_labels
    p_l  = pred_cls

    # WT = any tumour (labels 1,2,3)
    gt_wt = (gt_l  > 0).float();  p_wt = (p_l  > 0).float()
    # TC = NCR + ET (labels 1,3)
    gt_tc = ((gt_l == 1) | (gt_l == 3)).float()
    p_tc  = ((p_l  == 1) | (p_l  == 3)).float()
    # ET = label 3
    gt_et = (gt_l == 3).float();  p_et = (p_l == 3).float()

    return {
        'pixel_acc': pixel_acc,
        'mean_iou':  mean_iou,
        'dice_wt':   region_dice(p_wt, gt_wt),
        'dice_tc':   region_dice(p_tc, gt_tc),
        'dice_et':   region_dice(p_et, gt_et),
        'iou_wt':    region_iou(p_wt,  gt_wt),
        'iou_tc':    region_iou(p_tc,  gt_tc),
        'iou_et':    region_iou(p_et,  gt_et),
    }
