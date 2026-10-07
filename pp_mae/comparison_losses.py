"""
comparison_losses.py — benchmark arms for the PathologyLoss study
=================================================================

Four training objectives that let the PathologyLoss claim be tested against
published alternatives rather than against an unweighted control alone.

    PathologyLossClean   Equation (3) exactly as the manuscript states it:
                         L1 + sum_r w_r * ||M_r (X^ - X)||_1 / |M_r|.
                         No SSIM term, no cross-modal term, no learned
                         weighting. This is what the paper describes.

    BinaryROI            Sun et al. 2019 (ROIRecNet) principle: one binary
                         tumour mask, flat weighting, no per-region
                         normalisation, no sub-region hierarchy.
                         THE DECISIVE CONTROL: if this matches
                         PathologyLossClean, the nested multi-region design
                         contributes nothing and the novelty claim fails.

    ROIFeature           Chen et al. 2021 (LIDnet) principle, adapted: the
                         perceptual term is restricted to the lesion region
                         instead of the whole feature map. LIDnet draws its
                         regions from a detector's RPN over bounding boxes;
                         BraTS supplies segmentation masks and no boxes, so
                         the region comes from the mask and the features come
                         from the frozen segmentor's encoder. An adaptation of
                         the mechanism, not a reimplementation of LIDnet.

    TaskFeedback         LIDnet's central idea: put the downstream task's own
                         loss into the restoration objective, so the restorer
                         is optimised for the task rather than for a prior
                         about where the task looks.

Reference
---------
Sun L, Fan Z, Ding X, Huang Y, Paisley J. Region-of-interest undersampled MRI
reconstruction: a deep convolutional neural network approach. Magnetic
Resonance Imaging 2019;63:185-192.

Chen K, Long K, Ren Y, Sun J, Pu X. Lesion-Inspired Denoising Network:
Connecting Medical Image Denoising and Lesion Detection. arXiv:2104.08845.

Both are cited here for the mechanism being adapted. Neither is reimplemented
faithfully, and no arm below should be presented as "LIDnet" or "ROIRecNet" --
report them as the adapted principles they are.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
#  Region masks
# ═══════════════════════════════════════════════════════════════════════════
#
# BraTS labels on disk after the 4 -> 3 remap: 0=BG, 1=NCR, 2=ED, 3=ET.
# The three clinical regions are NESTED:  ET subset of TC subset of WT.
#
# That nesting has a consequence Equation (3) does not state. Summing three
# terms over nested masks with weights (1, 2, 3) gives every ET pixel a weight
# of 1+2+3 = 6, every NCR pixel 1+2 = 3, and every oedema pixel 1 -- an
# effective ratio of 6:3:1, not the 3:2:1 the manuscript claims.
#
# `disjoint` partitions the tumour so the stated weights are the applied
# weights. `nested` reproduces the original behaviour. Report whichever you
# run, and say which in the paper.

def region_masks(seg_map: torch.Tensor, mode: str = 'disjoint') -> Dict[str, torch.Tensor]:
    """seg_map: (B, 1, H, W) integer labels -> dict of float masks."""
    if mode == 'nested':
        return {
            'WT': (seg_map > 0).float(),
            'TC': ((seg_map == 1) | (seg_map == 3)).float(),
            'ET': (seg_map == 3).float(),
        }
    if mode == 'disjoint':
        return {
            'ED':  (seg_map == 2).float(),   # oedema only          -> w = 1
            'NCR': (seg_map == 1).float(),   # necrotic core only   -> w = 2
            'ET':  (seg_map == 3).float(),   # enhancing tumour     -> w = 3
        }
    raise ValueError(f"mode must be 'disjoint' or 'nested', got {mode!r}")


DEFAULT_WEIGHTS = {
    'disjoint': {'ED': 1.0, 'NCR': 2.0, 'ET': 3.0},
    'nested':   {'WT': 1.0, 'TC': 2.0,  'ET': 3.0},
}


def whole_tumour_mask(seg_map: torch.Tensor) -> torch.Tensor:
    """Single binary lesion mask -- the ROIRecNet-style region of interest."""
    return (seg_map > 0).float()


# ═══════════════════════════════════════════════════════════════════════════
#  Loss terms
# ═══════════════════════════════════════════════════════════════════════════

class RegionWeightedL1(nn.Module):
    """Equation (3): sum_r w_r * ||M_r (X^ - X)||_1 / |M_r|.

    Dividing by |M_r| is the point of the design: without it a region's
    contribution stays proportional to its area, which is the imbalance the
    weights are meant to remove. An empty region contributes zero rather than
    dividing by zero -- a slice with no enhancing tumour simply drops that
    term, it does not get a free perfect score.
    """

    def __init__(self, mode: str = 'disjoint', weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.mode = mode
        self.weights = dict(weights or DEFAULT_WEIGHTS[mode])

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                seg_map: torch.Tensor) -> torch.Tensor:
        err = (pred - target).abs()                      # (B, C, H, W)
        total = torch.zeros((), device=pred.device, dtype=pred.dtype)
        for name, mask in region_masks(seg_map, self.mode).items():
            w = self.weights.get(name)
            if w is None:
                continue
            n = mask.sum()
            if n.item() == 0:
                continue
            total = total + w * (err * mask).sum() / (n * pred.shape[1])
        return total


class BinaryROIL1(nn.Module):
    """ROIRecNet-style: one binary mask, flat extra weight, no normalisation.

    L = mean |X^ - X| over the image, with error inside the lesion counted
    (1 + k) times instead of once. This is the simplest thing that "weight the
    tumour more" can mean, and it is the baseline PathologyLoss has to beat to
    justify nesting, per-region normalisation and clinical priorities.
    """

    def __init__(self, k: float = 2.0):
        super().__init__()
        self.k = k

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                seg_map: torch.Tensor) -> torch.Tensor:
        err = (pred - target).abs()
        w = 1.0 + self.k * whole_tumour_mask(seg_map)
        return (err * w).sum() / (w.sum() * pred.shape[1])


class ROIFeatureLoss(nn.Module):
    """LIDnet-style ROI perceptual loss, adapted to segmentation masks.

    LIDnet computes a feature-space difference inside detector-proposed boxes,
    using the detector's own backbone as the feature extractor. Here the region
    is the ground-truth lesion mask and the extractor is the frozen segmentor's
    encoder, which is the closest analogue available in this pipeline: it is
    the network whose judgement the study is ultimately scored against.

    The segmentor must already be frozen. Its parameters are never updated
    here, and gradients flow only back into the restoration network.
    """

    def __init__(self, segmentor: nn.Module, stage: str = 'e3'):
        super().__init__()
        self.seg = segmentor
        self.stage = stage
        for p in self.seg.parameters():
            p.requires_grad_(False)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        s = self.seg
        e1 = s.e1(x)
        if self.stage == 'e1':
            return e1
        e2 = s.e2(s.pool(e1))
        if self.stage == 'e2':
            return e2
        e3 = s.e3(s.pool(e2))
        if self.stage == 'e3':
            return e3
        return s.bt(s.pool(e3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                seg_map: torch.Tensor) -> torch.Tensor:
        f_pred = self._features(pred)
        with torch.no_grad():
            f_true = self._features(target)

        mask = whole_tumour_mask(seg_map)
        mask = F.interpolate(mask, size=f_pred.shape[-2:], mode='nearest')
        n = mask.sum()
        if n.item() == 0:                       # no lesion in this batch
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        return ((f_pred - f_true) ** 2 * mask).sum() / (n * f_pred.shape[1])


def soft_dice_loss(logits: torch.Tensor, labels: torch.Tensor,
                   n_classes: int = 4, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable Dice over the three foreground classes."""
    probs = torch.softmax(logits, dim=1)
    onehot = F.one_hot(labels.long(), n_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice[1:].mean()                # skip background


# ═══════════════════════════════════════════════════════════════════════════
#  Trainers
# ═══════════════════════════════════════════════════════════════════════════
#
# Every trainer below matches the interface run_full_experiment.py expects:
#     trainer = trainer_fn(model)
#     losses  = trainer.step(batch)      # dict carrying 'total'
#     pred    = trainer.predict(noisy)
# and optimises ONLY the restoration network, so each arm differs from the L1
# control in its objective and in nothing else.

class _BaseTrainer:
    def __init__(self, model: nn.Module, device: str = 'cpu', lr: float = 1e-4):
        self.model = model.to(device)
        self.device = device
        self.optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    def _backward(self, loss: torch.Tensor) -> None:
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()

    @torch.no_grad()
    def predict(self, noisy: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(noisy.to(self.device)).cpu()


class PathologyLossCleanTrainer(_BaseTrainer):
    """ARM: PathologyLoss as Equation (3) states it.

    L_total = L1(global) + L_path

    Nothing else. In particular no SSIM term in the global loss, no
    cross-modal consistency term, and fixed numeric weights rather than
    weights emitted by a network. Use this arm for every PathologyLoss number
    the manuscript reports, so the equation and the experiment agree.
    """

    def __init__(self, model, device='cpu', lr=1e-4,
                 mode='disjoint', weights=None, lambda_path=1.0):
        super().__init__(model, device, lr)
        self.path = RegionWeightedL1(mode=mode, weights=weights)
        self.lambda_path = lambda_path

    def step(self, batch: Dict) -> Dict[str, float]:
        self.model.train()
        noisy = batch['noisy'].to(self.device)
        target = batch['target'].to(self.device)
        seg = batch['seg'].to(self.device)

        pred = self.model(noisy)
        l_global = F.l1_loss(pred, target)
        l_path = self.path(pred, target, seg)
        total = l_global + self.lambda_path * l_path
        self._backward(total)
        return {'total': total.item(),
                'global': l_global.item(),
                'path': l_path.item()}


class BinaryROITrainer(_BaseTrainer):
    """ARM: binary ROI weighting (ROIRecNet principle).

    One lesion mask, flat weight, no normalisation, no hierarchy. If this
    performs as well as PathologyLossCleanTrainer, the paper's contribution
    reduces to "weight the tumour", which was published in 2019.
    """

    def __init__(self, model, device='cpu', lr=1e-4, k=2.0):
        super().__init__(model, device, lr)
        self.loss_fn = BinaryROIL1(k=k)

    def step(self, batch: Dict) -> Dict[str, float]:
        self.model.train()
        noisy = batch['noisy'].to(self.device)
        target = batch['target'].to(self.device)
        seg = batch['seg'].to(self.device)

        pred = self.model(noisy)
        total = self.loss_fn(pred, target, seg)
        self._backward(total)
        return {'total': total.item()}


class ROIFeatureTrainer(_BaseTrainer):
    """ARM: ROI-restricted perceptual loss (LIDnet principle, adapted).

    L_total = L1 + lambda * ||phi(X^) - phi(X)||^2 restricted to the lesion.

    Isolates feature-space region weighting from the pixel-space region
    weighting of PathologyLoss. The segmentor passed in must be the frozen
    evaluation segmentor or a copy of it.
    """

    def __init__(self, model, segmentor, device='cpu', lr=1e-4,
                 lambda_feat=1.0, stage='e3'):
        super().__init__(model, device, lr)
        self.feat = ROIFeatureLoss(segmentor.to(device), stage=stage)
        self.lambda_feat = lambda_feat

    def step(self, batch: Dict) -> Dict[str, float]:
        self.model.train()
        noisy = batch['noisy'].to(self.device)
        target = batch['target'].to(self.device)
        seg = batch['seg'].to(self.device)

        pred = self.model(noisy)
        l_global = F.l1_loss(pred, target)
        l_feat = self.feat(pred, target, seg)
        total = l_global + self.lambda_feat * l_feat
        self._backward(total)
        return {'total': total.item(),
                'global': l_global.item(),
                'feat': l_feat.item()}


class TaskFeedbackTrainer(_BaseTrainer):
    """ARM: downstream task loss inside the restoration objective (LIDnet core).

    L_total = L1 + lambda * Dice(aux_segmentor(X^), Y)

    The restorer is optimised for what the segmentor actually needs, rather
    than for a prior about where the segmentor looks. This is the strongest
    version of the idea PathologyLoss approximates with a mask.

    LEAKAGE CONTROL -- the reason there are two segmentors.
    The segmentor inside this loop is an AUXILIARY copy, trained here on
    restored images. The frozen segmentor used to score every arm must be a
    different object, never updated by this trainer. Scoring with the network
    the restorer was trained against would measure how well the two co-adapted,
    not how well the images support segmentation. LIDnet handles this the same
    way: its reported AP comes from a detector pre-trained on normal-dose CT
    and held fixed (Section 4.1).

    Training alternates in LIDnet's style: the restorer updates every step, the
    auxiliary segmentor every `seg_every` steps on detached restored images.
    """

    def __init__(self, model, aux_segmentor, device='cpu', lr=1e-4,
                 lambda_task=0.5, seg_lr=5e-4, seg_every=4, n_classes=4):
        super().__init__(model, device, lr)
        self.aux = aux_segmentor.to(device)
        self.aux_optim = torch.optim.Adam(self.aux.parameters(), lr=seg_lr)
        self.lambda_task = lambda_task
        self.seg_every = seg_every
        self.n_classes = n_classes
        self._i = 0

    def step(self, batch: Dict) -> Dict[str, float]:
        noisy = batch['noisy'].to(self.device)
        target = batch['target'].to(self.device)
        labels = batch['seg'][:, 0].long().to(self.device)

        # ---- restorer update: auxiliary segmentor frozen -------------------
        self.model.train()
        self.aux.eval()
        for p in self.aux.parameters():
            p.requires_grad_(False)

        pred = self.model(noisy)
        l_global = F.l1_loss(pred, target)
        l_task = soft_dice_loss(self.aux(pred), labels, self.n_classes)
        total = l_global + self.lambda_task * l_task
        self._backward(total)

        # ---- auxiliary segmentor update on detached restorations ----------
        self._i += 1
        l_aux = float('nan')
        if self._i % self.seg_every == 0:
            for p in self.aux.parameters():
                p.requires_grad_(True)
            self.aux.train()
            self.aux_optim.zero_grad()
            loss_aux = F.cross_entropy(self.aux(pred.detach()), labels)
            loss_aux.backward()
            self.aux_optim.step()
            l_aux = loss_aux.item()

        return {'total': total.item(),
                'global': l_global.item(),
                'task': l_task.item(),
                'aux': l_aux}


# ═══════════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════════

def _self_test() -> None:
    """Shape and sanity checks. Run: python comparison_losses.py"""
    torch.manual_seed(0)
    B, C, H, W = 2, 4, 32, 32
    pred = torch.rand(B, C, H, W, requires_grad=True)
    target = torch.rand(B, C, H, W)
    seg = torch.zeros(B, 1, H, W, dtype=torch.long)
    seg[:, :, 4:12, 4:12] = 2      # oedema
    seg[:, :, 6:10, 6:10] = 1      # necrotic core
    seg[:, :, 7:9, 7:9] = 3        # enhancing tumour

    # effective weight check: disjoint applies the stated weights
    m = region_masks(seg, 'disjoint')
    assert (m['ET'] * m['NCR']).sum() == 0, 'disjoint masks must not overlap'
    assert (m['ET'] * m['ED']).sum() == 0, 'disjoint masks must not overlap'
    n = region_masks(seg, 'nested')
    assert (n['ET'] * n['TC']).sum() > 0, 'nested masks are expected to overlap'
    print('masks           ok   (disjoint are exclusive, nested overlap)')

    for name, fn in [('RegionWeightedL1 disjoint', RegionWeightedL1('disjoint')),
                     ('RegionWeightedL1 nested',   RegionWeightedL1('nested')),
                     ('BinaryROIL1',               BinaryROIL1(k=2.0))]:
        v = fn(pred, target, seg)
        assert v.ndim == 0 and torch.isfinite(v), name
        v.backward(retain_graph=True)
        print(f'{name:<26} ok   value={v.item():.5f}')

    # empty-region safety: a slice with no ET must not blow up
    empty = torch.zeros(B, 1, H, W, dtype=torch.long)
    v = RegionWeightedL1('disjoint')(pred, target, empty)
    assert torch.isfinite(v) and v.item() == 0.0, 'empty masks must give 0, not NaN'
    print('empty regions   ok   (returns 0, no division by zero)')

    # nested really does give ET a 6x effective weight
    w_nested = sum(DEFAULT_WEIGHTS['nested'].values())
    print(f'effective ET weight under nested masks = {w_nested:.0f}x '
          f'(manuscript claims 3x)')

    print('\nall checks passed')


if __name__ == '__main__':
    _self_test()
