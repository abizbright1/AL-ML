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
# Nesting has a consequence Equation (3) does not state, and it is NOT the
# naive one. Because every term carries its own 1/|M_r|, the coefficient on a
# single voxel x is
#
#     sum_r  w_r * M_r(x) / |M_r|
#
# so an ET voxel contributes  1/|WT| + 2/|TC| + 3/|ET| , a TC-but-not-ET voxel
# 1/|WT| + 2/|TC| , and oedema 1/|WT| . The effective emphasis therefore
# depends on the three region SIZES and varies slice by slice -- it is not a
# fixed 6:3:1 ratio. The manuscript should state this; the implementation
# should not change because of it.
#
# `nested` is the method that produced the published results and is the
# default for the primary arm. `disjoint` exists only for the PL-Disjoint
# ablation, which asks a different question and must never be presented as a
# correction to the primary method.

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


class FlatROIL1(nn.Module):
    """One binary lesion mask, one normalised term. ROIRecNet-INSPIRED.

        L_flat = lambda * ||M_WT (X^ - X)||_1 / |M_WT|

    Deliberately mirrors the STRUCTURE of RegionWeightedL1 with the region
    count reduced from three to one, so the contrast between this arm and the
    primary arm isolates exactly one thing: whether splitting the tumour into
    nested sub-regions with separate normalisation and separate clinical
    priorities buys anything over "weight the tumour".

    If the two arms match, the benefit comes from ROI-aware restoration
    itself, which Sun et al. established in 2019.

    NOT a reproduction of ROIRecNet. Sun et al. derive the ROI from a
    segmentation network and fine-tune with a binary weighted L2; this arm
    uses the ground-truth mask, an L1 penalty, and no fine-tuning stage. Label
    it "ROIRecNet-inspired" in the paper, never "ROIRecNet".
    """

    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                seg_map: torch.Tensor) -> torch.Tensor:
        err = (pred - target).abs()
        mask = whole_tumour_mask(seg_map)
        n = mask.sum()
        if n.item() == 0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        return self.lam * (err * mask).sum() / (n * pred.shape[1])


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


class CurrentPathologyLossTrainer(_BaseTrainer):
    """PRIMARY ARM: PathologyLoss with the NESTED ET/TC/WT masks.

        L_total = L1(global) + L_path

    with w_WT = 1, w_TC = 2, w_ET = 3 and each region normalised by its own
    size, over the nested BraTS regions ET subset TC subset WT.

    This is the method under test. It is NOT redesigned here: the benchmark
    exists to compare other objectives against the published method, so the
    masks stay nested and the weights stay as they are. A disjoint variant is
    available as the separately named PL-Disjoint ablation, which answers a
    different question and must never be presented as a correction.

    One caveat the regression test will surface: the original trainers wrap
    this term in PPMAELoss, which also carries an SSIM term in its global loss
    and a cross-modal consistency term, and emits w_r from an untrained
    network rather than as constants. This class implements Equation (3) as
    the manuscript writes it. The two therefore will NOT agree numerically --
    see verify_pathologyloss.py, which quantifies the gap so you can decide
    which object the paper benchmarks.
    """

    def __init__(self, model, device='cpu', lr=1e-4,
                 mode='nested', weights=None, lambda_path=1.0):
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


class FlatROITrainer(_BaseTrainer):
    """PRIMARY COMPETITOR: one lesion region instead of three.

        L_total = L1 + lambda * ||M_WT (X^ - X)||_1 / |M_WT|

    Same structure as the primary arm with the region count reduced to one.
    THE DECISIVE CONTRAST. If this matches CurrentPathologyLossTrainer, the
    nested hierarchy, the separate per-region normalisation and the 1:2:3
    priorities contribute nothing measurable, and the remaining novelty is
    ROI-aware restoration itself -- published by Sun et al. in 2019.

    ROIRecNet-inspired, not a reproduction. Do not name Sun et al. as the
    arm's identity in the paper.
    """

    def __init__(self, model, device='cpu', lr=1e-4, lam=1.0):
        super().__init__(model, device, lr)
        self.roi = FlatROIL1(lam=lam)

    def step(self, batch: Dict) -> Dict[str, float]:
        self.model.train()
        noisy = batch['noisy'].to(self.device)
        target = batch['target'].to(self.device)
        seg = batch['seg'].to(self.device)

        pred = self.model(noisy)
        l_global = F.l1_loss(pred, target)
        l_roi = self.roi(pred, target, seg)
        total = l_global + l_roi
        self._backward(total)
        return {'total': total.item(),
                'global': l_global.item(), 'roi': l_roi.item()}


class L1SSIMTrainer(_BaseTrainer):
    """CONVENTIONAL COMPETITOR: structure-aware reconstruction, no lesion prior.

        L_total = L1 + lambda_ssim * SSIMLoss

    Answers the question a reviewer asks before any lesion argument: does
    PathologyLoss beat simply using a stronger, standard structural objective?

    It matters doubly here. The ORIGINAL +PathologyLoss trainers already carry
    an SSIM term at weight 0.5 inside PPMAELoss while their L1 controls do not,
    so part of the published effect may be this term rather than the region
    weighting. This arm measures that part directly.
    """

    def __init__(self, model, device='cpu', lr=1e-4, lambda_ssim=0.5, channels=4):
        super().__init__(model, device, lr)
        from losses import SSIMLoss
        self.ssim = SSIMLoss(channel=channels).to(device)
        self.lambda_ssim = lambda_ssim

    def step(self, batch: Dict) -> Dict[str, float]:
        self.model.train()
        noisy = batch['noisy'].to(self.device)
        target = batch['target'].to(self.device)

        pred = self.model(noisy)
        l_global = F.l1_loss(pred, target)
        l_ssim = self.ssim(pred, target)
        total = l_global + self.lambda_ssim * l_ssim
        self._backward(total)
        return {'total': total.item(),
                'global': l_global.item(), 'ssim': l_ssim.item()}


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

    FROZEN BY DEFAULT (`seg_every=0`). A co-trained auxiliary network changes
    the restorer and the network generating its loss at the same time, which
    makes the result hard to attribute. Frozen, the arm asks one clean
    question: if the restorer is optimised directly for a fixed segmentation
    objective, does that beat optimising for a mask prior?

    Set `seg_every=4` for the alternating, LIDnet-style co-training variant,
    and report it as a separate arm rather than as this one.
    """

    def __init__(self, model, aux_segmentor, device='cpu', lr=1e-4,
                 lambda_task=0.5, seg_lr=5e-4, seg_every=0, n_classes=4):
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
        if self.seg_every and self._i % self.seg_every == 0:
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

    for name, fn in [('RegionWeightedL1 nested',   RegionWeightedL1('nested')),
                     ('RegionWeightedL1 disjoint', RegionWeightedL1('disjoint')),
                     ('FlatROIL1',                 FlatROIL1(lam=1.0))]:
        v = fn(pred, target, seg)
        assert v.ndim == 0 and torch.isfinite(v), name
        v.backward(retain_graph=True)
        print(f'{name:<26} ok   value={v.item():.5f}')

    # empty-region safety: a slice with no ET must not blow up
    empty = torch.zeros(B, 1, H, W, dtype=torch.long)
    v = RegionWeightedL1('disjoint')(pred, target, empty)
    assert torch.isfinite(v) and v.item() == 0.0, 'empty masks must give 0, not NaN'
    print('empty regions   ok   (returns 0, no division by zero)')

    # per-voxel coefficient under nested masks depends on region SIZES
    n = region_masks(seg, 'nested')
    sizes = {k: v.sum().item() for k, v in n.items()}
    coef_et = 1/sizes['WT'] + 2/sizes['TC'] + 3/sizes['ET']
    coef_ed = 1/sizes['WT']
    print(f"region sizes    WT={sizes['WT']:.0f}  TC={sizes['TC']:.0f}  "
          f"ET={sizes['ET']:.0f}")
    print(f'per-voxel coefficient  ET={coef_et:.5f}  oedema={coef_ed:.5f}  '
          f'ratio={coef_et/coef_ed:.1f}x')
    print('  (size-dependent, not a fixed 6:3:1 -- state this in the paper)')

    print('\nall checks passed')


if __name__ == '__main__':
    _self_test()
