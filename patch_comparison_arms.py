#!/usr/bin/env python3
"""
patch_comparison_arms.py — register the benchmark arms in run_full_experiment.py
================================================================================

Adds six arms so the PathologyLoss claim can be tested against published
alternatives and against the two references every restoration study needs:

    Noisy (no restoration)   floor   -- does restoration help segmentation at all?
    Clean (oracle)           ceiling -- how much headroom exists?
    PathologyLoss (clean)            -- Equation (3) exactly as written
    Binary ROI               ROIRecNet principle (Sun 2019)
    ROI feature              LIDnet ROI perceptual principle (Chen 2021), adapted
    Task feedback            LIDnet task-loss principle, adapted

Also trains a SECOND segmentor. That is not redundancy -- see below.

Idempotent: re-running is a no-op.

Usage
-----
    python3 patch_comparison_arms.py
    python3 -c "import ast; ast.parse(open('run_full_experiment.py').read())"
"""

import sys

P = 'run_full_experiment.py'
s = open(P).read()

if 'comparison_losses' in s:
    sys.exit('Already patched — comparison arms are present. Nothing to do.')

# ─────────────────────────────────────────────── 1. imports
old = "from segmentor import UNetSegmentor, SegTrainer"
new = """from segmentor import UNetSegmentor, SegTrainer
from comparison_losses import (CurrentPathologyLossTrainer, FlatROITrainer,
                               ROIFeatureTrainer, TaskFeedbackTrainer)"""
assert s.count(old) == 1, 'import anchor not found'
s = s.replace(old, new)

# ─────────────────────────────────────────────── 2. build_models signature
old = "def build_models(device: str) -> Dict[str, dict]:"
new = """def build_models(device: str,
                 seg_eval: 'UNetSegmentor' = None,
                 seg_aux:  'UNetSegmentor' = None) -> Dict[str, dict]:"""
assert s.count(old) == 1, 'build_models signature anchor not found'
s = s.replace(old, new)

# ─────────────────────────────────────────────── 3. the new arms
old = """        'SwinIR-lite (L1)': {"""
new = """        # ── reference arms: no training, inference only ────────────────
        # Without these two numbers every Dice figure in the paper is
        # unanchored. The floor says whether restoration helps segmentation at
        # all; the ceiling says how much of the achievable gap any method
        # closes. Both reuse the frozen evaluation segmentor.
        'Noisy input (no restoration) [FLOOR]': {
            'class':  _Identity,
            'config': {},
            'trainer_fn': lambda m: _NullTrainer(m),
            'infer_fn':   lambda m, noisy, seg: noisy,
            'uses_mask_at_inference': False,
            'no_train': True,
            'short': 'Noisy',
        },
        'Clean image (oracle) [CEILING]': {
            'class':  _Identity,
            'config': {},
            'trainer_fn': lambda m: _NullTrainer(m),
            'infer_fn':   lambda m, noisy, seg, target=None: target,
            'needs_target': True,
            'uses_mask_at_inference': False,
            'no_train': True,
            'short': 'Clean',
        },

        # ── PathologyLoss exactly as Equation (3) states it ────────────────
        # No SSIM term, no cross-modal term, fixed numeric weights, disjoint
        # masks so the stated 3:2:1 is the applied 3:2:1. Quote THIS arm for
        # every PathologyLoss number in the manuscript.
        'SwinIR + PathologyLoss (current, nested)': {
            'class':  SwinIRLite,
            'config': dict(in_ch=4, dim=64, n_blocks=4, window_size=4),
            'trainer_fn': lambda m: CurrentPathologyLossTrainer(
                m, device=device, lr=1e-4, mode='nested'),
            'infer_fn':   lambda m, noisy, seg: m(noisy),
            'uses_mask_at_inference': False,
            'short': 'SwinIR+PLcur',
        },
        'Uformer + PathologyLoss (current, nested)': {
            'class':  UformerLite,
            'config': dict(in_ch=4, dim=32, window_size=4),
            'trainer_fn': lambda m: CurrentPathologyLossTrainer(
                m, device=device, lr=1e-4, mode='nested'),
            'infer_fn':   lambda m, noisy, seg: m(noisy),
            'uses_mask_at_inference': False,
            'short': 'Uformer+PLcur',
        },

        # ── Binary ROI weighting — Sun et al. 2019 (ROIRecNet) principle ───
        # THE DECISIVE CONTROL. One binary lesion mask, flat weight, no
        # per-region normalisation, no sub-region hierarchy. If this matches
        # the arms above, nesting and clinical priorities contribute nothing
        # and the novelty claim does not survive.
        'SwinIR + FlatROI [ROIRecNet-inspired]': {
            'class':  SwinIRLite,
            'config': dict(in_ch=4, dim=64, n_blocks=4, window_size=4),
            'trainer_fn': lambda m: FlatROITrainer(m, device=device, lr=1e-4, k=2.0),
            'infer_fn':   lambda m, noisy, seg: m(noisy),
            'uses_mask_at_inference': False,
            'short': 'SwinIR+FlatROI',
        },
        'Uformer + FlatROI [ROIRecNet-inspired]': {
            'class':  UformerLite,
            'config': dict(in_ch=4, dim=32, window_size=4),
            'trainer_fn': lambda m: FlatROITrainer(m, device=device, lr=1e-4, k=2.0),
            'infer_fn':   lambda m, noisy, seg: m(noisy),
            'uses_mask_at_inference': False,
            'short': 'Uformer+FlatROI',
        },

        # ── ROI perceptual loss — Chen et al. 2021 (LIDnet) principle ──────
        # Feature-space region weighting instead of pixel-space. Features come
        # from seg_aux, NOT from the evaluation segmentor — see the note in
        # main() for why that distinction decides whether the comparison means
        # anything.
        'SwinIR + ROI feature [LIDnet-inspired]': {
            'class':  SwinIRLite,
            'config': dict(in_ch=4, dim=64, n_blocks=4, window_size=4),
            'trainer_fn': lambda m: ROIFeatureTrainer(
                m, seg_aux, device=device, lr=1e-4, lambda_feat=1.0),
            'infer_fn':   lambda m, noisy, seg: m(noisy),
            'uses_mask_at_inference': False,
            'needs_seg_aux': True,
            'short': 'SwinIR+ROIfeat',
        },
        'Uformer + ROI feature [LIDnet-inspired]': {
            'class':  UformerLite,
            'config': dict(in_ch=4, dim=32, window_size=4),
            'trainer_fn': lambda m: ROIFeatureTrainer(
                m, seg_aux, device=device, lr=1e-4, lambda_feat=1.0),
            'infer_fn':   lambda m, noisy, seg: m(noisy),
            'uses_mask_at_inference': False,
            'needs_seg_aux': True,
            'short': 'Uformer+ROIfeat',
        },

        # ── Task feedback — LIDnet's central idea ──────────────────────────
        # Put the downstream loss in the objective rather than a prior about
        # where the downstream task looks. Trains its own copy of seg_aux.
        'SwinIR + task feedback [LIDnet-inspired]': {
            'class':  SwinIRLite,
            'config': dict(in_ch=4, dim=64, n_blocks=4, window_size=4),
            'trainer_fn': lambda m: TaskFeedbackTrainer(
                m, _fresh_aux(seg_aux), device=device, lr=1e-4, lambda_task=0.5),
            'infer_fn':   lambda m, noisy, seg: m(noisy),
            'uses_mask_at_inference': False,
            'needs_seg_aux': True,
            'short': 'SwinIR+task',
        },
        'Uformer + task feedback [LIDnet-inspired]': {
            'class':  UformerLite,
            'config': dict(in_ch=4, dim=32, window_size=4),
            'trainer_fn': lambda m: TaskFeedbackTrainer(
                m, _fresh_aux(seg_aux), device=device, lr=1e-4, lambda_task=0.5),
            'infer_fn':   lambda m, noisy, seg: m(noisy),
            'uses_mask_at_inference': False,
            'needs_seg_aux': True,
            'short': 'Uformer+task',
        },

        'SwinIR-lite (L1)': {"""
assert s.count(old) == 1, 'arm-insertion anchor not found'
s = s.replace(old, new, 1)

# ─────────────────────────────────────────────── 4. helper classes
old = """def build_models(device: str,"""
new = '''class _Identity(nn.Module):
    """Placeholder for the reference arms, which train nothing."""
    def __init__(self):
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        return x


class _NullTrainer:
    """Satisfies the trainer interface without performing any optimisation."""
    def __init__(self, model):
        self.model = model

    def step(self, batch):
        return {'total': 0.0}

    def predict(self, noisy):
        return noisy


def _fresh_aux(template):
    """A newly initialised auxiliary segmentor of the same shape.

    The task-feedback arms train their own segmentor from scratch, so each one
    gets its own copy rather than sharing (and silently co-training) a single
    instance across arms.
    """
    import copy
    m = copy.deepcopy(template)
    for layer in m.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    return m


def build_models(device: str,'''
assert s.count(old) == 1, 'helper-class anchor not found'
s = s.replace(old, new, 1)

# ─────────────────────────────────────────────── 5. train the aux segmentor
old = """    specs, results, histories, all_slices = build_models(device), [], {}, []"""
new = """    # ── second segmentor: the reason the comparison is honest ───────────
    # seg_model scores every arm and is frozen. seg_aux supplies features to
    # the ROI-feature arms and seeds the task-feedback arms.
    #
    # They must be different networks. If a restorer were optimised against
    # the same network that scores it, the score would measure how well those
    # two co-adapted, not how well the restored images support segmentation —
    # the LIDnet-style arms would win by construction and the comparison would
    # be worthless. LIDnet handles this the same way: its reported AP comes
    # from a detector pre-trained on normal-dose CT and held fixed (Sec. 4.1).
    #
    # seg_aux sees the same clean training images but is initialised
    # independently, so it is a genuinely separate judge.
    seg_aux = UNetSegmentor(in_channels=4, n_classes=4, base_ch=32).to(device)
    _aux_tr = SegTrainer(seg_aux, device=device, lr=5e-4)
    print('\\n=== AUXILIARY SEGMENTOR (for LIDnet-style arms) ===', flush=True)
    for ep in range(1, args.seg_epochs + 1):
        tot = 0.0
        for b in train_loader:
            tot += _aux_tr.step(b['target'], b['seg'][:, 0].long())
        if ep % max(args.seg_epochs // 2, 1) == 0 or ep == 1:
            print(f'  [aux] Ep {ep:2d}/{args.seg_epochs}  '
                  f'loss={tot/max(len(train_loader),1):.4f}', flush=True)
    print('  Auxiliary segmentor is separate from the evaluation segmentor.\\n',
          flush=True)

    specs, results, histories, all_slices = build_models(device, seg_model, seg_aux), [], {}, []"""
assert s.count(old) == 1, 'aux-segmentor anchor not found'
s = s.replace(old, new)

# ─────────────────────────────────────────────── 6. skip training where asked
old = """        model = spec['class'](**spec['config'])
        model, hist, dev = train_model(model, spec, train_loader, args, device, name)"""
new = """        model = spec['class'](**spec['config'])
        if spec.get('no_train'):
            model, hist, dev = model.to(device), [], device
            print('    reference arm — no training', flush=True)
        else:
            model, hist, dev = train_model(model, spec, train_loader, args, device, name)"""
assert s.count(old) == 1, 'no-train anchor not found'
s = s.replace(old, new)

# ─────────────────────────────────────────────── 7. give the oracle its target
old = """        pred = spec['infer_fn'](model, noisy, seg)"""
new = """        if spec.get('needs_target'):
            pred = spec['infer_fn'](model, noisy, seg, target.to(device))
        else:
            pred = spec['infer_fn'](model, noisy, seg)"""
assert s.count(old) == 1, 'infer-call anchor not found'
s = s.replace(old, new)

open(P, 'w').write(s)
print('PATCHED run_full_experiment.py')
print('  + 2 reference arms  (Noisy floor, Clean ceiling — inference only)')
print('  + 2 PathologyLoss Eq.(3) arms  (fixed weights, disjoint masks)')
print('  + 2 binary-ROI arms            (ROIRecNet principle)')
print('  + 2 ROI-feature arms           (LIDnet perceptual principle)')
print('  + 2 task-feedback arms         (LIDnet task-loss principle)')
print('  + auxiliary segmentor, kept separate from the evaluation segmentor')
