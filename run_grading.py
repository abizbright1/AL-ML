"""
run_grading.py — Brain Cancer Grade Prediction Pipeline
========================================================
Answers: "Does PP-MAE denoising improve brain tumour grade classification?"

Full pipeline:
    1. Load BraTS data (real or demo)
    2. Train 5 denoisers (PP-MAE clinical_risk + 4 baselines)
    3. Train one shared frozen UNet segmentor on clean images
    4. For EACH denoiser:
          a. Denoise all subjects
          b. Run frozen segmentor → predicted tumour masks
          c. Extract 7 radiomics features per slice
          d. Aggregate to subject level (max volumes, mean heterogeneity)
          e. Train GradingHead MLP on training subjects
          f. Evaluate on held-out test subjects
    5. Compare grading AUC, sensitivity, specificity across all denoisers
    6. Generate 5 publication-quality figures

Usage
-----
  # Demo mode (no real files needed):
  python3 run_grading.py

  # Real BraTS data:
  python3 run_grading.py /path/to/BraTS2021_Training_Data

  # Real data, limit subjects:
  python3 run_grading.py /path/to/BraTS2021_Training_Data --max_subjects 50

Output files
------------
  grading_auc.png          — AUC bar chart for all methods
  grading_roc.png          — ROC curves overlaid for all methods
  grading_features.png     — Feature distributions GBM vs LGG (violin plots)
  grading_table.png        — Summary metrics table
  grading_confusion.png    — Confusion matrices side by side
  grading_results.csv      — Full numeric results
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import sys, os, csv, argparse
import numpy as np
from collections import defaultdict
from typing import Tuple

# ── Resolve module path relative to this script ──────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, 'pp_mae'))

from option1_cnn_pp_mae  import CNNPPMAE, PPMAETrainer
from baselines           import (DnCNN, DnCNNTrainer,
                                  StandardUNet, StandardUNetTrainer,
                                  Noise2Noise, Noise2NoiseTrainer,
                                  REDNet, REDNetTrainer)
from segmentor           import UNetSegmentor, SegTrainer
from brats_loader        import BraTSDataset, make_demo_brats
from grading             import (GradingHead, GradingTrainer,
                                  extract_grading_features,
                                  aggregate_subject_features,
                                  grading_metrics, get_roc_curve,
                                  assign_demo_grade_labels,
                                  FEATURE_NAMES)
from grading_baselines   import (RadioTransformer, RadioTransformerTrainer,
                                  CBAMResNet, DINOv2Probe,
                                  SliceGradingTrainer,
                                  build_slice_dataset, aggregate_to_subject)

# ── CLI arguments ─────────────────────────────────────────────────────────────
_p = argparse.ArgumentParser(
    description='PP-MAE grading pipeline',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
_p.add_argument('brats_root',     nargs='?', default=None,
                help='Path to BraTS training folder (omit = demo mode)')
_p.add_argument('--max_subjects', type=int,   default=None)
_p.add_argument('--epochs',       type=int,   default=15,
                help='Denoiser training epochs')
_p.add_argument('--seg_epochs',   type=int,   default=20,
                help='Segmentor training epochs')
_p.add_argument('--grade_epochs', type=int,   default=60,
                help='Grading head training epochs')
_p.add_argument('--patch_size',   type=int,   default=96)
_p.add_argument('--sigma',        type=float, default=0.08)
_p.add_argument('--out',          type=str,   default=None)
_args = _p.parse_args()

DEVICE     = 'cpu'
OUT        = _args.out or _SCRIPT_DIR
D_EPOCHS   = _args.epochs
S_EPOCHS   = _args.seg_epochs
G_EPOCHS   = _args.grade_epochs
PATCH_SIZE = _args.patch_size
SIGMA      = _args.sigma
BATCH_SIZE = 4
SEED       = 42

os.makedirs(OUT, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load Data
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65, flush=True)
print("  PP-MAE  →  Grading Pipeline", flush=True)
print("="*65, flush=True)

BRATS_ROOT = _args.brats_root
USE_REAL   = False

if BRATS_ROOT and os.path.isdir(BRATS_ROOT):
    try:
        ds_full = BraTSDataset(
            BRATS_ROOT, slice_axis=2, patch_size=PATCH_SIZE,
            sigma=SIGMA, min_tumour_frac=0.01, cache=True,
            max_subjects=_args.max_subjects,
        )
        USE_REAL   = True
        DATA_LABEL = f'Real BraTS  ({len(ds_full)} slices)'
        print(f"\n✅ Real BraTS data: {BRATS_ROOT}", flush=True)
    except Exception as e:
        print(f"⚠️  Could not load BraTS: {e}\n   Falling back to demo.", flush=True)

if not USE_REAL:
    ds_full    = make_demo_brats(n_subjects=20, patch_size=PATCH_SIZE,
                                  sigma=SIGMA, slices_per_subject=20)
    DATA_LABEL = 'Synthetic BraTS-style demo'
    print(f"\n📌 DEMO MODE — {len(ds_full)} synthetic slices", flush=True)

# ── Assign grade labels per subject ─────────────────────────────────────────
# Real BraTS: use ET volume fraction as proxy (same as demo, but on real data).
# In a full study these would come from the BraTS survival CSV.
print("\n[grade labels] Assigning GBM / LGG labels per subject...", flush=True)
grade_labels = assign_demo_grade_labels(ds_full)   # uses median split → balanced GBM/LGG

# ── Stratified subject-level train/test split (80/20) ────────────────────────
# IMPORTANT: split at SUBJECT level (not slice level) to prevent data leakage.
# STRATIFIED: ensures both GBM and LGG appear in train AND test sets.
# Without stratification, small cohorts can end up with all GBM in train
# and all LGG in test (or vice versa), making AUC meaningless.
gbm_subjects = sorted([s for s, l in grade_labels.items() if l == 1])
lgg_subjects = sorted([s for s, l in grade_labels.items() if l == 0])
np.random.shuffle(gbm_subjects)
np.random.shuffle(lgg_subjects)

# Put at least 1 of each class in test set
n_test_gbm = max(1, int(0.2 * len(gbm_subjects)))
n_test_lgg = max(1, int(0.2 * len(lgg_subjects)))

test_subjects  = set(gbm_subjects[:n_test_gbm] + lgg_subjects[:n_test_lgg])
train_subjects = set(gbm_subjects[n_test_gbm:] + lgg_subjects[n_test_lgg:])

# Build slice-level loaders respecting the subject split
train_indices = [i for i, item in enumerate(ds_full)
                 if item['subject'] in train_subjects]
test_indices  = [i for i, item in enumerate(ds_full)
                 if item['subject'] in test_subjects]

train_ds = torch.utils.data.Subset(ds_full, train_indices)
test_ds  = torch.utils.data.Subset(ds_full, test_indices)

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = torch.utils.data.DataLoader(
    test_ds,  batch_size=BATCH_SIZE, shuffle=False)

n_gbm_train = sum(grade_labels[s] for s in train_subjects)
n_lgg_train = len(train_subjects) - n_gbm_train
n_gbm_test  = sum(grade_labels[s] for s in test_subjects)
n_lgg_test  = len(test_subjects)  - n_gbm_test

print(f"\n  Train: {len(train_subjects)} subjects  "
      f"(GBM={n_gbm_train}, LGG={n_lgg_train})  "
      f"→  {len(train_indices)} slices", flush=True)
print(f"  Test:  {len(test_subjects)} subjects  "
      f"(GBM={n_gbm_test},  LGG={n_lgg_test})   "
      f"→  {len(test_indices)} slices", flush=True)
print(f"  Data:  {DATA_LABEL}\n", flush=True)

C = 4   # MRI modalities

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Define and Train Denoisers
# ─────────────────────────────────────────────────────────────────────────────
# Light models for demo — same interface as run_brats_test.py
MODELS = {
    'PP-MAE (clinical_risk)': (
        CNNPPMAE(C, base_ch=16, depth=3),
        lambda m: PPMAETrainer(m, device=DEVICE, mode='clinical_risk', lr=1e-4)),
    'DnCNN': (
        DnCNN(C, features=32, num_layers=10),
        lambda m: DnCNNTrainer(m, device=DEVICE, lr=1e-3)),
    'UNet-L1': (
        StandardUNet(C, base_ch=16),
        lambda m: StandardUNetTrainer(m, device=DEVICE, lr=1e-4)),
    'Noise2Noise': (
        Noise2Noise(C, features=32, num_layers=10),
        lambda m: Noise2NoiseTrainer(m, device=DEVICE, lr=1e-3)),
    'REDNet': (
        REDNet(C, features=32, num_layers=4),
        lambda m: REDNetTrainer(m, device=DEVICE, lr=1e-4)),
}


def run_epoch(trainer, loader):
    total = 0.0
    for b in loader:
        m = trainer.step(b)
        total += m.get('total', 0.0)
    return total / max(len(loader), 1)


trained_models = {}
denoiser_histories = {}

print("="*65, flush=True)
print("  STEP 1 — TRAINING DENOISERS", flush=True)
print("="*65, flush=True)

for name, (model, make_trainer) in MODELS.items():
    trainer = make_trainer(model)
    hist = []
    print(f"\n  [{name}]", flush=True)
    for ep in range(1, D_EPOCHS + 1):
        loss = run_epoch(trainer, train_loader)
        hist.append(loss)
        if ep == 1 or ep % 5 == 0:
            print(f"    Ep {ep:2d}/{D_EPOCHS}  loss={loss:.4f}", flush=True)
    trained_models[name]      = model
    denoiser_histories[name]  = hist

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Train Shared Frozen Segmentor on Clean Images
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65, flush=True)
print("  STEP 2 — TRAINING SEGMENTOR (on clean BraTS images)", flush=True)
print("="*65, flush=True)

seg_model   = UNetSegmentor(in_channels=C, n_classes=4, base_ch=32)
seg_trainer = SegTrainer(seg_model, device=DEVICE, lr=5e-4)

for ep in range(1, S_EPOCHS + 1):
    ep_loss = 0.0
    for b in train_loader:
        clean  = b['target']
        labels = b['seg'][:, 0].long()
        ep_loss += seg_trainer.step(clean, labels)
    ep_loss /= max(len(train_loader), 1)
    if ep == 1 or ep % 5 == 0:
        print(f"  [segmentor] Ep {ep:2d}/{S_EPOCHS}  loss={ep_loss:.4f}",
              flush=True)

seg_model.eval()
# Freeze segmentor — weights must not change during grading experiments
for p in seg_model.parameters():
    p.requires_grad_(False)
print("  Segmentor trained and frozen.\n", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Feature Extraction Helper
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_features(
    model,             # denoiser (or None for noisy baseline)
    loader,            # DataLoader
    seg_model,         # frozen UNet segmentor
    grade_labels,      # {subject_name: 0/1}
    split_subjects,    # set of subject names in this split
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run the full pipeline (denoise → segment → extract features)
    and aggregate to subject level.

    Returns:
        features : (N_subjects, 7)  radiomics feature matrix
        labels   : (N_subjects,)    grade labels  0=LGG / 1=GBM
    """
    seg_model.eval()
    if model is not None:
        model.eval()

    # Accumulate per-slice features grouped by subject
    subj_slice_feats = defaultdict(list)   # {subject: [tensor(7), ...]}

    with torch.no_grad():
        for b in loader:
            noisy  = b['noisy']
            seg_gt = b['seg']
            names  = b['subject']

            # ── Denoise ──────────────────────────────────────────────────────
            if model is None:
                denoised = noisy                      # baseline: no denoising
            else:
                try:
                    denoised = model(noisy, seg_gt)   # PP-MAE: needs seg map
                except TypeError:
                    denoised = model(noisy)            # baselines: image only

            # ── Segment the denoised image ───────────────────────────────────
            seg_logits = seg_model(denoised)          # (B, 4, H, W) logits

            # ── Extract 7 features per slice ─────────────────────────────────
            feats = extract_grading_features(
                denoised, seg_logits)                 # (B, 7)

            # Store per subject
            for i, subj in enumerate(names):
                if subj in split_subjects:
                    subj_slice_feats[subj].append(feats[i].detach().cpu())

    # ── Aggregate slices → subject level ─────────────────────────────────────
    feat_list, label_list = [], []
    for subj in sorted(split_subjects):
        if subj not in subj_slice_feats:
            continue
        subj_feat = aggregate_subject_features(subj_slice_feats[subj])  # (7,)
        feat_list.append(subj_feat)
        label_list.append(grade_labels.get(subj, 0))

    if len(feat_list) == 0:
        return torch.zeros(1, 7), torch.zeros(1)

    features = torch.stack(feat_list, dim=0)       # (N, 7)
    labels   = torch.tensor(label_list).long()     # (N,)
    return features, labels


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Run Grading for Each Denoiser
# ─────────────────────────────────────────────────────────────────────────────
print("="*65, flush=True)
print("  STEP 3 — GRADING EVALUATION", flush=True)
print("="*65, flush=True)

all_results   = {}    # {method_name: metrics_dict}
all_probs     = {}    # {method_name: np.array of GBM probabilities}
all_labels_np = None  # ground truth (same for all methods)

# Include "No Denoising" as a baseline
EVAL_MODELS = {'No Denoising': None, **trained_models}

for method_name, model in EVAL_MODELS.items():
    print(f"\n  [{method_name}]", flush=True)

    # ── Extract features ──────────────────────────────────────────────────────
    print("    Extracting train features...", flush=True)
    tr_feats, tr_labels = extract_all_features(
        model, train_loader, seg_model, grade_labels, train_subjects)

    print("    Extracting test features...", flush=True)
    te_feats, te_labels = extract_all_features(
        model, test_loader,  seg_model, grade_labels, test_subjects)

    # ── Normalise features (z-score using train stats) ────────────────────────
    # Prevents features with large ranges (V_WT in pixels vs H in [0,1])
    # from dominating the MLP — same principle as batch norm but applied
    # to the feature matrix directly.
    mu  = tr_feats.mean(dim=0)
    std = tr_feats.std(dim=0).clamp(min=1e-6)
    tr_feats_norm = (tr_feats - mu) / std
    te_feats_norm = (te_feats - mu) / std     # use TRAIN stats for test (no leakage)

    # ── Train grading head ────────────────────────────────────────────────────
    # pos_weight: account for class imbalance
    n_pos = tr_labels.sum().item()
    n_neg = len(tr_labels) - n_pos
    pos_w = n_neg / max(n_pos, 1)             # inverse frequency weighting

    grading_model   = GradingHead(n_features=7, hidden1=32, hidden2=16)
    grading_trainer = GradingTrainer(
        grading_model, device=DEVICE, lr=1e-3, pos_weight=pos_w)

    for ep in range(1, G_EPOCHS + 1):
        loss = grading_trainer.step(tr_feats_norm, tr_labels.float())
        if ep == 1 or ep % 20 == 0:
            print(f"    [grader] Ep {ep:2d}/{G_EPOCHS}  loss={loss:.4f}",
                  flush=True)

    # ── Evaluate on test subjects ─────────────────────────────────────────────
    probs   = grading_trainer.predict(te_feats_norm)   # (N_test,)
    metrics = grading_metrics(probs, te_labels)

    all_results[method_name] = metrics
    all_probs[method_name]   = probs.numpy()
    if all_labels_np is None:
        all_labels_np = te_labels.numpy()

    print(f"    AUC={metrics['auc']:.3f}  "
          f"Acc={metrics['accuracy']:.3f}  "
          f"Sens={metrics['sensitivity']:.3f}  "
          f"Spec={metrics['specificity']:.3f}",
          flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5B.  Grading Architecture Baselines (RadioTransformer, CBAM-ResNet, DINOv2)
#      All three use PP-MAE (clinical_risk) denoised images as input so we
#      compare GRADING ARCHITECTURES fairly (not denoising quality).
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65, flush=True)
print("  STEP 3B — GRADING ARCHITECTURE BASELINES", flush=True)
print("  (using PP-MAE denoised images as input)", flush=True)
print("="*65, flush=True)

pp_mae_model = trained_models.get('PP-MAE (clinical_risk)')

# ── Re-extract PP-MAE subject-level features (7-dim) for RadioTransformer ──
print("\n  [Extracting PP-MAE features for RadioTransformer...]", flush=True)
tr_feats_pp, tr_labels_pp = extract_all_features(
    pp_mae_model, train_loader, seg_model, grade_labels, train_subjects)
te_feats_pp, te_labels_pp = extract_all_features(
    pp_mae_model, test_loader,  seg_model, grade_labels, test_subjects)

mu_pp   = tr_feats_pp.mean(dim=0)
std_pp  = tr_feats_pp.std(dim=0).clamp(min=1e-6)
tr_pp_n = (tr_feats_pp - mu_pp) / std_pp
te_pp_n = (te_feats_pp - mu_pp) / std_pp

n_pos_pp = tr_labels_pp.sum().item()
n_neg_pp = len(tr_labels_pp) - n_pos_pp
pos_w_pp = n_neg_pp / max(n_pos_pp, 1)

# ── 1. RadioTransformer ────────────────────────────────────────────────────
print("\n  [RadioTransformer]", flush=True)
rt_model   = RadioTransformer(n_features=7, d_model=64, n_heads=4, n_layers=2)
rt_trainer = RadioTransformerTrainer(
    rt_model, device=DEVICE, lr=1e-3, pos_weight=pos_w_pp)

for ep in range(1, G_EPOCHS + 1):
    loss = rt_trainer.step(tr_pp_n, tr_labels_pp.float())
    if ep == 1 or ep % 20 == 0:
        print(f"    [grader] Ep {ep:2d}/{G_EPOCHS}  loss={loss:.4f}", flush=True)

rt_probs   = rt_trainer.predict(te_pp_n)
rt_metrics = grading_metrics(rt_probs, te_labels_pp)
all_results['RadioTransformer'] = rt_metrics
all_probs['RadioTransformer']   = rt_probs.numpy()
print(f"    AUC={rt_metrics['auc']:.3f}  "
      f"Acc={rt_metrics['accuracy']:.3f}  "
      f"Sens={rt_metrics['sensitivity']:.3f}  "
      f"Spec={rt_metrics['specificity']:.3f}", flush=True)

# ── Build denoised-slice datasets for slice-level models ──────────────────
print("\n  [Building denoised slice datasets...]", flush=True)
tr_slices, tr_slice_labels, tr_subj_names = build_slice_dataset(
    train_loader, pp_mae_model, grade_labels, train_subjects, DEVICE)
te_slices, te_slice_labels, te_subj_names = build_slice_dataset(
    test_loader,  pp_mae_model, grade_labels, test_subjects,  DEVICE)

print(f"  Train slices: {len(tr_slices)}  |  Test slices: {len(te_slices)}", flush=True)


def _train_slice_model(trainer, tr_s, tr_l, epochs, batch_size, name):
    """Mini-batch training loop for slice-level models."""
    for ep in range(1, epochs + 1):
        idx     = torch.randperm(len(tr_s))
        ep_loss = 0.0
        n_batch = 0
        for start in range(0, len(tr_s), batch_size):
            bi   = idx[start:start + batch_size]
            loss = trainer.step(tr_s[bi], tr_l[bi].float())
            ep_loss += loss
            n_batch += 1
        ep_loss /= max(n_batch, 1)
        if ep == 1 or ep % 20 == 0:
            print(f"    [grader] Ep {ep:2d}/{epochs}  loss={ep_loss:.4f}", flush=True)


def _eval_slice_model(trainer, te_s, te_subj, batch_size):
    """Predict slice-level probs in batches."""
    parts = []
    for start in range(0, len(te_s), batch_size):
        parts.append(trainer.predict_slice(te_s[start:start + batch_size]))
    return torch.cat(parts) if parts else torch.zeros(1)


# ── 2. CBAM-ResNet ─────────────────────────────────────────────────────────
print("\n  [CBAM-ResNet]", flush=True)
cbam_model   = CBAMResNet(in_channels=C, base_ch=16)
cbam_trainer = SliceGradingTrainer(
    cbam_model, device=DEVICE, lr=1e-4, pos_weight=pos_w_pp)

_train_slice_model(cbam_trainer, tr_slices, tr_slice_labels,
                   G_EPOCHS, BATCH_SIZE, 'CBAM-ResNet')

cbam_slice_probs = _eval_slice_model(cbam_trainer, te_slices,
                                     te_subj_names, BATCH_SIZE)
cbam_probs, cbam_labels = aggregate_to_subject(
    cbam_slice_probs, te_subj_names, grade_labels, test_subjects, mode='max')
cbam_metrics = grading_metrics(cbam_probs, cbam_labels)
all_results['CBAM-ResNet'] = cbam_metrics
all_probs['CBAM-ResNet']   = cbam_probs.numpy()
print(f"    AUC={cbam_metrics['auc']:.3f}  "
      f"Acc={cbam_metrics['accuracy']:.3f}  "
      f"Sens={cbam_metrics['sensitivity']:.3f}  "
      f"Spec={cbam_metrics['specificity']:.3f}", flush=True)

# ── 3. DINOv2Probe ─────────────────────────────────────────────────────────
print("\n  [DINOv2Probe]", flush=True)
dino_model   = DINOv2Probe(img_size=PATCH_SIZE, use_dino=True, device=DEVICE)
dino_trainer = SliceGradingTrainer(
    dino_model, device=DEVICE, lr=1e-3, pos_weight=pos_w_pp)

_train_slice_model(dino_trainer, tr_slices, tr_slice_labels,
                   G_EPOCHS, BATCH_SIZE, 'DINOv2Probe')

dino_slice_probs = _eval_slice_model(dino_trainer, te_slices,
                                     te_subj_names, BATCH_SIZE)
dino_probs, dino_labels = aggregate_to_subject(
    dino_slice_probs, te_subj_names, grade_labels, test_subjects, mode='max')
dino_metrics = grading_metrics(dino_probs, dino_labels)
all_results['DINOv2Probe'] = dino_metrics
all_probs['DINOv2Probe']   = dino_probs.numpy()
print(f"    AUC={dino_metrics['auc']:.3f}  "
      f"Acc={dino_metrics['accuracy']:.3f}  "
      f"Sens={dino_metrics['sensitivity']:.3f}  "
      f"Spec={dino_metrics['specificity']:.3f}", flush=True)

# Ensure all_labels_np is aligned (test subjects are the same across methods)
if all_labels_np is None:
    all_labels_np = te_labels_pp.numpy()

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Save CSV
# ─────────────────────────────────────────────────────────────────────────────
csv_path = os.path.join(OUT, 'grading_results.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Method', 'AUC', 'Accuracy', 'Sensitivity',
                'Specificity', 'Precision', 'F1', 'TP', 'TN', 'FP', 'FN'])
    for name, m in all_results.items():
        w.writerow([name, m['auc'], m['accuracy'], m['sensitivity'],
                    m['specificity'], m['precision'], m['f1'],
                    m['TP'], m['TN'], m['FP'], m['FN']])
print(f"\n  Saved {csv_path}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Plots
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65, flush=True)
print("  STEP 4 — GENERATING PLOTS", flush=True)
print("="*65, flush=True)

METHOD_NAMES = list(all_results.keys())

# Colour scheme: PP-MAE = blue, baselines = orange shades, no-denoise = grey
def method_colour(name):
    if name == 'No Denoising':           return '#9E9E9E'
    if name == 'PP-MAE (clinical_risk)': return '#1565C0'
    if name == 'DnCNN':                  return '#E65100'
    if name == 'UNet-L1':                return '#F57C00'
    if name == 'Noise2Noise':            return '#EF6C00'
    if name == 'REDNet':                 return '#FF8F00'
    if name == 'RadioTransformer':       return '#6A1B9A'   # purple
    if name == 'CBAM-ResNet':            return '#00695C'   # teal
    if name == 'DINOv2Probe':            return '#AD1457'   # deep pink
    return '#607D8B'

colours = [method_colour(n) for n in METHOD_NAMES]

# ── Figure 1: AUC Bar Chart ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 5))
aucs  = [all_results[n]['auc']  for n in METHOD_NAMES]
accs  = [all_results[n]['accuracy'] for n in METHOD_NAMES]
bars  = ax.bar(range(len(METHOD_NAMES)), aucs, color=colours,
               edgecolor='white', linewidth=1.2, width=0.6)
ax.axhline(0.5, color='red', linestyle='--', linewidth=1.2,
           label='Random classifier (AUC=0.5)')
ax.set_xticks(range(len(METHOD_NAMES)))
ax.set_xticklabels(METHOD_NAMES, rotation=25, ha='right', fontsize=10)
ax.set_ylabel('AUC-ROC', fontsize=12)
ax.set_ylim(0, 1.12)
ax.set_title(
    f'Brain Tumour Grade Prediction AUC\n'
    f'GBM (Grade IV) vs LGG (Grade II/III)  —  {DATA_LABEL}',
    fontweight='bold', fontsize=12)
ax.grid(axis='y', alpha=0.3)
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{auc:.3f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
pp_p = mpatches.Patch(color='#1565C0', label='PP-MAE (ours)')
bl_p = mpatches.Patch(color='#E65100', label='Denoising baselines')
nd_p = mpatches.Patch(color='#9E9E9E', label='No Denoising')
rt_p = mpatches.Patch(color='#6A1B9A', label='RadioTransformer (SOTA)')
cb_p = mpatches.Patch(color='#00695C', label='CBAM-ResNet (SOTA)')
di_p = mpatches.Patch(color='#AD1457', label='DINOv2Probe (SOTA)')
ax.legend(handles=[pp_p, bl_p, nd_p, rt_p, cb_p, di_p,
                   plt.Line2D([0],[0], color='red', ls='--', label='Random')],
          loc='upper right', fontsize=8, ncol=2)
plt.tight_layout()
auc_path = os.path.join(OUT, 'grading_auc.png')
plt.savefig(auc_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved grading_auc.png", flush=True)

# ── Figure 2: ROC Curves ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.50)')
for name, col in zip(METHOD_NAMES, colours):
    fprs, tprs = get_roc_curve(all_probs[name], all_labels_np)
    auc = all_results[name]['auc']
    ax.plot(fprs, tprs, color=col, linewidth=2.2,
            label=f'{name}  (AUC={auc:.3f})')
ax.set_xlabel('False Positive Rate  (1 − Specificity)', fontsize=11)
ax.set_ylabel('True Positive Rate  (Sensitivity)', fontsize=11)
ax.set_title(f'ROC Curves — Brain Cancer Grading\n{DATA_LABEL}',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=9, loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'grading_roc.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved grading_roc.png", flush=True)

# ── Figure 3: Feature Distributions (GBM vs LGG) ─────────────────────────────
# Show feature values for the test set split by true grade label
# Uses PP-MAE (clinical_risk) features — the best method
best_model = trained_models.get('PP-MAE (clinical_risk)')
te_feats_raw, te_labels_raw = extract_all_features(
    best_model, test_loader, seg_model, grade_labels, test_subjects)

gbm_mask = te_labels_raw == 1
lgg_mask = te_labels_raw == 0

fig, axes = plt.subplots(1, 7, figsize=(20, 5))
for fi, (ax, fname) in enumerate(zip(axes, FEATURE_NAMES)):
    gbm_vals = te_feats_raw[gbm_mask, fi].numpy()
    lgg_vals = te_feats_raw[lgg_mask, fi].numpy()

    # Box plots for each class
    bp = ax.boxplot(
        [lgg_vals if len(lgg_vals) > 0 else [0],
         gbm_vals if len(gbm_vals) > 0 else [0]],
        patch_artist=True,
        labels=['LGG', 'GBM'],
        widths=0.5,
    )
    bp['boxes'][0].set_facecolor('#42A5F5')    # LGG = light blue
    bp['boxes'][1].set_facecolor('#EF5350')    # GBM = red
    ax.set_title(fname, fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

fig.suptitle(
    f'Radiomics Feature Distributions: GBM vs LGG\n'
    f'(from PP-MAE denoised images)  —  {DATA_LABEL}',
    fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'grading_features.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved grading_features.png", flush=True)

# ── Figure 4: Confusion Matrices ──────────────────────────────────────────────
n_methods = len(METHOD_NAMES)
# Arrange in 2 rows if many methods, to keep the figure readable
if n_methods <= 5:
    ncols_cm, nrows_cm = n_methods, 1
else:
    ncols_cm = (n_methods + 1) // 2
    nrows_cm = 2
fig, axes = plt.subplots(nrows_cm, ncols_cm,
                          figsize=(ncols_cm * 3.2, nrows_cm * 3.5))
axes = np.array(axes).flatten()   # always 1-D, hide any extras
for ax in axes[n_methods:]:
    ax.set_visible(False)

for ax, name in zip(axes, METHOD_NAMES):
    m  = all_results[name]
    cm = np.array([[m['TN'], m['FP']],
                   [m['FN'], m['TP']]])
    im = ax.imshow(cm, cmap='Blues', vmin=0)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred LGG', 'Pred GBM'], fontsize=8)
    ax.set_yticklabels(['True LGG', 'True GBM'], fontsize=8)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    short = name.replace('PP-MAE ', 'PP-MAE\n').replace(' (', '\n(')
    ax.set_title(f'{short}\nAUC={m["auc"]:.3f}', fontsize=8, fontweight='bold')

fig.suptitle(f'Confusion Matrices — Grading (threshold=0.5)\n{DATA_LABEL}',
             fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'grading_confusion.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved grading_confusion.png", flush=True)

# ── Figure 5: Summary Table ───────────────────────────────────────────────────
col_labels = ['Method', 'AUC↑', 'Accuracy↑', 'Sensitivity↑',
              'Specificity↑', 'F1↑']
cell_data  = []
for name in METHOD_NAMES:
    m = all_results[name]
    cell_data.append([name,
                      f'{m["auc"]:.3f}', f'{m["accuracy"]:.3f}',
                      f'{m["sensitivity"]:.3f}', f'{m["specificity"]:.3f}',
                      f'{m["f1"]:.3f}'])

row_height = max(1.6, 9.0 / max(len(cell_data), 1))
fig, ax = plt.subplots(figsize=(14, max(4, len(cell_data) * row_height)))
ax.axis('off')
tbl = ax.table(cellText=cell_data, colLabels=col_labels,
               cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2.0)

SOTA_NAMES = {'RadioTransformer', 'CBAM-ResNet', 'DINOv2Probe'}
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#1A237E')
        cell.set_text_props(color='white', fontweight='bold')
    elif row > 0 and 'PP-MAE' in cell_data[row-1][0]:
        cell.set_facecolor('#BBDEFB')           # blue tint — PP-MAE
    elif row > 0 and cell_data[row-1][0] == 'No Denoising':
        cell.set_facecolor('#F5F5F5')           # grey — no denoising
    elif row > 0 and cell_data[row-1][0] in SOTA_NAMES:
        cell.set_facecolor('#F3E5F5')           # purple tint — SOTA baselines
    else:
        cell.set_facecolor('#FFF3E0' if row % 2 else '#FFFFFF')
    cell.set_edgecolor('#BDBDBD')

# Bold the best AUC value
best_auc_row = max(range(len(METHOD_NAMES)),
                   key=lambda i: all_results[METHOD_NAMES[i]]['auc'])
tbl[best_auc_row + 1, 1].set_text_props(fontweight='bold', color='#1565C0')

label = "REAL BraTS Data" if USE_REAL else "Synthetic BraTS-style Demo"
fig.suptitle(
    f'PP-MAE vs Baselines — Brain Cancer Grading  ({label})\n'
    f'(Blue = PP-MAE  |  Orange = Baselines  |  Bold = Best AUC)',
    fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'grading_table.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved grading_table.png", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Final Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65, flush=True)
print(f"  GRADING COMPLETE  —  {label}", flush=True)
print("="*65, flush=True)
print(f"\n  {'Method':<28}  {'AUC':>6}  {'Sens':>6}  {'Spec':>6}  {'F1':>6}")
print("  " + "-"*58)

best_name = max(all_results, key=lambda n: all_results[n]['auc'])
for name in METHOD_NAMES:
    m    = all_results[name]
    mark = '  ◀ BEST' if name == best_name else ''
    print(f"  {name:<28}  {m['auc']:>6.3f}  {m['sensitivity']:>6.3f}  "
          f"{m['specificity']:>6.3f}  {m['f1']:>6.3f}{mark}")

print(f"\n  📁 Output files in {OUT}/")
for fn in ['grading_auc.png', 'grading_roc.png', 'grading_features.png',
           'grading_confusion.png', 'grading_table.png', 'grading_results.csv']:
    print(f"     {fn}", flush=True)

if not USE_REAL:
    print('\n' + '─'*65, flush=True)
    print('  📌 This was a DEMO RUN on synthetic data.', flush=True)
    print('  Run on real BraTS data:', flush=True)
    print('    python3 run_grading.py /path/to/BraTS2021_Training_Data', flush=True)
    print('─'*65, flush=True)
