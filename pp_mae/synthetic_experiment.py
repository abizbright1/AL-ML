"""
Synthetic end-to-end experiment for PP-MAE validation.

What this does (and why):
    Since we have no real MRI data yet, we create a controlled synthetic
    dataset that mimics the key properties of glioma MRI:
      - 4 modality channels (T1W, T1Wce, T2W, FLAIR)
      - A tumour region with distinct signal characteristics
      - Gaussian noise degradation

    This lets us verify 3 things before touching real data:
      1. The training loop runs without errors
      2. The loss actually decreases (model is learning)
      3. The evaluation metrics produce sensible numbers

    In your PhD, this is called a "proof-of-concept experiment" and is
    legitimate to include in a methods paper as validation of the framework.

Experiment:
    We compare all 3 conditions from your study protocol (Section 6):
      Condition A : No denoising    (lower-bound baseline)
      Condition B : Standard U-Net  (no pathology-aware components)
      Condition C : PP-MAE          (full model with pathology loss)
"""

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from losses import PPMAELoss
from evaluation import (
    compute_image_quality_metrics, segmentation_metrics,
    icc, bland_altman, auroc
)


# ─────────────────────────────────────────────────────────────
# STEP 1: Synthetic dataset
# ─────────────────────────────────────────────────────────────

def make_synthetic_brain(H=128, W=128):
    """
    Creates a synthetic 4-channel MRI slice + segmentation map.

    The synthetic 'brain' has:
      - Background (signal ~0)
      - Normal parenchyma (signal 0.3–0.6)
      - Peritumoral oedema / label 2 (slightly elevated T2/FLAIR)
      - Tumour core / label 1 (hypointense T1, hyperintense T2)
      - Enhancing tumour / label 3 (hyperintense T1ce)

    This is a simplified but physically motivated representation.
    """
    # Start with all zeros
    img = np.zeros((4, H, W), dtype=np.float32)   # 4 modalities
    seg = np.zeros((1, H, W), dtype=np.int64)

    # Normal brain parenchyma — ellipse in center
    cy, cx = H // 2, W // 2
    ry, rx = H // 3, W // 3
    Y, X = np.ogrid[:H, :W]
    brain_mask = ((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2 <= 1.0
    img[:, brain_mask] = np.random.uniform(0.3, 0.6, (4, brain_mask.sum()))

    # Peritumoral oedema (label 2) — medium ellipse offset
    ry2, rx2 = H // 8, W // 8
    ocy, ocx = cy - H // 8, cx + W // 8
    ed_mask = brain_mask & (((Y - ocy) / ry2) ** 2 + ((X - ocx) / rx2) ** 2 <= 1.0)
    img[2, ed_mask] = 0.8   # T2 hyperintense
    img[3, ed_mask] = 0.75  # FLAIR hyperintense
    img[0, ed_mask] = 0.4   # T1 slightly low
    img[1, ed_mask] = 0.4   # T1ce similar
    seg[0, ed_mask] = 2

    # Tumour core (label 1) — smaller ellipse
    ry3, rx3 = H // 14, W // 14
    tc_mask = ed_mask & (((Y - ocy) / ry3) ** 2 + ((X - ocx) / rx3) ** 2 <= 1.0)
    img[0, tc_mask] = 0.2   # T1 hypointense (necrosis)
    img[1, tc_mask] = 0.3
    img[2, tc_mask] = 0.9   # T2 very high
    img[3, tc_mask] = 0.85
    seg[0, tc_mask] = 1

    # Enhancing tumour (label 3) — rim around necrotic core
    ry4, rx4 = H // 10, W // 10
    et_ring = ed_mask & (((Y - ocy) / ry4) ** 2 + ((X - ocx) / rx4) ** 2 <= 1.0) & ~tc_mask
    img[1, et_ring] = 0.95  # T1ce very bright (contrast enhancement)
    img[0, et_ring] = 0.5
    img[2, et_ring] = 0.7
    img[3, et_ring] = 0.6
    seg[0, et_ring] = 3

    return img.clip(0, 1), seg


def make_noisy(img, sigma=0.08):
    """Add Gaussian noise to simulate scanner noise."""
    return (img + np.random.normal(0, sigma, img.shape)).clip(0, 1).astype(np.float32)


def make_dataset(n=60, H=128, W=128, noise_sigma=0.08):
    """Generate n synthetic samples."""
    targets, noisys, segs = [], [], []
    for _ in range(n):
        target, seg = make_synthetic_brain(H, W)
        noisy = make_noisy(target, sigma=noise_sigma)
        targets.append(target)
        noisys.append(noisy)
        segs.append(seg)

    return (
        torch.tensor(np.stack(targets)),  # (N, 4, H, W)
        torch.tensor(np.stack(noisys)),   # (N, 4, H, W)
        torch.tensor(np.stack(segs)),     # (N, 1, H, W)
    )


# ─────────────────────────────────────────────────────────────
# STEP 2: Define the 3 experimental conditions
# ─────────────────────────────────────────────────────────────

class StandardUNet(nn.Module):
    """
    Condition B: plain U-Net denoiser — NO pathology-aware components.
    Same capacity as Option 1 but trained with L1 loss only.
    This isolates the contribution of the pathology-aware loss.
    """
    def __init__(self, ch=4, base=32):
        super().__init__()
        # Encoder
        self.e1 = self._block(ch,    base)
        self.e2 = self._block(base,  base*2)
        self.e3 = self._block(base*2,base*4)
        self.pool = nn.MaxPool2d(2)
        # Bottleneck
        self.bn = self._block(base*4, base*4)
        # Decoder
        self.up3 = nn.ConvTranspose2d(base*4, base*4, 2, stride=2)
        self.d3  = self._block(base*8, base*2)
        self.up2 = nn.ConvTranspose2d(base*2, base*2, 2, stride=2)
        self.d2  = self._block(base*4, base)
        self.up1 = nn.ConvTranspose2d(base, base, 2, stride=2)
        self.d1  = self._block(base*2, base)
        self.out = nn.Conv2d(base, ch, 1)

    @staticmethod
    def _block(ic, oc):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1, bias=False),
            nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            nn.Conv2d(oc, oc, 3, padding=1, bias=False),
            nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
        )

    def forward(self, x, seg=None):   # seg ignored — no pathology awareness
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        b  = self.bn(self.pool(s3))
        x  = self.d3(torch.cat([self.up3(b),  s3], 1))
        x  = self.d2(torch.cat([self.up2(x),  s2], 1))
        x  = self.d1(torch.cat([self.up1(x),  s1], 1))
        return torch.sigmoid(self.out(x))


# ─────────────────────────────────────────────────────────────
# STEP 3: Training function (shared across conditions)
# ─────────────────────────────────────────────────────────────

def train_model(model, targets, noisys, segs,
                n_epochs=30, batch_size=8,
                use_pathology_loss=False,
                lr=1e-3, verbose=True):
    """
    Trains a model and returns per-epoch loss history.

    Args:
        use_pathology_loss: if True → PP-MAE composite loss
                            if False → plain L1 (standard denoiser)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if use_pathology_loss:
        loss_fn = PPMAELoss(lambda1=1.0, lambda2=0.5)
    else:
        loss_fn_l1 = nn.L1Loss()

    N = len(targets)
    history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # Mini-batch loop
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            x_b  = noisys[idx]
            y_b  = targets[idx]
            s_b  = segs[idx]

            optimizer.zero_grad()
            pred = model(x_b, s_b)

            if use_pathology_loss:
                losses = loss_fn(pred, y_b, s_b)
                loss = losses["total"]
            else:
                loss = loss_fn_l1(pred, y_b)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / n_batches
        history.append(avg_loss)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"    Epoch {epoch:3d}/{n_epochs}  loss: {avg_loss:.4f}")

    return history


# ─────────────────────────────────────────────────────────────
# STEP 4: Evaluation function
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_condition(model, targets, noisys, segs, condition_name):
    """
    Runs inference and computes all metrics from the study protocol.
    Returns a results dict and prints a summary.
    """
    model.eval()
    psnr_list, ssim_list, nrmse_list = [], [], []
    dsc_wt_list, dsc_tc_list, dsc_et_list = [], [], []
    vol_pred_list, vol_true_list = [], []

    N = len(targets)
    for i in range(N):
        x_in  = noisys[i:i+1]
        y_true = targets[i].numpy()
        s_true = segs[i].numpy()

        pred = model(x_in, segs[i:i+1]).squeeze(0).numpy()

        # Image quality
        iq = compute_image_quality_metrics(pred, y_true)
        psnr_list.append(iq["psnr"])
        ssim_list.append(iq["ssim"])
        nrmse_list.append(iq["nrmse"])

        # Segmentation — threshold T1ce channel (index 1) as proxy segmentation
        # In real experiments this uses a trained segmentation network.
        # Here we threshold to get a binary tumour mask for WT/TC/ET evaluation.
        pred_wt = (pred[1] > 0.45).astype(bool)
        true_wt = (s_true[0] > 0).astype(bool)
        pred_tc = (pred[1] > 0.60).astype(bool)
        true_tc = ((s_true[0] == 1) | (s_true[0] == 3)).astype(bool)
        pred_et = (pred[1] > 0.75).astype(bool)
        true_et = (s_true[0] == 3).astype(bool)

        from evaluation import dice_score
        dsc_wt_list.append(dice_score(pred_wt, true_wt))
        dsc_tc_list.append(dice_score(pred_tc, true_tc))
        dsc_et_list.append(dice_score(pred_et, true_et))

        # Volume consistency (for ICC / Bland-Altman)
        vol_pred_list.append(float(pred_wt.sum()))
        vol_true_list.append(float(true_wt.sum()))

    vol_pred = np.array(vol_pred_list)
    vol_true = np.array(vol_true_list)

    results = {
        "PSNR":    np.mean(psnr_list),
        "SSIM":    np.mean(ssim_list),
        "NRMSE":   np.mean(nrmse_list),
        "DSC_WT":  np.mean(dsc_wt_list),
        "DSC_TC":  np.mean(dsc_tc_list),
        "DSC_ET":  np.mean(dsc_et_list),
        "ICC":     icc(vol_pred, vol_true),
        "BA_bias": bland_altman(vol_pred, vol_true)["bias"],
    }
    return results


# ─────────────────────────────────────────────────────────────
# STEP 5: Print results table (mirrors Table in Section 7)
# ─────────────────────────────────────────────────────────────

def print_results_table(all_results):
    conditions = list(all_results.keys())
    metrics    = list(all_results[conditions[0]].keys())

    col_w = 16
    header = f"{'Metric':<14}" + "".join(f"{c:>{col_w}}" for c in conditions)
    print("\n" + "=" * (14 + col_w * len(conditions)))
    print(header)
    print("-" * (14 + col_w * len(conditions)))

    # Direction of improvement
    higher_is_better = {"PSNR", "SSIM", "DSC_WT", "DSC_TC", "DSC_ET", "ICC"}

    for m in metrics:
        vals = {c: all_results[c][m] for c in conditions}
        best = max(vals, key=lambda c: vals[c] if m in higher_is_better else -vals[c])
        row = f"{m:<14}"
        for c in conditions:
            v = vals[c]
            mark = " *" if c == best else "  "
            row += f"{v:>{col_w - 2}.4f}{mark}"
        print(row)

    print("=" * (14 + col_w * len(conditions)))
    print("* = best value for this metric\n")


# ─────────────────────────────────────────────────────────────
# MAIN — runs the full comparative experiment
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PP-MAE Synthetic Comparative Experiment")
    print("Conditions: Baseline | Standard U-Net | PP-MAE")
    print("=" * 60)

    # ── Generate data ────────────────────────────────────────
    print("\n[1/5] Generating synthetic glioma MRI dataset...")
    N_TRAIN, N_TEST = 48, 20
    H, W = 128, 128
    NOISE = 0.08

    targets_tr, noisys_tr, segs_tr = make_dataset(N_TRAIN, H, W, NOISE)
    targets_te, noisys_te, segs_te = make_dataset(N_TEST,  H, W, NOISE)
    print(f"       Train: {N_TRAIN} slices | Test: {N_TEST} slices")
    print(f"       Image size: {H}x{W}, Modalities: 4, Noise σ: {NOISE}")
    print(f"       Tumour voxels (train mean): "
          f"{(segs_tr > 0).float().mean().item() * 100:.1f}% of pixels")

    # ── Condition A: No denoising (identity) ─────────────────
    print("\n[2/5] Condition A — No denoising (lower-bound baseline)")
    print("       Applying noisy images directly to evaluation...")

    class IdentityModel(nn.Module):
        """Pass-through — represents applying no denoising."""
        def forward(self, x, seg=None):
            return x

    baseline_model = IdentityModel()
    results_A = evaluate_condition(baseline_model, targets_te, noisys_te, segs_te,
                                   "No Denoising")
    print("       Done.")

    # ── Condition B: Standard U-Net ───────────────────────────
    print("\n[3/5] Condition B — Standard U-Net (no pathology loss)")
    print("       Training with plain L1 loss...")
    t0 = time.time()
    model_B = StandardUNet(ch=4, base=32)
    hist_B  = train_model(model_B, targets_tr, noisys_tr, segs_tr,
                          n_epochs=30, batch_size=8,
                          use_pathology_loss=False, lr=1e-3)
    print(f"       Training complete ({time.time()-t0:.1f}s) | "
          f"Final loss: {hist_B[-1]:.4f}")
    results_B = evaluate_condition(model_B, targets_te, noisys_te, segs_te,
                                   "Standard U-Net")

    # ── Condition C: PP-MAE ───────────────────────────────────
    print("\n[4/5] Condition C — PP-MAE (pathology-aware loss)")
    print("       Training with L_global + λ1·L_pathology + λ2·L_crossmodal...")
    t0 = time.time()

    # Import Option 1 CNN backbone for PP-MAE
    from option1_cnn_pp_mae import CNNPPMAE
    model_C = CNNPPMAE(in_channels=4, base_ch=32, depth=3)
    hist_C  = train_model(model_C, targets_tr, noisys_tr, segs_tr,
                          n_epochs=30, batch_size=8,
                          use_pathology_loss=True, lr=1e-3)
    print(f"       Training complete ({time.time()-t0:.1f}s) | "
          f"Final loss: {hist_C[-1]:.4f}")
    results_C = evaluate_condition(model_C, targets_te, noisys_te, segs_te,
                                   "PP-MAE")

    # ── Print results ─────────────────────────────────────────
    print("\n[5/5] Results Table")
    all_results = {
        "No Denoising":   results_A,
        "Standard U-Net": results_B,
        "PP-MAE":         results_C,
    }
    print_results_table(all_results)

    # ── Loss curves ───────────────────────────────────────────
    print("Loss Curves (every 5 epochs):")
    print(f"  {'Epoch':<8} {'Standard U-Net':>16} {'PP-MAE':>10}")
    for i in range(0, 30, 5):
        print(f"  {i+1:<8} {hist_B[i]:>16.4f} {hist_C[i]:>10.4f}")

    # ── Interpretation ────────────────────────────────────────
    print("\nInterpretation:")
    psnr_gain_b = results_B["PSNR"] - results_A["PSNR"]
    psnr_gain_c = results_C["PSNR"] - results_A["PSNR"]
    dsc_gain    = results_C["DSC_ET"] - results_B["DSC_ET"]
    print(f"  PSNR gain over no-denoising: "
          f"Standard={psnr_gain_b:+.2f} dB | PP-MAE={psnr_gain_c:+.2f} dB")
    print(f"  ET DSC gain (PP-MAE vs Standard): {dsc_gain:+.4f}")
    print(f"  ICC (volume consistency): "
          f"Standard={results_B['ICC']:.4f} | PP-MAE={results_C['ICC']:.4f}")

    print("\nNext steps:")
    print("  1. Download BraTS 2023 data → train at full scale on GPU cluster")
    print("  2. Replace threshold segmentation with trained nnU-Net head")
    print("  3. Run Option 3 end-to-end pipeline for grading AUROC")
    print("  4. Conduct radiologist qualitative review (Likert scale)")
