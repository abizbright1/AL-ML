"""
run_grading_round.py  —  Round 5: Joint Denoising + Segmentation + Grading
===========================================================================

Trains all models end-to-end (or sequentially) on real BraTS 2021 data,
evaluates reconstruction, segmentation, and grading, then saves results.

Pipeline variants compared
--------------------------
  PP-MAE-Joint        : SwinPPMAE denoiser + UNet seg + GradingHead,
                        trained jointly with combined loss
  PP-MAE-Seq          : SwinPPMAE denoiser → UNet seg → GradingHead,
                        trained sequentially (stage-wise)
  SwinIR-Seq          : SwinIR-lite + UNet seg + GradingHead (sequential)
  Uformer-Seq         : Uformer-lite + UNet seg + GradingHead (sequential)
  RadioTransformer    : stand-alone transformer grader on radiomic features
  CBAM-ResNet         : stand-alone CNN grader on denoised MRI slices

Saved outputs
-------------
  results/round5_grading/options_results.csv   — reconstruction + seg metrics
  results/round5_grading/grading_results.csv   — AUC, Acc, Sens, Spec per model
  results/round5_grading/grade_labels.json     — subject → grade mapping used

Usage (MacBook MPS)
-------------------
  python3 run_grading_round.py ~/Downloads/BraTS2021_data \\
      --device mps --epochs 30 --seg_epochs 20 --grade_epochs 20 \\
      --max_subjects 50 --out results/round5_grading/

Quick smoke-test (3 subjects, 2 epochs):
  python3 run_grading_round.py ~/Downloads/BraTS2021_data \\
      --device mps --epochs 2 --seg_epochs 2 --grade_epochs 2 \\
      --max_subjects 3 --out results/round5_smoke/
"""

from __future__ import annotations
import argparse, csv, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "pp_mae"))

from option4_swin_pp_mae import SwinPPMAE
from option_baselines    import (SwinIRLite, SwinIRTrainer,
                                  UformerLite, UformerTrainer,
                                  SwinIRPathologyTrainer,
                                  UformerPathologyTrainer)
from segmentor           import UNetSegmentor
from brats_loader        import BraTSDataset
from grading             import (extract_grading_features,
                                  aggregate_subject_features,
                                  GradingHead, GradingTrainer,
                                  grading_metrics, assign_demo_grade_labels,
                                  get_roc_curve)
from grading_baselines   import (RadioTransformer, RadioTransformerTrainer,
                                  CBAMResNet, SliceGradingTrainer)


# ── Dice / PSNR helpers ───────────────────────────────────────────────────────
def _psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = F.mse_loss(pred, gt).item()
    return 20 * np.log10(1.0 / np.sqrt(max(mse, 1e-10)))

def _ssim_approx(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mu_p, mu_g = pred.mean(), gt.mean()
    sig_p = pred.std(); sig_g = gt.std()
    sig_pg = ((pred - mu_p) * (gt - mu_g)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(((2*mu_p*mu_g + c1)*(2*sig_pg + c2)) /
                 ((mu_p**2 + mu_g**2 + c1)*(sig_p**2 + sig_g**2 + c2)))

def _nrmse(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return float(torch.sqrt(F.mse_loss(pred, gt)) / (gt.max() - gt.min() + 1e-8))

def _dice(pred_seg: np.ndarray, gt_seg: np.ndarray, region: str) -> float:
    masks = {"WT": gt_seg > 0,
             "TC": (gt_seg == 1) | (gt_seg == 3),
             "ET": gt_seg == 3}
    pred_masks = {"WT": pred_seg > 0,
                  "TC": (pred_seg == 1) | (pred_seg == 3),
                  "ET": pred_seg == 3}
    p, g = pred_masks[region].astype(float), masks[region].astype(float)
    inter = (p * g).sum()
    denom = p.sum() + g.sum()
    return float(2 * inter / max(denom, 1e-6))


# ── Joint PP-MAE model ────────────────────────────────────────────────────────
class JointPPMAE(nn.Module):
    """
    Single model trained with combined loss:
      L = L_path_recon + w_seg * L_dice_ce + w_grade * L_bce_grade
    """
    def __init__(self, in_ch: int, out_ch: int, device):
        super().__init__()
        self.denoiser = SwinPPMAE(in_channels=in_ch, out_channels=out_ch)
        self.segmentor = UNetSegmentor(in_channels=out_ch, num_classes=4)
        self.grader    = GradingHead(in_features=7, num_classes=2)
        self.w_seg     = 0.5
        self.w_grade   = 0.3

    def forward(self, noisy, seg_map=None):
        if seg_map is None:
            seg_map = torch.zeros(noisy.shape[0], 1, *noisy.shape[2:],
                                  device=noisy.device, dtype=torch.long)
        denoised    = self.denoiser(noisy, seg_map)
        seg_logits  = self.segmentor(denoised)
        grade_feats = extract_grading_features(denoised, seg_logits)
        grade_logits = self.grader(grade_feats)
        return denoised, seg_logits, grade_logits

    def loss(self, noisy, gt, seg_gt, grade_label,
             recon_loss_fn, seg_loss_fn, grade_loss_fn):
        denoised, seg_logits, grade_logits = self(noisy, seg_gt.unsqueeze(1).float())
        l_recon = recon_loss_fn(denoised, gt)
        l_seg   = seg_loss_fn(seg_logits, seg_gt)
        l_grade = grade_loss_fn(grade_logits,
                                grade_label.to(noisy.device))
        return (l_recon + self.w_seg * l_seg + self.w_grade * l_grade,
                l_recon.item(), l_seg.item(), l_grade.item())


# ── Dataset helpers ───────────────────────────────────────────────────────────
def _build_datasets(data_dir: str, max_subjects: int):
    n_train = max(int(max_subjects * 0.8), 2)
    n_val   = max(max_subjects - n_train, 1)
    train_ds = BraTSDataset(data_dir, max_subjects=n_train, mode="train")
    val_ds   = BraTSDataset(data_dir, max_subjects=n_val,   mode="val")
    return train_ds, val_ds


def _collate(batch):
    inp = torch.stack([b["input"]  for b in batch])
    tgt = torch.stack([b["target"] for b in batch])
    seg = torch.stack([torch.from_numpy(b["seg"]) if isinstance(b["seg"], np.ndarray)
                       else b["seg"] for b in batch])
    subj = [b.get("subject", f"subj_{i}") for i, b in enumerate(batch)]
    return inp, tgt, seg, subj


# ── Stage 1: train denoiser ───────────────────────────────────────────────────
def _train_denoiser(model, train_ds, epochs, device, loss_fn, desc=""):
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=4, shuffle=True,
        collate_fn=_collate, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for inp, tgt, seg, _ in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            opt.zero_grad()
            out = model(inp)
            if isinstance(out, (list, tuple)):
                out = out[0]
            loss = loss_fn(out, tgt)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        sched.step()
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    [{desc}] ep {ep+1}/{epochs}  loss={ep_loss/len(loader):.4f}")
    model.eval()


# ── Stage 2: train segmentor ──────────────────────────────────────────────────
def _train_segmentor(denoiser, seg_model, train_ds, epochs, device, desc=""):
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=4, shuffle=True,
        collate_fn=_collate, drop_last=False)
    opt  = torch.optim.Adam(seg_model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    seg_model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for inp, tgt, seg_gt, _ in loader:
            inp = inp.to(device)
            seg_gt = seg_gt.long().to(device)
            with torch.no_grad():
                dn = denoiser(inp)
                if isinstance(dn, (list, tuple)):
                    dn = dn[0]
            opt.zero_grad()
            pred = seg_model(dn)
            loss = loss_fn(pred, seg_gt)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        sched.step()
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    [{desc} seg] ep {ep+1}/{epochs}  loss={ep_loss/len(loader):.4f}")
    seg_model.eval()


# ── Stage 3: train grader ─────────────────────────────────────────────────────
def _train_grader(denoiser, seg_model, grader, train_ds,
                  grade_labels: dict, epochs: int, device, desc=""):
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=4, shuffle=True,
        collate_fn=_collate, drop_last=False)
    opt     = torch.optim.Adam(grader.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    grader.train()
    for ep in range(epochs):
        ep_loss = 0.0
        n_batches = 0
        for inp, tgt, seg_gt, subjs in loader:
            labels = [grade_labels.get(s, 0) for s in subjs]
            if all(l == labels[0] for l in labels):
                continue   # skip all-same-class batches — unhelpful for grader
            inp   = inp.to(device)
            label_t = torch.tensor(labels, dtype=torch.long, device=device)
            with torch.no_grad():
                dn = denoiser(inp)
                if isinstance(dn, (list, tuple)):
                    dn = dn[0]
                seg_logits = seg_model(dn)
                feats = extract_grading_features(dn, seg_logits)
            opt.zero_grad()
            logits = grader(feats)
            loss   = loss_fn(logits, label_t)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batches += 1
        if n_batches and (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    [{desc} grade] ep {ep+1}/{epochs}  "
                  f"loss={ep_loss/n_batches:.4f}")
    grader.eval()


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def _evaluate(denoiser, seg_model, grader, val_ds, grade_labels, device):
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=4, shuffle=False,
        collate_fn=_collate, drop_last=False)

    psnrs, ssims, nrmses = [], [], []
    dice_wt, dice_tc, dice_et = [], [], []
    all_probs, all_labels = [], []

    for inp, tgt, seg_gt, subjs in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        seg_gt_np = seg_gt.numpy()

        dn = denoiser(inp)
        if isinstance(dn, (list, tuple)):
            dn = dn[0]

        for b in range(inp.shape[0]):
            psnrs.append(_psnr(dn[b], tgt[b]))
            ssims.append(_ssim_approx(dn[b], tgt[b]))
            nrmses.append(_nrmse(dn[b], tgt[b]))

        seg_logits = seg_model(dn)
        seg_pred_np = seg_logits.argmax(1).cpu().numpy()
        for b in range(inp.shape[0]):
            for region, lst in [("WT", dice_wt), ("TC", dice_tc), ("ET", dice_et)]:
                lst.append(_dice(seg_pred_np[b], seg_gt_np[b], region))

        feats   = extract_grading_features(dn, seg_logits)
        logits  = grader(feats)
        probs   = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        labels  = np.array([grade_labels.get(s, 0) for s in subjs])
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())

    recon = dict(PSNR=np.mean(psnrs), SSIM=np.mean(ssims), NRMSE=np.mean(nrmses))
    seg   = dict(Dice_WT=np.mean(dice_wt), Dice_TC=np.mean(dice_tc),
                 Dice_ET=np.mean(dice_et))

    all_probs  = torch.tensor(all_probs)
    all_labels = torch.tensor(all_labels)
    grade = grading_metrics(all_probs, all_labels)
    fprs, tprs = get_roc_curve(all_probs.numpy(), all_labels.numpy())

    return recon, seg, grade, (fprs, tprs)


# ── Evaluate joint model ──────────────────────────────────────────────────────
@torch.no_grad()
def _evaluate_joint(joint_model, val_ds, grade_labels, device):
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=4, shuffle=False,
        collate_fn=_collate, drop_last=False)

    psnrs, ssims, nrmses = [], [], []
    dice_wt, dice_tc, dice_et = [], [], []
    all_probs, all_labels = [], []

    for inp, tgt, seg_gt, subjs in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        seg_gt_np = seg_gt.numpy()

        denoised, seg_logits, grade_logits = joint_model(
            inp, seg_gt.long().unsqueeze(1).to(device))

        for b in range(inp.shape[0]):
            psnrs.append(_psnr(denoised[b], tgt[b]))
            ssims.append(_ssim_approx(denoised[b], tgt[b]))
            nrmses.append(_nrmse(denoised[b], tgt[b]))

        seg_pred_np = seg_logits.argmax(1).cpu().numpy()
        for b in range(inp.shape[0]):
            for region, lst in [("WT", dice_wt), ("TC", dice_tc), ("ET", dice_et)]:
                lst.append(_dice(seg_pred_np[b], seg_gt_np[b], region))

        probs  = torch.softmax(grade_logits, 1)[:, 1].cpu().numpy()
        labels = np.array([grade_labels.get(s, 0) for s in subjs])
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())

    recon = dict(PSNR=np.mean(psnrs), SSIM=np.mean(ssims), NRMSE=np.mean(nrmses))
    seg   = dict(Dice_WT=np.mean(dice_wt), Dice_TC=np.mean(dice_tc),
                 Dice_ET=np.mean(dice_et))
    all_probs  = torch.tensor(all_probs)
    all_labels = torch.tensor(all_labels)
    grade = grading_metrics(all_probs, all_labels)
    fprs, tprs = get_roc_curve(all_probs.numpy(), all_labels.numpy())
    return recon, seg, grade, (fprs, tprs)


# ── Save results ──────────────────────────────────────────────────────────────
def _save_results(all_results: list, out_dir: str, roc_data: dict):
    os.makedirs(out_dir, exist_ok=True)

    recon_path = os.path.join(out_dir, "options_results.csv")
    with open(recon_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "PSNR", "SSIM", "NRMSE",
                    "Dice_WT", "Dice_TC", "Dice_ET"])
        for r in all_results:
            w.writerow([
                r["name"],
                f"{r['recon']['PSNR']:.4f}",
                f"{r['recon']['SSIM']:.4f}",
                f"{r['recon']['NRMSE']:.4f}",
                f"{r['seg']['Dice_WT']:.4f}",
                f"{r['seg']['Dice_TC']:.4f}",
                f"{r['seg']['Dice_ET']:.4f}",
            ])
    print(f"  Saved {recon_path}")

    grade_path = os.path.join(out_dir, "grading_results.csv")
    with open(grade_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "AUC", "Accuracy", "Sensitivity", "Specificity", "F1"])
        for r in all_results:
            g = r["grade"]
            w.writerow([r["name"],
                        f"{g['auc']:.4f}", f"{g['accuracy']:.4f}",
                        f"{g['sensitivity']:.4f}", f"{g['specificity']:.4f}",
                        f"{g['f1']:.4f}"])
    print(f"  Saved {grade_path}")

    roc_path = os.path.join(out_dir, "roc_curves.json")
    with open(roc_path, "w") as f:
        json.dump({k: {"fprs": v[0].tolist(), "tprs": v[1].tolist()}
                   for k, v in roc_data.items()}, f)
    print(f"  Saved {roc_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--device",       default="mps")
    ap.add_argument("--epochs",       type=int, default=30)
    ap.add_argument("--seg_epochs",   type=int, default=20)
    ap.add_argument("--grade_epochs", type=int, default=20)
    ap.add_argument("--max_subjects", type=int, default=50)
    ap.add_argument("--out",          default="results/round5_grading")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"\n[Round 5 — Grading] device={args.device}  "
          f"subjects={args.max_subjects}  "
          f"epochs={args.epochs}/{args.seg_epochs}/{args.grade_epochs}\n")

    # ── Data ──────────────────────────────────────────────────────
    train_ds, val_ds = _build_datasets(args.data_dir, args.max_subjects)
    in_ch  = train_ds[0]["input"].shape[0]
    out_ch = train_ds[0]["target"].shape[0]
    print(f"  Train slices: {len(train_ds)}  |  Val slices: {len(val_ds)}")
    print(f"  Channels: in={in_ch}  out={out_ch}")

    # Grade labels (ET-volume median split)
    grade_labels = assign_demo_grade_labels(train_ds)
    val_grade_labels = assign_demo_grade_labels(val_ds)
    all_grade_labels = {**grade_labels, **val_grade_labels}

    all_results, roc_data = [], {}

    # ── Model 1: PP-MAE Joint ──────────────────────────────────────
    print("\n[1/6] PP-MAE-Joint (end-to-end joint training)")
    joint = JointPPMAE(in_ch, out_ch, device).to(device)
    recon_loss = nn.L1Loss()
    seg_loss   = nn.CrossEntropyLoss()
    grade_loss = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(joint.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-5)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=4, shuffle=True, collate_fn=_collate)
    joint.train()
    for ep in range(args.epochs):
        ep_loss = 0.0
        for inp, tgt, seg_gt, subjs in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            seg_gt   = seg_gt.long().to(device)
            labels   = torch.tensor([all_grade_labels.get(s, 0) for s in subjs],
                                    dtype=torch.long)
            opt.zero_grad()
            total, lr, ls, lg = joint.loss(
                inp, tgt, seg_gt, labels,
                recon_loss, seg_loss, grade_loss)
            total.backward(); opt.step()
            ep_loss += total.item()
        sched.step()
        if (ep+1) % max(1, args.epochs//5) == 0:
            print(f"  ep {ep+1}/{args.epochs}  loss={ep_loss/len(loader):.4f}")
    joint.eval()
    r, s, g, roc = _evaluate_joint(joint, val_ds, all_grade_labels, device)
    all_results.append(dict(name="PP-MAE-Joint", recon=r, seg=s, grade=g))
    roc_data["PP-MAE-Joint"] = roc
    print(f"  → PSNR={r['PSNR']:.2f}  Dice_ET={s['Dice_ET']:.4f}  AUC={g['auc']:.4f}")

    # ── Model 2: PP-MAE Sequential ────────────────────────────────
    print("\n[2/6] PP-MAE-Sequential (stage-wise training)")
    pp_den = SwinPPMAE(in_channels=in_ch, out_channels=out_ch).to(device)
    pp_seg = UNetSegmentor(in_channels=out_ch, num_classes=4).to(device)
    pp_grd = GradingHead(in_features=7, num_classes=2).to(device)
    from pp_mae.losses import PathologyLoss
    path_loss = PathologyLoss(mode="clinical_risk").to(device)
    _train_denoiser(pp_den, train_ds, args.epochs, device, path_loss, "PP-MAE-Seq")
    _train_segmentor(pp_den, pp_seg, train_ds, args.seg_epochs, device, "PP-MAE-Seq")
    _train_grader(pp_den, pp_seg, pp_grd, train_ds,
                  all_grade_labels, args.grade_epochs, device, "PP-MAE-Seq")
    r, s, g, roc = _evaluate(pp_den, pp_seg, pp_grd, val_ds, all_grade_labels, device)
    all_results.append(dict(name="PP-MAE-Seq", recon=r, seg=s, grade=g))
    roc_data["PP-MAE-Seq"] = roc
    print(f"  → PSNR={r['PSNR']:.2f}  Dice_ET={s['Dice_ET']:.4f}  AUC={g['auc']:.4f}")

    # ── Model 3: SwinIR Sequential ────────────────────────────────
    print("\n[3/6] SwinIR-lite + GradingHead (sequential)")
    sw_den = SwinIRLite(in_channels=in_ch, out_channels=out_ch).to(device)
    sw_seg = UNetSegmentor(in_channels=out_ch, num_classes=4).to(device)
    sw_grd = GradingHead(in_features=7, num_classes=2).to(device)
    _train_denoiser(sw_den, train_ds, args.epochs, device, nn.L1Loss(), "SwinIR-Seq")
    _train_segmentor(sw_den, sw_seg, train_ds, args.seg_epochs, device, "SwinIR-Seq")
    _train_grader(sw_den, sw_seg, sw_grd, train_ds,
                  all_grade_labels, args.grade_epochs, device, "SwinIR-Seq")
    r, s, g, roc = _evaluate(sw_den, sw_seg, sw_grd, val_ds, all_grade_labels, device)
    all_results.append(dict(name="SwinIR-Seq", recon=r, seg=s, grade=g))
    roc_data["SwinIR-Seq"] = roc
    print(f"  → PSNR={r['PSNR']:.2f}  Dice_ET={s['Dice_ET']:.4f}  AUC={g['auc']:.4f}")

    # ── Model 4: Uformer Sequential ───────────────────────────────
    print("\n[4/6] Uformer-lite + GradingHead (sequential)")
    uf_den = UformerLite(in_channels=in_ch, out_channels=out_ch).to(device)
    uf_seg = UNetSegmentor(in_channels=out_ch, num_classes=4).to(device)
    uf_grd = GradingHead(in_features=7, num_classes=2).to(device)
    _train_denoiser(uf_den, train_ds, args.epochs, device, nn.L1Loss(), "Uformer-Seq")
    _train_segmentor(uf_den, uf_seg, train_ds, args.seg_epochs, device, "Uformer-Seq")
    _train_grader(uf_den, uf_seg, uf_grd, train_ds,
                  all_grade_labels, args.grade_epochs, device, "Uformer-Seq")
    r, s, g, roc = _evaluate(uf_den, uf_seg, uf_grd, val_ds, all_grade_labels, device)
    all_results.append(dict(name="Uformer-Seq", recon=r, seg=s, grade=g))
    roc_data["Uformer-Seq"] = roc
    print(f"  → PSNR={r['PSNR']:.2f}  Dice_ET={s['Dice_ET']:.4f}  AUC={g['auc']:.4f}")

    # ── Model 5: RadioTransformer (standalone grading baseline) ───
    print("\n[5/6] RadioTransformer (standalone grading baseline)")
    # Uses SwinIR-denoised MRI → radiomic features → transformer grader
    rt_trainer = RadioTransformerTrainer(
        in_features=7, device=str(device))
    for ep in range(args.grade_epochs):
        loader2 = torch.utils.data.DataLoader(
            train_ds, batch_size=4, shuffle=True, collate_fn=_collate)
        for inp, tgt, seg_gt, subjs in loader2:
            inp = inp.to(device)
            labels = torch.tensor([all_grade_labels.get(s, 0) for s in subjs],
                                   dtype=torch.long)
            with torch.no_grad():
                dn = sw_den(inp)
                if isinstance(dn, (list, tuple)):
                    dn = dn[0]
                seg_logits = sw_seg(dn)
                feats = extract_grading_features(dn, seg_logits)
            rt_trainer.step(feats, labels)
        if (ep+1) % max(1, args.grade_epochs//5) == 0:
            print(f"  RadioTransformer ep {ep+1}/{args.grade_epochs}")
    # Evaluate RadioTransformer (uses SwinIR recon + seg)
    loader3 = torch.utils.data.DataLoader(
        val_ds, batch_size=4, shuffle=False, collate_fn=_collate)
    rt_probs, rt_labels = [], []
    for inp, tgt, seg_gt, subjs in loader3:
        inp = inp.to(device)
        with torch.no_grad():
            dn = sw_den(inp)
            if isinstance(dn, (list, tuple)):
                dn = dn[0]
            seg_logits = sw_seg(dn)
            feats = extract_grading_features(dn, seg_logits)
            probs = rt_trainer.predict(feats)
        rt_probs.extend(probs[:, 1].numpy().tolist())
        rt_labels.extend([all_grade_labels.get(s, 0) for s in subjs])
    rt_probs_t  = torch.tensor(rt_probs)
    rt_labels_t = torch.tensor(rt_labels)
    g_rt = grading_metrics(rt_probs_t, rt_labels_t)
    roc_data["RadioTransformer"] = get_roc_curve(
        rt_probs_t.numpy(), rt_labels_t.numpy())
    all_results.append(dict(name="RadioTransformer",
                            recon={"PSNR": 0, "SSIM": 0, "NRMSE": 0},
                            seg={"Dice_WT": 0, "Dice_TC": 0, "Dice_ET": 0},
                            grade=g_rt))
    print(f"  → AUC={g_rt['auc']:.4f}  Acc={g_rt['accuracy']:.4f}")

    # ── Model 6: CBAM-ResNet (standalone) ─────────────────────────
    print("\n[6/6] CBAM-ResNet (end-to-end grading baseline)")
    cbam = CBAMResNet(in_channels=out_ch, num_classes=2).to(device)
    cbam_trainer = SliceGradingTrainer(cbam, device=str(device))
    for ep in range(args.grade_epochs):
        loader4 = torch.utils.data.DataLoader(
            train_ds, batch_size=4, shuffle=True, collate_fn=_collate)
        for inp, tgt, seg_gt, subjs in loader4:
            inp = inp.to(device)
            labels = torch.tensor([all_grade_labels.get(s, 0) for s in subjs],
                                   dtype=torch.long)
            with torch.no_grad():
                dn = sw_den(inp)
                if isinstance(dn, (list, tuple)):
                    dn = dn[0]
            cbam_trainer.step(dn, labels)
        if (ep+1) % max(1, args.grade_epochs//5) == 0:
            print(f"  CBAM-ResNet ep {ep+1}/{args.grade_epochs}")
    cbam_probs, cbam_labels = [], []
    for inp, tgt, seg_gt, subjs in loader3:
        inp = inp.to(device)
        with torch.no_grad():
            dn = sw_den(inp)
            if isinstance(dn, (list, tuple)):
                dn = dn[0]
            probs = cbam_trainer.predict(dn)
        cbam_probs.extend(probs[:, 1].numpy().tolist())
        cbam_labels.extend([all_grade_labels.get(s, 0) for s in subjs])
    cbam_probs_t  = torch.tensor(cbam_probs)
    cbam_labels_t = torch.tensor(cbam_labels)
    g_cbam = grading_metrics(cbam_probs_t, cbam_labels_t)
    roc_data["CBAM-ResNet"] = get_roc_curve(
        cbam_probs_t.numpy(), cbam_labels_t.numpy())
    all_results.append(dict(name="CBAM-ResNet",
                            recon={"PSNR": 0, "SSIM": 0, "NRMSE": 0},
                            seg={"Dice_WT": 0, "Dice_TC": 0, "Dice_ET": 0},
                            grade=g_cbam))
    print(f"  → AUC={g_cbam['auc']:.4f}  Acc={g_cbam['accuracy']:.4f}")

    # ── Save ──────────────────────────────────────────────────────
    _save_results(all_results, args.out, roc_data)

    print("\n" + "="*60)
    print("  ROUND 5 — FINAL RESULTS")
    print("="*60)
    print(f"  {'Method':<22} {'PSNR':>6} {'Dice_ET':>8} {'AUC':>7} {'Acc':>7}")
    print("  " + "-"*54)
    for r in all_results:
        print(f"  {r['name']:<22} "
              f"{r['recon']['PSNR']:>6.2f} "
              f"{r['seg']['Dice_ET']:>8.4f} "
              f"{r['grade']['auc']:>7.4f} "
              f"{r['grade']['accuracy']:>7.4f}")
    print()


if __name__ == "__main__":
    main()
