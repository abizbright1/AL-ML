"""
Unified training entrypoint — select option via --option flag.

Usage:
    python train.py --option 1 --data_root /path/to/data --epochs 200
    python train.py --option 2 --data_root /path/to/data --epochs 200 --patch_size 16
    python train.py --option 3 --data_root /path/to/data --stage 1 --epochs 100
    python train.py --option 4 --data_root /path/to/data --epochs 200

Checkpoints are saved to ./checkpoints/option{N}/epoch_{E}.pt
Metrics are logged to ./logs/option{N}_metrics.jsonl  (JSON-lines)
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_utils import GliomaSliceDataset, GliomaPatchDataset
from option1_cnn_pp_mae  import CNNPPMAE, PPMAETrainer
from option2_vit_pp_mae  import ViTPPMAE, ViTPPMAETrainer
from option3_full_pipeline import PPMAEPipeline, PipelineTrainer
from option4_swin_pp_mae import SwinPPMAE, SwinPPMAETrainer


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--option",    type=int, required=True, choices=[1, 2, 3, 4])
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--epochs",    type=int, default=200)
    p.add_argument("--batch_size",type=int, default=4)
    p.add_argument("--lr",        type=float, default=1e-4)
    p.add_argument("--lambda1",   type=float, default=1.0,
                   help="Weight for pathology loss")
    p.add_argument("--lambda2",   type=float, default=0.5,
                   help="Weight for cross-modal consistency loss")
    p.add_argument("--patch_size",type=int, default=96,
                   help="3D patch size for Options 2 & 4")
    p.add_argument("--stage",     type=int, default=1, choices=[1, 2],
                   help="Training stage for Option 3")
    p.add_argument("--ckpt_dir",  type=str, default="./checkpoints")
    p.add_argument("--log_dir",   type=str, default="./logs")
    p.add_argument("--save_every",type=int, default=10)
    p.add_argument("--device",    type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_loaders(args: argparse.Namespace):
    """Build train and validation dataloaders appropriate for the chosen option."""
    use_3d = args.option in (2,)   # ViT uses 3D patches
    DS = GliomaPatchDataset if use_3d else GliomaSliceDataset

    common = dict(data_root=args.data_root, degrade=True)
    if use_3d:
        common["patch_size"] = args.patch_size

    train_ds = DS(split="train", **common)
    val_ds   = DS(split="val",   degrade=False,
                  **({"data_root": args.data_root, "patch_size": args.patch_size}
                     if use_3d else {"data_root": args.data_root}))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


def build_model_trainer(args: argparse.Namespace):
    dev = args.device

    if args.option == 1:
        model   = CNNPPMAE(in_channels=4, base_ch=64, depth=4).to(dev)
        optim   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        trainer = PPMAETrainer(model, optim, device=dev,
                               lambda1=args.lambda1, lambda2=args.lambda2)
        return model, trainer, "step"

    elif args.option == 2:
        model = ViTPPMAE(
            vol_size=(args.patch_size,) * 3,
            patch_size=16,
            in_chans=4,
            embed_dim=384,
            depth=12,
            n_heads=6,
            decoder_dim=192,
            decoder_depth=4,
        ).to(dev)
        optim   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        trainer = ViTPPMAETrainer(model, optim, device=dev,
                                  lambda1=args.lambda1, lambda2=args.lambda2)
        return model, trainer, "step"

    elif args.option == 3:
        pipeline = PPMAEPipeline({"in_channels": 4, "base_ch": 64, "depth": 4})
        trainer  = PipelineTrainer(pipeline, device=dev)
        step_fn  = "stage1_step" if args.stage == 1 else "stage2_step"
        return pipeline, trainer, step_fn

    elif args.option == 4:
        model = SwinPPMAE(
            in_ch=4, embed_dim=96,
            depths=(2, 2, 6, 2),
            n_heads=(3, 6, 12, 24),
            window_size=7,
        ).to(dev)
        optim   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        trainer = SwinPPMAETrainer(model, optim, device=dev,
                                   lambda1=args.lambda1, lambda2=args.lambda2)
        return model, trainer, "step"


def main():
    args = argparse.Namespace(
        option=1, data_root=".", epochs=1, batch_size=2, lr=1e-4,
        lambda1=1.0, lambda2=0.5, patch_size=96, stage=1,
        ckpt_dir="./checkpoints", log_dir="./logs", save_every=10,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    # Override with CLI args when run directly
    import sys
    if len(sys.argv) > 1:
        args = get_args()

    ckpt_dir = Path(args.ckpt_dir) / f"option{args.option}"
    log_dir  = Path(args.log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True,  exist_ok=True)
    log_path = log_dir / f"option{args.option}_metrics.jsonl"

    model, trainer, step_fn = build_model_trainer(args)

    print(f"PP-MAE Option {args.option} | device: {args.device}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Metrics log: {log_path}")

    try:
        train_loader, val_loader = build_loaders(args)
    except Exception as e:
        print(f"[WARNING] Could not build real dataloaders ({e}). Using synthetic data.")
        train_loader = val_loader = None

    for epoch in range(1, args.epochs + 1):
        # Training
        train_metrics = {}
        if train_loader is not None:
            for batch in train_loader:
                train_metrics = getattr(trainer, step_fn)(batch)
        else:
            # Synthetic fallback for CI / sanity testing
            B, C, H, W = 2, 4, 128, 128
            batch = {
                "noisy":  torch.rand(B, C, H, W),
                "target": torch.rand(B, C, H, W),
                "seg":    torch.randint(0, 4, (B, 1, H, W)),
                "grade":  torch.randint(0, 2, (B,)),
                "idh":    torch.randint(0, 2, (B,)),
            }
            train_metrics = getattr(trainer, step_fn)(batch)

        # Validation
        val_metrics = {}
        if val_loader is not None and hasattr(trainer, "validate"):
            for batch in val_loader:
                val_metrics = trainer.validate(batch)

        log_entry = {
            "epoch": epoch,
            "train": {k: round(v, 5) for k, v in train_metrics.items()},
            "val":   {k: round(v, 5) for k, v in val_metrics.items()},
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(f"Epoch {epoch:04d} | "
              + " | ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items()))

        if epoch % args.save_every == 0:
            state = (model if args.option != 3 else trainer.pipeline).state_dict()
            torch.save(state, ckpt_dir / f"epoch_{epoch:04d}.pt")
            print(f"  → Checkpoint saved.")


if __name__ == "__main__":
    main()
