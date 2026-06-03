"""
MPS backward-crash locator.
Runs ONE training step of Swin PP-MAE on MPS with anomaly detection ON.
PyTorch will print the exact forward line that produced the failing backward
node, instead of the useless C++ engine traceback.

Usage:
    python3 diagnose_mps.py
"""
import sys, os
sys.path.insert(0, "pp_mae")
sys.path.insert(0, ".")

import torch

# This is the key line: it makes autograd remember where each op was created,
# so the backward error points at the real forward line.
torch.autograd.set_detect_anomaly(True)

from option4_swin_pp_mae import SwinPPMAE, SwinPPMAETrainer

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# Same config run_all_options.py uses for Round 4
model = SwinPPMAE(in_ch=4, embed_dim=96, depths=(2, 2, 6, 2),
                  n_heads=(3, 6, 12, 24), window_size=7)
trainer = SwinPPMAETrainer(model, device=device)

B, H, W = 2, 96, 96
batch = {
    "noisy":  torch.rand(B, 4, H, W),
    "target": torch.rand(B, 4, H, W),
    "seg":    torch.randint(0, 4, (B, 1, H, W)),
}

print("Running one step with anomaly detection...")
try:
    out = trainer.step(batch)
    print("NO CRASH — step succeeded:", {k: round(v, 4) for k, v in out.items()})
except RuntimeError as e:
    print("\n================= CRASH LOCATED =================")
    print(e)
    print("================================================")
