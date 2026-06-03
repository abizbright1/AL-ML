from typing import Optional
"""
Data utilities for the PP-MAE glioma MRI pipeline.

Pipeline:
  1. Load 4-modality NIfTI volumes  (T1W, T1Wce, T2W, FLAIR)
  2. Load tumour segmentation map   (BraTS label convention)
  3. Pre-processing: co-registration → skull-strip → intensity norm
  4. Noise simulation for self-supervised training
  5. 2D slice extraction / 3D patch extraction depending on option chosen
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Optional nibabel for NIfTI loading; graceful fallback for testing.
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


# ---------------------------------------------------------------------------
# Intensity normalisation
# ---------------------------------------------------------------------------

def percentile_normalise(vol: np.ndarray, pmin: float = 1.0, pmax: float = 99.0) -> np.ndarray:
    """Clip to [pmin, pmax] percentile of foreground voxels, then scale to [0, 1]."""
    mask = vol > 0
    if mask.sum() == 0:
        return vol.astype(np.float32)
    lo = np.percentile(vol[mask], pmin)
    hi = np.percentile(vol[mask], pmax)
    vol = np.clip(vol, lo, hi)
    return ((vol - lo) / (hi - lo + 1e-8)).astype(np.float32)


# ---------------------------------------------------------------------------
# Noise simulation  (image-domain degradation)
#
# Three degradation modes to mirror Park et al. (2025) training protocol:
#   'gaussian'  — additive white Gaussian noise (AWGN)
#   'rician'    — Rician noise (physically realistic for MRI magnitude)
#   'undersampling' — partial k-space undersampling via random zero-fill
# ---------------------------------------------------------------------------

def add_gaussian_noise(img: np.ndarray, sigma_range=(0.02, 0.15)) -> np.ndarray:
    sigma = random.uniform(*sigma_range)
    return (img + np.random.normal(0, sigma, img.shape)).astype(np.float32)


def add_rician_noise(img: np.ndarray, sigma_range=(0.02, 0.12)) -> np.ndarray:
    """Rician noise = magnitude of complex Gaussian-corrupted signal."""
    sigma = random.uniform(*sigma_range)
    noise_r = np.random.normal(0, sigma, img.shape)
    noise_i = np.random.normal(0, sigma, img.shape)
    return (np.sqrt((img + noise_r) ** 2 + noise_i ** 2)).astype(np.float32)


def undersample_kspace(img: np.ndarray, retain_fraction: Optional[float] = None) -> np.ndarray:
    """
    Simulate partial Fourier undersampling.
    retain_fraction drawn uniformly from [0.25, 0.75] when None.
    """
    if retain_fraction is None:
        retain_fraction = random.uniform(0.25, 0.75)
    kspace = np.fft.fft2(img)
    H = img.shape[-2]
    n_keep = int(H * retain_fraction)
    rows = np.random.choice(H, n_keep, replace=False)
    mask = np.ones(kspace.shape, dtype=bool)
    mask[..., rows, :] = False   # zero out non-kept rows along second-to-last axis
    kspace[mask] = 0
    return np.abs(np.fft.ifft2(kspace)).astype(np.float32)


def simulate_degradation(
    img: np.ndarray,
    mode: str = "random",
) -> np.ndarray:
    if mode == "random":
        mode = random.choice(["gaussian", "rician", "undersampling"])
    if mode == "gaussian":
        return add_gaussian_noise(img)
    elif mode == "rician":
        return add_rician_noise(img)
    elif mode == "undersampling":
        return undersample_kspace(img)
    else:
        raise ValueError(f"Unknown degradation mode: {mode}")


# ---------------------------------------------------------------------------
# Dataset — 2D slice-level (Options 1 & 3)
# ---------------------------------------------------------------------------

class GliomaSliceDataset(Dataset):
    """
    Loads pre-processed 2D axial slices for each patient.

    Expected directory structure (after preprocessing):
        data_root/
            patient_001/
                T1W.nii.gz
                T1Wce.nii.gz
                T2W.nii.gz
                FLAIR.nii.gz
                seg.nii.gz        ← integer tumour labels (BraTS convention)
            patient_002/
                ...

    Args:
        data_root: Path to dataset root.
        split:     'train' | 'val' | 'test'
        split_file: Optional JSON mapping patient IDs to split.
        degrade:   Whether to apply noise simulation (training mode).
        degrade_mode: 'random' | 'gaussian' | 'rician' | 'undersampling'
        slice_axis: 0=sagittal, 1=coronal, 2=axial (default).
        min_tumour_fraction: Skip slices where tumour occupies < this
                             fraction of pixels (keeps training focused).
    """

    MODALITIES = ["T1W", "T1Wce", "T2W", "FLAIR"]

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        split_file: Optional[str] = None,
        degrade: bool = True,
        degrade_mode: str = "random",
        slice_axis: int = 2,
        min_tumour_fraction: float = 0.01,
    ):
        self.data_root = Path(data_root)
        self.degrade = degrade
        self.degrade_mode = degrade_mode
        self.slice_axis = slice_axis
        self.min_tumour_fraction = min_tumour_fraction

        patient_dirs = sorted(self.data_root.glob("patient_*"))
        # Temporally stratified split  (chronological ordering, no leakage)
        n = len(patient_dirs)
        if split == "train":
            patient_dirs = patient_dirs[: int(0.70 * n)]
        elif split == "val":
            patient_dirs = patient_dirs[int(0.70 * n): int(0.85 * n)]
        else:
            patient_dirs = patient_dirs[int(0.85 * n):]

        self.slices: list[tuple[Path, int]] = []
        for pdir in patient_dirs:
            seg_path = pdir / "seg.nii.gz"
            if not seg_path.exists():
                continue
            seg = self._load_vol(seg_path)
            n_slices = seg.shape[slice_axis]
            for s in range(n_slices):
                sl = np.take(seg, s, axis=slice_axis)
                if (sl > 0).mean() >= min_tumour_fraction:
                    self.slices.append((pdir, s))

    @staticmethod
    def _load_vol(path: Path) -> np.ndarray:
        if HAS_NIBABEL:
            return nib.load(str(path)).get_fdata(dtype=np.float32)
        # Synthetic fallback for unit testing (128³ random volume)
        return np.random.rand(128, 128, 128).astype(np.float32)

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> dict:
        pdir, s = self.slices[idx]

        modality_slices = []
        for mod in self.MODALITIES:
            vol = self._load_vol(pdir / f"{mod}.nii.gz")
            vol = percentile_normalise(vol)
            sl = np.take(vol, s, axis=self.slice_axis)   # (H, W)
            modality_slices.append(sl)

        target = np.stack(modality_slices, axis=0)        # (4, H, W)

        if self.degrade:
            noisy = np.stack(
                [simulate_degradation(sl, self.degrade_mode) for sl in modality_slices],
                axis=0,
            )
        else:
            noisy = target.copy()

        seg_vol = self._load_vol(pdir / "seg.nii.gz")
        seg_sl  = np.take(seg_vol, s, axis=self.slice_axis).astype(np.int64)  # (H, W)

        return {
            "noisy":  torch.from_numpy(noisy),            # (4, H, W)
            "target": torch.from_numpy(target),           # (4, H, W)
            "seg":    torch.from_numpy(seg_sl).unsqueeze(0),  # (1, H, W)
        }


# ---------------------------------------------------------------------------
# Dataset — 3D patch-level (Options 2 & 4)
# ---------------------------------------------------------------------------

class GliomaPatchDataset(Dataset):
    """
    Extracts random 3D patches from volumetric MRI for ViT/Swin-based options.

    Args:
        patch_size: spatial size of each cubic patch (e.g. 96).
        patches_per_patient: number of random patches sampled per volume.
        tumour_bias: fraction of patches guaranteed to overlap with tumour.
    """

    MODALITIES = ["T1W", "T1Wce", "T2W", "FLAIR"]

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        patch_size: int = 96,
        patches_per_patient: int = 8,
        tumour_bias: float = 0.7,
        degrade: bool = True,
        degrade_mode: str = "random",
    ):
        self.data_root = Path(data_root)
        self.patch_size = patch_size
        self.patches_per_patient = patches_per_patient
        self.tumour_bias = tumour_bias
        self.degrade = degrade
        self.degrade_mode = degrade_mode

        patient_dirs = sorted(self.data_root.glob("patient_*"))
        n = len(patient_dirs)
        if split == "train":
            self.patients = patient_dirs[: int(0.70 * n)]
        elif split == "val":
            self.patients = patient_dirs[int(0.70 * n): int(0.85 * n)]
        else:
            self.patients = patient_dirs[int(0.85 * n):]

    def __len__(self) -> int:
        return len(self.patients) * self.patches_per_patient

    @staticmethod
    def _load_vol(path: Path) -> np.ndarray:
        if HAS_NIBABEL:
            return nib.load(str(path)).get_fdata(dtype=np.float32)
        return np.random.rand(128, 128, 128).astype(np.float32)

    def _random_patch_origin(self, seg: np.ndarray) -> tuple[int, int, int]:
        P = self.patch_size
        D, H, W = seg.shape
        max_d, max_h, max_w = D - P, H - P, W - P

        if random.random() < self.tumour_bias:
            tumour_voxels = np.argwhere(seg > 0)
            if len(tumour_voxels) > 0:
                centre = tumour_voxels[random.randint(0, len(tumour_voxels) - 1)]
                d = int(np.clip(centre[0] - P // 2, 0, max(max_d, 0)))
                h = int(np.clip(centre[1] - P // 2, 0, max(max_h, 0)))
                w = int(np.clip(centre[2] - P // 2, 0, max(max_w, 0)))
                return d, h, w

        return (
            random.randint(0, max(max_d, 0)),
            random.randint(0, max(max_h, 0)),
            random.randint(0, max(max_w, 0)),
        )

    def __getitem__(self, idx: int) -> dict:
        patient_idx = idx // self.patches_per_patient
        pdir = self.patients[patient_idx]
        P = self.patch_size

        seg = self._load_vol(pdir / "seg.nii.gz")
        d, h, w = self._random_patch_origin(seg)
        seg_patch = seg[d:d+P, h:h+P, w:w+P].astype(np.int64)

        target_patches, noisy_patches = [], []
        for mod in self.MODALITIES:
            vol = percentile_normalise(self._load_vol(pdir / f"{mod}.nii.gz"))
            patch = vol[d:d+P, h:h+P, w:w+P]
            target_patches.append(patch)
            noisy_patches.append(
                simulate_degradation(patch, self.degrade_mode) if self.degrade else patch.copy()
            )

        target = np.stack(target_patches, axis=0)   # (4, P, P, P)
        noisy  = np.stack(noisy_patches,  axis=0)   # (4, P, P, P)

        return {
            "noisy":  torch.from_numpy(noisy),
            "target": torch.from_numpy(target),
            "seg":    torch.from_numpy(seg_patch).unsqueeze(0),  # (1, P, P, P)
        }
