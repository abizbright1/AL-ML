"""
brats_loader.py — Real BraTS 2021 / 2023 data loader
======================================================
Supports two sources:
  A) Full BraTS dataset from disk  (provide root_dir)
  B) Tiny 1-subject NIfTI demo    (auto-generated synthetic volumes
     in the same shape / format as BraTS, so all downstream code
     works identically — swap in real files when you have them)

BraTS folder layout expected (Task1 / 2021 / 2023 style):
  root_dir/
    BraTS2021_00000/
      BraTS2021_00000_t1.nii.gz
      BraTS2021_00000_t1ce.nii.gz
      BraTS2021_00000_t2.nii.gz
      BraTS2021_00000_flair.nii.gz
      BraTS2021_00000_seg.nii.gz   (optional — absent at test time)
    BraTS2021_00001/
      ...

Usage
-----
  # Real data
  from brats_loader import BraTSDataset
  ds = BraTSDataset('/path/to/BraTS2021_Training_Data', slice_axis=2,
                    patch_size=128, sigma=0.08, cache=True)

  # Demo mode (no files needed)
  from brats_loader import make_demo_brats
  ds = make_demo_brats(n_subjects=4, patch_size=64)
"""

import os, glob, warnings
import numpy as np
import torch
from torch.utils.data import Dataset


# ── Modality file-name suffixes ────────────────────────────────────────────────
MODALITY_KEYS = ['t1', 't1ce', 't2', 'flair']


def _load_nii(path: str) -> np.ndarray:
    """Load a .nii / .nii.gz volume → float32 numpy array (D,H,W)."""
    try:
        import nibabel as nib
    except ImportError:
        raise RuntimeError("nibabel not installed. Run: pip install nibabel")
    return nib.load(path).get_fdata(dtype=np.float32)


def _normalise(vol: np.ndarray) -> np.ndarray:
    """Z-score → min-max rescale to [0,1] (brain mask only)."""
    mask = vol > 0
    if mask.sum() == 0:
        return vol
    mu  = vol[mask].mean()
    std = vol[mask].std() + 1e-8
    vol = (vol - mu) / std
    vmin, vmax = vol.min(), vol.max()
    return ((vol - vmin) / (vmax - vmin + 1e-8)).clip(0., 1.)


def _add_rician_noise(vol: np.ndarray, sigma: float) -> np.ndarray:
    """Rician noise: |x + n_r + i·n_i|."""
    n_r = np.random.randn(*vol.shape).astype('float32') * sigma
    n_i = np.random.randn(*vol.shape).astype('float32') * sigma
    return np.sqrt((vol + n_r)**2 + n_i**2).clip(0., 1.).astype('float32')


def _find_subjects(root_dir: str):
    """Return sorted list of subject directories under root_dir."""
    subjects = sorted([
        d for d in glob.glob(os.path.join(root_dir, '*'))
        if os.path.isdir(d)
    ])
    return subjects


def _build_subject_paths(subject_dir: str):
    """
    Returns dict {modality: path} for all 4 MRI modalities.
    Tries multiple naming conventions (BraTS 2021 and 2023).
    """
    name = os.path.basename(subject_dir)
    paths = {}
    for mod in MODALITY_KEYS:
        candidates = [
            os.path.join(subject_dir, f'{name}_{mod}.nii.gz'),
            os.path.join(subject_dir, f'{name}_{mod}.nii'),
            os.path.join(subject_dir, f'{mod}.nii.gz'),
            os.path.join(subject_dir, f'{mod}.nii'),
        ]
        for c in candidates:
            if os.path.exists(c):
                paths[mod] = c
                break
    # segmentation (optional)
    for seg_name in [f'{name}_seg.nii.gz', f'{name}_seg.nii',
                     'seg.nii.gz', 'seg.nii']:
        seg_path = os.path.join(subject_dir, seg_name)
        if os.path.exists(seg_path):
            paths['seg'] = seg_path
            break
    return paths


# ── Main Dataset ──────────────────────────────────────────────────────────────

class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS MRI data.

    Each __getitem__ returns one 2-D axial (or coronal/sagittal) patch:
        {
          'noisy' : (4, H, W)  float32  — noisy 4-channel MRI patch
          'target': (4, H, W)  float32  — clean ground-truth patch
          'seg'   : (1, H, W)  int64    — tumour labels (0-3), 0 if absent
          'subject': str                — subject folder name
          'slice_idx': int              — which slice
        }

    Parameters
    ----------
    root_dir   : Path to BraTS training/validation folder
    slice_axis : 0=sagittal, 1=coronal, 2=axial (default)
    patch_size : Spatial crop size (square). None = full slice.
    sigma      : Rician noise sigma to add to target (0 = no noise added,
                 test the model on already-noisy-looking MRI)
    min_tumour_frac : Only keep slices where tumour occupies at least this
                      fraction of pixels (avoids empty background slices)
    cache      : Cache loaded volumes in RAM (fast if RAM allows)
    max_subjects : Limit number of subjects (useful for quick tests)
    """

    def __init__(self, root_dir: str,
                 slice_axis:       int   = 2,
                 patch_size:       int   = 128,
                 sigma:            float = 0.08,
                 min_tumour_frac:  float = 0.005,
                 cache:            bool  = True,
                 max_subjects:     int   = None):
        super().__init__()
        self.root_dir       = root_dir
        self.slice_axis     = slice_axis
        self.patch_size     = patch_size
        self.sigma          = sigma
        self.min_tumour_frac = min_tumour_frac
        self.cache          = cache
        self._vol_cache     = {}

        subjects = _find_subjects(root_dir)
        if not subjects:
            raise FileNotFoundError(
                f"No subject directories found under {root_dir}.\n"
                "Expected layout: root_dir/SubjectID/SubjectID_t1.nii.gz ...")
        if max_subjects:
            subjects = subjects[:max_subjects]

        print(f"[BraTSDataset] Found {len(subjects)} subjects under {root_dir}", flush=True)

        # Build index: list of (subject_dir, paths, slice_idx)
        self.index = []
        for sub_dir in subjects:
            paths = _build_subject_paths(sub_dir)
            missing = [m for m in MODALITY_KEYS if m not in paths]
            if missing:
                warnings.warn(f"Skipping {sub_dir}: missing modalities {missing}")
                continue
            # Count slices along slice_axis
            ref_vol = _load_nii(paths[MODALITY_KEYS[0]])
            n_slices = ref_vol.shape[slice_axis]
            for sl in range(n_slices):
                # Pre-filter: only keep tumour-containing slices
                if 'seg' in paths:
                    seg_vol = _load_nii(paths['seg'])
                    seg_slice = np.take(seg_vol, sl, axis=slice_axis)
                    if seg_slice.sum() < min_tumour_frac * seg_slice.size:
                        continue
                self.index.append((sub_dir, paths, sl))

        print(f"[BraTSDataset] Total slices in index: {len(self.index)}", flush=True)
        if len(self.index) == 0:
            raise RuntimeError("No valid slices found. Check root_dir and file naming.")

    def _load_subject(self, sub_dir: str, paths: dict):
        if sub_dir in self._vol_cache:
            return self._vol_cache[sub_dir]
        vols = {}
        for mod in MODALITY_KEYS:
            vols[mod] = _normalise(_load_nii(paths[mod]))
        vols['seg'] = _load_nii(paths['seg']).astype('int64') if 'seg' in paths \
                      else np.zeros_like(vols[MODALITY_KEYS[0]], dtype='int64')
        if self.cache:
            self._vol_cache[sub_dir] = vols
        return vols

    def _extract_slice(self, vol: np.ndarray, sl: int) -> np.ndarray:
        return np.take(vol, sl, axis=self.slice_axis)

    def _centre_crop(self, arr: np.ndarray) -> np.ndarray:
        """Centre-crop a 2-D slice to (patch_size, patch_size)."""
        if self.patch_size is None:
            return arr
        H, W = arr.shape[-2], arr.shape[-1]
        p = self.patch_size
        h0 = max((H - p) // 2, 0)
        w0 = max((W - p) // 2, 0)
        if arr.ndim == 2:
            return arr[h0:h0+p, w0:w0+p]
        return arr[..., h0:h0+p, w0:w0+p]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        sub_dir, paths, sl = self.index[idx]
        vols = self._load_subject(sub_dir, paths)

        # Stack 4 modalities → (4, H, W) clean image
        clean_channels = []
        for mod in MODALITY_KEYS:
            s = self._extract_slice(vols[mod], sl).astype('float32')
            s = self._centre_crop(s)
            clean_channels.append(s)
        clean = np.stack(clean_channels, axis=0)           # (4,H,W)

        # Segmentation (must be augmented consistently with image)
        seg_slice = self._extract_slice(vols['seg'], sl).astype('int64')
        seg_slice = self._centre_crop(seg_slice)           # (H,W)

        # ── Data augmentation (training-time only) ──────────────────────────
        # Random horizontal flip (50% chance) — applied to both image & seg
        if np.random.rand() > 0.5:
            clean     = clean[:, :, ::-1].copy()
            seg_slice = seg_slice[:, ::-1].copy()
        # Random vertical flip (50% chance)
        if np.random.rand() > 0.5:
            clean     = clean[:, ::-1, :].copy()
            seg_slice = seg_slice[::-1, :].copy()
        # Random intensity jitter per modality (±5%) — simulates scanner variation
        for c in range(clean.shape[0]):
            scale = np.random.uniform(0.95, 1.05)
            clean[c] = (clean[c] * scale).clip(0., 1.)

        # Add Rician noise
        noisy = _add_rician_noise(clean, self.sigma) if self.sigma > 0 else clean.copy()

        seg_slice = seg_slice[np.newaxis]  # (1,H,W)
        # Clamp to valid labels [0,3]
        seg_slice = np.clip(seg_slice, 0, 3)

        return {
            'noisy':     torch.from_numpy(noisy),
            'target':    torch.from_numpy(clean),
            'seg':       torch.from_numpy(seg_slice),
            'subject':   os.path.basename(sub_dir),
            'slice_idx': sl,
        }


# ── Demo dataset (no real files needed) ──────────────────────────────────────

class _DemoBraTSDataset(Dataset):
    """
    Synthetic BraTS-shaped dataset.
    Volumes are 240×240×155 (same as real BraTS) but values are synthetic.
    Used when real data is not available — all downstream code is identical.
    """
    def __init__(self, n_subjects=4, patch_size=128, sigma=0.08,
                 slices_per_subject=20, seed=42):
        rng = np.random.default_rng(seed)
        self.items = []
        for s in range(n_subjects):
            for sl in range(slices_per_subject):
                H = W = patch_size
                # Background tissue
                base = rng.random((4, H, W)).astype('float32') * 0.35 + 0.05
                # Tumour blob (realistic BraTS intensities)
                cy, cx = rng.integers(H//4, 3*H//4), rng.integers(W//4, 3*W//4)
                Y, X   = np.ogrid[:H, :W]
                r_wt   = rng.integers(H//8, H//5)
                r_tc   = rng.integers(H//12, H//8)
                r_et   = rng.integers(H//20, H//12)
                wt = ((Y-cy)**2 + (X-cx)**2) < r_wt**2
                tc = ((Y-cy)**2 + (X-cx)**2) < r_tc**2
                et = ((Y-cy)**2 + (X-cx)**2) < r_et**2
                # T1W: NCR dark, surrounding oedema variable
                base[0][wt] = rng.uniform(0.4, 0.7, wt.sum())
                base[0][tc] = rng.uniform(0.2, 0.5, tc.sum())
                # T1CE: ET enhancing bright
                base[1][tc] = rng.uniform(0.5, 0.8, tc.sum())
                base[1][et] = rng.uniform(0.75, 1.0, et.sum())
                # T2W: oedema bright
                base[2][wt] = rng.uniform(0.6, 0.95, wt.sum())
                # FLAIR: oedema very bright
                base[3][wt] = rng.uniform(0.7, 1.0, wt.sum())
                base = base.clip(0., 1.)
                # Seg map
                seg = np.zeros((1, H, W), dtype='int64')
                seg[0][wt] = 2   # ED
                seg[0][tc] = 1   # NCR
                seg[0][et] = 3   # ET
                # Rician noise
                noisy = _add_rician_noise(base, sigma)
                self.items.append({
                    'noisy':     torch.from_numpy(noisy),
                    'target':    torch.from_numpy(base),
                    'seg':       torch.from_numpy(seg),
                    'subject':   f'DemoSubject_{s:03d}',
                    'slice_idx': sl,
                })

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def make_demo_brats(n_subjects: int = 4, patch_size: int = 128,
                    sigma: float = 0.08, slices_per_subject: int = 20):
    """
    Returns a BraTS-compatible Dataset without any real files.
    Drop-in replacement for BraTSDataset for testing pipelines.
    """
    print(f"[make_demo_brats] Creating synthetic BraTS dataset: "
          f"{n_subjects} subjects × {slices_per_subject} slices = "
          f"{n_subjects*slices_per_subject} total", flush=True)
    return _DemoBraTSDataset(n_subjects, patch_size, sigma, slices_per_subject)
