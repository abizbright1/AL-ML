# PP-MAE: Pathology-Preserving Masked Autoencoder
### Multi-Modal Brain MRI Reconstruction & Tumour-Aware Segmentation

**Louisiana Tech University** | Department of Electrical Engineering

---

## Overview

PP-MAE is a unified deep learning framework for brain MRI denoising and glioma segmentation. It couples a **hierarchical Swin Transformer encoder** with three tumour-oriented components:

1. **Cross-modal attention** — captures T1ce↔T2W and T2W↔FLAIR relationships  
2. **Saliency-aware feature reweighting** — amplifies tumour regions at every encoder stage  
3. **PathologyLoss** — penalises errors in ET (×3), TC (×2), WT (×1) with clinical weights  

**Key result:** PP-MAE achieves the highest Dice on Enhancing Tumour (ET=0.7903) and Tumour Core (TC=0.8383) on BraTS 2021, despite lower PSNR — demonstrating the **PSNR paradox**: higher reconstruction fidelity ≠ better tumour detection.

---

## Repository Structure

```
├── model/          Core PP-MAE model code (4 architecture options)
├── notebooks/      Colab & Kaggle notebooks for running experiments
├── scripts/        Training, evaluation, and visualization scripts
├── results/        Experiment results (CSV files, round-by-round)
├── figures/        All generated figures and plots
├── paper/          IEEE paper draft (LaTeX) and supplementary docs
└── data/           Dataset loaders and preprocessing utilities
```

---

## Quick Start

### Run on Google Colab
Open `notebooks/PP_MAE_AllOptions_Colab.ipynb` — connects to BraTS 2021 automatically.

### Run Locally (Apple MPS / CUDA)
```bash
python3 scripts/run_all_options.py ~/Downloads/BraTS2021_data \
    --device mps --rounds 4 --epochs 30 --seg_epochs 20 \
    --max_subjects 50 --out results/round4/
```

### Generate Paper Figures (from pre-computed results)
```bash
python3 scripts/paper_figures.py --out figures/paper/
```

### Generate Figures with Visual Samples (3 BraTS subjects)
```bash
python3 scripts/paper_figures.py \
    --data_dir ~/Downloads/BraTS2021_data \
    --device mps --n_samples 3 --out figures/paper/
```

---

## Results (BraTS 2021, Round 4, n=50 subjects)

| Method | PSNR | SSIM | Dice WT | Dice TC | Dice ET |
|--------|------|------|---------|---------|---------|
| **PP-MAE (Proposed)** | 28.06 | 0.9565 | 0.8788 | **0.8383** | **0.7903** |
| SwinIR-lite (L1) | **31.45** | 0.9796 | 0.8788 | 0.7943 | 0.7601 |
| Uformer-lite (L1) | 31.70 | **0.9801** | 0.8819 | 0.8101 | 0.7707 |
| SwinIR + PathologyLoss | 31.20 | 0.9784 | 0.8811 | 0.7670 | 0.7198 |
| Uformer + PathologyLoss | 31.68 | 0.9802 | **0.8836** | 0.8258 | 0.7780 |

Statistical significance (Wilcoxon signed-rank, n=543 slices):  
PP-MAE vs SwinIR-lite: Dice_TC p=4.37×10⁻¹⁰ ***, Dice_ET p=1.94×10⁻⁴ ***

---

## Architecture Options

| Option | Architecture | Key Feature |
|--------|-------------|-------------|
| Option 1 | CNN-based PP-MAE | Lightweight baseline |
| Option 2 | ViT PP-MAE | Global attention |
| Option 3 | Full Pipeline | End-to-end denoising+segmentation |
| **Option 4** | **Swin PP-MAE** | **Proposed — best performance** |

---

## Dataset

**BraTS 2021** — Multi-parametric MRI (T1W, T1ce, T2W, FLAIR) with expert glioma annotations.  
Label convention: 0=Background, 1=NCR, 2=Oedema, 3=Enhancing Tumour  
Regions: WT={1,2,3}, TC={1,3}, ET={3}

---

## Paper

Full IEEE-format paper in `paper/paper_draft.tex`.  
Compile with: `pdflatex paper/paper_draft.tex`  
Or paste into [Overleaf](https://overleaf.com) for instant preview.
