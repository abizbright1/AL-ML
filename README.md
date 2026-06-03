# PP-MAE: Pathology-Preserving Masked Autoencoder
### Multi-Modal Brain MRI Reconstruction, Segmentation & Tumour Grading

**Louisiana Tech University** | Department of Electrical Engineering

---

## Folder Structure by Date

Each folder is named by the date the work was first committed.

| Folder | What was done |
|--------|--------------|
| `2025-06-30_initial_setup/` | Project created, README |
| `2025-07-01_model_skeleton/` | Initial model, dataset, training scaffolding |
| `2026-05-04_pp_mae_core/` | Full PP-MAE implementation — all 4 architecture options |
| `2026-05-16_kaggle_notebooks_losses/` | Kaggle notebooks, derived PathologyLoss components |
| `2026-05-17_colab_notebook/` | Google Colab notebook |
| `2026-05-20_experiment_runner/` | Standalone experiment runner script |
| `2026-05-21_complete_notebook_figures/` | Complete experiment notebook + first paper figures |
| `2026-05-22_baselines_segmentation/` | DnCNN/UNet/REDNet baselines + segmentation benchmark |
| `2026-05-23_brats_loader_pipeline/` | BraTS 2021 NIfTI loader + end-to-end test pipeline |
| `2026-05-24_grading_options_comparison/` | Grading pipeline (LGG/GBM) + 4-option comparison |
| `2026-05-25_miccai_paper_sota/` | MICCAI paper draft + SOTA 2021–2026 baselines |
| `2026-05-26_option3_miccai_colab/` | Option 3 MICCAI Colab notebook + paper figures module |
| `2026-05-30_mps_diagnostics/` | Apple MPS GPU diagnostics |
| `2026-05-31_mps_fixes_round4_results/` | All MPS crash fixes + Round 4 experiment results |
| `2026-06-01_real_brats_integration/` | Real BraTS data validation + ablation figures |
| `2026-06-02_option4_full_analysis/` | Full Option 4 (Swin PP-MAE) analysis + roadmap |
| `2026-06-03_final_paper_gpu_results/` | IEEE paper draft + final GPU results + significance tests |

---

## Key Results (BraTS 2021, Round 4, n=50 subjects, MPS GPU)

| Method | PSNR | Dice TC | Dice ET |
|--------|------|---------|---------|
| **PP-MAE Swin [PROPOSED]** | 28.06 | **0.8383** | **0.7903** |
| SwinIR-lite (L1) | **31.45** | 0.7943 | 0.7601 |
| Uformer-lite (L1) | 31.70 | 0.8101 | 0.7707 |
| SwinIR + PathologyLoss | 31.20 | 0.7670 | 0.7198 |
| Uformer + PathologyLoss | 31.68 | 0.8258 | 0.7780 |

PP-MAE vs SwinIR-lite: Dice_TC p=4.37×10⁻¹⁰ \*\*\*, Dice_ET p=1.94×10⁻⁴ \*\*\* (Wilcoxon signed-rank, n=543 slices)

---

## Paper
See `2026-06-03_final_paper_gpu_results/paper_draft.tex` — IEEE conference format.  
Open in [Overleaf](https://overleaf.com) for instant PDF preview.
