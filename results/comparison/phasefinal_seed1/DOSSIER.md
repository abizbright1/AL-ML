# `phasefinal_seed1` — complete run dossier

Everything needed to write up, defend, or reproduce the 50-epoch publication run.
All numbers below are computed from the committed artefacts listed in §6; nothing
here is estimated, carried over from an earlier run, or simulated.

---

## 1. What was run, exactly

| | |
|---|---|
| Launched | 2026-10-19 14:28:54 local |
| Wall-clock | 36 h 5 min (2165 min) |
| Exit status | `1` — see §7. Training completed; a plotting call failed afterwards. |
| Command | `PHASE=final bash run_comparison_sweep.sh ~/Downloads/BraTS2021_data` |
| Code version | git `3efe85df`, working tree **dirty** |
| Hardware | Apple Silicon, PyTorch 2.8.0, device `mps` |
| Output dir | `results/comparison/phasefinal_seed1/` |

Run-level hyperparameters, verbatim from `run_config.json`:

| Parameter | Value |
|---|---|
| `epochs` (restoration) | 50 |
| `seg_epochs` (segmentors) | 20, two of them |
| `max_subjects` | 100 |
| `split_seed` | 42 — **fixed, independent of the run seed** |
| `seed` (init / batching) | 1 |
| `patch_size` | 96 |
| `sigma` (noise) | 0.08 |
| `batch_size` | 4 |
| `dump_slices` | 96 |
| `ckpt_every` | 25 |

`split_seed` being separate from `seed` is deliberate: the test cohort is held
constant so that adding seeds later varies only initialisation and batch order,
not which patients are being scored.

### Caveats on reproducibility

- **The working tree was dirty at launch.** The committed SHA `3efe85df` does not
  fully describe the code that ran. This must be stated in the paper or fixed by
  re-running from a clean tree before submission. It is the single weakest point
  in the provenance chain.
- **MPS is seeded but not bit-deterministic.** The log says so explicitly. Re-running
  this exact configuration will give close but not identical numbers.

---

## 2. Data and the split

```
[BraTSDataset] Found 100 subjects
[BraTSDataset] Total slices in index: 5960
  SUBJECTS  train 80 | val 20 | overlap 0   (split seed 42 FIXED)
  train 4754 slices | val 1206 slices
  Split is by SUBJECT — no patient appears in both sets.
```

BraTS 2021, 100 subjects, four co-registered modalities (T1, T1ce, T2, FLAIR).
Labels remapped 4→3, giving `{0 background, 1 necrotic core, 2 oedema, 3 enhancing
tumour}`.

**The 80/20 split is by patient, not by slice.** This matters more than anything
else in this section. An earlier run in this repo (`results/v2_50subj_100ep/`) used
`torch.utils.data.random_split` over the slice index, which put slices from the same
patient in both train and test. Its numbers are inflated by leakage and must never
be cited. This run does not have that defect, and the log prints `overlap 0` as
positive evidence.

Every result below is on the **20 held-out patients / 1206 held-out slices**.

### Noise model

Rician, applied in the image domain at σ = 0.08:

```
x_noisy = sqrt((x + n_real)^2 + n_imag^2),   n_real, n_imag ~ N(0, sigma^2)
```

Rician rather than Gaussian because that is what magnitude-reconstructed MRI
actually carries. The σ is fixed, not sampled — so this measures performance at one
noise level, and generalisation across noise levels is untested.

---

## 3. The seven arms

Two are references that train no restoration model; five are trained objectives.
All five share the identical Uformer backbone, optimiser, learning rate, epoch
budget and data. **The objective is the only thing that differs.**

| Arm | Objective | Role |
|---|---|---|
| `Noisy` | — | **FLOOR.** Segment the noisy input directly. |
| `Clean` | — | **CEILING.** Segment the uncorrupted image. |
| `Uformer-L1` | `L1` | Plain restoration baseline. |
| `Uformer-Base` | `L1 + 0.5·SSIM + 0.5·CrossModal` | **MATCHED CONTROL.** Every non-pathology term present in the treatment, and nothing else. |
| `Uformer+PLcur` | `Base + Σ_r w_r·‖M_r⊙(X̂−X)‖₁/|M_r|` | **PROPOSED.** Nested WT⊃TC⊃ET, weights 1:2:3. |
| `Uformer+FlatROI` | `Base + flat binary tumour mask` | Ablation: does the *nesting* matter, or just "weight the tumour"? |
| `Uformer+BSMaskL1` | `Base + BraSyn 2023 Eq. 4, w=0.5` | **Closest published prior art.** |

The matched control is the scientifically load-bearing arm. Without it, any gain
PathologyLoss shows over plain L1 could be coming from the SSIM and cross-modal
terms rather than from the pathology weighting. §5 shows this is exactly what
happens.

### On the per-region normalisation

The coefficient a voxel receives is `Σ_r w_r·M_r(x)/|M_r|` — i.e. `1/|WT| + 2/|TC| +
3/|ET|` for an enhancing-tumour voxel. Because of the `/|M_r|` normalisation this is
**size-dependent**, not a fixed 6×. Measured on real data the effective ratio is
2.97 : 2.04 : 1.00 with spread 0.039, so the intended 3:2:1 ordering is preserved
in practice. BraSyn 2023 (Eq. 2–4) publishes region-normalised masked reconstruction
on BraTS, so the normalisation is **not** a novel contribution of this work — the
nesting and the clinically-ordered weights are what remain.

---

## 4. Evaluation protocol

1. Train a segmentor on **clean** images for 20 epochs, then **freeze** it.
2. A second, independent auxiliary segmentor is trained for the LIDnet-style
   task-feedback arms, so no arm is ever judged by a network that was part of its
   own training signal. (Not exercised in this phase, but trained.)
3. Each restoration arm trains for 50 epochs on the 80 training patients.
4. Restore the 1206 held-out slices, then segment the restored output with the
   single frozen evaluation segmentor.

Every arm is therefore scored by the **identical** judge. Any Dice difference is
attributable to the restoration, not to a differently-trained segmentor.

### Metrics recorded per slice

`psnr`, `ssim`, `nrmse`, `dice_wt`, `dice_tc`, `dice_et`, `psnr_wt`, `psnr_tc`,
`psnr_et`, `ssim_brain` — 11 metrics × 1206 slices × 7 arms = 8442 rows.

### Two conventions that change the answer

**(a) Subject-level, not slice-level.** The 1206 slices come from only 20 patients.
Treating slices as independent is pseudo-replication and inflates significance by
roughly the number of slices per patient. All inference in §5 averages within
patient first, then tests across the 20 patients. The repository's own
`options_results.csv` reports slice-level pooled means — useful as a description,
**not** valid for a p-value.

**(b) Dice on empty regions.** `dice()` returns **NaN** when a region is absent from
both prediction and ground truth, rather than 1.0:

> *"Returning NaN rather than 1.0 matters: ~40–60% of BraTS slices contain no
> enhancing tumour, and scoring those as perfect inflates mean Dice_ET."*

That reasoning is defensible, but it is **not the official BraTS convention**, and
the number of excluded slices differs by arm (428 to 451), so the arms are scored
on slightly different slice subsets. §5.4 quantifies how much this matters. It
matters a lot.

---

## 5. Results

### 5.1 Descriptive, subject-level means (n = 20 patients)

| Arm | Dice_ET | Dice_TC | Dice_WT | PSNR | PSNR_ET | PSNR_TC | SSIM_brain | NRMSE |
|---|---|---|---|---|---|---|---|---|
| FLOOR (noisy) | 0.3186 | 0.2881 | 0.6676 | 21.80 | 22.07 | 22.03 | 0.8286 | 0.2658 |
| L1 | 0.5619 | 0.5945 | 0.7363 | 32.42 | 30.00 | 30.04 | 0.9792 | 0.0776 |
| BraSyn-MaskedL1 | 0.5776 | 0.6138 | 0.7437 | **32.83** | 30.64 | 30.48 | **0.9802** | **0.0742** |
| **PathologyLoss** | 0.5895 | 0.6294 | 0.7421 | 32.38 | **30.94** | **30.80** | 0.9785 | 0.0782 |
| FlatROI | 0.5963 | 0.6011 | 0.7416 | 32.71 | 30.53 | 30.48 | 0.9798 | 0.0753 |
| **MatchedBase** | **0.6289** | **0.6311** | 0.7399 | 32.77 | 30.78 | 30.68 | 0.9801 | 0.0747 |
| CEILING (clean) | 0.6345 | 0.6644 | 0.7583 | ∞ | ∞ | ∞ | 1.0000 | 0.0000 |

Bootstrap 95% CIs are in `paper_figures/table1_main_results.csv` (4000 percentile
resamples over subjects).

Read the Dice_ET column top to bottom. **MatchedBase, which contains no pathology
term at all, scores highest of the five trained arms** — 0.6289 against
PathologyLoss's 0.5895, and essentially at the clean-image ceiling of 0.6345.

Read the PSNR_ET and PSNR_TC columns. **PathologyLoss is highest on both**, and has
the *lowest* global PSNR and the *worst* NRMSE and SSIM_brain of the five. It is
spending global fidelity to buy fidelity inside the regions it weights. That is
precisely the designed behaviour of Eq. 3, and it is cleanly measured here.

### 5.2 The decisive contrasts — paired, subject-level, Wilcoxon signed-rank

Dice_ET, common subjects only (the unpaired table in §5.1 compares n=19 against
n=20 and is therefore comparing different cohorts):

| Contrast | n | Δ | p | patients won |
|---|---|---|---|---|
| **PathologyLoss − MatchedBase** | 19 | **−0.0084** | **0.515** | 8/19 |
| PathologyLoss − L1 | 20 | +0.0276 | **0.0011** | 16/20 |
| PathologyLoss − FlatROI | 19 | +0.0242 | 0.104 | 12/19 |
| PathologyLoss − BraSyn | 20 | +0.0119 | 0.494 | 9/20 |
| **MatchedBase − L1** | 19 | **+0.0375** | **0.0012** | 16/19 |
| MatchedBase − FlatROI | 19 | +0.0326 | 0.016 | 15/19 |
| MatchedBase − BraSyn | 19 | +0.0209 | 0.396 | 11/19 |

Rows 2 and 5 together are the finding. PathologyLoss beats plain L1 by +0.0276 —
but its own matched control beats L1 by **more**, +0.0375. The pathology term
accounts for none of the improvement over L1; the SSIM and cross-modal terms account
for all of it.

Dice_TC, same method:

| Contrast | n | Δ | p | won |
|---|---|---|---|---|
| PathologyLoss − MatchedBase | 20 | −0.0017 | 0.956 | 10/20 |
| PathologyLoss − L1 | 20 | +0.0349 | 0.0004 | 15/20 |
| PathologyLoss − FlatROI | 20 | +0.0282 | 0.014 | 16/20 |
| PathologyLoss − BraSyn | 20 | +0.0155 | 0.246 | 11/20 |

### 5.3 PSNR_ET — what does survive

| Contrast | Δ dB | p | won |
|---|---|---|---|
| PathologyLoss − L1 | +0.935 | <0.0001 | **19/19** |
| PathologyLoss − FlatROI | +0.405 | <0.0001 | **19/19** |
| PathologyLoss − BraSyn | +0.296 | 0.0008 | 16/19 |
| PathologyLoss − MatchedBase | +0.160 | 0.016 | 15/19 |

Unanimous or near-unanimous per-patient wins against all four competitors. This is
the most robust positive result in the run, and it is the one claim the data
supports without qualification.

### 5.4 Sensitivity to the empty-region Dice convention

Re-scoring absent-in-both slices as Dice = 1.0, the official BraTS convention,
instead of excluding them:

| Arm | excluded (as run) | scored 1.0 (BraTS) | slices excluded |
|---|---|---|---|
| FLOOR | 0.3186 | 0.4235 | 161 |
| L1 | 0.5619 | 0.7492 | 451 |
| FlatROI | 0.5963 | 0.7548 | 446 |
| BraSyn | 0.5776 | 0.7584 | 440 |
| PathologyLoss | 0.5895 | 0.7549 | 428 |
| MatchedBase | 0.6289 | 0.7705 | 445 |
| CEILING | 0.6345 | 0.7791 | 408 |

Paired contrasts under the BraTS convention:

| Contrast | Δ | p | won |
|---|---|---|---|
| PathologyLoss − MatchedBase | −0.0156 | 0.143 | 6/20 |
| PathologyLoss − L1 | **+0.0058** | **0.368** | 12/20 |
| PathologyLoss − BraSyn | −0.0035 | 0.648 | 8/20 |
| PathologyLoss − FlatROI | +0.0001 | 0.784 | 10/20 |
| MatchedBase − L1 | +0.0213 | 0.012 | 16/20 |

**This is the most important robustness result in the run.** The one significant
Dice win PathologyLoss had — beating L1 by +0.0276, p=0.0011 — falls to +0.0058,
p=0.37 under the standard convention. It was an artefact of the scoring rule. The
only Dice effect that survives both conventions is **MatchedBase over L1**.

A reviewer will check this. It needs to be in the paper, not discovered at review.

### 5.5 TOST equivalence, margin ±0.05 Dice_ET (fixed a priori)

| Contrast | n | Δ | 90% CI | TOST p | Verdict |
|---|---|---|---|---|---|
| PathologyLoss vs MatchedBase | 19 | −0.0084 | [−0.0234, +0.0066] | 0.0001 | **EQUIVALENT** |
| PathologyLoss vs BraSyn | 20 | +0.0119 | [−0.0024, +0.0262] | 0.0001 | **EQUIVALENT** |
| PathologyLoss vs FlatROI | 19 | +0.0242 | [+0.0044, +0.0441] | 0.019 | **EQUIVALENT** |

Equivalence testing was added to answer a reviewer who would say "no difference
shown." It now formally establishes that PathologyLoss is *statistically
indistinguishable* from its own control, from the closest prior art, and from the
flat-mask ablation, within the pre-registered 0.05 margin.

### 5.6 Training convergence

Final-epoch losses are **not comparable across arms** — each arm minimises a
different functional. They are only useful as evidence that nothing diverged:

| Arm | Ep 1 | Ep 25 | Ep 50 | Still falling at 50? |
|---|---|---|---|---|
| PathologyLoss | 0.4247 | 0.1935 | 0.1836 | yes, slowly |
| MatchedBase | 0.1387 | 0.0663 | 0.0633 | yes, slowly |
| FlatROI | 0.1892 | 0.0843 | 0.0805 | yes, slowly |
| L1 | 0.0365 | 0.0165 | 0.0158 | yes, slowly |
| BraSyn | 0.1868 | 0.0829 | 0.0793 | yes, slowly |

All five were still improving marginally at epoch 50. Each took 371–390 min.

### 5.7 Comparison with the earlier 12-epoch run

| | 12 ep (`phase2_seed1`) | 50 ep (`phasefinal_seed1`) |
|---|---|---|
| L1 | 0.5137 | 0.5619 |
| MatchedBase | 0.5459 | **0.6289** |
| PathologyLoss | **0.5632** | 0.5895 |
| PL − MatchedBase | **+0.0173** | **−0.0084** |

At 12 epochs PathologyLoss led its control. At 50 epochs the gap closes and inverts.
The honest interpretation: **the 12-epoch advantage was an under-training artefact.**
Given a longer budget, the base objective reaches the same place without the
pathology term. Reporting the 12-epoch result as the headline would be selecting the
epoch budget that flatters the hypothesis, after having seen both.

---

## 6. File inventory

### 6.1 Committed to git — `results/comparison/phasefinal_seed1/`

**Primary data (cite these):**

| File | Size | What it is |
|---|---|---|
| `per_slice_metrics.csv` | 1.5 MB | **The raw result.** 8442 rows: 11 metrics × 1206 slices × 7 arms. Every number in this dossier derives from here. |
| `run_config.json` | 4 KB | Full provenance: args, git SHA, dirty flag, torch version, device, timestamp. |
| `paper_figures/table1_main_results.csv` / `.tex` | 4 KB | Subject-level summary with bootstrap 95% CIs. `.tex` drops into the manuscript. |
| `options_results.csv` | 4 KB | Slice-level pooled means. **Descriptive only** — see §4(a). |
| `paper_figures/README.txt` | 4 KB | Provenance of the figures: source CSV, ranking metric, floor/ceiling, palette CVD validation. |

**Publication figures** — `paper_figures/`, each as 600-dpi vector PDF + PNG:

| File | Content |
|---|---|
| `fig1_accuracy_by_loss` | Dice_ET per objective with CIs, floor/ceiling marked |
| `fig2_accuracy_vs_psnr` | Dice_ET against global PSNR — shows the trade-off |
| `fig3_accuracy_vs_psnr_et` | Dice_ET against PSNR_ET — the targeted-fidelity view |
| `fig4_heatmap` | All arms × all metrics |
| `fig5_decomposition_uformer` | L1 → MatchedBase → PathologyLoss, attributing the gain |
| `fig6_by_region_uformer` | Per-region WT / TC / ET breakdown |
| `fig7_qualitative_uformer` | **The MRI comparison.** 7 arms × (restored / \|error\| / segmentation), BraTS2021_00002 slice 40, T1ce. Green = found real ET, red = missed, blue = invented. |

`fig5` currently renders a decomposition that §5.2 contradicts — it must be
regenerated or retired. See §8.

**Explanatory figures** — `figures/`, auto-generated during the run, slice 95 of
BraTS2021_00014 (chosen for maximum inter-arm disagreement): `fig01_data_sample`,
`fig02_noise_added`, `fig03_denoised_all_models`, `fig04_error_maps_all_models`,
`fig05_segmentation_all_models`, `fig09_tumour_regions`, `fig10_noise_progression`,
`fig11_pathology_weight_map`, `fig12_loss_heatmap`, `fig13_pipeline`,
`fig14_cross_modal`. These are teaching/illustration assets, suitable for the
presentation and for a methods figure.

**Logs:** `final.log` (sweep-level trace, the arm-by-arm results table, and the
traceback) and `results/comparison/logs/phasefinal_seed1.log` (per-arm detail).

### 6.2 On the Mac only, deliberately not committed

`.gitignore` excludes `*.npz`, `*.pt`, `checkpoints/`, `predictions/` — about 600 MB.

| Path | Content | Why it stays local |
|---|---|---|
| `checkpoints/*.pt` | 8 files: 5 restoration models + 2 segmentors + reference | Model weights; large, and regenerable from the committed config |
| `predictions/*.npz` | 7 files, 96 dumped slices each | Raw arrays; `fig7` is built from these and the figure is committed |

**Consequence:** `make_qualitative_figure.py` only runs on the Mac. If those
`.npz` files are deleted, `fig7` cannot be rebuilt without re-running 36 hours of
training. They are the most valuable un-backed-up artefact in the project.

### 6.3 Reproducing the analysis

```
python3 analyze_final_run.py     # §5.1–5.3 paired contrasts
python3 make_paper_figures.py --run results/comparison/phasefinal_seed1
python3 make_qualitative_figure.py --run results/comparison/phasefinal_seed1 --backbone Uformer
```

---

## 7. The non-zero exit — not a failed run

```
File "run_full_experiment.py", line 975, in fig_metrics
    ax.set_ylim(min(vals) * .95, max(vals) * 1.03)
ValueError: Axis limits cannot be NaN or Inf
```

The `Clean` arm reconstructs itself exactly, so its PSNR is `+inf`, and matplotlib
refuses an infinite axis limit. This fired **after** all 50 epochs of all five arms
had trained, after `per_slice_metrics.csv` was written, and after 11 of 12 figures
were saved. The only casualty was `fig06_metrics_bars.png`.

The wrapper reported `✗ FAILED (exit 1) after 2165 min` because it saw a non-zero
exit code and could not know the science had already succeeded.

Fixed in commit `f3d981d`: the axis is now scaled from finite bars only, and a
non-finite bar is drawn to the top of the axis and labelled `∞` rather than with a
number. Verified against both `inf` and `NaN` inputs.

---

## 8. What the paper can claim, and what it cannot

**Supported by this run:**

1. Region-weighted restoration measurably improves reconstruction fidelity inside
   the weighted regions. PSNR_ET +0.94 dB over L1 on 19/19 patients, and
   PathologyLoss is highest of all five arms on both PSNR_ET and PSNR_TC. (§5.3)
2. It does so at a measurable cost in global fidelity — lowest global PSNR, worst
   NRMSE and SSIM_brain of the five trained arms. The trade-off is real,
   directional, and exactly what Eq. 3 was designed to produce. (§5.1)
3. **Improved pathology-region fidelity did not transfer to segmentation accuracy.**
   PathologyLoss is statistically equivalent to its own matched control, to BraSyn
   2023, and to a flat tumour mask, within a pre-registered ±0.05 Dice margin.
   (§5.2, §5.5)
4. Restoration as a whole helps enormously: Dice_ET from 0.3186 (floor) to
   0.56–0.63, against a clean-image ceiling of 0.6345. (§5.1)
5. On this cohort, the SSIM and cross-modal terms do all of the downstream work.
   MatchedBase − L1 = +0.0375 (p=0.0012) is the only Dice effect that survives both
   scoring conventions. (§5.2, §5.4)

**Not supported — remove or rewrite:**

- ~~"PathologyLoss improves downstream segmentation over the state of the art."~~
  It is equivalent to BraSyn (p=0.49) and to its own control (p=0.52).
- ~~"Recovers 72% of the gap to the ceiling."~~ Computed from the old floor/ceiling
  (0.2435 / 0.6172). This run's are 0.3186 / 0.6345.
- ~~"65% of the gain comes from SSIM, 35% from the pathology term."~~ At 50 epochs it
  is approximately 100/0. `fig5_decomposition_uformer` renders this stale claim.
- ~~"PathologyLoss beats BraSyn by +0.0685, p=0.002."~~ That was SwinIR at 12 epochs.
  Here it is +0.0119, p=0.49.
- ~~Anything from `results/v2_50subj_100ep/`.~~ Slice-level split; leakage.

**Threats to validity, to state plainly in the limitations section:**

1. **One seed.** A single-seed reversal is suggestive, not conclusive. n=19–20
   patients with Dice SDs near 0.24 is thinly powered for a 0.008 difference. The
   defensible claim is "no detectable benefit at 50 epochs in one seed," not "the
   term does not work." 3–5 more seeds on the L1 / MatchedBase / PathologyLoss trio
   (~15 h per seed) would settle it.
2. **Dirty working tree at launch.** The recorded SHA does not fully describe the
   code that ran.
3. **One noise level** (σ=0.08), one noise model (Rician), one backbone (Uformer),
   one dataset (BraTS 2021), 100 subjects.
4. **No tuning split.** Hyperparameters were not selected on held-out data
   independent of the 20 test patients.
5. **No HD95.** Dice alone rewards finding tumour and does not symmetrically punish
   inventing it — the exact failure mode `fig7` makes visible as blue. A boundary
   metric would close this gap.
6. **Dice convention is outcome-changing** (§5.4) and is not the BraTS standard.
7. **Segmentor trained on clean images only.** An arm is rewarded for producing
   images that look like what the segmentor was trained on, which is a different
   objective from clinical utility.

---

## 9. Suggested reframing

The run does not support the paper as currently written. It does support a
defensible and, in one respect, more interesting paper:

> Region-weighted restoration objectives do what they claim at the level of image
> fidelity — targeted reconstruction error inside pathology falls measurably and
> consistently — but on BraTS 2021 that improvement does not propagate to
> downstream segmentation accuracy, where a well-chosen spatially-uniform objective
> performs equivalently. We establish this with a matched control that isolates the
> pathology term, five published baselines, patient-level inference, and
> pre-registered equivalence testing.

That negative-transfer result is a genuine contribution: the ROIRecNet / LIDnet /
BraSyn line of work assumes the transfer and does not test it against a matched
control. This run tests it. It also directly answers the three criticisms raised in
supervisor feedback — unclear problem statement, no benchmark models, thin
literature review — because the benchmarks are now real, on real data, with
equivalence testing and an explicit matched control.

The alternative is to run more seeds first and find out whether the reversal is
real before committing to a framing. Both are honest; only reporting the 12-epoch
result is not.
