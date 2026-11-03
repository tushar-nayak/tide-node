# Carrying the last scan forward: a leakage-free evaluation of next-visit glioblastoma forecasting on LUMIERE

**Draft. All numbers are reproduced by the scripts named in [EXPERIMENTS.md](EXPERIMENTS.md).**

## Abstract

**Background.** Deep models that forecast a glioblastoma's next MRI are usually compared
with "persistence" (copying the last scan), and modest gains over it are reported as
evidence of learned tumor dynamics. We asked how much of such a gain survives a strict
evaluation on a public longitudinal dataset.

**Methods.** LUMIERE: 79 patients, 399 scan-to-scan transitions after excluding pre-operative
scans and forecasts across surgery. Every analysis uses patient-level 5-fold cross-validation
and patient-level bootstrap CIs. We compared Neural ODE models with persistence and
simple controls on five next-visit targets:
1. multi-modal slice intensities (re-testing an earlier +6.7% claim);
2. compartment volumes;
3. 3D tumor masks;
4. the noise ceiling of those masks;
5. the expert RANO rating.

**Results.**
* **Intensity.** On held-out patients the intensity model beats persistence on whole-slice
  MSE by 8.7% [5.7, 11.7]. In the tumor region the gain is 2.5% [-1.0, 6.0], and a
  1-pixel Gaussian blur of the last scan does at least as well (4.1%).
* **Volume.** Learned volume forecasts reduce error 5-7% below persistence. A population
  Gompertz fit gets 3.4%, and the Neural ODE is tied with ridge regression and gradient
  boosting.
* **Alignment.** LUMIERE's per-scan atlas space leaves consecutive scans misaligned by a
  median 2.5 mm. That misalignment accounted for 40-45% of the change that mask forecasters
  appeared to predict.
* **3D masks.** After registration, the Neural ODE mask forecast matches copy-last-mask:
  whole-tumor Dice 0.573 vs 0.570, difference +0.003 [-0.003, +0.010].
* **Noise ceiling.** Two automated segmenters agree on the same scan only to Dice 0.65, and
  visits rated stable at most 12 weeks apart overlap at 0.58.
* **RANO rating.** The last visit's rating predicts next-visit progression with AUROC 0.684.
  No learned model improves on it significantly (best +0.040 [-0.015, 0.092]).

**Conclusion.** On LUMIERE, carrying the last observation forward is close to the best
next-visit forecast. Reported gains over persistence should be checked with three things:
held-out patients, registration of the target to the input frame, and a smoothing control.

## 1. Introduction

Forecasting how a glioblastoma will look at the next visit would help response assessment
and treatment planning, and several deep-learning approaches model it as a continuous-time
dynamical system. Because the tumor and the surrounding brain change slowly relative to
typical imaging intervals, persistence is a strong baseline. A small improvement over it
can therefore come from sources other than tumor dynamics:
* scoring on training data;
* misalignment between visits that a model learns to undo;
* the generic benefit of predicting a smoothed image under squared error;
* label noise that caps attainable accuracy.

This project began with a 2D Attention U-Net + Neural ODE intensity forecaster reported at
+6.7% MSE over persistence across 81 per-patient runs. On inspection, every run had been
scored on its own training pairs. We re-tested that claim and then asked the same question
of progressively more clinically meaningful targets, each time including the controls
needed to read the result.

## 2. Data and cohort

* **Dataset.** LUMIERE (v202211): longitudinal FLAIR, T1, T2 and contrast-enhanced T1, with
  DeepBraTumIA and HD-GLIO-AUTO segmentations and expert RANO ratings.
* **Visit resolution.** Several visits share a week number (pre-/post-operative and
  re-resection scans). We resolve these deterministically to the latest complete scan of the
  week.
* **Cohort rules** (`lumiere_cohort.py`):
  * Pre-operative scans are dropped (39).
  * A transition never crosses surgery: a post-operative scan starts a new segment, and the
    31 post-operative targets are excluded.
  * History is the earlier scans of the same segment.
* **Result:** 79 patients and 399 transitions. The exclusion log is in
  `results/volume_trajectory/metrics.md`.
* **Folds.** Patient-level, identical across all experiments: 5 folds; volume and RANO
  experiments repeat with 3 fold seeds.
* **Reporting.** Differences are reported as the mean over patients of the per-patient
  difference, with 95% bootstrap CIs over patients.

## 3. Methods

### 3.1 Alignment between visits

Step 2 was first run in LUMIERE's per-scan DeepBraTumIA atlas space, on the assumption
that it makes visits voxel-aligned. We tested this assumption (`alignment_check.py`) by
correlating healthy tissue between the last scan and the target. Healthy tissue here means
the brain eroded by 2 mm, minus both tumors dilated by 5 mm.

Atlas-space alignment is poor:
* median T1 correlation 0.57;
* median estimated shift 2.5 mm (Figure 1);
* a dedicated affine registration reaches 0.85.

We therefore registered each target, and the second-to-last scan, into the frame of the
**last** scan (`register_step2_pairs.py`, 691 pairs):
* one rigid-then-affine transform per pair on T1, Mattes mutual information;
* tumors masked out of the metric;
* the transform applied to all modalities and the segmentation;
* brain-overlap QC with fallbacks.

Because the forecast is made from the last scan, the inputs carry no target information.
After registration, median healthy-tissue T1 correlation is 0.85 and brain Dice is 0.981.

![Figure 1](results/figures/alignment_atlas_vs_affine.png)
*Figure 1. Alignment of the last scan to the target in healthy tissue: LUMIERE atlas space
vs affine registration, 399 transitions.*

### 3.2 Experiments

**A. Original intensity model, held-out retest** (`intensity_forecast_cv.py`). The
original model and metric are kept: 5 centre slices × 4 modalities, history registered to
the target, zero-initialised warp and residual heads. Only the evaluation changes:
* patient-level CV;
* the checkpoint is chosen on held-out training patients.

Controls:
* tumor-region ("focal") MSE;
* a blurred-persistence baseline, with the Gaussian width tuned on training patients only.

**B. Volume trajectory** (`volume_trajectory_experiment.py`). Targets are log1p(mL) of
necrosis, enhancing tumor and edema at the next visit. Methods:
* persistence;
* per-patient log-linear extrapolation;
* population Gompertz;
* ridge regression and gradient boosting on history features;
* an ODE-RNN (GRU updates at scans, latent ODE between them, zero-initialised residual
  decoder), as a 5-seed ensemble.

**C. 3D mask forecast** (`mask_forecast_experiment.py`).
* Input: 2 mm volumes. The last scan (4 modalities + WT/TC/ET masks), the previous scan's
  masks and gap when available, and the interval.
* Model: a 3D U-Net whose bottleneck is evolved either by a Neural ODE over the interval or
  by an additive interval embedding (ablation). It predicts a displacement field that warps
  the last mask plus residual logits. Both heads are zero-initialised, so the untrained
  model is exactly persistence.
* Selection: early stopping on validation-patient Dice; patient folds as in B (seed 0).
* Metrics: Dice, and change-Dice (overlap of predicted and real change relative to the last
  mask).
* Frames: run in both the atlas frame and the registered frame.

**D. Noise ceiling and stratification** (`step2_followup_analysis.py`).
* (a) Same-scan agreement between DeepBraTumIA, the step-2 label, and HD-GLIO-AUTO on all
  399 target scans. HD-GLIO is mapped by registering its own CT1 to DeepBraTumIA's atlas
  CT1, with 4 starting orientations and no tumor masks used; brain-mask Dice median 0.978.
  LUMIERE's HD-GLIO native-space files are mis-oriented for many scans, so they were not
  used.
* (b) Copy-last-mask Dice on visits the expert rated stable disease.
* (c) Step 2 results split by RANO rating, volume change, overlap change and interval.

**E. Next-visit RANO rating** (`rano_forecast_experiment.py`). The task is PD vs SD/PR/CR
on 374 rated transitions. Features use only information known at the last scan:
* volume history, including height above the nadir;
* the interval;
* the last visit's rating;
* age, sex and MGMT.

Comparators:
* trivial rules: last volume, interval, last rating;
* logistic regression and gradient boosting;
* the step-B forecasts.

## 4. Results

### 4.1 The original claim on held-out patients (A)

| metric | model vs persistence [95% CI] | patients better | 1-px blurred persistence | model minus blur [95% CI] |
| :--- | :---: | ---: | :---: | :---: |
| whole-slice MSE (original metric) | +8.7% [+5.7, +11.7] | 64/79 | +3.7% | +5.0% [+2.1, +8.0] |
| tumor-region MSE | +2.5% [-1.0, +6.0] | 50/79 | +4.1% | -1.6% [-5.1, +2.0] |
| whole-slice MAE | +5.4% [+3.6, +7.2] | 64/79 | -1.2% | +6.7% [+4.8, +8.5] |

The headline number is reproduced on unseen patients. Two things show it is not tumor
forecasting:
* inside the tumor region the gain is indistinguishable from zero and below a blur of the
  last scan;
* the remaining whole-slice gain over blur is outside the tumor, consistent with global
  intensity and normalisation correction.

T1 MSE gets worse (-9.9%).

### 4.2 Volume (B)

| method | MAE, log1p(mL) | vs persistence | patient diff [95% CI] | AUROC, next-visit PD |
| :--- | ---: | ---: | :---: | ---: |
| persistence | 0.745 | - | - | 0.500 |
| log-linear extrapolation | 0.993 | -33.4% | +0.219 [+0.164, +0.277] | 0.541 |
| Gompertz (population) | 0.719 | +3.4% | -0.076 [-0.114, -0.038] | 0.584 |
| ridge | 0.721 | +3.2% | -0.088 [-0.136, -0.040] | 0.653 |
| gradient boosting | 0.693 | +6.9% | -0.085 [-0.119, -0.051] | 0.635 |
| Neural ODE (ODE-RNN) | 0.708 | +4.9% | -0.081 [-0.122, -0.044] | 0.628 |

Learned models reduce volume error significantly, but by a few percent, and much of it is
mean reversion captured by a two-parameter population model. Per-patient exponential
extrapolation is far worse than persistence. The Neural ODE is statistically tied with the
tabular models.

![Figure 2](results/figures/step1_volume_vs_persistence.png)
*Figure 2. Next-visit volume error relative to persistence (95% CI over patients).*

### 4.3 3D masks (C)

| frame | variant | WT Dice, model / persistence | TC | ET | Dice diff [95% CI] | change-Dice diff [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| atlas | Neural ODE | 0.521 / 0.506 | 0.355 / 0.350 | 0.231 / 0.234 | +0.010 [+0.000, +0.019] | +0.127 [+0.107, +0.148] |
| atlas | interval embedding | 0.521 / 0.506 | 0.360 / 0.350 | 0.239 / 0.234 | +0.011 [+0.005, +0.017] | +0.118 [+0.101, +0.135] |
| registered | Neural ODE | 0.573 / 0.570 | 0.408 / 0.406 | 0.266 / 0.278 | +0.003 [-0.003, +0.010] | +0.071 [+0.053, +0.092] |
| registered | interval embedding | 0.574 / 0.570 | 0.408 / 0.406 | 0.277 / 0.278 | +0.006 [+0.001, +0.012] | +0.076 [+0.058, +0.093] |

Registration alone raises persistence Dice more than either model does: 0.506 → 0.570.
It also removes roughly 40-45% of the change-Dice gain, i.e. the part that was
misalignment. In the registered frame:
* the Neural ODE matches persistence on whole tumor and is slightly below it on enhancing
  tumor;
* early stopping kept the persistence initialisation in 2 of 5 folds;
* the Neural ODE adds nothing over a plain interval embedding.

![Figure 3](results/figures/step2_mask_dice.png)
*Figure 3. Next-visit mask Dice, model vs persistence, atlas frame (top) and registered
frame (bottom).*

![Figure 4](results/figures/step2_forecast_registered_ode_fold0_final.png)
*Figure 4. Registered-frame forecasts for held-out patients (fold 0, last-epoch weights, to show
what the network learned; the early-stopped fold-0 model used for the metrics is persistence):
last scan with its outline, real next scan, forecast vs real outline, and change map.*

### 4.4 Noise ceiling and where the mask model helps (D)

| comparison | whole-tumor Dice |
| :--- | :---: |
| DeepBraTumIA vs HD-GLIO on the same scan (patient mean [95% CI]) | 0.648 [0.612, 0.680] |
| same, excluding DeepBraTumIA's necrosis/cavity label | 0.689 [0.655, 0.721] |
| copy-last-mask, visits rated stable, ≤ 12 weeks apart (mean, n=32) | 0.581 |
| copy-last-mask, all transitions (mean) | 0.570 |

Two segmenters reading the same image agree only modestly better than copying the last
visit's mask. Even between clinically stable visits, rescan, segmentation and registration
noise costs about 0.4 Dice. The attainable headroom over persistence for whole tumor is
therefore small (on the order of 0.08 Dice), and the models use a small fraction of it.

Stratified (Figure 5), the mask models behave like a population growth prior:
* They enlarge the tumor in 78-82% of transitions where the forecast differs from the last
  mask; 53% actually grew. This includes tumors that later shrank by more than 20%.
* They gain slightly where tumors grew (+0.009 to +0.019 Dice) and lose where nothing
  changed (-0.014) and under complete response (-0.03).
* Their correlation with the true volume change (+0.38 to +0.46) is matched by the last
  volume alone ("small tumors grow"; +0.47 to +0.48). After removing it, the correlation
  is +0.08 to +0.11.

![Figure 5](results/figures/step2_strata_registered.png)
*Figure 5. Registered-frame gain over copy-last-mask by RANO rating, volume change, overlap
with the last mask, and interval.*

### 4.5 Next-visit RANO rating (E)

| score | AUROC [95% CI] | minus last-rating rule [95% CI] |
| :--- | :---: | :---: |
| last volume ("small tumor grows") | 0.565 [0.491, 0.642] | -0.118 [-0.207, -0.026] |
| last visit's rating | 0.684 [0.621, 0.740] | - |
| logistic, volumes + timing + clinical | 0.593 [0.530, 0.657] | -0.090 [-0.158, -0.020] |
| logistic, all features | 0.686 [0.620, 0.747] | +0.002 [-0.042, +0.053] |
| gradient boosting, all features | 0.723 [0.660, 0.781] | +0.040 [-0.015, +0.092] |
| step 1 Neural ODE, predicted volume change | 0.633 [0.563, 0.708] | -0.051 [-0.133, +0.037] |

A patient rated PD is rated PD at the next visit 87% of the time. That rule is as strong as
any learned model, and as well calibrated: Brier 0.201 vs 0.204 for gradient boosting, and
0.233 for always predicting the base rate.

The clinically useful case is predicting new progression in a patient currently not
progressing (238 transitions, 50% PD). There, every score has AUROC ≤ 0.61 and none is
distinguishable from the last-volume rule.

![Figure 6](results/figures/step3_rano_auroc.png)
*Figure 6. AUROC for next-visit RANO progression: trivial rules, models trained on the
label, and step 1 volume forecasts.*

## 5. Discussion

The Neural ODE formulation, in three settings (2D intensity, 3D mask, volume trajectory),
does not beat the best simple comparator once each confound is removed.

**What inflated the original result:**
* Scoring on training pairs. This turned out to be minor: the gain survives on held-out
  patients.
* Smoothing. An MSE-trained forecaster under uncertainty benefits from blurring, and a
  tuned blur of the last scan explains the tumor-region gain.
* Global intensity correction outside the tumor.

**What inflated the first mask result:**
* Misalignment between visits in a space that was assumed to be aligned.

**Why there is little left to find:**
* Automated tumor masks of the same scan agree only to Dice 0.65.
* Stable visits overlap at 0.58.
* Expert RANO ratings are highly persistent.

**Practical recommendations for next-visit forecasting studies:**
1. Report persistence and a tuned blurred-persistence control.
2. Register the target into the input frame and quantify the residual misalignment.
3. Report a same-scan label-agreement ceiling.
4. Report stratified results (growing, stable and responding tumors), because a growth prior
   looks like skill in pooled metrics.

**Limitations:**
* Single dataset, 79 patients, and no external validation.
* Labels are automated, apart from the RANO ratings.
* The models are deliberately conservative (zero-initialised at persistence) and trained
  for a fixed budget. Larger models or pretraining might extract more. The noise ceiling
  limits how much more is measurable on this data.
* Same-segmenter test-retest data would give a tighter ceiling than the two-segmenter
  comparison.

## 6. Conclusion

On LUMIERE, carrying the last scan, mask or rating forward is close to the best available
next-visit forecast. Neural ODE models offer no measurable advantage over simple
baselines. The evaluation protocol and controls in this repository are the main
contribution.

## Data and code availability

* LUMIERE is available from its authors under their data-use terms. It is not
  redistributed; see `DATA_SETUP.md`.
* All code, derived result tables and figures are in this repository. EXPERIMENTS.md gives
  every command.

---
**Status**: draft, results frozen
**Updated**: November 3, 2026
