# Tumor Interval Dynamics Estimation via Neural ODE Sequences
## (previously glioblastoma-evolution)

A leakage-free evaluation of next-visit glioblastoma forecasting on the LUMIERE
longitudinal MRI dataset (79 patients, 399 scan-to-scan transitions). Every experiment is
scored against **carrying the last observation forward**, on patients the model never saw,
with patient-level bootstrap confidence intervals.

**Bottom line: on LUMIERE, carrying the last observation forward is close to the best
next-visit forecast available.** Once the evaluation is leakage-free, the scans are properly
aligned and trivial controls are included, no Neural ODE model in this repository beats the
best simple baseline for its target. Full details, numbers and commands are in
[EXPERIMENTS.md](EXPERIMENTS.md); the draft paper is [MANUSCRIPT.md](MANUSCRIPT.md).

## Findings

| Question | Result | EXPERIMENTS.md § |
| :--- | :--- | :---: |
| Does the original "+6.7% over persistence" hold on unseen patients? | The number holds: whole-slice MSE is +8.7% [+5.7, +11.7]. The tumor-forecasting reading does not: in the tumor region it is +2.5% [-1.0, +6.0], and a 1-pixel blur of the last scan does at least as well (+4.1%). | 8 |
| Can tumor volume be forecast? | Learned models cut volume error 5-7% below persistence (CIs exclude 0). A population Gompertz fit alone gets 3.4%, and the Neural ODE is tied with ridge regression and gradient boosting. | 7 |
| Can the 3D tumor mask be forecast? | In LUMIERE's atlas space, consecutive scans are misaligned by a median 2.5 mm. That misalignment was 40-45% of the change the first models appeared to predict. After registration, Dice is 0.573 vs 0.570 for copy-last-mask (+0.003 [-0.003, +0.010]). | 2b, 7 |
| How much headroom is there? | Little. Two segmenters on the *same* scan agree only to whole-tumor Dice 0.65. Visits rated stable at most 12 weeks apart overlap at 0.58. | 9a |
| Where does the mask model help? | Nowhere specific. It enlarges the tumor in about 80% of cases (truth: 53%), including tumors that later shrank. Its sense of direction is "small tumors grow, big ones shrink." | 9b |
| Can the next RANO rating be forecast? | The last visit's rating alone gives AUROC 0.684. No learned model beats it significantly (best +0.04 [-0.015, +0.092]). For patients not currently progressing, every score is ≤ 0.61. | 10 |

Figures: `results/figures/`, listed at the end of EXPERIMENTS.md.

### What changed from the earlier claim

Earlier versions of this README reported a +6.7% mean MSE improvement over persistence
across 81 patient-level runs (`results/lumiere_full_v1_metrics.*`, kept for reference).
Every row of that table was scored on the patient's own training pairs
(`metric_split=all`). Re-tested on held-out patients, the number survives, but the gain is
smoothing and global intensity correction, not tumor forecasting (section 8).

Three data issues found along the way are documented in EXPERIMENTS.md:
* **Duplicate scans per week** were resolved by filesystem order. Fixed in
  `lumiere_weeks.py` (section 1).
* **Atlas-space misalignment** between visits. Fixed by `register_step2_pairs.py`
  (section 2b).
* **LUMIERE's HD-GLIO native-space masks are mis-oriented** for many scans. Worked around by
  image registration (section 9a).

## Repository map

| Step | Script | Output |
| :--- | :--- | :--- |
| cohort rules (no pre-op scans, no forecasting across surgery) | `scripts/lumiere_cohort.py`, `scripts/lumiere_weeks.py` | - |
| original intensity model, held-out retest + blur control | `scripts/intensity_forecast_cv.py` (model: `scripts/run_neural_ode_pipeline.py`) | `results/intensity_forecast/` |
| step 1: volume trajectory | `scripts/volume_trajectory_experiment.py` | `results/volume_trajectory/` |
| step 2: 3D mask forecast (atlas or registered frame) | `scripts/build_atlas_cache.py`, `scripts/register_step2_pairs.py`, `scripts/mask_forecast_experiment.py` | `results/mask_forecast{,_registered}/` |
| alignment check | `scripts/alignment_check.py` | `results/alignment/` |
| noise ceiling + stratified step 2 | `scripts/step2_followup_analysis.py` | `results/step2_followups/` |
| step 3: next-visit RANO rating | `scripts/rano_forecast_experiment.py` | `results/rano_forecast/` |
| figures | `scripts/plot_step_results.py`, `scripts/visualize_mask_forecasts.py` | `results/figures/` |

## Reproducing

Data: see [DATA_SETUP.md](DATA_SETUP.md). LUMIERE is not redistributed here.

Long jobs run in tmux with logs in `runs/logs/` and a `*.DONE` sentinel. All are resumable,
and registrations are cached and never repeated. The exact commands are in
EXPERIMENTS.md section 6. In short:

```bash
.venv/bin/python scripts/volume_trajectory_experiment.py           # step 1, CPU
.venv/bin/python scripts/build_atlas_cache.py                      # step 2 input cache, once (CPU)
.venv/bin/python scripts/register_step2_pairs.py                   # step 2 registration, once (~1 h CPU)
tmux new -d -s gbm_followups 'bash scripts/run_followup_experiments.sh'   # intensity retest + step 2 registered (GPU, ~6 h)
.venv/bin/python scripts/step2_followup_analysis.py ceiling        # ~1 h CPU
.venv/bin/python scripts/step2_followup_analysis.py report
.venv/bin/python scripts/rano_forecast_experiment.py               # step 3, ~1 min CPU
.venv/bin/python scripts/plot_step_results.py
```

Tests:

```bash
python3 -m py_compile scripts/*.py
python3 -m unittest discover -s tests
```

## Legacy pipeline

`scripts/run_neural_ode_pipeline.py` is the original 2D Attention U-Net + Neural ODE
intensity forecaster (per-patient runs, `results/lumiere_full_v1_metrics.*`). It still
runs, and the smoke test below exercises it on synthetic data, but its per-patient
evaluation scores training pairs. Use `scripts/intensity_forecast_cv.py` for held-out
numbers.

```bash
python3 scripts/generate_dummy_lumiere.py --output-dir test_data --num-weeks 3
python3 scripts/run_neural_ode_pipeline.py --lumiere --data-dir test_data --patients Patient-999 \
  --epochs 1 --batch-size 1 --model-size tiny --run-name smoke_default_holdout_3wk
```

## Status and limitations

* One dataset (LUMIERE, single centre), 79 patients. There is no external validation.
* All tumor labels are automated (DeepBraTumIA). Their reliability is the main limit on any
  mask-level claim (section 9a).
* Results are complete and frozen for the analyses above. `PUBLICATION_PROTOCOL.md`
  describes the original intensity-forecasting protocol and is superseded by
  EXPERIMENTS.md for what is reported here.
