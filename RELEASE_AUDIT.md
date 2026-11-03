# Release evaluation audit

Audit date: 2026-11-03. Branch: `release/audit`. Source and worktree started at
`66c6f7f93af5233247eb337c8451de12c5facc4c`. Main project inputs were read only;
no training, manuscript edits, push, or publication were performed.

**Disposition: saved-table evaluation completed; full prediction provenance remains
blocked.** The final volume and both mask experiments have complete expected case
coverage. Directly paired effects do not establish an ODE advantage in mean mask
Dice or volume MAE over static comparators. Some secondary mask effects differ;
“tied everywhere” is not supported. CPU inference works but fails exact metric
reproduction on all four sampled trained checkpoints.

## Reproduce

Run from this worktree (requires the existing main data, caches and checkpoints):

```bash
PY=/home/tusharn/ppr/glioblastoma-evolution/.venv/bin/python
SOURCE=/home/tusharn/ppr/glioblastoma-evolution
export MPLCONFIGDIR=/tmp/tide-audit-mpl
"$PY" scripts/release_audit.py --source "$SOURCE" --smoke
"$PY" scripts/release_reference_audit.py --source "$SOURCE"
"$PY" -m unittest discover -s tests
"$PY" -m py_compile scripts/*.py
```

The scripts write only to `results/release_audit` by default and reject outputs
inside the source directory. Integrity failures raise errors; CPU metric mismatches
are explicitly recorded in `audit.json` and its status, rather than aborting the
saved-table analysis. `--bootstrap` and `--seed` are configurable. No GPU is used
by the audit inference. The reference check reads mask channels from every scored
cache; it does not rerun registration.

Artifacts:

- [effects.md](results/release_audit/effects.md): all direct comparisons and CIs.
- [audit.json](results/release_audit/audit.json): checks, reconstructed mask split
  membership, checkpoint inspection, software versions, CPU metric differences,
  SHA-256 hashes of prediction tables, source scripts, checkpoints, logs, registration
  metadata and transforms. These hashes freeze the inputs observed now; they are
  not evidence of what originally generated a table.
- [references.json](results/release_audit/references.json): independently recomputed
  mask reference metrics and hashes of the unthresholded mask channels consumed.
- [tests.log](results/release_audit/tests.log), [run.log](results/release_audit/run.log),
  and [references.log](results/release_audit/references.log): execution evidence.

## Cohort, folds and repeat handling

Rebuilding the shared cohort yields **79 patients and 399 transitions**, excluding
39 preoperative scans and 31 surgery targets, with no missing-volume or missing-mask
exclusions. All mask variants contain exactly these cases in five folds. Each
patient's test assignment matches `patient_folds(seed=0)`, and reconstructed mask
fit, validation and test patient sets are disjoint. Predictions do not contain
training membership, so this establishes consistency with the current code, not a
historical execution attestation.

Final volume predictions are in `results/volume_trajectory/predictions.csv`, not
`runs/volume_trajectory_quick`. They contain **1,197 rows = 399 cases × three repeats**,
with five folds per repeat and assignments matching seeds 0, 1 and 2. The protocol
metadata describes five ensemble members and up to 400 epochs. The quick run has
399 rows, one repeat, two members and up to 100 epochs; its effects are reported
separately and must not replace the final results.

Duplicate transition keys, missing/unexpected folds or repeats, missing cohort
cases, incorrect patient assignments, and nonfinite prediction/metric values are
fatal checks. No such failures were found. Volume last-scan and target compartment
values were checked against the source volume JSONs. Paired mask models must agree
on case identity, interval, history length, target rating, reference volumes and
all persistence metrics; no disagreement was found.

Volume uses the same outer assignment as masks only for repeat 0. Inner validation
seeds differ: volume uses `1000*repeat + fold`; masks use `1000 + fold`. Volume ODE
normalization and fitting use fit patients only, and checkpoint selection uses
validation patients. Static volume comparators use all outer-training patients,
including those reserved for ODE validation. `RidgeCV` is not configured with
patient-grouped inner CV. This does not expose outer-test patients, but the model
comparison is not an equal-training-set, matched-architecture ODE ablation.
There is no saved no-ODE volume neural model; ridge, gradient boosting and Gompertz
are the available non-ODE comparators.

## Directly paired inference and manuscript-ready findings

Each effect subtracts the comparator's metric from the ODE's metric on the exact
same transition (and repeat for volume), averages compartments/regions and
transitions within patient, and then resamples the **79 patient differences** with
replacement. Volume repeat losses are averaged within patient; repeats are neither
independent patients nor averaged predictions. Complete repeat coverage prevents
unequal repeat weighting. We use 20,000 percentile bootstrap draws, seed 20261103.
Intervals are conditional on the fitted models, do not capture retraining/split
uncertainty, and are not adjusted for multiple comparisons. No formal equivalence
margin or multiplicity-controlled primary testing plan was supplied.

Suggested concise findings:

> Among 79 patients (399 longitudinal transitions), the ODE's patient-weighted
> volume MAE differed from ridge by +0.0079 log1p(mL) (95% patient bootstrap CI
> −0.0137 to +0.0290) and from gradient boosting by +0.0040 (−0.0122 to +0.0190).
> Its difference from persistence was −0.0805 (−0.1208 to −0.0420). These paired
> comparisons support improvement over persistence, but do not establish superiority
> over the non-ODE comparators or equivalence to them.

> For next-scan masks, ODE minus static mean Dice was −0.0019 (95% CI −0.0078 to
> +0.0041) in atlas space and −0.0031 (−0.0080 to +0.0013) in the registered frame.
> Mean change-Dice differences were +0.0085 (−0.0010 to +0.0181) and −0.0042
> (−0.0175 to +0.0086), respectively. These results do not establish an ODE benefit
> in mean overlap. Exploratory registered-frame mean log-volume error favored ODE
> by −0.0213 (−0.0428 to −0.0048), with an ET effect of −0.0628 (−0.1156 to −0.0213).

Atlas WT and TC change-Dice also have unadjusted intervals above zero (WT +0.0239
[+0.0128, +0.0375]; TC +0.0088 [+0.0008, +0.0174]). These secondary findings preclude
“tied everywhere,” but do not establish general or clinical superiority. Consult
the complete table rather than selecting only favorable endpoints. Patient-weighted
effects need not equal differences between transition-weighted headline means.

## Checkpoint selection and prediction provenance

The current mask trainer runs 40 epochs, retains the checkpoint with best
validation-transition mean Dice, includes the zero-output persistence initialization
as a candidate, and reloads `best.pt` before writing test CSVs. It does not stop
training early; “validation-selected checkpoint” is more precise than “early
stopping.” Saved `best.pt` files and 40 finite, uniquely numbered log epochs exist
for all 20 model/fold combinations. Validation logs omit the initialization score,
round Dice to five decimals, and do not store selected epoch metadata. Accordingly,
`best_logged_epoch` in the audit is not an authenticated checkpoint epoch.

Zero flow/residual heads and exact model/persistence equality in the saved CSVs
identify fallback to persistence in:

| Frame | ODE folds | Static folds |
|---|---|---|
| Atlas | 0 | 0 |
| Registered | 0, 4 | **0, 4** |

The registered static fold-4 fallback is missing from the existing documentation.
Atlas `final.pt` files are absent for folds 0–2 in both variants; registered final
weights all exist. Final-weight figures do not illustrate the evaluated best
checkpoint unless explicitly regenerated with `--weights best`.

Four CPU forward passes use the first fold-1 test case (Patient-003, target week-014),
one per frame/variant, with actual cached inputs and saved best weights. All load
and produce finite metrics. Maximum absolute Dice differences range from 0.000046 to
0.000138. All fail exact agreement at 1e-6: maximum absolute
metric differences are 0.024, 0.032, 0.016 and 0.056, respectively, in predicted
volume mL. CPU fp32 versus original CUDA autocast is a plausible cause; it is not
proven. Per-metric deltas are retained rather than treating mixed-unit maxima as
accuracy measures. This is a smoke check, not an exhaustive checkpoint replay.

Historical CSVs lack run configuration, frame, checkpoint hash, source revision,
precision/device, selected epoch and source-cache hashes. Mask CSV fold/variant/frame
are inferred from directory names. Volume fitting returns prediction closures and
does not archive its ensemble checkpoints, limiting independent reconstruction
without forbidden retraining. The mask trainer also skips a fold solely because its
CSV exists, and resumes without validating configuration or restoring RNG/scaler
state. Existing files therefore cannot certify exact original execution provenance.

## Reference alignment

The atlas and registered experiments use identical transition identities but
**different target representations**; effects must be computed separately within
frame. Registered targets and previous-history masks are mapped into the unchanged
last-history frame; the forecast inputs do not use target intensities or labels.
Target registration itself uses future anatomy/segmentation to construct the
reference, so evaluation remains conditional on that retrospective registration.

All **691 required target/previous pairs** have `.npy`, `.h5` and matching patient,
moving-source and fixed-source JSON metadata. QC values are finite. Methods are
534 rigid+affine, 132 rigid, 13 unmasked rigid+affine, 11 unmasked rigid, and one atlas
fallback. Registration code shares one transform across modalities and segmentation,
uses nearest-neighbor label interpolation followed by pooling, and thresholds mask
fractions at `>0.5` for scoring. QC tolerates up to 0.002 brain-Dice degradation;
“never worse” is not literally the implemented criterion.

The independent reference script recomputes persistence Dice, change-Dice, volume
and log-volume error and target volumes from all 399 cached last/target pairs per
frame. Agreement with the ODE table, plus the audited equality of ODE/static
reference columns, checks both variants. All 798 frame-specific cases passed with
maximum absolute metric difference **0.0**. This does not certify native-image
orientation, transform quality, or clinical ground truth: the registration code
assumes the common atlas voxel grid, and no registration was rerun.

Inter-segmenter Dice is **agreement between two automated labeling methods**, not
a formal accuracy ceiling, attainable performance bound, or estimate of biological
forecastability. Different label definitions, shared errors, real interval change
and registration errors prevent interpreting its gap above persistence as available
headroom. Likewise, true-volume-change AUROC against RANO is a reference score,
not an upper bound on every volume-based predictor.

## Existing claims needing correction (documents left unchanged)

| Location | Required correction |
|---|---|
| `EXPERIMENTS.md:181–184` | Replace “statistically tied (overlapping CIs)” with direct paired estimates; CI overlap is not a test of paired difference or equivalence. |
| `EXPERIMENTS.md:202,234` | Replace “tied everywhere”/“adds nothing” with endpoint-specific paired findings, including exploratory volume-error effects. |
| `EXPERIMENTS.md:231–233` | Registered static persistence fallback is folds 0 **and 4**. Best-versus-final behavior does not prove longer training could never help. |
| `EXPERIMENTS.md:225–229` | Registration changes the reference and trained models; reduced gains do not identify a causal percentage of previously learned “misalignment.” |
| `MANUSCRIPT.md:75–76` | Identical folds applies to outer repeat 0; volume has three assignments, and inner validation splits differ. |
| `MANUSCRIPT.md:19,34,141,226–239,299,307–310`; `EXPERIMENTS.md:282–300` | Rename “noise ceiling” as inter-segmenter agreement and remove the formal headroom/ceiling interpretation. Stable-visit differences cannot all be assigned to measurement noise. |
| `EXPERIMENTS.md:174–186` | True-volume-change AUROC 0.713 is a reference, not an oracle ceiling for any possible volume-change score. |
| `scripts/build_atlas_cache.py:3–6` | Common atlas grid does not establish longitudinal anatomical alignment; later registration findings demonstrate residual mismatch. |
| `PUBLICATION_PROTOCOL.md` | It describes the older slice-intensity pipeline, not a locked protocol for these volume and 3D-mask analyses. Do not treat it as preregistration of these comparisons. |

## Validation, blockers and next steps

The full existing suite plus seven audit tests passes (27 tests). Tests exercise
patient weighting, row-order-independent pairing, zero paired differences,
duplicates, unmatched cases, nonfinite metrics, missing repeats/folds/cases and
wrong patient assignments. All scripts also pass `py_compile`. Existing tests cover frame paths and affine composition.
CPU inference completed without retraining. Exact saved prediction reproduction
failed as detailed above; this is retained as a failure rather than relaxed away.

Before a fully provenance-certified release:

1. Recover original run configurations/logs, code revision and numerical environment;
   attach selected checkpoint and cache hashes to each prediction table. Validate
   each selected epoch against unrounded validation records where available.
2. Replay saved mask best checkpoints on all cases in the original precision/device
   environment, writing new artifacts in an authorized isolated output directory.
   Compare with frozen tables and explain discrepancies. No retraining is required.
3. For volume, recover the original ensemble states if they exist elsewhere. If not,
   label the release as reproducible saved-prediction analysis, not reproducible
   model inference; any retraining requires separate authorization.
4. Correct the documentation claims above, explicitly label final-weight figures,
   and lock endpoint priorities before confirmatory statistical interpretation.

The audit completes the requested evaluation of available outputs; these remaining
historical-provenance and interpretation limits prevent an unconditional release
certification. No new model training or manuscript changes were attempted.
