# TIDE authorized research release integration

Completed 2026-11-03. The local research workflow is usable. This is not a clinical
validation or an unconditional certification of historical inference reproducibility.
No push, external release, messaging, dataset/checkpoint upload, or mask retraining
was performed. All integration inference/training ran on CPU in the existing `.venv`.
No GPU jobs were needed; future GPU jobs must use `flock /tmp/tide-release-gpu.lock`.

## Worker review and commits

Read all three `runs/release_build/*.exit` and `*.report.txt`, inspected the branch
diffs, worker log endings, packaging contract and complete audit conclusions.
All worker exits were zero; correctness was assessed independently below.
No failed worker required recovering uncommitted changes. Main and manuscript
working trees were initially clean. Existing ignored worker outputs and sibling
worktrees were retained; no reset, cleanup or overwrite of original inputs occurred.

Integrated committed changes:

- `302d151`: prediction worker, fast-forwarded into main.
- `335ab93`: packaging worker, merged by `0493960`.
- `74be41b`: audit worker, merged by `74dbc57`.
- `3e6ac71`: final integrated model contracts, history-only volume/mask inference,
  fixed all-cohort GBM configuration, CV ensemble support, reproduction script,
  validation evidence and corrected scientific interpretations.
- Manuscript repository `/home/tusharn/ppr/tide-node-report`: `dece578`, updated
  `main.tex`, evidence, README and built `main.pdf`, citing analysis `3e6ac71`.
- The subsequent main-project report commit records this report and execution logs;
  find its ID with `git log -1 --format=%H -- runs/release_build/FINAL_REPORT.md`.

## Integrated behavior and fixed contracts

`scripts/predict.py` accepts historical inputs and an explicit positive forecast
horizon. Masks require historical atlas-grid MRI and existing historical
segmentations. Registered models require prepared history-only previous-to-last
alignment. No future scan, target segmentation or cohort lookup is used for
forecast inputs. Mask outputs include NIfTI WT/TC/ET regions, pooled affine, mL
volumes, metadata and PNG overlay. Regions are not forced to nest. A trusted cache
override is a prepared input whose relationship to MRI is not certified by the CLI.

The entrypoint now accepts checksum-verified packaging directories and five-member
CV descriptors. Members must be distinct compatible folds 0–4 in one frame;
logits are averaged before thresholding at zero. Atlas/registered and ODE/static
remain explicit alternatives; no test-based final mask configuration was selected.
Legacy explicit checkpoint descriptors remain supported.

Volume input is an ordered, surgery-separated history of non-overlapping compartment
volumes in mm3 and optional clinical strings. It emits compartment mL, not mask
WT/TC/ET. The exact 22-feature experiment implementation is reused. Loading local
sklearn pickle requires `--trust-volume-bundle`; bundle checksums, implementation
hashes and sklearn version are checked. See PREDICTION.md for schemas.

The all-cohort GBM uses the existing baseline settings, fixed before fitting:
max_depth=3, learning_rate=0.05, max_iter=200, loss=absolute_error, random_state=0,
early_stopping=false. This refit uses all 399 eligible transitions from 79 patients;
39 pre-op scans and 31 surgery targets are excluded. There is no held-out performance
estimate for these refit weights. No test-data retuning was performed.

## Commands and validation

From `/home/tusharn/ppr/glioblastoma-evolution`, using:

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/tide-mpl
PY=.venv/bin/python
$PY -m unittest discover -s tests
$PY -m py_compile scripts/*.py
$PY -m pip check
bash -n scripts/build_research_release.sh
$PY scripts/release_audit.py --source . --output /tmp/tide-release-final-audit --smoke
$PY scripts/release_reference_audit.py --source . --output /tmp/tide-release-final-audit/references.json
$PY scripts/package_models.py fit-volume-all \
  --contract runs/packaging/release-20261103/volume-contract.json \
  --data-dir data/lumiere/extracted/Imaging --clinical-dir data/lumiere/clinical \
  --output runs/packaging/release-20261103/volume-all
# Same contract, independent fitting/serialization:
$PY scripts/package_models.py fit-volume-all \
  --contract runs/packaging/release-20261103/volume-contract.json \
  --data-dir data/lumiere/extracted/Imaging --clinical-dir data/lumiere/clinical \
  --output runs/packaging/release-20261103/volume-replay
$PY scripts/package_models.py inventory-masks --runs-dir runs \
  --data-dir data/lumiere/extracted/Imaging --clinical-dir data/lumiere/clinical \
  --output runs/packaging/release-20261103/masks
$PY scripts/verify_release_inference.py --source . \
  --release runs/packaging/release-20261103 \
  --output runs/release_build/export-equivalence.json
$PY scripts/predict.py --history runs/release_build/mask-history.json \
  --horizon-weeks 12 --bundle runs/packaging/release-20261103/atlas-static-cv-ensemble.json \
  --output-dir runs/release_build/mask-cli-smoke
$PY scripts/predict.py --history runs/release_build/volume-history.json \
  --horizon-weeks 12 --bundle runs/packaging/release-20261103/volume-all \
  --trust-volume-bundle --output-dir runs/release_build/volume-cli-smoke
```

Outputs are exclusive-create: for another run, use new output paths. Histories for
these smoke commands were generated from the first fold-1 test transition's historical
visits only; the 12-week horizon was explicit and no target was used as CLI input.
For a complete fresh local package, run
`scripts/build_research_release.sh NEW_OUTPUT_DIRECTORY`; its checked-in contracts
and all source/estimator/environment hashes preserve the reproduction recipe.

Results:

- **42 tests pass**, final log `integration-tests-final.log`; compilation, shell
  syntax and `pip check` pass. Tests cover strict history inputs, preprocessing,
  affine, checksum/source guards, volume serialization and nonzero-head CV aggregation.
- All 34 mask checkpoint exports pass exact CPU round trips and checksum checks.
  Twenty are validation-selected best CV fold models; fourteen are available final
  weights retained separately. Ensemble descriptors use best weights only.
- Four trained fold-1 models (both variants/frames) match original checkpoint logits
  and CLI masks **exactly at full spatial resolution** on Patient-003's historical
  input. All four CV ensemble forwards succeed. This is packaging equivalence,
  not a re-estimate of generalization and not exact historical CSV reproduction.
- Both real-history CLI smoke runs succeed. Mask outputs include PNG and NIfTI;
  volume output is finite and includes provenance metadata.
- Independent all-cohort GBM fits and exports produce identical estimator bytes:
  SHA-256 `8c75e49351a98f82c56f7ff6b67fee144a85bbc0ee854785fb2c15fa85f0dd80`.
  Export/reload predictions are exact on all fitting transitions. This supports
  local deterministic reproduction in the captured numerical environment only.
- All **798** frame-specific cached reference cases match exactly, max difference 0.
- All **1,459** audited original data/cache metadata, transforms, checkpoints,
  logs and prediction tables retain their worker-audit hashes.

One reference-audit invocation mistakenly passed the output directory rather than
its JSON filename and failed when writing. It was rerun with the explicit filename
above and completed successfully. No input artifacts were changed. This command
error is not counted as a successful check.

## Artifacts and scientific findings

Local weights remain ignored by Git under `runs/packaging/release-20261103/`:
`volume-all`, independent `volume-replay`, 34 bundles under `masks`, and four labeled
`*-cv-ensemble.json` descriptors. Each bundle includes source snapshots, environment,
input provenance and checksums. `release/contracts/` and `release/validation.json`
record fixed settings, counts and artifact hashes in version control.

Audits are frozen under `results/release_audit/integration_{audit,references,effects}.*`
and `export_equivalence.json`. Original worker evidence remains alongside them.
Main manuscript documentation and the separate PDF manuscript now distinguish
inter-segmenter agreement from an accuracy bound, checkpoint selection from early
stopping, and paired effects from overlapping separate confidence intervals.

The paired patient-weighted ODE-minus-ridge volume MAE is +0.0079 log1p(mL)
(95% CI -0.0137 to +0.0290); ODE-minus-GBM is +0.0040 (-0.0122 to +0.0190).
ODE-minus-persistence is -0.0805 (-0.1208 to -0.0420). Mean Dice differences versus
static are -0.0019 (-0.0078 to +0.0041) atlas and -0.0031 (-0.0080 to +0.0013)
registered. These establish neither general ODE superiority nor equivalence.
Secondary exploratory effects vary by endpoint; the full table is retained.

## Manuscript build

Ran `pdflatex -interaction=nonstopmode -halt-on-error main.tex` twice after the
final edits in `/home/tusharn/ppr/tide-node-report`. The installed command invokes
the local TeX engine; both passes exit successfully. Final PDF has 14 pages;
`pdftotext` confirms the paired audit, deterministic-fit statement and analysis
revision. All eight referenced figures exist and remain PNG. No figure formats or
author declarations were invented. Remaining TeX messages are an ignored inputenc
package warning and two underfull paragraph boxes; no unresolved references or
overfull boxes were found. Build logs are copied beside this report.

## Concrete unresolved blockers and limitations

1. Four historical CPU checkpoint replays still fail exact saved-metric agreement
   at 1e-6. Dice differences range 0.000046–0.000138; volume differences range
   0.016–0.056 mL. Original device/precision settings are not authenticated;
   CPU versus CUDA autocast is plausible, not proven. Recover historical run
   metadata and replay in the original numerical environment before certification.
2. Historical volume ensembles were never saved in the inspected outputs. The
   new reproducible GBM refit does not recover original CV/ODE estimator states.
3. Historical source revisions, selected checkpoint epochs, numerical environments
   and cache hashes were not captured contemporaneously. Export-time snapshots and
   reconstructed split manifests do not retroactively authenticate training runs.
4. Mask artifacts are CV ensembles, not all-cohort final fits. Their aggregate
   component CV metrics are not an independent accuracy estimate of the ensembles.
5. Registered evaluation references use retrospective target registration. Automated
   labels, single-center data, exploratory secondary endpoints, unequal static/ODE
   volume training subsets and non-grouped ridge inner CV limit interpretation.
6. External validation, calibration, clinical utility, independent anatomical review,
   broader related work and author-confirmed declarations remain outstanding.

These do not prevent local research inference or reproducing the saved-table paired
analysis. They prevent a claim of exact historical inference certification, formal
ODE equivalence/superiority, submission readiness, or clinical validation.
