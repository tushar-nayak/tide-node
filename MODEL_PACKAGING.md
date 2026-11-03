# TIDE model packaging contract (schema 1)

This work is confined to `release/packaging`. Prediction worker owns the inference
entrypoint. No mask training code was changed. No all-cohort training or GPU jobs
were run. Existing runs, data, and `.venv` under
`/home/tusharn/ppr/glioblastoma-evolution` were read only.

## Artifacts and inventory

Local exports live at `runs/packaging/masks/` in this worktree (ignored by Git).
`inventory.json` lists every discovered mask forecast checkpoint with its source
SHA-256, destination and round-trip result. Each bundle contains:

- Original `weights.pt`, byte-for-byte copied from the source checkpoint.
- `manifest.json`: architecture, constructor arguments, expanded layers, temporal
  solver/warp constants, channel order, preprocessing and training provenance.
- `source/`: exact export-time Python implementation and preprocessing sources.
- `environment.json`: interpreter, platform, CUDA build, installed versions and
  complete `pip freeze --all` output.
- Original `train_log.csv` and `test_predictions.csv`, when present.
- `SHA256SUMS.json`: hashes of every other bundle file, including the manifest.

The checksum index detects corruption; it is not an authenticity signature.
Only load trusted artifacts. Volume pickle loading requires `trusted=True`.

| Source under original `runs/` | Best weights | Final weights | Architecture |
| --- | --- | --- | --- |
| `mask_forecast/ode/fold{0..4}` | 5 | 2 (folds 3,4) | 12 input channels, base 16, ODE |
| `mask_forecast/static/fold{0..4}` | 5 | 2 (folds 3,4) | 12 input channels, base 16, static |
| `mask_forecast_registered/ode/fold{0..4}` | 5 | 5 | 12 input channels, base 16, ODE |
| `mask_forecast_registered/static/fold{0..4}` | 5 | 5 | 12 input channels, base 16, static |

Total: **34 weight files, 20 patient-fold models**. These are cross-validation
weights, **not models trained on all patients**. Best weights can be the initial
persistence model (epoch -1); final weights do not imply validation selection.
Missing final weights are not reconstructed or silently replaced with best weights.
Training-state files, if encountered on subsequent runs, are inventoried but not
exported as deployable weights.

Current cohort reconstruction yields 79 eligible patients, 399 transitions.
The five outer folds use seed 0; inner validation uses seed `1000 + fold`.
Every existing held-out CSV matches the reconstructed patient/last/target tuples.
Manifests list fit, validation and test patients and transitions, and hash the
ratings and volume JSON inputs. This supports reconstruction, not proof of the
historical training cohort. Original command lines, exact epochs/hyperparameters,
source revision, historical environment and historical imaging/cache hashes are
unknown. Current source hashes and environment are explicitly labeled export-time.

## Reproduce the mask exports

Run from this worktree. Output directories must not already exist; choose a new
path for each run. Exports run on CPU. Existing artifacts are never overwritten.

```bash
TIDE_SOURCE=/home/tusharn/ppr/glioblastoma-evolution
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
"$TIDE_SOURCE/.venv/bin/python" scripts/package_models.py inventory-masks \
  --runs-dir "$TIDE_SOURCE/runs" \
  --data-dir "$TIDE_SOURCE/data/lumiere/extracted/Imaging" \
  --clinical-dir "$TIDE_SOURCE/data/lumiere/clinical" \
  --output runs/packaging/masks
"$TIDE_SOURCE/.venv/bin/python" scripts/package_models.py verify \
  runs/packaging/masks/mask_forecast_registered/ode/fold0/best
```

Every mask export strictly loads its tensor-only state dictionary, checks the
architecture, reloads the exported bundle and compares logits and flow exactly on
a deterministic 16³ CPU probe. This exercises all layers; it is not full-resolution
inference validation or a clinical quality check. The original weight bytes and
source hashes are reproducible; machine-specific environment/path fields may differ
on another host. A failed inventory run may leave already verified bundles; the
final `inventory.json` exists only after all exports finish.

## Interface for the prediction worker

`load_mask(bundle)` returns `(model.eval(), manifest)` on CPU. It verifies file
hashes and the implementation hash before loading with `weights_only=True`.
`load_volume(bundle, trusted=True)` returns `(three_estimators, manifest)` and
requires the same scikit-learn version. Callers should use the captured numerical
environment. Both loaders refuse changed relevant source. For an independent
consumer process, put the bundle's `source/` directory first on `sys.path` before
importing `package_models`; do not mix already-imported modules from other versions.

Mask input order is FLAIR, T1, T2, CT1, last WT/TC/ET, previous WT/TC/ET,
previous-present flag, previous gap in years. Inputs are float32; source caches
are float16. Per-modality positive voxels define normalization (not a shared brain
mask), followed by clipping [-5,5] and 2³ mean pooling. Label definitions are
WT={1,2,3}, TC={1,2}, ET={1}. Cache spatial shape is 91×109×91, padded at high
ends to 96×112×96, including constant channels. Tensor spatial axes follow the
atlas array x/y/z order; flow channels follow PyTorch W/H/D order. Forecast dt
and previous gap use weeks/52. Missing history gives zero previous masks, flag
and gap. Registered models require the previous scan in the last scan's frame;
the last scan is unchanged. Registration details are in bundled
`register_step2_pairs.py`. Never use target-derived information for prediction.

Outputs are WT/TC/ET logits and normalized displacement; threshold logits >0,
crop padding, and use 0.008 mL/voxel. The model does not enforce region nesting.
Integration must decide output affine/orientation handling, nesting policy, input
validation, missing modality behavior, and selection of frame/variant/fold or
ensemble. The manifest does not select a production model.

Volume features and clinical encoding are included as executable source plus a
22-column named feature contract. `volume_features()` delegates to the existing
experiment's `history_features` and `static_features`; there is no learned scaler.
Volumes enter as mm³ and become log1p(mL). The three GBMs predict changes in
necrosis/enhancing/edema log volumes; `predict_volume()` adds the last observation
and clips at zero. Apply `expm1` to obtain mL. This library helper supports export
validation; it is not a competing product inference entrypoint.
`export_volume(models, output, provenance)` serializes already-fitted estimators,
checks their type and feature count, bundles preprocessing, and verifies exact
prediction equality after reload. Existing volume experiments emitted predictions
but did not retain GBM estimators; there are no historical GBM weights to recover.
No real-cohort volume artifact has been fabricated from those predictions.

## Deferred all-cohort fitting

A reproducible CPU GBM refit command is implemented. **Do not run it until integration
has fixed the settings and model contract.** The command requires an explicit JSON
contract; there is no implicit production configuration. For example, after review,
write `runs/packaging/volume-contract.json` with these keys (values below reflect
experiment settings and are a proposal, not a selected release configuration):

```json
{
  "contract_id": "replace-with-fixed-integration-contract",
  "max_depth": 3,
  "learning_rate": 0.05,
  "max_iter": 200,
  "loss": "absolute_error",
  "random_state": 0,
  "early_stopping": false
}
```

```bash
"$TIDE_SOURCE/.venv/bin/python" scripts/package_models.py fit-volume-all \
  --contract runs/packaging/volume-contract.json \
  --data-dir "$TIDE_SOURCE/data/lumiere/extracted/Imaging" \
  --clinical-dir "$TIDE_SOURCE/data/lumiere/clinical" \
  --output runs/packaging/volume-all
```

Use the thread limits and interpreter from the export recipe. This fits all eligible
surgery-separated transitions, not every enrolled patient's every scan. Internal
early stopping must be disabled so no implicit validation split removes training
rows. The bundle records exact estimator parameters (including library defaults),
seed, contract hash, cohort input hashes, exclusions, patients and transitions.
There are no held-out metrics for this refit. The command compares predictions on
all fitting transitions before and after serialization. Exact reproducibility is
limited to the captured software/platform and thread configuration.

Mask all-cohort training is intentionally not implemented by altering the CV trainer.
It requires integration to fix frame, variant, base width, seed, epoch budget,
augmentation, optimizer/scheduler, mixed precision, and selection policy without
a held-out validation set. Its current CLI always removes test and validation
patients; passing fewer folds is not an all-patient training command. A future
trainer should record these settings, actual cohort/cache hashes and a full fitting
manifest at training time. All future GPU training/inference jobs must use:

```bash
flock /tmp/tide-release-gpu.lock <gpu-command-and-arguments>
```

## Environment and validation

`environments/packaging-2026-11-03/environment.json` and
`requirements.freeze.txt` capture the exact inspected environment: Python 3.14.4,
PyTorch 2.13.0+cu129, NumPy 2.5.4, scikit-learn 1.9.1 (see files for all packages).
This is an environment inventory, not a portable wheel lock or a claim about the
original training environment. CUDA-specific versions require the matching wheel
source/platform; wheel URLs/hashes and the GPU driver are not captured. Retain the
broad compatible `requirements.txt` for normal installs. Scikit-learn is now
explicitly declared; no shared environment packages were installed or changed.
The original environment passes `pip check`. Fresh installation on other Python
versions/platforms has not been tested.

```bash
"$TIDE_SOURCE/.venv/bin/python" scripts/package_models.py snapshot-environment \
  --output runs/packaging/environment-new
"$TIDE_SOURCE/.venv/bin/python" -m unittest discover -s tests -p test_package_models.py -v
```

Validation covers all 34 real mask exports with exact CPU round trips, both mask
variants with nonzero heads, rejection of incompatible channels, checksum and
extra-file tampering, refusal to overwrite, source mismatch, volume preprocessing
and trusted serialization, and the explicit all-cohort command on a **synthetic**
48-row fixture. No real-cohort refitting was performed. Remaining integration work
is to fix the production contract, implement the prediction entrypoint and
full-resolution checks, then deliberately schedule any approved refits.

## Integrated research release, 2026-11-03

The deferred decisions above are now fixed for this local release. The existing
GBM baseline hyperparameters were retained without further outcome-driven tuning;
seed 0 and no internal early stopping define the all-eligible-cohort refit.
The checked-in contract is `release/contracts/volume-gbm-v1.json`. Two independent
fits in the existing `.venv`, with `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`, produced
byte-identical `estimators.pkl` files (SHA-256
`8c75e49351a98f82c56f7ff6b67fee144a85bbc0ee854785fb2c15fa85f0dd80`).
This verifies local deterministic fitting/serialization in the captured environment,
not portability to another numerical environment or held-out refit performance.

Local artifacts: `runs/packaging/release-20261103/{volume-all,volume-replay,masks}`.
All 34 mask exports have exact CPU round trips. Four full-resolution fold-1 checks
also match original checkpoint logits and CLI masks exactly, and all four five-fold
CV ensemble forwards succeed. See `results/release_audit/export_equivalence.json`.
The prediction CLI accepts packaged directories and the four ensemble descriptors
at the release root. Copy the descriptors from `release/contracts/` to the release
root after reproducing exports; their member paths are relative to that root.
Mean logits are thresholded at zero, without enforcing nesting.

These mask ensembles are explicitly CV ensembles, with no new independent ensemble
evaluation. No final mask configuration was selected using test outcomes and no
mask models were retrained. Exact CPU packaging equivalence is distinct from
historical CSV reproduction, which still fails on the four audited trained cases.
Original volume ensembles were not archived; the new GBM is a separately identified
refit and does not recreate historical ODE or GBM CV weights.
