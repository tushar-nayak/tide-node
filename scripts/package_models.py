"""Versioned TIDE exports. CPU only; prediction entrypoint belongs to integration.

Bundles contain trusted sklearn pickle or tensor-only torch weights, a JSON contract,
source snapshots, environment inventory and SHA256SUMS.json. See MODEL_PACKAGING.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import platform
import shutil
import subprocess
import sys
import tempfile

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
SCHEMA = 1
SOURCES = ("package_models.py", "mask_forecast_experiment.py", "build_atlas_cache.py",
           "register_step2_pairs.py", "volume_trajectory_experiment.py",
           "lumiere_cohort.py", "lumiere_weeks.py")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def environment():
    import torch
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(),
            "torch_cuda_build": torch.version.cuda,
            "packages": dict(sorted((d.metadata["Name"], d.version)
                                    for d in importlib.metadata.distributions())),
            "pip_freeze_all": subprocess.check_output(
                [sys.executable, "-m", "pip", "--no-cache-dir", "freeze", "--all"], text=True)}


def seal(path, manifest):
    manifest["schema_version"] = SCHEMA
    manifest["environment_scope"] = ("fitting and export environment" if manifest["provenance"].get("all_cohort")
                                     else "export environment; historical training environment unknown")
    src = path / "source"
    src.mkdir()
    for name in SOURCES:
        shutil.copyfile(SCRIPTS / name, src / name)
    manifest["source_sha256"] = {name: sha256(src / name) for name in SOURCES}
    manifest["source_scope"] = ("fitting and export implementation" if manifest["provenance"].get("all_cohort")
                                else "export-time implementation, not proof of historical training source")
    write_json(path / "manifest.json", manifest)
    write_json(path / "environment.json", environment())
    write_json(path / "SHA256SUMS.json", {
        str(p.relative_to(path)): sha256(p) for p in sorted(path.rglob("*")) if p.is_file()
    })


def verify(path):
    path = Path(path)
    checks = json.loads((path / "SHA256SUMS.json").read_text())
    files = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
    if files != set(checks) | {"SHA256SUMS.json"}:
        raise ValueError("Bundle file inventory mismatch")
    for name, digest in checks.items():
        p = (path / name).resolve()
        if not p.is_relative_to(path.resolve()) or sha256(p) != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["schema_version"] != SCHEMA:
        raise ValueError("Unsupported bundle schema")
    return manifest


def check_source(manifest, names):
    for name in names:
        if sha256(SCRIPTS / name) != manifest["source_sha256"][name]:
            raise ValueError(f"Implementation differs from bundle: {name}; use bundled source")


def load_mask(path):
    import torch
    from mask_forecast_experiment import MaskForecastNet
    manifest = verify(path)
    if manifest["kind"] != "mask":
        raise ValueError("Not a mask bundle")
    check_source(manifest, ["package_models.py", "mask_forecast_experiment.py"])
    model = MaskForecastNet(**manifest["architecture"]["constructor"])
    model.load_state_dict(torch.load(Path(path) / "weights.pt", map_location="cpu", weights_only=True), strict=True)
    return model.eval(), manifest


def mask_contract(state, frame):
    from mask_forecast_experiment import MaskForecastNet
    if frame not in ("atlas", "registered"):
        raise ValueError("Unknown coordinate frame")
    variant = "ode" if "temporal.f.0.weight" in state else "static"
    base, in_ch = state["enc.0.0.weight"].shape[:2]
    if in_ch != 12:
        raise ValueError("Unsupported input channel count")
    config = dict(in_ch=in_ch, base=base, variant=variant)
    model = MaskForecastNet(**config).eval()
    model.load_state_dict(state, strict=True)
    return model, {
        "constructor": config, "class": "mask_forecast_experiment.MaskForecastNet",
        "modules": repr(model), "ode_solver": "fixed RK4, 2 steps, float32" if variant == "ode" else None,
        "pool": "max_pool3d kernel/stride=2, three levels", "flow_scale": 0.1,
        "flow_channels": ["W", "H", "D"], "persist_logit_scale": 8.0,
        "warp": {"mode": "bilinear", "padding_mode": "border", "align_corners": True},
    }


def export_mask(checkpoint, output, frame, provenance):
    import torch
    torch.set_num_threads(2)
    checkpoint, output = Path(checkpoint), Path(output)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model, architecture = mask_contract(state, frame)
    manifest = {
        "kind": "mask", "architecture": architecture, "provenance": provenance,
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "preprocessing": {
            "frame": frame, "cache_shape": [7, 91, 109, 91], "padded_shape": [96, 112, 96],
            "axis_order": "N,C,atlas-x,atlas-y,atlas-z (torch D,H,W)",
            "channels": ["FLAIR", "T1", "T2", "CT1", "last_WT", "last_TC", "last_ET",
                         "previous_WT", "previous_TC", "previous_ET", "previous_present", "previous_gap_years"],
            "normalization": "Per modality: mask v>0; (v-mean(v[mask]))/(std(v[mask])+1e-6); clip [-5,5]; zero outside mask",
            "downsample": "2x2x2 mean pooling after truncation to even shape; float16 cache then float32 tensors",
            "mask_labels": {"WT": [1, 2, 3], "TC": [1, 2], "ET": [1]},
            "padding": "Zero at high ends of all spatial axes, including constant channels",
            "missing_previous": "Zero previous masks, flag and gap",
            "time": "dt and previous gap in weeks / 52",
            "registration": "Previous scan into last frame via register_step2_pairs.py; last untouched" if frame == "registered" else "Atlas coordinates; no pair registration",
            "output": "WT,TC,ET logits; threshold >0; crop high-end padding; 0.008 mL per voxel; no enforced nesting",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir()
        shutil.copyfile(checkpoint, stage / "weights.pt")
        for name in ("train_log.csv", "test_predictions.csv"):
            if (checkpoint.parent / name).exists():
                shutil.copyfile(checkpoint.parent / name, stage / name)
        seal(stage, manifest)
        restored, _ = load_mask(stage)
        # Small CPU probe exercises every layer and temporal/warp path, not full-resolution quality.
        g = torch.Generator().manual_seed(1729)
        x = torch.randn((1, 12, 16, 16, 16), generator=g)
        lm = (torch.rand((1, 3, 16, 16, 16), generator=g) > 0.5).float()
        dt = torch.tensor([0.25])
        with torch.no_grad():
            for a, b in zip(model(x, lm, dt), restored(x, lm, dt)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                if not torch.isfinite(b).all():
                    raise ValueError("Nonfinite mask output")
        stage.rename(output)
    return manifest


def cohort_inputs(data, clinical, mask=False):
    from lumiere_cohort import build_cohort
    ratings = clinical / "LUMIERE-ExpertRating-v202211.csv"
    if not data.is_dir() or not ratings.is_file():
        raise FileNotFoundError("Imaging directory and ratings CSV are required")
    cohort = build_cohort(data, ratings, require_seg_mask=mask)
    if not cohort.transitions:
        raise ValueError("Empty cohort")
    paths = {ratings}
    for scans in cohort.scans.values():
        paths.update(s.scan.volumes_json_path for s in scans if s.scan.volumes_json_path)
    return cohort, {str(p): sha256(p) for p in sorted(paths)}


def transition_records(transitions):
    return [{"patient": t.patient_id, "history": [s.source_name for s in t.history],
             "weeks": [s.week for s in t.history], "target": t.target.source_name,
             "target_week": t.target.week} for t in transitions]


def inventory_masks(runs, data, clinical, output):
    from mask_forecast_experiment import fold_split
    cohort, hashes = cohort_inputs(data, clinical, mask=True)
    if output.exists():
        raise FileExistsError(output)
    paths = sorted(p for root in (runs / "mask_forecast", runs / "mask_forecast_registered")
                   for variant in ("ode", "static") for p in (root / variant).glob("fold*/*.pt"))
    if not paths:
        raise ValueError("No mask checkpoints found")
    output.mkdir(parents=True)
    entries = []
    for path in paths:
        relative = path.relative_to(runs)
        fold = int(path.parent.name.removeprefix("fold"))
        if path.name not in ("best.pt", "final.pt"):
            entries.append({"path": str(relative), "sha256": sha256(path), "status": "not exported: training state"})
            continue
        fit, val, test = fold_split(cohort.transitions, 5, fold)
        pred_path = path.parent / "test_predictions.csv"
        observed = None
        if pred_path.exists():
            with pred_path.open() as f:
                observed = sorted((r["patient_id"], r["last_source"], r["target_source"]) for r in csv.DictReader(f))
        expected = sorted((t.patient_id, t.last.source_name, t.target.source_name) for t in test)
        provenance = {
            "training_scope": "cross_validation_fold", "all_cohort": False,
            "fold": fold, "folds": 5, "fold_seed": 0, "validation_seed": 1000 + fold,
            "split_status": "reconstructed from export-time cohort/code; not historical training attestation",
            "test_rows_match_reconstruction": observed == expected if observed is not None else None,
            "fit_patients": sorted({t.patient_id for t in fit}),
            "validation_patients": sorted({t.patient_id for t in val}),
            "test_patients": sorted({t.patient_id for t in test}),
            "transitions": {k: transition_records(v) for k, v in (("fit", fit), ("validation", val), ("test", test))},
            "cohort_input_sha256": hashes,
            "unknown": ["historical source revision", "training environment", "exact CLI/hyperparameters",
                        "historical imaging/cache hashes", "best checkpoint epoch (may be persistence initialization)"],
            "selection": "best validation mean Dice, including epoch -1 initialization" if path.name == "best.pt" else "final epoch",
        }
        if observed is not None and observed != expected:
            raise ValueError(f"Test cohort mismatch: {path}")
        dest = output / relative.with_suffix("")
        manifest = export_mask(path, dest, "registered" if "registered" in relative.parts[0] else "atlas", provenance)
        entries.append({"path": str(relative), "sha256": sha256(path), "bundle": str(dest.relative_to(output)),
                        "architecture": manifest["architecture"]["constructor"], "round_trip": "exact CPU probe"})
        print(f"Verified {relative}", flush=True)
    write_json(output / "inventory.json", entries)


VOLUME_FEATURES = ([f"last_log1p_ml_{c}" for c in ("necrosis", "enhancing", "edema")]
                   + [f"slope_per_year_{c}" for c in ("necrosis", "enhancing", "edema")]
                   + [f"slope_times_dt_{c}" for c in ("necrosis", "enhancing", "edema")]
                   + [f"last_minus_first_{c}" for c in ("necrosis", "enhancing", "edema")]
                   + ["dt_years", "log1p_dt_weeks", "history_span_years", "history_count", "previous_gap_years",
                      "age_center60_scale12", "age_missing", "male", "mgmt_methylated", "mgmt_known"])


def volume_features(transitions, clinical):
    from volume_trajectory_experiment import history_features, static_features
    return np.stack([history_features(t, static_features(t.patient_id, clinical)) for t in transitions])


def predict_volume(models, transitions, clinical):
    """Library round-trip helper; returns log1p(mL), not the product inference API."""
    from volume_trajectory_experiment import y_of
    x = volume_features(transitions, clinical)
    last = np.stack([y_of(t.last) for t in transitions])
    return np.maximum(last + np.stack([m.predict(x) for m in models], axis=1), 0.0)


def export_volume(models, output, provenance):
    import sklearn
    from sklearn.utils.validation import check_is_fitted
    from sklearn.ensemble import HistGradientBoostingRegressor
    if len(models) != 3:
        raise ValueError("Expected three compartment estimators")
    for m in models:
        if not isinstance(m, HistGradientBoostingRegressor):
            raise TypeError("Expected HistGradientBoostingRegressor")
        check_is_fitted(m)
        if m.n_features_in_ != len(VOLUME_FEATURES):
            raise ValueError("Incorrect feature count")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir()
        _write_volume(models, stage, provenance, sklearn.__version__)
        restored, _ = load_volume(stage, trusted=True)
        probe = np.random.default_rng(1729).normal(size=(8, len(VOLUME_FEATURES)))
        for before, after in zip(models, restored):
            np.testing.assert_array_equal(before.predict(probe), after.predict(probe))
        stage.rename(output)


def _write_volume(models, output, provenance, sklearn_version):
    with (output / "estimators.pkl").open("wb") as f:
        pickle.dump(models, f, protocol=5)
    seal(output, {
        "kind": "volume_gbm", "sklearn_version": sklearn_version, "provenance": provenance,
        "estimator_parameters": [m.get_params() for m in models],
        "preprocessing": {"implementation": "volume_trajectory_experiment.history_features/static_features/y_of",
                          "features": VOLUME_FEATURES, "compartments": ["necrosis", "enhancing", "edema"],
                          "input_volume_unit": "mm3", "target": "next minus last log1p(mL)",
                          "output": "max(last log1p(mL) + predicted delta, 0); expm1 for mL",
                          "scaler": None, "history": "entire surgery-separated segment; slope uses last two scans, gap clamped >=1 week",
                          "missing_clinical": "age=0 with missing flag=1; unknown sex/MGMT indicators=0"},
    })


def load_volume(path, *, trusted=False):
    import sklearn
    manifest = verify(path)
    if manifest["kind"] != "volume_gbm":
        raise ValueError("Not a volume bundle")
    if not trusted:
        raise ValueError("Pickle can execute code; explicitly trust the bundle before loading")
    if sklearn.__version__ != manifest["sklearn_version"]:
        raise ValueError("scikit-learn version mismatch")
    check_source(manifest, ["package_models.py", "volume_trajectory_experiment.py", "lumiere_cohort.py", "lumiere_weeks.py"])
    with (Path(path) / "estimators.pkl").open("rb") as f:
        return pickle.load(f), manifest


def fit_volume(args):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from lumiere_cohort import load_clinical
    from volume_trajectory_experiment import y_of
    # Explicit configuration is mandatory: integration must first fix the model contract.
    contract = json.loads(args.contract.read_text())
    required = {"contract_id", "max_depth", "learning_rate", "max_iter", "loss", "random_state", "early_stopping"}
    if set(contract) != required or not contract["contract_id"] or contract["early_stopping"] is not False:
        raise ValueError(f"Contract must contain {sorted(required)} and early_stopping=false for all-cohort fitting")
    if args.output.exists():
        raise FileExistsError(args.output)
    cohort, hashes = cohort_inputs(args.data_dir, args.clinical_dir)
    demographics = args.clinical_dir / "LUMIERE-Demographics_Pathology.csv"
    if not demographics.exists():
        raise FileNotFoundError(demographics)
    hashes[str(demographics)] = sha256(demographics)
    clinical = load_clinical(demographics)
    x = volume_features(cohort.transitions, clinical)
    delta = np.stack([y_of(t.target) - y_of(t.last) for t in cohort.transitions])
    params = {k: v for k, v in contract.items() if k != "contract_id"}
    models = [HistGradientBoostingRegressor(**params).fit(x, delta[:, k]) for k in range(3)]
    export_volume(models, args.output, {
        "training_scope": "all_eligible_cohort_transitions", "all_cohort": True,
        "contract": contract, "contract_sha256": sha256(args.contract),
        "patients": cohort.patients, "transitions": transition_records(cohort.transitions),
        "cohort_input_sha256": hashes, "exclusions": cohort.excluded,
        "validation": "No held-out evaluation; these are refit weights, not CV metrics",
    })
    restored, _ = load_volume(args.output, trusted=True)
    np.testing.assert_array_equal(predict_volume(models, cohort.transitions, clinical),
                                  predict_volume(restored, cohort.transitions, clinical))
    print(f"Verified volume round trip: {args.output}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("inventory-masks", "fit-volume-all"):
        s = sub.add_parser(name)
        s.add_argument("--data-dir", required=True, type=Path)
        s.add_argument("--clinical-dir", required=True, type=Path)
        s.add_argument("--output", required=True, type=Path)
        if name == "inventory-masks":
            s.add_argument("--runs-dir", required=True, type=Path)
        else:
            s.add_argument("--contract", required=True, type=Path)
    s = sub.add_parser("snapshot-environment")
    s.add_argument("--output", required=True, type=Path)
    s = sub.add_parser("verify")
    s.add_argument("bundle", type=Path)
    a = p.parse_args()
    if a.command == "inventory-masks":
        inventory_masks(a.runs_dir, a.data_dir, a.clinical_dir, a.output)
    elif a.command == "fit-volume-all":
        fit_volume(a)
    elif a.command == "snapshot-environment":
        a.output.mkdir(parents=True, exist_ok=False)
        env = environment()
        write_json(a.output / "environment.json", env)
        (a.output / "requirements.freeze.txt").write_text(env["pip_freeze_all"])
    else:
        print(json.dumps(verify(a.bundle), indent=2))


if __name__ == "__main__":
    main()
