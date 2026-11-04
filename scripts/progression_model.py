#!/usr/bin/env python3
"""Released progression-call model: probability that the current scan is RANO progressive disease.

Inputs, all available when the scan is read:
  * automated DeepBraTumIA compartment volumes of the current scan and of the earlier scans
    of the same surgery-free segment;
  * the previous scan's expert rating;
  * optionally age, sex and MGMT status.
This is feature set V+R+S of progression_call_experiment.py. On held-out LUMIERE patients
it scored AUROC 0.82 [0.78, 0.86], against 0.68 for carrying the previous rating forward
(results/progression_call/metrics.md).

Command:
  fit-all  Fit on every expert-rated LUMIERE follow-up scan and write a checksummed bundle.
           The bundle includes operating thresholds taken from patient-level
           cross-validation, so its sensitivity and specificity are held-out estimates.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
REPO = SCRIPTS.parent

from lumiere_cohort import COMPARTMENT_SHORT, COMPARTMENTS  # noqa: E402
from volume_trajectory_experiment import static_features, y_of  # noqa: E402

KIND = "progression_gbm"
SCHEMA = 1
RATINGS = ("PD", "SD", "PR", "CR", "Post-Op", "none")
INPUT_RATINGS = ("PD", "SD", "PR", "CR", "Post-Op", "unknown")
FEATURES = ([f"current_log1p_ml_{c}" for c in COMPARTMENT_SHORT] + [f"change_vs_previous_{c}" for c in COMPARTMENT_SHORT]
            + [f"change_vs_nadir_{c}" for c in COMPARTMENT_SHORT]
            + ["total_log1p_ml", "total_change_vs_previous", "total_change_vs_nadir", "interval_years", "history_count",
               "years_since_segment_start"]
            + [f"previous_rating_{r}" for r in RATINGS]
            + ["age_center60_scale12", "age_missing", "male", "mgmt_methylated", "mgmt_known"])
# More enhancing / edema / total growth can never lower the PD probability (same held-out AUROC as unconstrained).
MONOTONE_UP = ("change_vs_previous_enhancing", "change_vs_previous_edema", "change_vs_nadir_enhancing", "change_vs_nadir_edema",
               "total_change_vs_previous", "total_change_vs_nadir")
GBM_PARAMS = dict(max_depth=2, learning_rate=0.05, max_iter=150, min_samples_leaf=15, random_state=0,
                  monotonic_cst=[int(f in MONOTONE_UP) for f in FEATURES])
SOURCES = ("progression_model.py", "volume_trajectory_experiment.py", "lumiere_cohort.py", "lumiere_weeks.py")


def norm_rating(rating: str | None) -> str:
    r = (rating or "").replace(" ", "")
    if r.startswith("Post-Op"):
        return "Post-Op"
    if r.startswith("PD"):
        return "PD"
    return r if r in ("SD", "PR", "CR") else "none"


def tlog(y) -> float:
    return float(np.log1p(np.expm1(np.asarray(y, dtype=np.float64)).sum(axis=-1)))


def features(history: list, current, static: np.ndarray) -> np.ndarray:
    """One row. history: earlier scans of the segment (oldest first; .week, .volumes_mm3, .rating)."""
    y_c, y_p = y_of(current), y_of(history[-1])
    past = np.stack([y_of(s) for s in history])
    previous = norm_rating(history[-1].rating)
    return np.concatenate([
        y_c, y_c - y_p, y_c - past.min(axis=0),
        [tlog(y_c), tlog(y_c) - tlog(y_p), tlog(y_c) - min(tlog(r) for r in past), (current.week - history[-1].week) / 52.0,
         len(history), (current.week - history[0].week) / 52.0],
        [float(previous == r) for r in RATINGS], static])


def cohort_rows(data_dir: Path, clinical_dir: Path):
    from lumiere_cohort import build_cohort, load_clinical

    cohort = build_cohort(data_dir, clinical_dir / "LUMIERE-ExpertRating-v202211.csv")
    clinical = load_clinical(clinical_dir / "LUMIERE-Demographics_Pathology.csv")
    rows = [t for t in cohort.transitions if norm_rating(t.target.rating) in ("PD", "SD", "PR", "CR")]
    x = np.stack([features(t.history, t.target, static_features(t.patient_id, clinical)) for t in rows])
    y = np.array([norm_rating(t.target.rating) == "PD" for t in rows], dtype=int)
    return cohort, rows, x, y


def new_model():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(**GBM_PARAMS)


def held_out_scores(cohort, rows, x, y, seeds=(0, 1, 2), folds=5) -> np.ndarray:
    from lumiere_cohort import patient_folds

    pid = np.array([t.patient_id for t in rows])
    total = np.zeros(len(rows))
    for seed in seeds:
        for test in patient_folds(cohort.patients, folds, seed):
            held = np.isin(pid, test)
            total[held] += new_model().fit(x[~held], y[~held]).predict_proba(x[held])[:, 1]
    return total / len(seeds)


def tier(p: float, thr: dict) -> str:
    if p >= thr["specificity_80"]["threshold"]:
        return "likely PD"
    return "uncertain" if p >= thr["sensitivity_90"]["threshold"] else "PD less likely"


def tier_rates(scores: np.ndarray, y: np.ndarray, thr: dict) -> dict:
    """Held-out share of scans in each output band and how often they were rated PD."""
    tiers = np.array([tier(float(v), thr) for v in scores])
    return {name: {"fraction_of_scans": float((tiers == name).mean()), "scans": int((tiers == name).sum()),
                   "pd_rate": float(y[tiers == name].mean()) if (tiers == name).any() else None}
            for name in ("likely PD", "uncertain", "PD less likely")}


def thresholds(scores: np.ndarray, y: np.ndarray) -> dict:
    out = {}
    for name, thr in (("specificity_80", float(np.quantile(scores[y == 0], 0.8))),
                      ("sensitivity_90", float(np.quantile(scores[y == 1], 0.1)))):
        out[name] = {"threshold": thr, "held_out_sensitivity": float((scores[y == 1] >= thr).mean()),
                     "held_out_specificity": float((scores[y == 0] < thr).mean())}
    return out


def fit_all(args) -> None:
    import sklearn
    from approach_screen import fast_auroc
    from package_models import sha256

    if args.output.exists():
        raise FileExistsError(args.output)
    cohort, rows, x, y = cohort_rows(args.data_dir, args.clinical_dir)
    scores = held_out_scores(cohort, rows, x, y)
    model = new_model().fit(x, y)
    manifest = {
        "kind": KIND, "schema_version": SCHEMA, "sklearn_version": sklearn.__version__, "features": FEATURES,
        "estimator_parameters": model.get_params(),
        "training": {"scope": "all expert-rated LUMIERE follow-up scans (PD/SD/PR/CR)", "scans": len(rows),
                     "patients": len({t.patient_id for t in rows}), "pd_rate": float(y.mean())},
        "held_out": {"protocol": "patient-level 5-fold CV x 3 fold seeds, probabilities averaged over seeds",
                     "auroc": fast_auroc(scores, y), "thresholds": thresholds(scores, y)},
        "label": "expert RANO rating of the current scan is PD",
        "inputs": "current scan + earlier scans of the same surgery-free segment (DeepBraTumIA volumes, mm3), "
                  "previous scan's expert rating, optional age/sex/MGMT",
        "excluded_inputs": "LessThan3Months flag (selectively annotated; leaks the label)",
        "source_sha256": {name: sha256(SCRIPTS / name) for name in SOURCES},
        "clinical_validation": False,
    }
    manifest["held_out"]["tiers"] = tier_rates(scores, y, manifest["held_out"]["thresholds"])
    write_bundle(model, manifest, args.output, probe=x)
    t = manifest["held_out"]["thresholds"]["specificity_80"]
    print(f"Wrote {args.output}: held-out AUROC {manifest['held_out']['auroc']:.3f}; at the 80%-specificity threshold "
          f"{t['threshold']:.3f}, held-out sensitivity {t['held_out_sensitivity']:.2f}")


def write_bundle(model, manifest: dict, output: Path, probe: np.ndarray) -> None:
    """Stage, checksum, reload and compare, then move into place; refuses an existing output."""
    from package_models import sha256, write_json

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir()
        with (stage / "estimator.pkl").open("wb") as h:
            pickle.dump(model, h, protocol=5)
        write_json(stage / "manifest.json", manifest)
        write_json(stage / "SHA256SUMS.json", {n: sha256(stage / n) for n in ("estimator.pkl", "manifest.json")})
        restored, _ = load(stage, trusted=True)
        np.testing.assert_array_equal(model.predict_proba(probe), restored.predict_proba(probe))
        stage.rename(output)


def load(bundle: Path, *, trusted: bool = False):
    import sklearn
    from package_models import sha256

    bundle = Path(bundle)
    sums = json.loads((bundle / "SHA256SUMS.json").read_text())
    if {p.name for p in bundle.iterdir() if p.is_file()} != set(sums) | {"SHA256SUMS.json"}:
        raise ValueError("Bundle file inventory mismatch")
    for name, digest in sums.items():
        if sha256(bundle / name) != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("kind") != KIND or manifest.get("schema_version") != SCHEMA:
        raise ValueError("Not a progression-call bundle")
    if not trusted:
        raise ValueError("Pickle can execute code; explicitly trust the bundle before loading")
    if manifest["sklearn_version"] != sklearn.__version__:
        raise ValueError("scikit-learn version mismatch")
    for name, digest in manifest["source_sha256"].items():
        if sha256(SCRIPTS / name) != digest:
            raise ValueError(f"Implementation differs from bundle: {name}")
    if manifest["features"] != FEATURES:
        raise ValueError("Feature contract differs from this code")
    with (bundle / "estimator.pkl").open("rb") as h:
        return pickle.load(h), manifest


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def read_history(path: Path):
    doc = json.loads(Path(path).read_text())
    allowed = {"version", "patient_id", "history", "clinical"}
    if not isinstance(doc, dict) or set(doc) - allowed or not {"version", "patient_id", "history"} <= set(doc):
        raise ValueError(f"Expected fields {sorted(allowed)}")
    if doc["version"] != 1 or not isinstance(doc["patient_id"], str) or not doc["patient_id"]:
        raise ValueError("Invalid version/patient_id")
    visits = doc["history"]
    if not isinstance(visits, list) or len(visits) < 2:
        raise ValueError("Need the current scan plus at least one earlier scan of the same segment")
    scans = []
    for i, v in enumerate(visits):
        if not isinstance(v, dict) or set(v) - {"week", "volumes_mm3", "rating"} or not {"week", "volumes_mm3"} <= set(v):
            raise ValueError("Each visit needs week and volumes_mm3, optionally rating")
        week = _number(v["week"], "week")
        if not isinstance(v["volumes_mm3"], dict) or set(v["volumes_mm3"]) != set(COMPARTMENTS):
            raise ValueError(f"volumes_mm3 must have exactly {list(COMPARTMENTS)}")
        vols = {k: _number(x, k) for k, x in v["volumes_mm3"].items()}
        if min(vols.values()) < 0 or (scans and week <= scans[-1].week):
            raise ValueError("Nonnegative volumes and strictly increasing weeks required")
        rating = v.get("rating")
        if rating is not None and rating not in INPUT_RATINGS:
            raise ValueError(f"rating must be one of {list(INPUT_RATINGS)}")
        if i > 0 and rating == "Post-Op":
            raise ValueError("A Post-Op scan starts a new segment; supply only scans since the last surgery")
        scans.append(SimpleNamespace(week=week, volumes_mm3=vols, rating=None if rating == "unknown" else rating))
    if visits[-2].get("rating") is None:
        raise ValueError("The previous scan's expert rating is required (use \"unknown\" if there is none)")
    if visits[-1].get("rating") is not None:
        raise ValueError("The current scan must not carry a rating; that is what the model estimates")
    clinical = doc.get("clinical", {})
    if not isinstance(clinical, dict) or set(clinical) - {"Age at surgery (years)", "Sex", "MGMT qualitative"} or \
            any(not isinstance(v, str) for v in clinical.values()):
        raise ValueError("clinical may contain string values for Age at surgery (years), Sex, MGMT qualitative")
    return doc["patient_id"], scans, clinical


def predict_progression(history_path: Path, bundle: Path, output: Path, trusted: bool = False) -> dict:
    from package_models import sha256

    patient, scans, clinical = read_history(history_path)
    model, manifest = load(bundle, trusted=trusted)
    x = features(scans[:-1], scans[-1], static_features(patient, {patient: clinical}))[None]
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite features")
    p = float(model.predict_proba(x)[0, 1])
    thr = manifest["held_out"]["thresholds"]
    cur, prev = scans[-1].volumes_mm3, scans[-2].volumes_mm3
    nadir_enh = min(s.volumes_mm3["Enhancing_Core"] for s in scans[:-1])

    def pct(a, b):
        return None if b <= 0 else round(100.0 * (a - b) / b, 1)

    result = {
        "patient_id": patient, "assessed_week": scans[-1].week, "probability_pd": p,
        "call": tier(p, thr),
        "held_out_pd_rate_in_this_band": (manifest["held_out"].get("tiers") or {}).get(tier(p, thr), {}).get("pd_rate"),
        "thresholds": thr, "held_out_auroc": manifest["held_out"]["auroc"],
        "measurements_ml": {
            "enhancing_current": cur["Enhancing_Core"] / 1000, "enhancing_previous": prev["Enhancing_Core"] / 1000,
            "enhancing_nadir": nadir_enh / 1000, "total_current": sum(cur.values()) / 1000, "total_previous": sum(prev.values()) / 1000,
            "enhancing_change_vs_previous_pct": pct(cur["Enhancing_Core"], prev["Enhancing_Core"]),
            "enhancing_change_vs_nadir_pct": pct(cur["Enhancing_Core"], nadir_enh),
            "total_change_vs_previous_pct": pct(sum(cur.values()), sum(prev.values())),
        },
        "previous_rating": scans[-2].rating or "unknown",
        "interpretation": "Probability that an expert would rate this scan PD under RANO, estimated from automated volumes, "
                          "the previous rating and clinical factors. 'likely PD' is above the threshold with 80% held-out "
                          "specificity; 'PD less likely' is below the one with 90% held-out sensitivity; 'uncertain' is "
                          "in between. held_out_pd_rate_in_this_band says how often such scans were rated PD. "
                          "Research use only.",
        "history_sha256": sha256(history_path), "bundle": str(Path(bundle).resolve()), "clinical_validation": False,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "progression.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit-all")
    f.add_argument("--data-dir", type=Path, default=REPO / "data/lumiere/extracted/Imaging")
    f.add_argument("--clinical-dir", type=Path, default=REPO / "data/lumiere/clinical")
    f.add_argument("--output", type=Path, required=True, help="New bundle directory (must not exist)")
    args = ap.parse_args()
    fit_all(args)


if __name__ == "__main__":
    main()
