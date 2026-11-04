#!/usr/bin/env python3
"""Calibrated ranges for the next-visit tumor volume forecast (step 1 gradient boosting).

The packaged volume model (``package_models.py fit-volume-all``) gives one number per
compartment. This script turns it into ranges (50/80/90/95%) that, across LUMIERE-like
patients, contain the next-visit volume at the stated rate.

Method: normalized split-conformal prediction on patient-grouped out-of-fold residuals.
  1. Out-of-fold forecasts: patient-level 5-fold CV over three fold seeds, using exactly
     the released GBM contract. Residual r = true - forecast in log1p(mL), per compartment
     and for the total.
  2. Scale model: a log-link Poisson GLM predicts |r| from the forecast horizon, the last
     observed volume, a last-volume-is-zero flag and a single-scan-history flag.
  3. Conformal step: equal-tailed order statistics of r / scale, with the (n + 1)
     finite-sample correction, give the lower and upper multipliers for each level.
  Range = forecast + scale * [lower, upper] in log1p(mL), clipped at 0 mL.

Commands:
  evaluate   Nested patient-level CV. The whole procedure is refit inside each training
             split and scored on held-out patients (coverage, width, interval score). It
             is compared with ranges centred on the last observed volume.
  calibrate  Write the calibration file for a packaged all-cohort volume bundle.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
REPO = SCRIPTS.parent

from lumiere_cohort import COMPARTMENT_SHORT, COMPARTMENTS, patient_folds  # noqa: E402
from package_models import VOLUME_FEATURES  # noqa: E402

KIND = "volume_forecast_intervals"
VERSION = 1
QUANTITIES = (*COMPARTMENT_SHORT, "total")
OUTPUT_NAMES = dict(zip(QUANTITIES, (*COMPARTMENTS, "Total")))
LEVELS = (0.5, 0.8, 0.9, 0.95)
METHODS = ("persistence_constant", "persistence_scaled", "gbm_constant", "gbm_scaled")
DEPLOYED = "gbm_scaled"
FOLDS = 5
OUTER_SEEDS = (0, 1, 2)  # same outer fold assignments as the step-1 experiment
CALIBRATION_SEEDS = (0, 1, 2)
SIGMA_FLOOR = 0.02  # log1p(mL); keeps r / scale finite
HORIZON_BINS = ((0, 8), (9, 16), (17, 26), (27, 10_000))
WELL_COVERED_MAX_WEEKS = HORIZON_BINS[-1][0] - 1  # held-out coverage was checked bin by bin up to here
SCALE_FEATURES = {q: ("log1p_dt_weeks", f"last_log1p_ml_{q}", f"last_{q}_is_zero", "single_scan_history")
                  for q in COMPARTMENT_SHORT}
SCALE_FEATURES["total"] = ("log1p_dt_weeks", "last_log1p_ml_total", "single_scan_history")
_COL = {name: i for i, name in enumerate(VOLUME_FEATURES)}
assert VOLUME_FEATURES[:3] == [f"last_log1p_ml_{c}" for c in COMPARTMENT_SHORT]


def level_key(level: float) -> str:
    return f"{round(level * 100)}%"


# ----------------------------------------------------------------------------- core math


def total_log(y: np.ndarray) -> np.ndarray:
    """log1p of the summed compartment volumes, from per-compartment log1p(mL)."""
    return np.log1p(np.expm1(np.asarray(y, dtype=np.float64)).sum(axis=-1))


def centres(last3: np.ndarray, delta3: np.ndarray) -> np.ndarray:
    """Forecast in log1p(mL): three compartments plus their total (as the bundle computes it)."""
    comp = np.maximum(last3 + delta3, 0.0)
    return np.column_stack([comp, total_log(comp)])


def scale_design(x: np.ndarray, quantity: str) -> np.ndarray:
    """Scale-model inputs, computed from the 22-column VOLUME_FEATURES matrix."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    log_dt = x[:, _COL["log1p_dt_weeks"]]
    single = (x[:, _COL["history_count"]] == 1).astype(np.float64)
    if quantity == "total":
        return np.column_stack([log_dt, total_log(x[:, :3]), single])
    last = x[:, _COL[f"last_log1p_ml_{quantity}"]]
    return np.column_stack([log_dt, last, (last == 0).astype(np.float64), single])


def fit_scale(design: np.ndarray, abs_resid: np.ndarray) -> dict:
    from sklearn.linear_model import PoissonRegressor

    mean = design.mean(axis=0)
    scale = design.std(axis=0)
    scale[scale == 0] = 1.0
    glm = PoissonRegressor(alpha=1e-4, max_iter=1000).fit((design - mean) / scale, abs_resid)
    return {"mean": mean.tolist(), "scale": scale.tolist(), "coef": glm.coef_.tolist(), "intercept": float(glm.intercept_)}


def scale_of(params: dict | None, design: np.ndarray) -> np.ndarray:
    if params is None:
        return np.ones(len(design))
    z = (design - np.asarray(params["mean"])) / np.asarray(params["scale"])
    return np.maximum(np.exp(params["intercept"] + z @ np.asarray(params["coef"])), SIGMA_FLOOR)


def conformal_multipliers(scores: np.ndarray, level: float) -> tuple[float, float]:
    """Equal-tailed split-conformal bounds on the scores, with the (n + 1) correction."""
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n, tail = len(s), (1.0 - level) / 2.0
    k_lo = math.floor(round((n + 1) * tail, 9))
    k_hi = math.ceil(round((n + 1) * (1.0 - tail), 9))
    if k_lo < 1 or k_hi > n:
        raise ValueError(f"{n} calibration scores are too few for a {level_key(level)} range")
    return float(s[k_lo - 1]), float(s[k_hi - 1])


def apply_ranges(entry: dict, centre: np.ndarray, design: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Ranges in log1p(mL) for one quantity: level -> (lower, upper)."""
    sigma = scale_of(entry["scale"], design)
    out = {}
    for key, (lo, hi) in entry["multipliers"].items():
        lower = np.maximum(centre + sigma * lo, 0.0)
        out[key] = (lower, np.maximum(centre + sigma * hi, lower))
    return out


def interval_score(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, level: float) -> np.ndarray:
    """Gneiting & Raftery (2007) interval score: width plus 2/alpha times any miss distance."""
    alpha = 1.0 - level
    return (hi - lo) + (2.0 / alpha) * (np.maximum(lo - y, 0.0) + np.maximum(y - hi, 0.0))


# ----------------------------------------------------------------------------- data and fitting


def load_data(data_dir: Path, clinical_dir: Path):
    from lumiere_cohort import build_cohort, load_clinical
    from package_models import volume_features
    from volume_trajectory_experiment import y_of

    cohort = build_cohort(data_dir, clinical_dir / "LUMIERE-ExpertRating-v202211.csv")
    clinical = load_clinical(clinical_dir / "LUMIERE-Demographics_Pathology.csv")
    ts = cohort.transitions
    last3 = np.stack([y_of(t.last) for t in ts])
    true3 = np.stack([y_of(t.target) for t in ts])
    data = {
        "x": volume_features(ts, clinical),
        "last": np.column_stack([last3, total_log(last3)]),
        "true": np.column_stack([true3, total_log(true3)]),
        "delta": true3 - last3,
        "patient": np.array([t.patient_id for t in ts]),
        "dt": np.array([t.dt_weeks for t in ts], dtype=np.float64),
        "n_history": np.array([len(t.history) for t in ts]),
        "rating": np.array([(t.target.rating or "").replace(" ", "") for t in ts]),
        "last_source": np.array([t.last.source_name for t in ts]),
        "target_source": np.array([t.target.source_name for t in ts]),
    }
    return cohort, data


def read_contract(path: Path) -> tuple[dict, dict]:
    contract = json.loads(Path(path).read_text())
    params = {k: v for k, v in contract.items() if k != "contract_id"}
    if params.get("early_stopping") is not False:
        raise ValueError("The interval calibration assumes the deterministic early_stopping=false contract")
    return contract, params


def gbm_fit_predict(params: dict, x_tr, d_tr, x_te) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor

    return np.stack([HistGradientBoostingRegressor(**params).fit(x_tr, d_tr[:, k]).predict(x_te) for k in range(3)], axis=1)


def calibrate(data: dict, idx: np.ndarray, params: dict, seeds=CALIBRATION_SEEDS, folds: int = FOLDS) -> dict:
    """Fit every method's scale model and conformal multipliers on the transitions in idx only."""
    patients = sorted(set(data["patient"][idx]))
    oof_idx, oof = [], []
    for seed in seeds:
        for test_patients in patient_folds(patients, folds, seed):
            held = np.isin(data["patient"][idx], test_patients)
            te, tr = idx[held], idx[~held]
            oof.append(centres(data["last"][te, :3], gbm_fit_predict(params, data["x"][tr], data["delta"][tr], data["x"][te])))
            oof_idx.append(te)
    oof_idx, oof = np.concatenate(oof_idx), np.concatenate(oof)
    out = {}
    for method in METHODS:
        # Persistence has no fitted part, so its residuals are the same for every fold seed.
        rows, centre = (oof_idx, oof) if method.startswith("gbm") else (idx, data["last"][idx])
        resid = data["true"][rows] - centre
        out[method] = {}
        for qi, q in enumerate(QUANTITIES):
            design = scale_design(data["x"][rows], q)
            scale = fit_scale(design, np.abs(resid[:, qi])) if method.endswith("scaled") else None
            scores = resid[:, qi] / scale_of(scale, design)
            out[method][q] = {
                "scale": scale,
                "multipliers": {level_key(lv): conformal_multipliers(scores, lv) for lv in LEVELS},
                "n_scores": int(len(scores)),
            }
    return out


_STATE: dict = {}


def _init_worker(data: dict, params: dict) -> None:
    import sklearn.ensemble  # noqa: F401  (load OpenMP before limiting it)
    from threadpoolctl import threadpool_limits

    _STATE.update(data=data, params=params, limits=threadpool_limits(1))


def outer_job(job: tuple[int, int]) -> dict:
    seed, fold = job
    data, params = _STATE["data"], _STATE["params"]
    test_patients = patient_folds(sorted(set(data["patient"])), FOLDS, seed)[fold]
    held = np.isin(data["patient"], test_patients)
    tr, te = np.flatnonzero(~held), np.flatnonzero(held)
    cal = calibrate(data, tr, params)
    gbm = centres(data["last"][te, :3], gbm_fit_predict(params, data["x"][tr], data["delta"][tr], data["x"][te]))
    bounds = {}
    for method in METHODS:
        centre = gbm if method.startswith("gbm") else data["last"][te]
        for qi, q in enumerate(QUANTITIES):
            bounds[(method, q)] = apply_ranges(cal[method][q], centre[:, qi], scale_design(data["x"][te], q))
    return {"seed": seed, "fold": fold, "idx": te, "gbm": gbm, "bounds": bounds, "calibration": cal}


# ----------------------------------------------------------------------------- evaluation


def patient_mean(values: np.ndarray, patient: np.ndarray, patients: list[str]) -> np.ndarray:
    return np.array([values[patient == p].mean() for p in patients])


def boot_mean_ci(per_patient: np.ndarray, rng, n_boot: int) -> list[float]:
    draws = rng.integers(0, len(per_patient), size=(n_boot, len(per_patient)))
    means = per_patient[draws].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def rating_group(rating: str) -> str:
    if rating.startswith("PD"):
        return "PD"
    if rating == "SD":
        return "SD"
    if rating in ("PR", "CR"):
        return "PR/CR"
    return "unrated"


def summarize(rows: list[dict], data_dt: np.ndarray, n_boot: int) -> dict:
    rng = np.random.default_rng(20261103)
    pid = np.array([r["patient_id"] for r in rows])
    patients = sorted(set(pid))
    col = lambda name: np.array([r[name] for r in rows], dtype=np.float64)  # noqa: E731
    out: dict = {"coverage": {}, "paired_interval_score": {}, "conditional_coverage_80": {}, "direction_calls": {}, "typical_range": {}}
    for method in METHODS:
        out["coverage"][method] = {}
        for q in QUANTITIES:
            y = col(f"true_{q}")
            out["coverage"][method][q] = {}
            for lv in LEVELS:
                key = level_key(lv)
                lo, hi = col(f"{method}_{q}_{key}_lo"), col(f"{method}_{q}_{key}_hi")
                covered = ((lo <= y) & (y <= hi)).astype(np.float64)
                per_patient = patient_mean(covered, pid, patients)
                score = interval_score(lo, hi, y, lv)
                out["coverage"][method][q][key] = {
                    "coverage": float(covered.mean()),
                    "patient_coverage": float(per_patient.mean()),
                    "patient_coverage_ci95": boot_mean_ci(per_patient, rng, n_boot),
                    "below": float((y < lo).mean()),
                    "above": float((y > hi).mean()),
                    "width_mean": float((hi - lo).mean()),
                    "width_median": float(np.median(hi - lo)),
                    "interval_score": float(score.mean()),
                    "patient_interval_score": float(patient_mean(score, pid, patients).mean()),
                }
    # Is the deployed range sharper than the alternatives at the same nominal level?
    for q in QUANTITIES:
        y = col(f"true_{q}")
        out["paired_interval_score"][q] = {}
        for lv in LEVELS:
            key = level_key(lv)
            ref = patient_mean(interval_score(col(f"{DEPLOYED}_{q}_{key}_lo"), col(f"{DEPLOYED}_{q}_{key}_hi"), y, lv), pid, patients)
            out["paired_interval_score"][q][key] = {}
            for other in METHODS:
                if other == DEPLOYED:
                    continue
                cmp = patient_mean(interval_score(col(f"{other}_{q}_{key}_lo"), col(f"{other}_{q}_{key}_hi"), y, lv), pid, patients)
                diff = ref - cmp
                out["paired_interval_score"][q][key][other] = {
                    "diff": float(diff.mean()),
                    "diff_ci95": boot_mean_ci(diff, rng, n_boot),
                    "relative": float(diff.mean() / cmp.mean()),
                    "patients_better": int((diff < 0).sum()),
                }
    # Coverage of the 80% ranges inside clinically meaningful subgroups (total volume).
    key, y = level_key(0.8), col("true_total")
    dt, nh, last = col("dt_weeks"), col("n_history"), col("last_total")
    terciles = np.percentile(last[col("repeat") == 0], [100 / 3, 200 / 3])
    groups = {
        **{f"horizon {a}-{b} wk" if b < 10_000 else f"horizon >{a - 1} wk": (dt >= a) & (dt <= b) for a, b in HORIZON_BINS},
        "single-scan history": nh == 1,
        "two or more history scans": nh >= 2,
        "last total volume, lowest third": last <= terciles[0],
        "last total volume, middle third": (last > terciles[0]) & (last <= terciles[1]),
        "last total volume, highest third": last > terciles[1],
        **{f"target rating {g}": np.array([rating_group(r["target_rating"]) == g for r in rows]) for g in ("PD", "SD", "PR/CR", "unrated")},
    }
    for name, mask in groups.items():
        entry = {"n_transitions": int(mask[col("repeat") == 0].sum())}
        for method in METHODS:
            lo, hi = col(f"{method}_total_{key}_lo"), col(f"{method}_total_{key}_hi")
            entry[method] = float(((lo <= y) & (y <= hi))[mask].mean()) if mask.any() else None
            entry[f"{method}_width"] = float((hi - lo)[mask].mean()) if mask.any() else None
        out["conditional_coverage_80"][name] = entry
    # When the range excludes "no change", it is making a growth or shrinkage call.
    for method in ("gbm_constant", "gbm_scaled"):
        out["direction_calls"][method] = {}
        for q in ("total", "enhancing"):
            yq, lq = col(f"true_{q}"), col(f"last_{q}")
            out["direction_calls"][method][q] = {}
            for lv in (0.5, 0.8):
                k = level_key(lv)
                lo, hi = col(f"{method}_{q}_{k}_lo"), col(f"{method}_{q}_{k}_hi")
                up, down = lo > lq, hi < lq
                out["direction_calls"][method][q][k] = {
                    "growth_calls_per_seed": float(up.sum() / len(OUTER_SEEDS)),
                    "shrink_calls_per_seed": float(down.sum() / len(OUTER_SEEDS)),
                    "growth_base_rate": float((yq > lq).mean()),
                    "shrink_base_rate": float((yq < lq).mean()),
                    "growth_calls": float(up.mean()),
                    "growth_call_precision": float((yq > lq)[up].mean()) if up.any() else None,
                    "shrink_calls": float(down.mean()),
                    "shrink_call_precision": float((yq < lq)[down].mean()) if down.any() else None,
                }
    # Typical range relative to the forecast (for log1p this is close to a fold change for tumors >> 1 mL).
    for q in QUANTITIES:
        c = col(f"gbm_{q}")
        out["typical_range"][q] = {}
        for lv in LEVELS:
            k = level_key(lv)
            lo, hi = col(f"{DEPLOYED}_{q}_{k}_lo"), col(f"{DEPLOYED}_{q}_{k}_hi")
            out["typical_range"][q][k] = {
                "median_lower_factor": float(np.median(np.exp(lo - c))),
                "median_upper_factor": float(np.median(np.exp(hi - c))),
                "median_width_ml": float(np.median(np.expm1(hi) - np.expm1(lo))),
            }
    out["horizon_weeks"] = {"min": float(data_dt.min()), "p95": float(np.percentile(data_dt, 95)),
                            "p99": float(np.percentile(data_dt, 99)), "max": float(data_dt.max())}
    return out


def consistency_check(rows: list[dict]) -> dict:
    """Outer GBM forecasts should reproduce the step-1 experiment's held-out GBM forecasts."""
    path = REPO / "results/volume_trajectory/predictions.csv"
    if not path.exists():
        return {"status": "step-1 predictions.csv not found"}
    with path.open() as h:
        ref = {(int(r["repeat"]), r["patient_id"], r["target_source"]): r for r in csv.DictReader(h)}
    diffs, missing = [], 0
    for r in rows:
        s = ref.get((r["repeat"], r["patient_id"], r["target_source"]))
        if s is None:
            missing += 1
            continue
        diffs += [abs(r[f"gbm_{c}"] - float(s[f"gbm_{c}"])) for c in COMPARTMENT_SHORT]
    return {"status": "compared", "rows": len(rows), "missing": missing, "max_abs_diff_log1p_ml": float(max(diffs)) if diffs else None}


def evaluate(args) -> None:
    contract, params = read_contract(args.contract)
    cohort, data = load_data(args.data_dir, args.clinical_dir)
    print(f"Cohort: {len(cohort.patients)} patients, {len(data['patient'])} transitions", flush=True)
    jobs = [(s, f) for s in OUTER_SEEDS for f in range(FOLDS)]
    workers = max(1, min(len(jobs), args.workers))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"  # one thread per worker process; inherited by the pool
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(data, params)) as pool:
        results = []
        for res in pool.map(outer_job, jobs):
            results.append(res)
            print(f"  outer seed {res['seed']} fold {res['fold']}: {len(res['idx'])} held-out transitions", flush=True)
    rows = []
    for res in results:
        for j, i in enumerate(res["idx"]):
            row = {"repeat": res["seed"], "fold": res["fold"], "patient_id": str(data["patient"][i]),
                   "last_source": str(data["last_source"][i]), "target_source": str(data["target_source"][i]),
                   "dt_weeks": float(data["dt"][i]), "n_history": int(data["n_history"][i]),
                   "target_rating": str(data["rating"][i])}
            for qi, q in enumerate(QUANTITIES):
                row[f"true_{q}"] = float(data["true"][i, qi])
                row[f"last_{q}"] = float(data["last"][i, qi])
                row[f"gbm_{q}"] = float(res["gbm"][j, qi])
            for (method, q), ranges in res["bounds"].items():
                for key, (lo, hi) in ranges.items():
                    row[f"{method}_{q}_{key}_lo"] = float(lo[j])
                    row[f"{method}_{q}_{key}_hi"] = float(hi[j])
            rows.append(row)
    summary = summarize(rows, data["dt"], args.bootstrap)
    multipliers = {
        q: {level_key(lv): [res["calibration"][DEPLOYED][q]["multipliers"][level_key(lv)] for res in results] for lv in (0.8, 0.9)}
        for q in QUANTITIES
    }
    meta = {
        "contract": contract,
        "contract_sha256": hashlib.sha256(Path(args.contract).read_bytes()).hexdigest(),
        "cohort": {"patients": len(cohort.patients), "transitions": int(len(data["patient"])), "exclusions": cohort.excluded},
        "protocol": {
            "outer": f"patient-level {FOLDS}-fold CV, fold seeds {list(OUTER_SEEDS)} (same assignments as step 1)",
            "inner": f"calibration refit on training patients only: {FOLDS}-fold out-of-fold GBM residuals, fold seeds {list(CALIBRATION_SEEDS)}",
            "methods": {
                "persistence_constant": "last observed volume +/- conformal residual quantiles",
                "persistence_scaled": "last observed volume, range scaled by the Poisson-GLM scale model",
                "gbm_constant": "GBM forecast +/- conformal residual quantiles",
                "gbm_scaled": "GBM forecast, range scaled by the Poisson-GLM scale model (deployed)",
            },
            "scale_features": SCALE_FEATURES,
            "levels": [level_key(lv) for lv in LEVELS],
            "units": "log1p(mL); total = log1p(sum of compartment mL)",
            "bootstrap": f"{args.bootstrap} patient resamples",
        },
        "consistency_with_step1": consistency_check(rows),
        "summary": summary,
        "deployed_multipliers_by_outer_fold": multipliers,
    }
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    (out / "metrics.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
    write_markdown(meta, out / "metrics.md")
    figure(meta, REPO / "results/figures/step1_volume_intervals.png")
    print((out / "metrics.md").read_text())


def write_markdown(meta: dict, path: Path) -> None:
    s = meta["summary"]
    cov, paired = s["coverage"], s["paired_interval_score"]
    label = {"persistence_constant": "last scan, fixed width", "persistence_scaled": "last scan, scaled width",
             "gbm_constant": "GBM forecast, fixed width", "gbm_scaled": "GBM forecast, scaled width (deployed)"}
    c = meta["consistency_with_step1"]
    lines = [
        "# Step 1b: calibrated ranges for the next-visit tumor volume forecast",
        "",
        f"- Cohort: {meta['cohort']['patients']} patients, {meta['cohort']['transitions']} transitions; exclusions {meta['cohort']['exclusions']}",
        f"- Outer: {meta['protocol']['outer']}. Every transition is scored once per fold seed on a model and calibration that never saw its patient.",
        f"- Inner: {meta['protocol']['inner']}.",
        f"- Point model: released contract `{meta['contract']['contract_id']}`. Check against step 1 held-out GBM forecasts: "
        f"max abs difference {c.get('max_abs_diff_log1p_ml')} log1p(mL) over {c.get('rows')} rows ({c.get('missing')} unmatched).",
        "- Units: log1p(mL). Total = log1p(sum of compartment mL). Interval score = width + (2/alpha) x miss distance (lower is better).",
        "- Patient coverage = mean over patients of each patient's coverage; 95% CI from patient bootstrap.",
        "",
        "## Held-out coverage, total tumor volume",
        "",
        "| Range | Nominal | Coverage | Patient coverage [95% CI] | Below / above | Mean width | Interval score |",
        "| :--- | ---: | ---: | :---: | :---: | ---: | ---: |",
    ]
    for m in METHODS:
        for lv in LEVELS:
            k = level_key(lv)
            v = cov[m]["total"][k]
            ci = v["patient_coverage_ci95"]
            lines.append(f"| {label[m]} | {k} | {v['coverage']:.3f} | {v['patient_coverage']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
                         f"{v['below']:.3f} / {v['above']:.3f} | {v['width_mean']:.3f} | {v['interval_score']:.3f} |")
    lines += ["", "## Deployed ranges, every quantity", "",
              "| Quantity | Nominal | Coverage | Patient coverage [95% CI] | Mean width | Median range vs forecast | Median width (mL) |",
              "| :--- | ---: | ---: | :---: | ---: | :---: | ---: |"]
    for q in QUANTITIES:
        for lv in LEVELS:
            k = level_key(lv)
            v, t = cov[DEPLOYED][q][k], s["typical_range"][q][k]
            ci = v["patient_coverage_ci95"]
            lines.append(f"| {q} | {k} | {v['coverage']:.3f} | {v['patient_coverage']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {v['width_mean']:.3f} | "
                         f"x{t['median_lower_factor']:.2f} to x{t['median_upper_factor']:.2f} | {t['median_width_ml']:.1f} |")
    lines += ["", "## Is the deployed range sharper than the alternatives?", "",
              "Patient-level mean interval score, deployed minus alternative (negative = deployed better), 95% CI over patients.", "",
              "| Quantity | Nominal | vs last scan, fixed | vs last scan, scaled | vs GBM, fixed |", "| :--- | ---: | :---: | :---: | :---: |"]
    for q in QUANTITIES:
        for lv in (0.8, 0.9):
            k = level_key(lv)
            cells = []
            for other in ("persistence_constant", "persistence_scaled", "gbm_constant"):
                v = paired[q][k][other]
                cells.append(f"{v['diff']:+.3f} [{v['diff_ci95'][0]:+.3f}, {v['diff_ci95'][1]:+.3f}] ({v['relative']:+.0%})")
            lines.append(f"| {q} | {k} | " + " | ".join(cells) + " |")
    lines += ["", "## Coverage of the 80% total-volume range by subgroup", "",
              "| Subgroup | Transitions | last scan, fixed | last scan, scaled | GBM, fixed | GBM, scaled (deployed) | deployed mean width |",
              "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, v in s["conditional_coverage_80"].items():
        lines.append(f"| {name} | {v['n_transitions']} | " + " | ".join(f"{v[m]:.3f}" for m in METHODS) + f" | {v[DEPLOYED + '_width']:.3f} |")
    lines += ["", "## Direction calls: ranges that exclude 'no change'", "",
              "Calls per fold seed are out of 399 transitions. 'Correct' = the volume then moved in the called direction; "
              "base rate = how often it moved that way among all transitions.", "",
              "| Range | Quantity | Nominal | Growth calls | per seed | correct (base rate) | Shrink calls | per seed | correct (base rate) |",
              "| :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: | :---: |"]
    fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
    for m, by_q in s["direction_calls"].items():
        for q, by_level in by_q.items():
            for k, v in by_level.items():
                lines.append(f"| {label[m]} | {q} | {k} | {v['growth_calls']:.1%} | {v['growth_calls_per_seed']:.0f} | "
                             f"{fmt(v['growth_call_precision'])} ({v['growth_base_rate']:.2f}) | {v['shrink_calls']:.1%} | "
                             f"{v['shrink_calls_per_seed']:.0f} | {fmt(v['shrink_call_precision'])} ({v['shrink_base_rate']:.2f}) |")
    h = s["horizon_weeks"]
    lines += ["", "## Calibration stability", "",
              "Deployed multipliers (lower, upper) of the scaled residual, total volume, across the 15 outer training splits:", ""]
    for k, values in meta["deployed_multipliers_by_outer_fold"]["total"].items():
        lo, hi = np.array(values).T
        lines.append(f"- {k}: lower {lo.min():.2f} to {lo.max():.2f}, upper {hi.min():.2f} to {hi.max():.2f}")
    lines += ["", f"Observed horizons: {h['min']:.0f}-{h['max']:.0f} weeks (95th percentile {h['p95']:.0f}, 99th {h['p99']:.0f}). "
              "Ranges for horizons beyond the 99th percentile rest on very few transitions.", ""]
    path.write_text("\n".join(lines) + "\n")


def figure(meta: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    series = (("persistence_scaled", "last scan, scaled width", "#8a8984", "s"),
              ("gbm_constant", "GBM forecast, fixed width", "#eb6834", "^"),
              ("gbm_scaled", "GBM forecast, scaled width (deployed)", "#2a78d6", "o"))
    s = meta["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), dpi=200)
    nominal = np.array(LEVELS)
    ax = axes[0]
    ax.plot([0.35, 1.0], [0.35, 1.0], color=muted, linewidth=1, linestyle=":")
    for m, name, color, marker in series:
        observed = [s["coverage"][m]["total"][level_key(lv)]["patient_coverage"] for lv in LEVELS]
        ax.plot(nominal, observed, color=color, linewidth=2, marker=marker, markersize=7, markeredgecolor="white",
                markeredgewidth=1.5, label=name)
    ax.set_xlim(0.35, 1.0)
    ax.set_ylim(0.35, 1.0)
    ax.set_xticks(nominal, [level_key(lv) for lv in LEVELS])
    ax.set_yticks(np.arange(0.4, 1.01, 0.1), [f"{round(v * 100)}%" for v in np.arange(0.4, 1.01, 0.1)])
    ax.set_xlabel("Nominal coverage", color=muted, fontsize=8)
    ax.set_ylabel("Held-out coverage (patient-weighted)", color=muted, fontsize=8)
    ax.set_title("A  Ranges hold their stated coverage", loc="left", fontsize=9, color=ink)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    groups = [g for g in s["conditional_coverage_80"] if g.startswith("horizon")]
    xs = np.arange(len(groups))
    for panel, (key, title, ylabel) in zip(axes[1:], (("", "B  80% range coverage by horizon", "Held-out coverage"),
                                                      ("_width", "C  80% range width by horizon", "Mean width, log1p(mL)"))):
        for j, (m, name, color, marker) in enumerate(series):
            vals = [s["conditional_coverage_80"][g][m + key] if key else s["conditional_coverage_80"][g][m] for g in groups]
            panel.plot(xs + (j - 1) * 0.08, vals, color=color, linewidth=2, marker=marker, markersize=7,
                       markeredgecolor="white", markeredgewidth=1.5, label=name)
        if not key:
            panel.axhline(0.8, color=muted, linewidth=1, linestyle=":")
        panel.set_xticks(xs, [g.replace("horizon ", "") for g in groups])
        panel.set_xlabel(f"Weeks to next scan (transitions: {', '.join(str(s['conditional_coverage_80'][g]['n_transitions']) for g in groups)})",
                         color=muted, fontsize=8)
        panel.set_ylabel(ylabel, color=muted, fontsize=8)
        panel.set_title(title, loc="left", fontsize=9, color=ink)
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
        ax.grid(color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Next-visit total tumor volume: calibrated ranges on held-out LUMIERE patients", x=0.01, ha="left",
                 fontsize=10, color=ink)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- deployment


def bundle_estimators_sha256(bundle: Path) -> str:
    return json.loads((Path(bundle) / "SHA256SUMS.json").read_text())["estimators.pkl"]


def calibrate_bundle(args) -> None:
    from package_models import transition_records, verify, write_json

    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = verify(args.bundle)  # checksums only; the pickle is not loaded
    provenance = manifest["provenance"]
    if manifest["kind"] != "volume_gbm" or provenance.get("all_cohort") is not True:
        raise ValueError("Expected a packaged all-cohort volume_gbm bundle")
    contract = provenance["contract"]
    params = {k: v for k, v in contract.items() if k != "contract_id"}
    if params.get("early_stopping") is not False:
        raise ValueError("The interval calibration assumes the deterministic early_stopping=false contract")
    cohort, data = load_data(args.data_dir, args.clinical_dir)
    if transition_records(cohort.transitions) != provenance["transitions"]:
        raise ValueError("Cohort transitions differ from those the bundle was fitted on")
    cal = calibrate(data, np.arange(len(data["patient"])), params)[DEPLOYED]
    held_out, long_horizon = None, None
    if args.evaluation.exists():
        ev = json.loads(args.evaluation.read_text())
        if ev["contract"] == contract:
            held_out = {OUTPUT_NAMES[q]: {k: round(v["patient_coverage"], 3) for k, v in ev["summary"]["coverage"][DEPLOYED][q].items()}
                        for q in QUANTITIES}
            last_bin = ev["summary"]["conditional_coverage_80"][f"horizon >{WELL_COVERED_MAX_WEEKS} wk"]
            long_horizon = {"total_80_coverage": round(last_bin[DEPLOYED], 3), "transitions": last_bin["n_transitions"]}
    doc = calibration_document(cal, bundle=args.bundle, contract=contract, dt=data["dt"], patients=len(cohort.patients),
                               held_out=held_out, long_horizon=long_horizon)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, doc)
    k = level_key(0.8)
    lo, hi = doc["quantities"]["Total"]["multipliers"][k]
    print(f"Wrote {args.output}; total-volume {k} multipliers {lo:+.3f} / {hi:+.3f}; held-out coverage {held_out and held_out['Total']}")


def calibration_document(cal: dict, *, bundle: Path, contract: dict, dt: np.ndarray, patients: int,
                         held_out: dict | None = None, long_horizon: dict | None = None) -> dict:
    """The calibration file: the deployed method's fitted entries plus what predict.py needs to check and explain."""
    return {
        "version": VERSION,
        "kind": KIND,
        "method": "normalized split-conformal on patient-grouped out-of-fold residuals (log1p mL)",
        "range_formula": "forecast_log1p + max(exp(intercept + coef . (features - mean) / scale), sigma_floor) * multiplier; "
                         "lower end clipped at 0; expm1 for mL",
        "levels": [level_key(lv) for lv in LEVELS],
        "quantities": {OUTPUT_NAMES[q]: {"scale_features": list(SCALE_FEATURES[q]), **cal[q]} for q in QUANTITIES},
        "sigma_floor": SIGMA_FLOOR,
        "calibration": {"folds": FOLDS, "fold_seeds": list(CALIBRATION_SEEDS), "patients": int(patients),
                        "transitions": int(len(dt))},
        "bundle_estimators_sha256": bundle_estimators_sha256(bundle),
        "contract": contract,
        "horizon_weeks": {"min": float(dt.min()), "well_covered_max": WELL_COVERED_MAX_WEEKS,
                          "fraction_longer": float((dt > WELL_COVERED_MAX_WEEKS).mean()), "max": float(dt.max())},
        "held_out_patient_coverage": held_out,
        "held_out_long_horizon": long_horizon,
        "scope": "Marginal coverage across LUMIERE-like patients and visits, not a per-patient guarantee. Assumes "
                 "DeepBraTumIA-style compartment volumes and a surgery-free interval. Calibrated on models fitted to 80% "
                 "of patients, so the all-cohort model's ranges are, if anything, slightly conservative.",
    }


def load_calibration(path: Path, bundle: Path) -> dict:
    """Read a calibration file and check it matches this code and the given (already verified) bundle."""
    doc = json.loads(Path(path).read_text())
    if not isinstance(doc, dict) or doc.get("kind") != KIND or doc.get("version") != VERSION:
        raise ValueError("Not a version-1 volume interval calibration file")
    try:
        if doc["bundle_estimators_sha256"] != bundle_estimators_sha256(bundle):
            raise ValueError("Interval calibration was built for a different volume bundle")
        if doc["sigma_floor"] != SIGMA_FLOOR:
            raise ValueError("Interval calibration uses a different scale floor than this code")
        for q in QUANTITIES:
            entry = doc["quantities"][OUTPUT_NAMES[q]]
            if entry["scale_features"] != list(SCALE_FEATURES[q]) or len(entry["scale"]["coef"]) != len(SCALE_FEATURES[q]):
                raise ValueError(f"Interval calibration scale features for {q} differ from this code")
            if set(entry["multipliers"]) != set(doc["levels"]) or not all(
                    math.isfinite(lo) and math.isfinite(hi) and lo <= hi for lo, hi in entry["multipliers"].values()):
                raise ValueError(f"Malformed interval multipliers for {q}")
        h = doc["horizon_weeks"]
        if not h["min"] <= h["well_covered_max"] <= h["max"] or not 0 <= h["fraction_longer"] <= 1:
            raise ValueError("Malformed calibrated horizon range")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Malformed interval calibration file: missing or invalid {exc}") from None
    return doc


def forecast_ranges(doc: dict, x: np.ndarray, centre: np.ndarray, last: np.ndarray, horizon: float) -> dict:
    """Ranges in mL for one forecast. centre/last: log1p(mL) of the three compartments."""
    centre4 = np.append(centre, total_log(centre))
    last4 = np.append(last, total_log(last))
    volumes = {}
    for qi, q in enumerate(QUANTITIES):
        entry = doc["quantities"][OUTPUT_NAMES[q]]
        ranges = apply_ranges(entry, centre4[qi:qi + 1], scale_design(x, q))
        volumes[OUTPUT_NAMES[q]] = {
            "last_observed": float(np.expm1(last4[qi])),
            "forecast": float(np.expm1(centre4[qi])),
            "ranges": {k: [float(np.expm1(lo[0])), float(np.expm1(hi[0]))] for k, (lo, hi) in ranges.items()},
        }
    h = doc["horizon_weeks"]
    inside = h["min"] <= horizon <= h["well_covered_max"]
    warning = None
    if horizon > h["max"]:
        warning = f"Horizon {horizon:g} weeks is longer than any LUMIERE interval ({h['max']:g} weeks); the ranges are an extrapolation."
    elif not inside:
        warning = (f"Only {h['fraction_longer']:.0%} of LUMIERE scan intervals were longer than {h['well_covered_max']:g} weeks"
                   if horizon > h["well_covered_max"] else f"Horizon is shorter than any LUMIERE interval ({h['min']:g} weeks)")
        long = doc.get("held_out_long_horizon")
        if long and horizon > h["well_covered_max"]:
            warning += (f"; for those, the 80% total-volume range held {long['total_80_coverage']:.0%} of next volumes in "
                        f"held-out testing ({long['transitions']} transitions)")
        warning += ". Treat the ranges as optimistic."
    return {
        "unit": "mL",
        "levels": doc["levels"],
        "volumes": volumes,
        "held_out_patient_coverage": doc.get("held_out_patient_coverage"),
        "horizon_within_calibration": bool(inside),
        "well_calibrated_horizon_weeks": [h["min"], h["well_covered_max"]],
        "warning": warning,
        "interpretation": doc["scope"],
    }


# ----------------------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "calibrate"):
        s = sub.add_parser(name)
        s.add_argument("--data-dir", type=Path, default=REPO / "data/lumiere/extracted/Imaging")
        s.add_argument("--clinical-dir", type=Path, default=REPO / "data/lumiere/clinical")
    e = sub.choices["evaluate"]
    e.add_argument("--contract", type=Path, default=REPO / "release/contracts/volume-gbm-v1.json")
    e.add_argument("--output-dir", type=Path, default=REPO / "results/volume_intervals")
    e.add_argument("--bootstrap", type=int, default=2000)
    e.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    c = sub.choices["calibrate"]
    c.add_argument("--bundle", type=Path, required=True, help="Packaged all-cohort volume bundle directory")
    c.add_argument("--output", type=Path, required=True, help="New calibration JSON (must not exist)")
    c.add_argument("--evaluation", type=Path, default=REPO / "results/volume_intervals/metrics.json",
                   help="Held-out evaluation to quote coverage from, if its contract matches")
    a = p.parse_args()
    if a.command == "evaluate":
        evaluate(a)
    else:
        calibrate_bundle(a)


if __name__ == "__main__":
    main()
