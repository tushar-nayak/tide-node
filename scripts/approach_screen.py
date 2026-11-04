#!/usr/bin/env python3
"""Screen alternative prediction tasks on LUMIERE for real signal.

Next-visit forecasting barely beats copying the last scan (EXPERIMENTS.md sections 7-11).
This screen asks which other questions the same data can answer, each scored against its
own trivial baseline under the same rules as every other experiment here:
  * patient-level 5-fold CV over 3 fold seeds;
  * out-of-fold predictions averaged over the seeds;
  * 95% CIs from a patient bootstrap.

  A. Long-horizon volume forecasting. Every (history, later scan) pair within a
     surgery-free segment, scored by horizon. Baselines: copy-last, population drift by
     horizon, population Gompertz, and per-patient slope extrapolation.
  B. Current-visit progression call (automated RANO). Is this scan PD, given the volumes of
     this and earlier scans? Baselines: previous rating carried forward, enhancing-volume
     change from the segment nadir, total-volume change from the last scan.
  C. Dynamic prognosis. Remaining survival from each visit, given the history so far.
     Baseline: age, sex, MGMT, weeks since the first scan and surgeries so far. Scored by
     C-index over pairs of visits from different patients.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
REPO = SCRIPTS.parent

from lumiere_cohort import COMPARTMENT_SHORT, Transition, build_cohort, load_clinical, patient_folds  # noqa: E402
from volume_trajectory_experiment import (  # noqa: E402
    auroc, fit_gompertz, history_features, predict_gompertz, predict_loglinear, static_features, y_of)

SEEDS = (0, 1, 2)
FOLDS = 5
N_BOOT = 2000
HORIZON_BINS = ((1, 13), (14, 26), (27, 52), (53, 10_000))
RATINGS = ("PD", "SD", "PR", "CR", "Post-Op", "none")
METHODS_A = ("persistence", "patient_slope", "drift", "gompertz", "gbm")


def norm_rating(rating: str | None) -> str:
    r = (rating or "").replace(" ", "")
    if r.startswith("Post-Op"):
        return "Post-Op"
    if r.startswith("PD"):
        return "PD"
    return r if r in ("SD", "PR", "CR") else "none"


def onehot(rating: str | None) -> list[float]:
    r = norm_rating(rating)
    return [float(r == k) for k in RATINGS]


def tlog(y: np.ndarray) -> np.ndarray:
    return np.log1p(np.expm1(np.asarray(y, dtype=np.float64)).sum(axis=-1))


def oof(row_patient: np.ndarray, patients: list[str], fit_predict) -> np.ndarray:
    """Out-of-fold predictions averaged over fold seeds; fit_predict(train_idx, test_idx) -> array."""
    total = None
    for seed in SEEDS:
        pred = None
        for test_patients in patient_folds(patients, FOLDS, seed):
            held = np.isin(row_patient, test_patients)
            out = fit_predict(np.flatnonzero(~held), np.flatnonzero(held))
            if pred is None:
                pred = np.zeros((len(row_patient),) + out.shape[1:])
            pred[held] = out
        total = pred if total is None else total + pred
    return total / len(SEEDS)


def boot_ci(values: np.ndarray, rng) -> list[float]:
    draws = values[rng.integers(0, len(values), size=(N_BOOT, len(values)))].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def hgb_reg(**kw):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.05, **kw)


# ----------------------------------------------------------------------------- A. long-horizon volume


def task_a(cohort, statics, rng) -> dict:
    pairs = []
    for pid in cohort.patients:
        scans = cohort.scans[pid]
        for seg in sorted({s.segment for s in scans}):
            ss = [s for s in scans if s.segment == seg]
            for j in range(1, len(ss)):
                for i in range(j):
                    pairs.append(Transition(pid, ss[: i + 1], ss[j]))
    pid = np.array([t.patient_id for t in pairs])
    x = np.stack([history_features(t, statics[t.patient_id]) for t in pairs])
    last = np.stack([y_of(t.last) for t in pairs])
    true = np.stack([y_of(t.target) for t in pairs])
    delta = true - last
    dt = np.array([t.dt_weeks for t in pairs], dtype=np.float64)
    ldt = np.log1p(dt)[:, None]

    def fit_predict(tr, te):
        g = fit_gompertz([pairs[i] for i in tr])
        out = {
            "persistence": last[te],
            "patient_slope": np.stack([predict_loglinear(pairs[i]) for i in te]),
            "gompertz": np.stack([predict_gompertz(g, pairs[i]) for i in te]),
            "drift": np.maximum(last[te] + np.stack([hgb_reg(max_depth=2, max_iter=100).fit(ldt[tr], delta[tr, k]).predict(ldt[te])
                                                     for k in range(3)], axis=1), 0.0),
            "gbm": np.maximum(last[te] + np.stack([hgb_reg(max_depth=3, max_iter=200).fit(x[tr], delta[tr, k]).predict(x[te])
                                                   for k in range(3)], axis=1), 0.0),
        }
        return np.stack([out[m] for m in METHODS_A], axis=1)

    pred = oof(pid, cohort.patients, fit_predict)  # (n, methods, 3)
    err = np.abs(pred - true[:, None, :]).mean(axis=2)  # mean over compartments
    err_total = np.abs(tlog(pred) - tlog(true)[:, None])
    bins = {"all horizons": np.ones(len(dt), bool),
            **{(f"{a}-{b} weeks" if b < 10_000 else f">{a - 1} weeks"): (dt >= a) & (dt <= b) for a, b in HORIZON_BINS}}
    res = {"pairs": len(pairs), "patients": len(set(pid)), "bins": {}}
    for name, mask in bins.items():
        pats = sorted(set(pid[mask]))
        entry = {"pairs": int(mask.sum()), "patients": len(pats), "median_horizon_weeks": float(np.median(dt[mask])),
                 "mae": {m: float(err[mask, k].mean()) for k, m in enumerate(METHODS_A)},
                 "mae_total": {m: float(err_total[mask, k].mean()) for k, m in enumerate(METHODS_A)}}
        base = err[mask, METHODS_A.index("persistence")].mean()
        entry["skill_vs_persistence"] = {m: float(1 - err[mask, k].mean() / base) for k, m in enumerate(METHODS_A)}
        g = METHODS_A.index("gbm")
        for ref in ("persistence", "drift", "gompertz"):
            r = METHODS_A.index(ref)
            per_patient = np.array([(err[mask & (pid == p), g] - err[mask & (pid == p), r]).mean() for p in pats])
            entry[f"gbm_minus_{ref}"] = {"diff": float(per_patient.mean()), "ci95": boot_ci(per_patient, rng),
                                         "relative": float(per_patient.mean() / np.mean([err[mask & (pid == p), r].mean() for p in pats])),
                                         "patients_better": int((per_patient < 0).sum())}
        res["bins"][name] = entry
    return res


# ----------------------------------------------------------------------------- B. current-visit progression call


def assess_features(t: Transition, static: np.ndarray) -> np.ndarray:
    y_t, y_l = y_of(t.target), y_of(t.last)
    hist = np.stack([y_of(s) for s in t.history])
    nadir = hist.min(axis=0)
    tot_hist = tlog(hist)
    return np.concatenate([
        y_t, y_t - y_l, y_t - nadir,
        [tlog(y_t), tlog(y_t) - tlog(y_l), tlog(y_t) - tot_hist.min(), t.dt_weeks / 52.0, len(t.history),
         (t.target.week - t.history[0].week) / 52.0],
        onehot(t.last.rating), static])


def fast_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties (same value as volume_trajectory_experiment.auroc)."""
    from scipy.stats import rankdata

    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auroc_boot(scores: dict[str, np.ndarray], label: np.ndarray, pid: np.ndarray, rng, ref: str) -> dict:
    pats = sorted(set(pid))
    rows = {p: np.flatnonzero(pid == p) for p in pats}
    point = {k: fast_auroc(v, label) for k, v in scores.items()}
    draws = {k: [] for k in scores}
    for _ in range(N_BOOT):
        idx = np.concatenate([rows[pats[i]] for i in rng.integers(0, len(pats), len(pats))])
        for k, v in scores.items():
            draws[k].append(fast_auroc(v[idx], label[idx]))
    out = {}
    for k in scores:
        d = np.array(draws[k])
        diff = d - np.array(draws[ref])
        out[k] = {"auroc": point[k], "ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
                  f"minus_{ref}": float(point[k] - point[ref]),
                  f"minus_{ref}_ci95": [float(np.nanpercentile(diff, 2.5)), float(np.nanpercentile(diff, 97.5))]}
    return out


def task_b(cohort, statics, rng) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = [t for t in cohort.transitions if norm_rating(t.target.rating) in ("PD", "SD", "PR", "CR")]
    pid = np.array([t.patient_id for t in rows])
    label = np.array([norm_rating(t.target.rating) == "PD" for t in rows], dtype=int)
    x = np.stack([assess_features(t, statics[t.patient_id]) for t in rows])
    prev = np.array([onehot(t.last.rating) for t in rows])
    vol_cols = list(range(0, 15))  # this scan's volumes, changes vs last and nadir, totals, timing

    def fit(model_fn, cols):
        def fp(tr, te):
            m = model_fn().fit(x[tr][:, cols] if cols is not None else prev[tr], label[tr])
            return m.predict_proba(x[te][:, cols] if cols is not None else prev[te])[:, 1]
        return fp

    gbm = lambda: HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=150, min_samples_leaf=15)  # noqa: E731
    logit = lambda: make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=10, max_iter=5000))  # noqa: E731
    all_cols = list(range(x.shape[1]))
    scores = {
        "previous_rating": oof(pid, cohort.patients, fit(lambda: LogisticRegression(max_iter=1000), None)),
        "enhancing_change_vs_nadir": x[:, 7],
        "total_change_vs_last": x[:, 10],
        "gbm_volumes_only": oof(pid, cohort.patients, fit(gbm, vol_cols)),
        "logistic_all": oof(pid, cohort.patients, fit(logit, all_cols)),
        "gbm_all": oof(pid, cohort.patients, fit(gbm, all_cols)),
    }
    out = {"rows": len(rows), "patients": len(set(pid)), "pd_rate": float(label.mean()),
           "all": auroc_boot(scores, label, pid, rng, "previous_rating")}
    sub = prev[:, RATINGS.index("PD")] == 0
    out["previous_not_pd"] = {"rows": int(sub.sum()), "pd_rate": float(label[sub].mean()),
                              **auroc_boot({k: v[sub] for k, v in scores.items()}, label[sub], pid[sub], rng, "enhancing_change_vs_nadir")}
    return out


# ----------------------------------------------------------------------------- C. dynamic prognosis


def concordance_matrices(pred: np.ndarray, remaining: np.ndarray, pid: np.ndarray, pats: list[str]):
    """K[a,b], N[a,b]: concordant / comparable visit pairs (visit of a lives shorter than visit of b)."""
    idx = [np.flatnonzero(pid == p) for p in pats]
    n = len(pats)
    K, N = np.zeros((n, n)), np.zeros((n, n))
    for a in range(n):
        ra, pa = remaining[idx[a]][:, None], pred[idx[a]][:, None]
        for b in range(n):
            if a == b:
                continue
            rb, pb = remaining[idx[b]][None, :], pred[idx[b]][None, :]
            comp = ra < rb
            N[a, b] = comp.sum()
            K[a, b] = ((pa < pb) & comp).sum() + 0.5 * ((pa == pb) & comp).sum()
    return K, N


def cindex_boot(preds: dict[str, np.ndarray], remaining, pid, rng, ref: str, mask=None) -> dict:
    mask = np.ones(len(pid), bool) if mask is None else mask
    pats = sorted(set(pid[mask]))
    mats = {k: concordance_matrices(v[mask], remaining[mask], pid[mask], pats) for k, v in preds.items()}
    point = {k: float(K.sum() / N.sum()) for k, (K, N) in mats.items()}
    draws = {k: [] for k in preds}
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(pats), len(pats)), minlength=len(pats)).astype(float)
        ww = np.outer(w, w)
        for k, (K, N) in mats.items():
            draws[k].append((ww * K).sum() / (ww * N).sum())
    out = {}
    for k in preds:
        d, diff = np.array(draws[k]), np.array(draws[k]) - np.array(draws[ref])
        out[k] = {"c_index": point[k], "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                  f"minus_{ref}": point[k] - point[ref],
                  f"minus_{ref}_ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
    return out


def task_c(cohort, clinical, statics, rng) -> dict:
    surv, inconsistent = {}, []
    for p in cohort.patients:
        try:
            s = float(clinical.get(p, {}).get("Survival time (weeks)", "na"))
        except ValueError:
            continue
        if s <= max(sc.week for sc in cohort.scans[p]):
            inconsistent.append(p)  # death recorded before the last scan: survival origin unclear
            continue
        surv[p] = s
    feats, base_cols, rating_cols, pid, remaining, first = [], None, None, [], [], []
    for p, s in surv.items():
        scans = cohort.scans[p]
        for i, sc in enumerate(scans):
            seg = [x for x in scans[: i + 1] if x.segment == sc.segment]
            y = y_of(sc)
            prev = y_of(seg[-2]) if len(seg) >= 2 else y
            nadir = np.stack([y_of(x) for x in seg]).min(axis=0)
            ratings = [norm_rating(x.rating) for x in scans[: i + 1]]
            base = list(statics[p]) + [sc.week / 52.0, sc.segment, i + 1]
            rate = onehot(sc.rating) + [float("PD" in ratings), float(ratings.count("PD"))]
            vol = list(y) + [tlog(y)] + list(y - prev) + [tlog(y) - tlog(prev), float(len(seg) >= 2)] + list(y - nadir) + \
                [tlog(y) - tlog(np.stack([y_of(x) for x in seg])).min(), (sc.week - seg[0].week) / 52.0, len(seg)]
            feats.append(base + rate + vol)
            base_cols = list(range(len(base)))
            rating_cols = list(range(len(base) + len(rate)))
            pid.append(p)
            remaining.append(s - sc.week)
            first.append(i == 0)
    x, pid, remaining, first = np.array(feats), np.array(pid), np.array(remaining), np.array(first)
    target = np.log(remaining)
    patients = sorted(surv)

    def fit(cols):
        def fp(tr, te):
            return hgb_reg(max_depth=3, max_iter=200, min_samples_leaf=20).fit(x[tr][:, cols], target[tr]).predict(x[te][:, cols])
        return fp

    preds = {
        "clinical_and_time": oof(pid, patients, fit(base_cols)),
        "plus_ratings": oof(pid, patients, fit(rating_cols)),
        "plus_ratings_and_volumes": oof(pid, patients, fit(list(range(x.shape[1])))),
    }
    return {"patients": len(patients), "visits": len(pid), "excluded_inconsistent_survival": inconsistent,
            "median_remaining_weeks": float(np.median(remaining)),
            "all_visits": cindex_boot(preds, remaining, pid, rng, "clinical_and_time"),
            "first_postop_visit": cindex_boot(preds, remaining, pid, rng, "clinical_and_time", mask=first)}


# ----------------------------------------------------------------------------- report


def write_markdown(res: dict, path: Path) -> None:
    a, b, c = res["A"], res["B"], res["C"]
    L = ["# Approach screen: which questions can LUMIERE answer?", "",
         f"Patient-level {FOLDS}-fold CV x {len(SEEDS)} fold seeds; out-of-fold predictions averaged over seeds; "
         f"95% CIs from {N_BOOT} patient resamples.", "",
         f"## A. Long-horizon volume forecasting ({a['pairs']} within-segment pairs, {a['patients']} patients)", "",
         "MAE in log1p(mL), mean of necrosis / enhancing / edema. Skill = 1 - MAE / copy-last MAE.", "",
         "| horizon | pairs (patients) | copy-last | patient slope | drift | Gompertz | GBM | GBM skill | GBM - drift [95% CI] | GBM - Gompertz [95% CI] |",
         "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |"]
    for name, e in a["bins"].items():
        m = e["mae"]
        d, g = e["gbm_minus_drift"], e["gbm_minus_gompertz"]
        L.append(f"| {name} | {e['pairs']} ({e['patients']}) | {m['persistence']:.3f} | {m['patient_slope']:.3f} | {m['drift']:.3f} | "
                 f"{m['gompertz']:.3f} | {m['gbm']:.3f} | {e['skill_vs_persistence']['gbm']:+.1%} | "
                 f"{d['diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] | {g['diff']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] |")
    L += ["", f"## B. Current-visit progression call ({b['rows']} rated visits, {b['patients']} patients, PD rate {b['pd_rate']:.2f})", "",
          "| score | AUROC, all visits [95% CI] | minus previous rating | AUROC, previous visit not PD [95% CI] | minus enhancing-vs-nadir rule |",
          "| :--- | :---: | :---: | :---: | :---: |"]
    s = b["previous_not_pd"]
    for k, v in b["all"].items():
        w = s[k]
        L.append(f"| {k} | {v['auroc']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | {v['minus_previous_rating']:+.3f} "
                 f"[{v['minus_previous_rating_ci95'][0]:+.3f}, {v['minus_previous_rating_ci95'][1]:+.3f}] | "
                 f"{w['auroc']:.3f} [{w['ci95'][0]:.3f}, {w['ci95'][1]:.3f}] | {w['minus_enhancing_change_vs_nadir']:+.3f} "
                 f"[{w['minus_enhancing_change_vs_nadir_ci95'][0]:+.3f}, {w['minus_enhancing_change_vs_nadir_ci95'][1]:+.3f}] |")
    L += ["", f"Previous visit not PD: {s['rows']} visits, PD rate {s['pd_rate']:.2f}.", "",
          f"## C. Dynamic prognosis ({c['patients']} patients, {c['visits']} visits, median remaining survival "
          f"{c['median_remaining_weeks']:.0f} weeks)", "",
          f"Excluded (death recorded before last scan): {', '.join(c['excluded_inconsistent_survival']) or 'none'}.", "",
          "| model | C-index, every visit [95% CI] | minus clinical + time | C-index, first post-op visit [95% CI] | minus clinical + time |",
          "| :--- | :---: | :---: | :---: | :---: |"]
    for k, v in c["all_visits"].items():
        w = c["first_postop_visit"][k]
        L.append(f"| {k} | {v['c_index']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | {v['minus_clinical_and_time']:+.3f} "
                 f"[{v['minus_clinical_and_time_ci95'][0]:+.3f}, {v['minus_clinical_and_time_ci95'][1]:+.3f}] | "
                 f"{w['c_index']:.3f} [{w['ci95'][0]:.3f}, {w['ci95'][1]:.3f}] | {w['minus_clinical_and_time']:+.3f} "
                 f"[{w['minus_clinical_and_time_ci95'][0]:+.3f}, {w['minus_clinical_and_time_ci95'][1]:+.3f}] |")
    path.write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data/lumiere/extracted/Imaging")
    ap.add_argument("--clinical-dir", type=Path, default=REPO / "data/lumiere/clinical")
    ap.add_argument("--output-dir", type=Path, default=REPO / "results/approach_screen")
    args = ap.parse_args()
    cohort = build_cohort(args.data_dir, args.clinical_dir / "LUMIERE-ExpertRating-v202211.csv")
    clinical = load_clinical(args.clinical_dir / "LUMIERE-Demographics_Pathology.csv")
    statics = {p: static_features(p, clinical) for p in cohort.patients}
    rng = np.random.default_rng(20261103)
    res = {"A": task_a(cohort, statics, rng)}
    print("A done", flush=True)
    res["B"] = task_b(cohort, statics, rng)
    print("B done", flush=True)
    res["C"] = task_c(cohort, clinical, statics, rng)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(res, indent=2, allow_nan=False) + "\n")
    write_markdown(res, args.output_dir / "summary.md")
    print((args.output_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
