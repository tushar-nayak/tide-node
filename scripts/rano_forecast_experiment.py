"""Step 3: forecast the expert RANO rating of the next visit (progression or not).

Question: given only what is known at the last scan, will the neuroradiologist rate the
next scan PD (progressive disease)? The reference is the expert rating, not an automated
mask, so the label-noise ceiling of step 2 does not apply.

Cohort and folds: exactly step 1's (lumiere_cohort.py; 5-fold patient CV, fold seeds 0-2).
Labels: target rated PD -> 1; SD / PR / CR -> 0; unrated targets are dropped.
Secondary: response (PR or CR) vs not, on the same rated targets.

Features (all known at the last scan; survival and the target's rating rationale are never used):
  volumes   log1p(mL) of necrosis / enhancing / edema at the last scan, total, slope over the
            last gap, slope x interval, change since segment start, height above the segment's
            nadir (smallest volume so far; RANO measures progression against the nadir)
  timing    interval to the target, log interval, time since segment start, #scans, last gap
  rating    expert rating of the last scan (one-hot: PD / SD / PR / CR / Post-Op / unrated)
  clinical  age, sex, MGMT (as in step 1)

Scores compared (higher = more likely PD):
  trivial   small_tumor   -log1p(last total volume)   ("small tumors grow")
            interval      weeks to the target
            last_rating   logistic regression on the last scan's rating only
  learned   logistic_vol  logistic regression, volume + timing + clinical features
            logistic_all  ... + last rating
            gbm_all       gradient-boosted trees on all features
  step 1    predicted change in total tumor volume from step 1's gbm / neural_ode forecasts
            (results/volume_trajectory/predictions.csv, same folds)

Each transition's score is its out-of-fold prediction averaged over the 3 fold seeds.
AUROC, Brier score; 95% patient-bootstrap CIs; paired patient-bootstrap AUROC differences
against the trivial scores.

  .venv/bin/python scripts/rano_forecast_experiment.py      # ~1 min, CPU
Output: results/rano_forecast/{metrics.md,metrics.json,predictions.csv},
        results/figures/step3_rano_auroc.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lumiere_cohort import Transition, build_cohort, load_clinical, patient_folds  # noqa: E402
from volume_trajectory_experiment import auroc, static_features, y_of  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RATINGS = ("PD", "SD", "PR", "CR", "Post-Op")
TRIVIAL = ("small_tumor", "interval", "last_rating")
LEARNED = ("logistic_vol", "logistic_all", "gbm_all")
STEP1 = ("step1_gbm", "step1_neural_ode")
LABELS = {
    "small_tumor": "small tumor grows (-last volume)",
    "interval": "interval length",
    "last_rating": "last visit's RANO rating",
    "logistic_vol": "logistic, volumes + timing + clinical",
    "logistic_all": "logistic, + last rating",
    "gbm_all": "gradient boosting, all features",
    "step1_gbm": "step 1 GBM predicted volume change",
    "step1_neural_ode": "step 1 Neural ODE predicted volume change",
}


def norm_rating(r: str | None) -> str:
    r = (r or "").replace(" ", "")
    for k in RATINGS:
        if r.startswith(k):
            return k
    return "unrated"


def label(t: Transition) -> int:
    r = norm_rating(t.target.rating)
    return 1 if r == "PD" else (0 if r in ("SD", "PR", "CR") else -1)


def total(y: np.ndarray) -> float:
    return math.log1p(float(np.expm1(y).sum()))


def features(t: Transition, static: np.ndarray) -> dict[str, np.ndarray]:
    ys = [y_of(s) for s in t.history]
    y_last = ys[-1]
    tot = np.array([total(y) for y in ys])
    if len(t.history) >= 2:
        gap = max(t.last.week - t.history[-2].week, 1)
        slope = (y_last - ys[-2]) / (gap / 52.0)
        tslope = (tot[-1] - tot[-2]) / (gap / 52.0)
    else:
        gap, slope, tslope = 0, np.zeros(3), 0.0
    dt = t.dt_weeks / 52.0
    nadir = np.min(np.stack(ys), axis=0)
    vol = np.concatenate([y_last, [tot[-1]], slope, [tslope], slope * dt, y_last - ys[0],
                          y_last - nadir, [tot[-1] - tot.min()]])
    timing = np.array([dt, math.log1p(t.dt_weeks), (t.last.week - t.history[0].week) / 52.0, len(t.history), gap / 52.0])
    r = norm_rating(t.last.rating)
    rating = np.array([float(r == k) for k in RATINGS + ("unrated",)])
    return {"vol": np.concatenate([vol, timing, static]), "rating": rating}


def patient_auc_boot(scores: dict[str, np.ndarray], y: np.ndarray, pids: np.ndarray, n: int, rng) -> dict:
    """Patient-cluster bootstrap: AUROC CI per score, and paired differences."""
    ps = np.array(sorted(set(pids)))
    idx = {p: np.where(pids == p)[0] for p in ps}
    boots = {k: [] for k in scores}
    for _ in range(n):
        rows = np.concatenate([idx[p] for p in rng.choice(ps, len(ps))])
        if y[rows].min() == y[rows].max():
            continue
        for k, s in scores.items():
            boots[k].append(auroc(s[rows], y[rows]))
    return {k: np.array(v) for k, v in boots.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=REPO / "data/lumiere/extracted/Imaging")
    p.add_argument("--clinical-dir", type=Path, default=REPO / "data/lumiere/clinical")
    p.add_argument("--output-dir", type=Path, default=REPO / "results/rano_forecast")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cohort = build_cohort(args.data_dir, args.clinical_dir / "LUMIERE-ExpertRating-v202211.csv")
    clinical = load_clinical(args.clinical_dir / "LUMIERE-Demographics_Pathology.csv")
    patients = cohort.patients  # fold assignment over all step-1 patients, as in step 1
    statics = {q: static_features(q, clinical) for q in patients}
    trans = [t for t in cohort.transitions if label(t) >= 0]
    key = lambda t: (t.patient_id, t.target.week)  # noqa: E731
    y = np.array([label(t) for t in trans])
    resp = np.array([1 if norm_rating(t.target.rating) in ("PR", "CR") else 0 for t in trans])
    pids = np.array([t.patient_id for t in trans])
    F = [features(t, statics[t.patient_id]) for t in trans]
    Xv = np.stack([f["vol"] for f in F])
    Xr = np.stack([f["rating"] for f in F])
    Xa = np.concatenate([Xv, Xr], axis=1)
    print(f"{len(trans)} rated transitions ({y.mean():.0%} PD), {len(set(pids))} patients; "
          f"{len(cohort.transitions) - len(trans)} unrated targets dropped")

    oof = {k: np.zeros((args.repeats, len(trans))) for k in ("last_rating",) + LEARNED}
    oof_resp = {k: np.zeros((args.repeats, len(trans))) for k in ("last_rating", "logistic_all", "gbm_all")}
    for rep in range(args.repeats):
        for test_p in patient_folds(patients, args.folds, seed=rep):
            te = np.isin(pids, test_p)
            tr = ~te
            for target, store in ((y, oof), (resp, oof_resp)):
                models = {
                    "last_rating": (LogisticRegression(C=1.0, max_iter=2000), Xr),
                    "logistic_vol": (make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-3, 2, 12), max_iter=5000)), Xv),
                    "logistic_all": (make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-3, 2, 12), max_iter=5000)), Xa),
                    "gbm_all": (HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=150,
                                                               min_samples_leaf=15, random_state=rep), Xa),
                }
                for k in store:
                    m, X = models[k]
                    store[k][rep, te] = m.fit(X[tr], target[tr]).predict_proba(X[te])[:, 1]
    scores = {
        "small_tumor": -Xv[:, 3],
        "interval": np.array([t.dt_weeks for t in trans], dtype=float),
        **{k: v.mean(0) for k, v in oof.items()},
    }
    # step 1 forecasts, same folds: predicted change in total volume, averaged over the 3 seeds
    s1: dict[tuple, dict[str, list[float]]] = {}
    for r in csv.DictReader((REPO / "results/volume_trajectory/predictions.csv").open()):
        d = s1.setdefault((r["patient_id"], int(r["target_week"])), {"step1_gbm": [], "step1_neural_ode": []})
        for m in ("gbm", "neural_ode"):
            d[f"step1_{m}"].append(float(r[f"{m}_total"]) - float(r["persistence_total"]))
    for k in STEP1:
        scores[k] = np.array([np.mean(s1[key(t)][k]) for t in trans])
    true_change = np.array([total(y_of(t.target)) - total(y_of(t.last)) for t in trans])

    rng = np.random.default_rng(0)
    boot = patient_auc_boot({**scores, "oracle": true_change}, y, pids, args.bootstrap, rng)
    res: dict = {"n_transitions": int(len(trans)), "n_patients": int(len(set(pids))), "pd_rate": float(y.mean()),
                 "oracle_true_volume_change_auroc": auroc(true_change, y), "pd": {}, "response": {}}
    for k, s in scores.items():
        b = boot[k]
        e = {"auroc": auroc(s, y), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
        for ref in TRIVIAL:
            if k != ref:
                dlt = b - boot[ref]
                e[f"minus_{ref}"] = float(e["auroc"] - auroc(scores[ref], y))
                e[f"minus_{ref}_ci95"] = [float(np.percentile(dlt, 2.5)), float(np.percentile(dlt, 97.5))]
        if k in oof:
            e["brier"] = float(np.mean((s - y) ** 2))
        res["pd"][k] = e
    res["brier_base_rate"] = float(np.mean((y.mean() - y) ** 2))  # in-sample base rate: optimistic for the baseline
    bresp = patient_auc_boot({k: v.mean(0) for k, v in oof_resp.items()} | {"small_tumor_neg": Xv[:, 3]}, resp, pids, args.bootstrap, rng)
    for k in list(oof_resp) + ["small_tumor_neg"]:
        s = oof_resp[k].mean(0) if k in oof_resp else Xv[:, 3]
        res["response"][k] = {"auroc": auroc(s, resp), "ci95": [float(np.percentile(bresp[k], 2.5)), float(np.percentile(bresp[k], 97.5))]}
    res["response_rate"] = float(resp.mean())
    # what the last rating alone implies
    last_r = np.array([norm_rating(t.last.rating) for t in trans])
    res["pd_rate_by_last_rating"] = {r: {"n": int((last_r == r).sum()), "pd_rate": float(y[last_r == r].mean())}
                                     for r in RATINGS + ("unrated",) if (last_r == r).any()}

    # within the last rating: does anything predict progression beyond "PD stays PD"?
    res["within_last_rating"] = {}
    for name, sub in (("last not PD (SD/PR/CR/Post-Op/unrated)", last_r != "PD"), ("last PD", last_r == "PD")):
        b = patient_auc_boot({k: scores[k][sub] for k in ("small_tumor",) + LEARNED + STEP1}, y[sub], pids[sub], args.bootstrap, rng)
        res["within_last_rating"][name] = {"n": int(sub.sum()), "pd_rate": float(y[sub].mean()), **{
            k: {"auroc": auroc(scores[k][sub], y[sub]), "ci95": [float(np.percentile(b[k], 2.5)), float(np.percentile(b[k], 97.5))]}
            for k in b}}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    with (args.output_dir / "predictions.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["patient_id", "last_source", "target_source", "dt_weeks", "last_rating", "target_rating", "target_pd", *scores])
        for i, t in enumerate(trans):
            w.writerow([t.patient_id, t.last.source_name, t.target.source_name, t.dt_weeks, last_r[i],
                        norm_rating(t.target.rating), y[i], *[f"{scores[k][i]:.5f}" for k in scores]])
    write_md(res, args.output_dir / "metrics.md")
    figure(res)
    print((args.output_dir / "metrics.md").read_text())


def write_md(res: dict, path: Path) -> None:
    L = ["# Step 3: next-visit RANO progression (PD vs SD/PR/CR)", "",
         f"{res['n_transitions']} expert-rated transitions, {res['n_patients']} patients, PD rate {res['pd_rate']:.0%}. "
         "Patient-level 5-fold CV x 3 fold seeds (step 1's folds); out-of-fold scores averaged over seeds; "
         "95% patient-bootstrap CIs.", "",
         "| score | AUROC [95% CI] | minus small-tumor rule [95% CI] | minus last-rating rule [95% CI] | Brier |",
         "| :--- | :---: | :---: | :---: | ---: |"]
    for k in TRIVIAL + LEARNED + STEP1:
        e = res["pd"][k]
        cell = lambda ref: (f"{e[f'minus_{ref}']:+.3f} [{e[f'minus_{ref}_ci95'][0]:+.3f}, {e[f'minus_{ref}_ci95'][1]:+.3f}]"  # noqa: E731
                            if f"minus_{ref}" in e else "-")
        L.append(f"| {LABELS[k]} | {e['auroc']:.3f} [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}] | {cell('small_tumor')} | "
                 f"{cell('last_rating')} | {e.get('brier', float('nan')):.3f} |")
    L += ["", f"Brier of always predicting the PD rate: {res['brier_base_rate']:.3f}. "
          f"Oracle (true next-scan volume change, not available at forecast time): AUROC {res['oracle_true_volume_change_auroc']:.3f}.", "",
          "PD rate by the last visit's rating: " + ", ".join(f"{r} {v['pd_rate']:.0%} (n={v['n']})" for r, v in res["pd_rate_by_last_rating"].items()) + ".", "",
          "Within the last visit's rating (does anything predict PD beyond \"PD stays PD\"?):", "",
          "| subset | n | PD rate | " + " | ".join(LABELS[k] for k in ("small_tumor",) + LEARNED + STEP1) + " |",
          "| :--- | ---: | ---: |" + " :---: |" * (1 + len(LEARNED) + len(STEP1))] + [
          f"| {name} | {v['n']} | {v['pd_rate']:.0%} | " + " | ".join(
              f"{v[k]['auroc']:.3f} [{v[k]['ci95'][0]:.3f}, {v[k]['ci95'][1]:.3f}]" for k in ("small_tumor",) + LEARNED + STEP1) + " |"
          for name, v in res["within_last_rating"].items()] + ["",
          f"Secondary, response (PR/CR, rate {res['response_rate']:.0%}) vs not: " +
          "; ".join(f"{LABELS.get(k, 'large tumor (+last volume)')} {v['auroc']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"
                    for k, v in res["response"].items()) + "."]
    path.write_text("\n".join(L) + "\n")


def figure(res: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    groups = (("trivial rule", TRIVIAL, "#8a8984"), ("learned on the label", LEARNED, "#2a78d6"), ("step 1 volume forecast", STEP1, "#eb6834"))
    fig, ax = plt.subplots(figsize=(7.6, 3.8), dpi=200)
    yy, ticks = 0, []
    for gname, keys, color in groups:
        for k in keys:
            e = res["pd"][k]
            ax.plot(e["ci95"], [yy, yy], color=color, linewidth=2, solid_capstyle="round")
            ax.plot(e["auroc"], yy, "o", color=color, markersize=7, markeredgecolor="white", markeredgewidth=1.5,
                    label=gname if k == keys[0] else None)
            ax.text(e["ci95"][1] + 0.008, yy, f"{e['auroc']:.3f}", va="center", fontsize=8, color=INK)
            ticks.append((yy, LABELS[k]))
            yy += 1
        yy += 0.5
    ax.axvline(0.5, color=MUTED, linewidth=1)
    ax.axvline(res["oracle_true_volume_change_auroc"], color=MUTED, linewidth=1, linestyle=":")
    ax.text(res["oracle_true_volume_change_auroc"] + 0.004, -0.75, "oracle: true volume change", fontsize=7, color=MUTED, va="center")
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks], color=INK, fontsize=8)
    ax.set_ylim(yy - 0.3, -1.2)
    ax.set_xlim(0.4, 0.95)
    ax.set_xlabel("AUROC for next-visit RANO progression (95% CI over patients)", color=MUTED, fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02), frameon=False, fontsize=7)
    ax.set_title(f"Step 3 - will the next visit be rated progression? ({res['n_transitions']} transitions, {res['n_patients']} patients)",
                 loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(REPO / "results/figures/step3_rano_auroc.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
