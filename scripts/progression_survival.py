#!/usr/bin/env python3
"""Does an automated progression call predict survival as well as the expert's RANO call?

The question: if progression is read automatically from tumor volumes and masks, does
"has progressed" carry the same prognostic information as the expert's reading? A good
surrogate should identify patients at higher risk of death once they progress.

Progression definitions, all within the first surgery-free segment:
  expert                 first follow-up scan the expert rated PD
  automated (+masks)     first follow-up scan whose held-out probability from
                         progression_call_experiment.py's automated_volumes_masks model is
                         >= 0.5 (pre-specified). This model uses no expert rating and no
                         clinical factors.
  automated (volumes)    the same with automated_volumes
  volume rule            enhancing volume >= 40% above the segment nadir and >= 1 mL above it
A re-operation (start of a second segment) counts as progression for every definition if
none was called earlier. Only expert-rated follow-ups (PD/SD/PR/CR) are scanned, so every
definition sees the same visits. A sensitivity analysis instead sets each automated
threshold so that it calls as many progressions as the expert.

Analyses (time origin: the segment's first scan; outcome: death, LUMIERE survival weeks):
  * time-dependent Cox: hazard ratio of death after progression (progression as a
    time-varying covariate), alone and expert + automated jointly;
  * time-dependent concordance of progression status with death order;
  * landmark at 26 and 52 weeks: survival of patients alive at the landmark, by whether
    they had progressed by then;
  * agreement of expert and automated progression times.
95% CIs from a patient bootstrap (2000 resamples). Survival is never used for model fitting.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
REPO = SCRIPTS.parent

from lumiere_cohort import build_cohort, load_clinical  # noqa: E402
from progression_model import norm_rating  # noqa: E402

N_BOOT = 2000  # overridden by --bootstrap
LANDMARKS = (26, 52)
AUTOMATED = {"automated (+masks)": "automated_volumes_masks", "automated (volumes)": "automated_volumes"}


# ----------------------------------------------------------------------------- Cox and concordance


def cox_fit(entry, exit_, z_at_events, weights=None, iters=50):
    """Breslow partial likelihood; every subject dies at exit_ (no censoring in LUMIERE survival).

    z_at_events: (k, n_subjects, n_events) covariate of subject j at the death time of event i."""
    n = len(exit_)
    w = np.ones(n) if weights is None else weights
    at_risk = (entry[:, None] < exit_[None, :]) & (exit_[:, None] >= exit_[None, :])  # [j, i]
    z = np.asarray(z_at_events, dtype=np.float64)
    k = z.shape[0]
    z_self = z[:, np.arange(n), np.arange(n)]  # (k, n): each subject's covariate at its own death
    events = w > 0  # bootstrap weights: resampled-out deaths do not enter the likelihood
    z, z_self, at_risk, w_ev = z[:, :, events], z_self[:, events], at_risk[:, events], w[events]

    def evaluate(beta):
        r = at_risk * w[:, None] * np.exp(np.einsum("k,kji->ji", beta, z))  # [j, i]
        s0 = r.sum(axis=0)
        mean = np.einsum("kji,ji->ki", z, r) / s0
        ll = float((w_ev * ((beta @ z_self) - np.log(s0))).sum())
        grad = ((z_self - mean) * w_ev).sum(axis=1)
        info = ((np.einsum("kji,lji,ji->kli", z, z, r) / s0 - np.einsum("ki,li->kli", mean, mean)) * w_ev).sum(axis=2)
        return ll, grad, info

    beta = np.zeros(k)
    ll, grad, info = evaluate(beta)
    for _ in range(iters):  # Newton-Raphson with backtracking
        step = np.linalg.solve(info + 1e-9 * np.eye(k), grad)
        t = 1.0
        while True:
            cand = beta + t * step
            ll_c, grad_c, info_c = evaluate(cand)
            if ll_c >= ll - 1e-12 or t < 1e-6:
                break
            t /= 2
        beta, ll, grad, info = cand, ll_c, grad_c, info_c
        if np.abs(t * step).max() < 1e-9:
            break
    return beta


def concordance(entry, exit_, z_at_events, weights=None) -> float:
    """Pairs (dying subject i, subject j still at risk who dies later); marker = covariate at i's death."""
    n = len(exit_)
    w = np.ones(n) if weights is None else weights
    later = (entry[:, None] < exit_[None, :]) & (exit_[:, None] > exit_[None, :])  # j at risk at i's death, dies later
    zi = z_at_events[np.arange(n), np.arange(n)][None, :]  # marker of the dying subject at its death
    zj = z_at_events  # marker of j at i's death
    pair_w = later * w[:, None] * w[None, :]
    conc = ((zi > zj) * pair_w).sum() + 0.5 * ((zi == zj) * pair_w).sum()
    return float(conc / pair_w.sum())


# ----------------------------------------------------------------------------- progression times


def progression_times(cohort, survival, predictions, thresholds):
    """Per patient: entry, death, and progression week under each definition."""
    pred = {(r["patient_id"], r["cur_source"]): r for r in predictions}
    out = []
    for pid in cohort.patients:
        if pid not in survival:
            continue
        scans = cohort.scans[pid]
        seg0 = [s for s in scans if s.segment == 0]
        reop = next((s.week for s in scans if s.segment == 1), None)
        rated = [s for s in seg0[1:] if norm_rating(s.rating) in ("PD", "SD", "PR", "CR")]
        nadir, rule_week = seg0[0].volumes_mm3["Enhancing_Core"], None
        for s in seg0[1:]:
            enh = s.volumes_mm3["Enhancing_Core"]
            if s in rated and rule_week is None and enh >= 1.4 * nadir and enh - nadir >= 1000:
                rule_week = s.week
            nadir = min(nadir, enh)

        def first(pred_fn):
            week = next((s.week for s in rated if pred_fn(s)), None)
            return week if week is not None else reop

        times = {"expert": first(lambda s: norm_rating(s.rating) == "PD"),
                 "volume rule": rule_week if rule_week is not None else reop}
        for name, col in AUTOMATED.items():
            for tag, thr in thresholds[name].items():
                times[f"{name} {tag}"] = first(lambda s, c=col, t=thr: float(pred[(pid, s.source_name)][c]) >= t)
        out.append({"patient_id": pid, "entry": float(seg0[0].week), "death": survival[pid], "reop": reop,
                    "rated_followups": len(rated), "progression": times})
    return out


def z_matrix(prog_weeks: np.ndarray, death: np.ndarray) -> np.ndarray:
    """z[j, i] = 1 if subject j had progressed by the death time of subject i."""
    p = np.where(np.isnan(prog_weeks), np.inf, prog_weeks)
    return (p[:, None] <= death[None, :]).astype(np.float64)


# ----------------------------------------------------------------------------- analyses


def km(times: np.ndarray):
    """Kaplan-Meier for uncensored times: step function points."""
    t = np.sort(times)
    return np.concatenate([[0.0], t]), np.concatenate([[1.0], 1.0 - np.arange(1, len(t) + 1) / len(t)])


def analyse(patients, definitions, rng) -> dict:
    n = len(patients)
    entry = np.array([p["entry"] for p in patients])
    death = np.array([p["death"] for p in patients])
    prog = {d: np.array([np.nan if p["progression"][d] is None else p["progression"][d] for p in patients]) for d in definitions}
    Z = {d: z_matrix(prog[d], death) for d in definitions}
    reop = np.array([np.nan if p["reop"] is None else p["reop"] for p in patients])
    point = {}
    for d in definitions:
        done = np.isfinite(prog[d])
        point[d] = {"hr": float(np.exp(cox_fit(entry, death, Z[d][None])[0])), "c_index": concordance(entry, death, Z[d]),
                    "progressed_patients": int(done.sum()), "progression_from_reoperation": int((done & (prog[d] == reop)).sum()),
                    "median_weeks_start_to_progression": float(np.median((prog[d] - entry)[done])),
                    "median_weeks_progression_to_death": float(np.median((death - prog[d])[done]))}
    joint_name = next(d for d in definitions if d.startswith("automated (+masks)"))
    joint = np.exp(cox_fit(entry, death, np.stack([Z["expert"], Z[joint_name]])))
    draws = {d: {"hr": [], "c": []} for d in definitions}
    joint_draws = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        for d in definitions:
            draws[d]["hr"].append(np.exp(cox_fit(entry, death, Z[d][None], weights=w)[0]))
            draws[d]["c"].append(concordance(entry, death, Z[d], weights=w))
        joint_draws.append(np.exp(cox_fit(entry, death, np.stack([Z["expert"], Z[joint_name]]), weights=w)))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    res = {"patients": n, "deaths": n, "definitions": {}}
    for d in definitions:
        dc = np.array(draws[d]["c"]) - np.array(draws["expert"]["c"])
        res["definitions"][d] = {**point[d], "hr_ci95": ci(draws[d]["hr"]), "c_index_ci95": ci(draws[d]["c"]),
                                 "c_minus_expert": point[d]["c_index"] - point["expert"]["c_index"], "c_minus_expert_ci95": ci(dc)}
    jd = np.array(joint_draws)
    res["joint_cox"] = {"model": f"expert + {joint_name}", "hr_expert": float(joint[0]), "hr_expert_ci95": ci(jd[:, 0]),
                        "hr_automated": float(joint[1]), "hr_automated_ci95": ci(jd[:, 1])}
    # Landmarks: patients alive at entry + L; fixed covariate "progressed by the landmark".
    res["landmarks"] = {}
    for L in LANDMARKS:
        alive = death > entry + L
        e = (entry + L)[alive]
        dd = death[alive]
        entry_l = {"alive_patients": int(alive.sum())}
        groups = {d: (prog[d][alive] <= (entry + L)[alive]).astype(np.float64) for d in definitions}
        boot = {d: [] for d in definitions}
        for _ in range(N_BOOT):
            w = np.bincount(rng.integers(0, len(dd), len(dd)), minlength=len(dd)).astype(np.float64)
            for d in definitions:
                g = groups[d]
                if 0 < (w * g).sum() < w.sum():
                    boot[d].append(np.exp(cox_fit(e, dd, np.repeat(g[:, None], len(dd), axis=1)[None], weights=w)[0]))
        for d in definitions:
            g = groups[d]
            ok = 0 < g.sum() < len(g)
            hr = float(np.exp(cox_fit(e, dd, np.repeat(g[:, None], len(dd), axis=1)[None])[0])) if ok else None
            entry_l[d] = {"progressed": int(g.sum()), "median_remaining_weeks_progressed": float(np.median((dd - e)[g == 1])) if g.any() else None,
                          "median_remaining_weeks_not_progressed": float(np.median((dd - e)[g == 0])) if (g == 0).any() else None,
                          "hr": hr, "hr_ci95": ci(boot[d]) if ok and boot[d] else None}
        a = (prog["expert"][alive] <= (entry + L)[alive])
        b = (prog[joint_name][alive] <= (entry + L)[alive])
        po, pe = float((a == b).mean()), float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
        entry_l["agreement_expert_vs_automated"] = {"percent": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else None}
        res["landmarks"][str(L)] = entry_l
    both = np.isfinite(prog["expert"]) & np.isfinite(prog[joint_name])
    diff = prog[joint_name][both] - prog["expert"][both]
    res["timing_automated_minus_expert_weeks"] = {"patients_both_progressed": int(both.sum()), "median": float(np.median(diff)),
                                                  "earlier": int((diff < 0).sum()), "same_scan": int((diff == 0).sum()),
                                                  "later": int((diff > 0).sum())}
    res["_curves"] = {"entry": entry.tolist(), "death": death.tolist(), "prog": {d: prog[d].tolist() for d in definitions}}
    return res


def write_markdown(res: dict, meta: dict, path: Path) -> None:
    L = ["# Does the automated progression call predict survival like the expert's?", "",
         f"- {res['patients']} patients with consistent survival data, all {res['deaths']} deaths observed; time origin = first scan "
         f"of the first surgery-free segment. {meta['reop_patients']} patients had a re-operation, counted as progression for every "
         "definition if none was called earlier.",
         "- Automated calls use held-out probabilities from models with no expert rating and no clinical factors.",
         f"- Thresholds: 0.5 (pre-specified) and 'rate-matched', where the automated call flags as many patients as the expert "
         f"({json.dumps(meta['thresholds'])}).",
         "- 95% CIs from 2000 patient resamples.", "",
         "## Progression as a time-varying covariate (all follow-up)", "",
         "A time-varying HR rewards late calls (a call just before death looks maximally 'prognostic'), so read it with the "
         "median time from the call to death; the landmark analyses below avoid this bias.", "",
         "| definition | patients progressed (via re-operation) | median weeks start -> call | median weeks call -> death | HR for death after progression [95% CI] | time-dependent C-index [95% CI] | C minus expert [95% CI] |",
         "| :--- | ---: | ---: | ---: | :---: | :---: | :---: |"]
    for d, v in res["definitions"].items():
        L.append(f"| {d} | {v['progressed_patients']} ({v['progression_from_reoperation']}) | {v['median_weeks_start_to_progression']:.0f} | "
                 f"{v['median_weeks_progression_to_death']:.0f} | {v['hr']:.2f} [{v['hr_ci95'][0]:.2f}, {v['hr_ci95'][1]:.2f}] | "
                 f"{v['c_index']:.3f} [{v['c_index_ci95'][0]:.3f}, {v['c_index_ci95'][1]:.3f}] | "
                 f"{v['c_minus_expert']:+.3f} [{v['c_minus_expert_ci95'][0]:+.3f}, {v['c_minus_expert_ci95'][1]:+.3f}] |")
    j = res["joint_cox"]
    L += ["", f"Joint Cox ({j['model']}): expert HR {j['hr_expert']:.2f} [{j['hr_expert_ci95'][0]:.2f}, {j['hr_expert_ci95'][1]:.2f}], "
          f"automated HR {j['hr_automated']:.2f} [{j['hr_automated_ci95'][0]:.2f}, {j['hr_automated_ci95'][1]:.2f}].", ""]
    for Lw, v in res["landmarks"].items():
        L += [f"## Landmark {Lw} weeks after the segment start ({v['alive_patients']} patients alive)", "",
              "| definition | progressed by landmark | median remaining weeks, progressed | not progressed | landmark HR [95% CI] |",
              "| :--- | ---: | ---: | ---: | :---: |"]
        for d in res["definitions"]:
            e = v[d]
            fmt = lambda x: "-" if x is None else f"{x:.0f}"  # noqa: E731
            hr = "-" if e["hr"] is None else f"{e['hr']:.2f} [{e['hr_ci95'][0]:.2f}, {e['hr_ci95'][1]:.2f}]"
            L.append(f"| {d} | {e['progressed']} | {fmt(e['median_remaining_weeks_progressed'])} | "
                     f"{fmt(e['median_remaining_weeks_not_progressed'])} | {hr} |")
        a = v["agreement_expert_vs_automated"]
        L += ["", f"Expert vs automated (+masks, 0.5) status at the landmark: {a['percent']:.0%} agreement, kappa {a['kappa']:.2f}.", ""]
    t = res["timing_automated_minus_expert_weeks"]
    L += [f"Timing, {t['patients_both_progressed']} patients progressed under both: automated call earlier in {t['earlier']}, "
          f"same scan in {t['same_scan']}, later in {t['later']} (median difference {t['median']:+.0f} weeks).", ""]
    path.write_text("\n".join(L) + "\n")


def figure(res: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    c = res["_curves"]
    entry, death = np.array(c["entry"]), np.array(c["death"])
    L = 26
    alive = death > entry + L
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=200, sharey=True)
    for ax, (d, title) in zip(axes, (("expert", "A  Expert RANO call"), ("automated (+masks) p>=0.5", "B  Automated call (+masks)"))):
        prog = np.array([np.nan if v is None else v for v in c["prog"][d]], dtype=np.float64)
        g = prog[alive] <= (entry + L)[alive]
        rem = (death - entry - L)[alive]
        for mask, name, color, ls in ((g, "progressed by 26 weeks", "#eb6834", "-"), (~g, "not progressed", "#2a78d6", "--")):
            t, s = km(rem[mask])
            ax.step(t, s, where="post", color=color, linewidth=2, linestyle=ls,
                    label=f"{name} (n={int(mask.sum())}, median {np.median(rem[mask]):.0f} wk)")
        ax.set_title(title, loc="left", fontsize=9, color=ink)
        ax.set_xlabel("Weeks after the 26-week landmark", color=muted, fontsize=8)
        ax.legend(frameon=False, fontsize=7, loc="upper right")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
        ax.grid(color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, None)
    axes[0].set_ylabel("Surviving fraction", color=muted, fontsize=8)
    fig.suptitle("Survival after a 26-week landmark, by progression status (LUMIERE, held-out automated calls)",
                 x=0.01, ha="left", fontsize=10, color=ink)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data/lumiere/extracted/Imaging")
    ap.add_argument("--clinical-dir", type=Path, default=REPO / "data/lumiere/clinical")
    ap.add_argument("--predictions", type=Path, default=REPO / "results/progression_call/predictions.csv")
    ap.add_argument("--output-dir", type=Path, default=REPO / "results/progression_survival")
    ap.add_argument("--bootstrap", type=int, default=N_BOOT)
    args = ap.parse_args()
    globals()["N_BOOT"] = args.bootstrap
    cohort = build_cohort(args.data_dir, args.clinical_dir / "LUMIERE-ExpertRating-v202211.csv")
    clinical = load_clinical(args.clinical_dir / "LUMIERE-Demographics_Pathology.csv")
    survival, inconsistent = {}, []
    for p in cohort.patients:
        try:
            s = float(clinical.get(p, {}).get("Survival time (weeks)", "na"))
        except ValueError:
            continue
        if s <= max(sc.week for sc in cohort.scans[p]):
            inconsistent.append(p)
            continue
        survival[p] = s
    with args.predictions.open() as h:
        predictions = list(csv.DictReader(h))
    # Rate-matched thresholds: the automated call flags as many first-segment progressions as the expert.
    base = progression_times(cohort, survival, predictions, {k: {"p>=0.5": 0.5} for k in AUTOMATED})
    n_expert = sum(p["progression"]["expert"] is not None for p in base)
    thresholds = {}
    for name in AUTOMATED:
        grid = np.linspace(0.3, 0.95, 131)
        counts = [sum(p["progression"][f"{name} t"] is not None
                      for p in progression_times(cohort, survival, predictions, {k: {"t": t} for k in AUTOMATED}))
                  for t in grid]
        best = float(grid[int(np.argmin(np.abs(np.array(counts) - n_expert)))])
        thresholds[name] = {"p>=0.5": 0.5, "rate-matched": best}
    patients = progression_times(cohort, survival, predictions, thresholds)
    definitions = ["expert", "volume rule"] + [f"{n} {t}" for n in AUTOMATED for t in ("p>=0.5", "rate-matched")]
    rng = np.random.default_rng(20261104)
    res = analyse(patients, definitions, rng)
    meta = {"thresholds": {k: {t: round(v, 3) for t, v in d.items()} for k, d in thresholds.items()},
            "excluded_inconsistent_survival": inconsistent, "reop_patients": sum(p["reop"] is not None for p in patients)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = {k: v for k, v in res.items() if not k.startswith("_")}
    (args.output_dir / "metrics.json").write_text(json.dumps({"meta": meta, **out}, indent=2, allow_nan=False) + "\n")
    write_markdown(res, meta, args.output_dir / "metrics.md")
    figure(res, REPO / "results/figures/progression_survival_landmark.png")
    print((args.output_dir / "metrics.md").read_text())


if __name__ == "__main__":
    main()
