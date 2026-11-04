#!/usr/bin/env python3
"""Automated progression call: is the current scan progressive disease (PD) under RANO?

The approach screen (scripts/approach_screen.py) found this the question LUMIERE can
answer: forecasting the next visit barely beats carrying the last one forward, but the
current scan's status can be read from automated measurements.

Rows: every follow-up scan with an expert rating of PD, SD, PR or CR (374 scans, 79
patients). Label: PD. Inputs are only what a reader has when the scan is taken: this scan,
the earlier scans of the same surgery-free segment, the previous expert rating, age, sex
and MGMT. The next scan and survival are never used.

Feature groups:
  V  automated volumes: this scan's compartments, change since the previous scan and since
     the segment nadir, totals, interval and time since surgery.
  R  the previous scan's expert rating.
  S  age, sex, MGMT.
  M  mask comparison of the current with the previous scan: new enhancing tumor outside the
     prior tumor core, new FLAIR abnormality, new enhancing components, largest axial
     enhancing area (a stand-in for RANO's 2D measurement). Two versions:
       atlas space, 8 mm margin  (released; needs only the two atlas segmentations)
       registered frame, 4 mm margin  (reference; current scan registered by
                                       register_step2_pairs.py)
     Atlas-space margins of 4/6/8 mm were tried; 8 mm was chosen (AUROC 0.822/0.831/0.835
     against 0.838 registered), so the released mask model carries a little selection optimism.

Models named automated_* use no expert rating and no clinical factors; their calls are fully
automated (used by progression_survival.py).

The rating file's LessThan3Months flag is NOT used: raters set it selectively, mostly on
early scans that grew but that they chose not to call PD (see window_flag_check in
metrics.json), so it leaks the label. True radiotherapy dates are not in LUMIERE.

Evaluation: patient-level 5-fold CV over 3 fold seeds, out-of-fold probabilities averaged
over seeds, patient-bootstrap 95% CIs, paired against the trivial rules.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
REPO = SCRIPTS.parent

import progression_model as pm  # noqa: E402  (released feature code: one implementation for training and the CLI)
from approach_screen import N_BOOT, assess_features, auroc_boot, norm_rating, oof  # noqa: E402
from lumiere_cohort import build_cohort, load_clinical  # noqa: E402
from volume_trajectory_experiment import static_features  # noqa: E402

MASK_FEATURES = pm.MASK_FEATURES
V_COLS, R_COLS, S_COLS = list(range(0, 15)), list(range(15, 21)), list(range(21, 26))
MR_COLS = list(range(26, 36))  # registered frame, 4 mm margin (reference)
MA_COLS = list(range(36, 46))  # atlas space, released margin


def mask_job(job: tuple[str, str, str, str, str]) -> tuple[list[float], list[float]]:
    """Registered-frame (4 mm margin) and atlas-space (released margin) mask features for one scan."""
    pid, prev_src, cur_src, atlas_dir, aligned_dir = job
    prev, cur_atlas = (np.load(Path(atlas_dir) / pid / f"{src}.npy")[4:7].astype(np.float32) > 0.5 for src in (prev_src, cur_src))
    cur_reg = np.load(Path(aligned_dir) / pid / f"{cur_src}__in__{prev_src}.npy")[4:7].astype(np.float32) > 0.5
    return pm.mask_features(prev, cur_reg, 2), pm.mask_features(prev, cur_atlas, pm.MASK_MARGIN_VOXELS)


def load_window_flags(ratings_csv: Path) -> dict[tuple[str, str], float]:
    with ratings_csv.open(newline="") as h:
        rows = list(csv.reader(h))
    return {(r[0].strip(), r[1].strip()): float(r[2].strip().lower() == "x") for r in rows[1:] if len(r) >= 3}


def load_rationales(ratings_csv: Path) -> dict[tuple[str, str], str]:
    with ratings_csv.open(newline="") as h:
        rows = list(csv.reader(h))
    return {(r[0].strip(), r[1].strip()): (r[5].strip() if len(r) > 5 else "") for r in rows[1:]}


def rationale_group(text: str) -> str:
    t = text.lower()
    t2, lesion = "t2" in t, ("l." in t or "lesion" in t or t.endswith(" l"))
    if t2 and lesion:
        return "T2 progression + lesion"
    if t2:
        return "T2 progression"
    if lesion:
        return "new / non-target lesion"
    return "enhancing growth (no rationale text)"


def operating_points(score: np.ndarray, label: np.ndarray) -> dict:
    order = np.sort(np.unique(score))
    out = {}
    for spec_target in (0.8, 0.9):
        thr = np.quantile(score[label == 0], spec_target)
        out[f"sensitivity_at_{round(spec_target * 100)}pct_specificity"] = float((score[label == 1] > thr).mean())
    thr = np.quantile(score[label == 1], 0.1)
    out["specificity_at_90pct_sensitivity"] = float((score[label == 0] < thr).mean())
    out["n_thresholds"] = int(len(order))
    return out


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data/lumiere/extracted/Imaging")
    ap.add_argument("--clinical-dir", type=Path, default=REPO / "data/lumiere/clinical")
    ap.add_argument("--atlas-dir", type=Path, default=REPO / "data/lumiere_atlas2mm")
    ap.add_argument("--aligned-dir", type=Path, default=REPO / "data/lumiere_step2_aligned")
    ap.add_argument("--output-dir", type=Path, default=REPO / "results/progression_call")
    ap.add_argument("--workers", type=int, default=11)
    args = ap.parse_args()

    ratings_csv = args.clinical_dir / "LUMIERE-ExpertRating-v202211.csv"
    cohort = build_cohort(args.data_dir, ratings_csv)
    clinical = load_clinical(args.clinical_dir / "LUMIERE-Demographics_Pathology.csv")
    statics = {p: static_features(p, clinical) for p in cohort.patients}
    window = load_window_flags(ratings_csv)
    rationale = load_rationales(ratings_csv)
    rows = [t for t in cohort.transitions if norm_rating(t.target.rating) in ("PD", "SD", "PR", "CR")]
    pid = np.array([t.patient_id for t in rows])
    label = np.array([norm_rating(t.target.rating) == "PD" for t in rows], dtype=int)
    base = np.stack([assess_features(t, statics[t.patient_id]) for t in rows])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    keys = [(t.patient_id, t.last.source_name, t.target.source_name) for t in rows]
    caches = {"registered": args.output_dir / "mask_features.csv",
              "atlas": args.output_dir / f"mask_features_atlas_{2 * pm.MASK_MARGIN_VOXELS}mm.csv"}
    stored = {name: {} for name in caches}
    for name, cache in caches.items():
        if cache.exists():
            with cache.open() as h:
                stored[name] = {(r["patient_id"], r["prev_source"], r["cur_source"]): [float(r[f]) for f in MASK_FEATURES]
                                for r in csv.DictReader(h)}
    missing = [k for k in keys if any(k not in stored[name] for name in caches)]
    if missing:
        print(f"Computing mask features for {len(missing)} scans", flush=True)
        jobs = [(*k, str(args.atlas_dir), str(args.aligned_dir)) for k in missing]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for k, (reg, atl) in zip(missing, pool.map(mask_job, jobs, chunksize=4)):
                stored["registered"][k], stored["atlas"][k] = reg, atl
        for name, cache in caches.items():
            with cache.open("w", newline="") as h:
                w = csv.writer(h)
                w.writerow(["patient_id", "prev_source", "cur_source", *MASK_FEATURES])
                for k in sorted(stored[name]):
                    w.writerow([*k, *[round(v, 6) for v in stored[name][k]]])
    wflag = np.array([[window.get((t.patient_id, t.target.source_name), 0.0)] for t in rows])
    x = np.hstack([base, np.array([stored["registered"][k] for k in keys]), np.array([stored["atlas"][k] for k in keys]), wflag])
    assert np.allclose(base, np.stack([pm.features(t.history, t.target, statics[t.patient_id]) for t in rows]))

    gbm = lambda: HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=150, min_samples_leaf=15)  # noqa: E731
    unconstrained = {k: v for k, v in pm.GBM_PARAMS.items() if k != "monotonic_cst"}
    vol_cst = [pm.GBM_PARAMS["monotonic_cst"][i] for i in V_COLS]
    automated = lambda cst: (lambda: HistGradientBoostingClassifier(**unconstrained, monotonic_cst=cst))  # noqa: E731

    def fit(model_fn, cols):
        def fp(tr, te):
            return model_fn().fit(x[tr][:, cols], label[tr]).predict_proba(x[te][:, cols])[:, 1]
        return fp

    models = {
        "previous_rating": (lambda: LogisticRegression(max_iter=1000), R_COLS),
        "unconstrained_volumes": (gbm, V_COLS + R_COLS + S_COLS),
        "released_volumes": (lambda: pm.new_model(False), V_COLS + R_COLS + S_COLS),
        "released_volumes_masks": (lambda: pm.new_model(True), V_COLS + R_COLS + S_COLS + MA_COLS),
        "reference_registered_masks": (lambda: pm.new_model(True), V_COLS + R_COLS + S_COLS + MR_COLS),
        "automated_volumes": (automated(vol_cst), V_COLS),
        "automated_volumes_masks": (automated(vol_cst + list(pm.MASK_MONOTONE)), V_COLS + MA_COLS),
    }
    scores = {k: oof(pid, cohort.patients, fit(fn, cols)) for k, (fn, cols) in models.items()}
    probs = dict(scores)
    scores["rule_enhancing_change_vs_nadir"] = x[:, 7]
    scores["rule_new_enhancing_ml"] = x[:, MA_COLS[0]]
    rng = np.random.default_rng(20261104)
    prev_pd = x[:, R_COLS[0]] == 1
    res = {
        "rows": len(rows), "patients": len(set(pid)), "pd_rate": float(label.mean()),
        "mask_margin_mm": {"atlas": 2 * pm.MASK_MARGIN_VOXELS, "registered": 4},
        "all": auroc_boot(scores, label, pid, rng, "previous_rating"),
        "vs_volume_model": auroc_boot(scores, label, pid, rng, "released_volumes"),
        "previous_not_pd": {"rows": int((~prev_pd).sum()), "pd_rate": float(label[~prev_pd].mean()),
                            **auroc_boot({k: v[~prev_pd] for k, v in scores.items()}, label[~prev_pd], pid[~prev_pd], rng,
                                         "rule_enhancing_change_vs_nadir")},
        "brier": {k: brier(v, label) for k, v in probs.items()} | {"pd_rate_constant": brier(np.full(len(label), label.mean()), label)},
        "operating_points": {k: operating_points(v, label) for k, v in scores.items()},
    }
    # Which PD calls are found? Sensitivity by the expert's stated reason, at 80% specificity.
    groups = np.array([rationale_group(rationale.get((t.patient_id, t.target.source_name), "")) for t in rows])
    res["sensitivity_by_rationale_at_80pct_specificity"] = {
        g: {"pd_scans": int(((groups == g) & (label == 1)).sum()),
            **{k: float((scores[k][(groups == g) & (label == 1)] > np.quantile(scores[k][label == 0], 0.8)).mean())
               for k in ("released_volumes_masks", "released_volumes")}}
        for g in sorted(set(groups[label == 1]))}
    # Why the LessThan3Months flag is excluded: among first follow-ups 9-26 weeks after surgery
    # (roughly the post-radiotherapy window) it marks mostly scans that grew but were not called PD.
    win = wflag[:, 0] == 1
    early = (np.array([len(t.history) for t in rows]) == 1) & (x[:, 14] * 52 >= 9) & (x[:, 14] * 52 <= 26)
    res["window_flag_check"] = {
        "flagged_scans": int(win.sum()), "flagged_pd": int(label[win].sum()),
        "early_first_followups": int(early.sum()), "early_first_followups_flagged": int((early & win).sum()),
        "median_enhancing_change_vs_nadir_flagged": float(np.median(x[win, 7])),
        "median_enhancing_change_vs_nadir_unflagged_early": float(np.median(x[early & ~win, 7])),
        "pd_rate_unflagged_early": float(label[early & ~win].mean()),
        "conclusion": "selective annotation correlated with the label; excluded as leakage",
    }
    with (args.output_dir / "predictions.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["patient_id", "prev_source", "cur_source", "expert_rating", "label_pd", "rationale_group", *scores])
        for i, t in enumerate(rows):
            w.writerow([t.patient_id, t.last.source_name, t.target.source_name, norm_rating(t.target.rating), label[i], groups[i],
                        *[round(float(scores[k][i]), 6) for k in scores]])
    (args.output_dir / "metrics.json").write_text(json.dumps(res, indent=2, allow_nan=False) + "\n")
    write_markdown(res, args.output_dir / "metrics.md")
    figure(scores, label, ~prev_pd, res, REPO / "results/figures/progression_call_roc.png")
    print((args.output_dir / "metrics.md").read_text())


def write_markdown(res: dict, path: Path) -> None:
    L = ["# Automated progression call (RANO PD at the current scan)", "",
         f"- {res['rows']} expert-rated follow-up scans (PD / SD / PR / CR), {res['patients']} patients, PD rate {res['pd_rate']:.2f}.",
         "- Patient-level 5-fold CV x 3 fold seeds; out-of-fold probabilities averaged over seeds; 95% CIs from "
         f"{N_BOOT} patient resamples. Inputs: this and earlier scans only, plus the previous expert rating and age/sex/MGMT.",
         "", "| score | AUROC [95% CI] | minus previous rating | minus released volume model | AUROC, previous scan not PD | minus enhancing-vs-nadir rule |",
         "| :--- | :---: | :---: | :---: | :---: | :---: |"]
    for k, v in res["all"].items():
        u, s = res["vs_volume_model"][k], res["previous_not_pd"][k]
        L.append(f"| {k} | {v['auroc']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | "
                 f"{v['minus_previous_rating']:+.3f} [{v['minus_previous_rating_ci95'][0]:+.3f}, {v['minus_previous_rating_ci95'][1]:+.3f}] | "
                 f"{u['minus_released_volumes']:+.3f} [{u['minus_released_volumes_ci95'][0]:+.3f}, "
                 f"{u['minus_released_volumes_ci95'][1]:+.3f}] | {s['auroc']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] | "
                 f"{s['minus_rule_enhancing_change_vs_nadir']:+.3f} [{s['minus_rule_enhancing_change_vs_nadir_ci95'][0]:+.3f}, "
                 f"{s['minus_rule_enhancing_change_vs_nadir_ci95'][1]:+.3f}] |")
    p = res["previous_not_pd"]
    L += ["", f"Previous scan not PD: {p['rows']} scans, PD rate {p['pd_rate']:.2f}.", "",
          "## Operating points (pooled out-of-fold scores)", "",
          "| score | sensitivity at 80% specificity | sensitivity at 90% specificity | specificity at 90% sensitivity | Brier |",
          "| :--- | ---: | ---: | ---: | ---: |"]
    for k, v in res["operating_points"].items():
        b = res["brier"].get(k)
        L.append(f"| {k} | {v['sensitivity_at_80pct_specificity']:.2f} | {v['sensitivity_at_90pct_specificity']:.2f} | "
                 f"{v['specificity_at_90pct_sensitivity']:.2f} | {'-' if b is None else f'{b:.3f}'} |")
    L += [f"| constant PD rate | - | - | - | {res['brier']['pd_rate_constant']:.3f} |", "",
          "## PD scans found, by the expert's stated reason (80% specificity)", "",
          "| reason | PD scans | released, with atlas masks | released, volumes only |", "| :--- | ---: | ---: | ---: |"]
    for g, v in res["sensitivity_by_rationale_at_80pct_specificity"].items():
        L.append(f"| {g} | {v['pd_scans']} | {v['released_volumes_masks']:.2f} | {v['released_volumes']:.2f} |")
    w = res["window_flag_check"]
    L += ["", "## Excluded input: the LessThan3Months flag", "",
          f"The rating file flags {w['flagged_scans']} scans ({w['flagged_pd']} rated PD). Of {w['early_first_followups']} first "
          f"follow-ups 9-26 weeks after surgery, only {w['early_first_followups_flagged']} are flagged. Flagged scans grew "
          f"(median enhancing log-change vs nadir {w['median_enhancing_change_vs_nadir_flagged']:+.2f}) while unflagged early "
          f"scans did not ({w['median_enhancing_change_vs_nadir_unflagged_early']:+.2f}; {w['pd_rate_unflagged_early']:.0%} PD). "
          "The flag records the rater's decision not to call growth PD, so using it would leak the label.", ""]
    path.write_text("\n".join(L) + "\n")


def figure(scores: dict, label: np.ndarray, prev_not_pd: np.ndarray, res: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    series = (("previous_rating", "previous rating carried forward", "#8a8984", "--"),
              ("rule_enhancing_change_vs_nadir", "enhancing volume vs nadir (rule)", "#eb6834", ":"),
              ("released_volumes", "released: volumes + rating + clinical", "#2a78d6", "-"),
              ("released_volumes_masks", "released: + new-tumor masks (atlas, 8 mm)", "#1baf7a", "-."))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), dpi=200)
    panels = (("A  All rated follow-up scans", np.ones(len(label), bool), "all"),
              ("B  Previous scan not PD (new progression)", prev_not_pd, "previous_not_pd"))
    for ax, (title, mask, key) in zip(axes, panels):
        for k, name, color, ls in series:
            s, y = scores[k][mask], label[mask]
            thr = np.unique(s)[::-1]
            tpr = [0.0] + [float((s[y == 1] >= t).mean()) for t in thr]
            fpr = [0.0] + [float((s[y == 0] >= t).mean()) for t in thr]
            ax.plot(fpr, tpr, color=color, linewidth=2, linestyle=ls, label=f"{name}  (AUROC {res[key][k]['auroc']:.2f})")
        ax.plot([0, 1], [0, 1], color=grid, linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False-positive rate (non-PD scans called PD)", color=muted, fontsize=8)
        ax.set_ylabel("Sensitivity (PD scans found)", color=muted, fontsize=8)
        ax.set_title(title, loc="left", fontsize=9, color=ink)
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
        ax.grid(color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Automated RANO progression call on held-out LUMIERE patients", x=0.01, ha="left", fontsize=10, color=ink)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
