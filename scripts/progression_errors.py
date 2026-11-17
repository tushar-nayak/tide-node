#!/usr/bin/env python3
"""Where and why the automated progression call disagrees with the expert.

Uses the out-of-fold (held-out) probabilities of scripts/progression_call_experiment.py
(results/progression_call/predictions.csv), so every call analysed here was made on a patient
the model had not seen. The expert's written rationale, the LessThan3Months flag and the
next scan's rating are used ONLY to explain errors, never as inputs: the first two leak the
label and the third is the future.

Operating point: the released "likely PD" threshold (80% specificity on held-out scores).
  miss         PD scan below the threshold
  false alarm  non-PD scan at or above it
Analyses:
  A. misses by the expert's stated reason for PD
  B. false alarms by the expert's stated reason for not calling PD
  C. was the model early? the next scan after a false alarm, against true negatives
  D. segmentation quality: accuracy where the two LUMIERE segmenters agree or disagree
     (needs results/progression_robustness/native_volumes.csv from progression_robustness.py)
  E. time since surgery
  F. are errors concentrated in a few patients?
CPU, under a minute.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
REPO = SCRIPTS.parent

import progression_model as pm  # noqa: E402
from approach_screen import N_BOOT, auroc_boot, norm_rating  # noqa: E402
from lumiere_cohort import build_cohort  # noqa: E402

MODELS = ("released_volumes_masks", "released_volumes")
MEASUREMENT = re.compile(r"\d+(\.\d+)?\s*(mm)?\s*x\s*\d")


def pd_reason(text: str) -> str:
    """Exclusive group for the expert's reason a scan is PD (most specific first)."""
    t = text.lower()
    if "new" in t:
        return "new lesion"
    if MEASUREMENT.search(t) or re.search(r"\btarget l", t.replace("non-target", "")):
        return "measurable lesion growth"
    if "t2" in t and "non-target" not in t and "non-measurable" not in t:
        return "T2/FLAIR progression only"
    if "non-target" in t or "non-measurable" in t or "progressing" in t:
        return "non-measurable / non-target lesion"
    return "other or no text"


def non_pd_reason(text: str) -> str:
    t = text.lower()
    if any(s in t for s in ("less than 3 months", "irradiat", "high dose", "high-dose", "pseudo", "radiation")):
        return "growth attributed to treatment (<3 months / irradiated field)"
    return "no rationale" if not t.strip() else "other rationale"


def load_rating_table(path: Path) -> dict[tuple[str, str], tuple[str, bool, bool]]:
    with path.open(newline="") as h:
        rows = list(csv.reader(h))

    def why(r):  # the file writes "None" for a blank rationale
        t = r[5].strip() if len(r) > 5 else ""
        return "" if t.lower() in ("none", "nan") else t
    return {(r[0].strip(), r[1].strip()): (why(r), r[2].strip().lower() == "x", r[3].strip().lower() == "x") for r in rows[1:]}


def rate(mask: np.ndarray, flag: np.ndarray) -> float | None:
    return float(flag[mask].mean()) if mask.any() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data/lumiere/extracted/Imaging")
    ap.add_argument("--clinical-dir", type=Path, default=REPO / "data/lumiere/clinical")
    ap.add_argument("--predictions", type=Path, default=REPO / "results/progression_call/predictions.csv")
    ap.add_argument("--mask-features", type=Path, default=REPO / "results/progression_call/mask_features_atlas_8mm.csv")
    ap.add_argument("--native-volumes", type=Path, default=REPO / "results/progression_robustness/native_volumes.csv")
    ap.add_argument("--output-dir", type=Path, default=REPO / "results/progression_errors")
    args = ap.parse_args()

    ratings_csv = args.clinical_dir / "LUMIERE-ExpertRating-v202211.csv"
    cohort = build_cohort(args.data_dir, ratings_csv)
    table = load_rating_table(ratings_csv)
    with args.predictions.open() as h:
        pred = {(r["patient_id"], r["prev_source"], r["cur_source"]): r for r in csv.DictReader(h)}
    with args.mask_features.open() as h:
        masks = {(r["patient_id"], r["prev_source"], r["cur_source"]): r for r in csv.DictReader(h)}
    rows = [t for t in cohort.transitions if norm_rating(t.target.rating) in ("PD", "SD", "PR", "CR")]
    keys = [(t.patient_id, t.last.source_name, t.target.source_name) for t in rows]
    assert len(rows) == len(pred) and all(k in pred for k in keys)
    pid = np.array([t.patient_id for t in rows])
    label = np.array([int(pred[k]["label_pd"]) for k in keys])
    scores = {m: np.array([float(pred[k][m]) for k in keys]) for m in MODELS}
    auto = np.array([float(pred[k]["automated_volumes_masks"]) for k in keys])  # same inputs minus rating and clinical
    auto_called = auto >= pm.thresholds(auto, label)["specificity_80"]["threshold"]
    rating = np.array([norm_rating(t.target.rating) for t in rows])
    text = [table.get((t.patient_id, t.target.source_name), ("", False, False))[0] for t in rows]
    flagged = np.array([table.get((t.patient_id, t.target.source_name), ("", False, False))[1] for t in rows])
    reason = np.array([pd_reason(x) if y else non_pd_reason(x) for x, y in zip(text, label)])
    prev_pd = np.array([norm_rating(t.last.rating) == "PD" for t in rows])
    weeks = np.array([t.target.week - t.history[0].week for t in rows])
    first_in_segment = np.array([norm_rating(t.history[0].rating) == "Post-Op" for t in rows])
    enh_nadir = np.array([pm.features(t.history, t.target, np.zeros(5))[pm.FEATURES.index("change_vs_nadir_enhancing")] for t in rows])
    new_et = np.array([float(masks[k]["et_new_ml"]) for k in keys])
    enh = "Enhancing_Core"
    enh_prev_pct = np.array([100.0 * (t.target.volumes_mm3[enh] - t.last.volumes_mm3[enh]) / t.last.volumes_mm3[enh]
                             if t.last.volumes_mm3[enh] > 0 else np.nan for t in rows])

    # Next scan of the same patient after the rated scan: PD, re-operation, not PD, or none.
    def next_outcome(t) -> tuple[str, int | None]:
        scans = cohort.scans[t.patient_id]
        i = next(j for j, s in enumerate(scans) if s.source_name == t.target.source_name)
        if i + 1 >= len(scans):
            return "no later scan", None
        nxt = scans[i + 1]
        r = norm_rating(nxt.rating)
        gap = nxt.week - t.target.week
        return ("re-operation" if r == "Post-Op" else "PD" if r == "PD" else "not PD" if r in ("SD", "PR", "CR") else "unrated"), gap

    nxt = [next_outcome(t) for t in rows]
    nxt_kind = np.array([n[0] for n in nxt])
    nxt_gap = np.array([np.nan if n[1] is None else n[1] for n in nxt], dtype=float)

    res: dict = {"rows": len(rows), "patients": len(set(pid)), "pd_scans": int(label.sum())}
    rng = np.random.default_rng(20261116)
    per_model = {}
    for m in MODELS:
        s = scores[m]
        thr = pm.thresholds(s, label)
        t80, t90 = thr["specificity_80"]["threshold"], thr["sensitivity_90"]["threshold"]
        called = s >= t80
        miss, false_alarm = (label == 1) & ~called, (label == 0) & called
        out = {"threshold_likely_pd": t80, "threshold_pd_less_likely": t90,
               "confusion": {"hits": int(((label == 1) & called).sum()), "misses": int(miss.sum()),
                             "false_alarms": int(false_alarm.sum()), "true_negatives": int(((label == 0) & ~called).sum())},
               "confident_misses_pd_in_lowest_band": int(((label == 1) & (s < t90)).sum())}
        # A. misses by reason
        out["pd_by_reason"] = {
            g: {"pd_scans": int(((label == 1) & (reason == g)).sum()), "misses": int((miss & (reason == g)).sum()),
                "sensitivity": rate((label == 1) & (reason == g), called),
                "in_lowest_band": rate((label == 1) & (reason == g), s < t90),
                "median_new_enhancing_ml": float(np.median(new_et[(label == 1) & (reason == g)])),
                "median_enhancing_log_change_vs_nadir": float(np.median(enh_nadir[(label == 1) & (reason == g)]))}
            for g in sorted(set(reason[label == 1]))}
        out["misses_share_previous_pd"] = rate(miss, prev_pd)
        out["hits_share_previous_pd"] = rate((label == 1) & called, prev_pd)
        # B. false alarms by reason and by rating
        out["non_pd_by_reason"] = {
            g: {"scans": int(((label == 0) & (reason == g)).sum()), "false_alarm_rate": rate((label == 0) & (reason == g), called),
                "false_alarms": int((false_alarm & (reason == g)).sum()),
                "mean_probability": float(s[(label == 0) & (reason == g)].mean())}
            for g in sorted(set(reason[label == 0]))}
        out["non_pd_by_rating"] = {r: {"scans": int((rating == r).sum()), "false_alarm_rate": rate(rating == r, called)}
                                   for r in ("SD", "PR", "CR")}
        out["non_pd_by_previous_rating"] = {
            name: {"scans": int(((label == 0) & mask).sum()), "false_alarm_rate": rate((label == 0) & mask, called),
                   "false_alarms": int((false_alarm & mask).sum())}
            for name, mask in (("previous scan PD", prev_pd), ("previous scan not PD", ~prev_pd))}
        after_pd = false_alarm & prev_pd
        out["false_alarms_after_pd"] = {
            "false_alarms": int(after_pd.sum()), "of_them_pr": int((after_pd & (rating == "PR")).sum()),
            "enhancing_fell_since_previous": int((after_pd & (enh_prev_pct < 0)).sum()),
            "no_enhancing_tumor_on_either_scan": int((after_pd & np.isnan(enh_prev_pct)).sum()),
            "pr_also_called_without_rating": int((after_pd & (rating == "PR") & auto_called).sum()),
            "not_pr_also_called_without_rating": int((after_pd & (rating != "PR") & auto_called).sum()),
            "enhancing_pct_change_since_previous_range": [float(np.nanmin(enh_prev_pct[after_pd])), float(np.nanmax(enh_prev_pct[after_pd]))]
            if np.isfinite(enh_prev_pct[after_pd]).any() else None}
        out["false_alarms_flagged_less_than_3_months"] = int((false_alarm & flagged).sum())
        # C. was the model early?
        out["next_scan"] = {}
        treat = np.char.startswith(reason.astype(str), "growth attributed")
        for name, mask in (("false alarms", false_alarm), ("true negatives", (label == 0) & ~called),
                           ("growth attributed to treatment, called", false_alarm & treat),
                           ("growth attributed to treatment, not called", (label == 0) & ~called & treat)):
            known = mask & np.isin(nxt_kind, ("PD", "re-operation", "not PD"))
            prog = mask & np.isin(nxt_kind, ("PD", "re-operation"))
            out["next_scan"][name] = {
                "scans": int(mask.sum()), "with_rated_next_scan": int(known.sum()),
                "next_pd_or_reoperation": int(prog.sum()),
                "share_next_pd_or_reoperation": float(prog.sum() / known.sum()) if known.any() else None,
                "median_weeks_to_that_scan": float(np.nanmedian(nxt_gap[prog])) if prog.any() else None,
                "counts": {k: int((mask & (nxt_kind == k)).sum()) for k in ("PD", "re-operation", "not PD", "unrated", "no later scan")}}
        # E. time since surgery
        bins = (("0-13 weeks", weeks <= 13), ("14-26 weeks", (weeks > 13) & (weeks <= 26)), (">26 weeks", weeks > 26))
        out["by_weeks_since_surgery"] = {}
        for name, mask in bins:
            b = auroc_boot({"m": s[mask]}, label[mask], pid[mask], rng, "m")["m"] if len(set(label[mask])) == 2 else None
            out["by_weeks_since_surgery"][name] = {
                "scans": int(mask.sum()), "pd_rate": float(label[mask].mean()), "auroc": None if b is None else b["auroc"],
                "ci95": None if b is None else b["ci95"], "false_alarm_rate": rate(mask & (label == 0), called),
                "sensitivity": rate(mask & (label == 1), called)}
        # F. concentration
        err = miss | false_alarm
        per_patient = np.array([err[pid == p].sum() for p in sorted(set(pid))])
        top = np.sort(per_patient)[::-1]
        out["concentration"] = {"errors": int(err.sum()), "patients_with_an_error": int((per_patient > 0).sum()),
                                "patients": len(per_patient), "share_of_errors_in_top_10_patients": float(top[:10].sum() / max(err.sum(), 1)),
                                "error_rate_overall": float(err.mean())}
        out["_masks"] = (miss, false_alarm, called)
        per_model[m] = out

    # D. segmentation quality (two segmenters on the native grid)
    if args.native_volumes.exists():
        with args.native_volumes.open() as h:
            nv = {(r["patient_id"], r["source"]): r for r in csv.DictReader(h)}

        def delta(t, a):
            c, p = nv.get((t.patient_id, t.target.source_name)), nv.get((t.patient_id, t.last.source_name))
            if c is None or p is None:
                return np.nan
            return np.log1p(float(c[a]) / 1000) - np.log1p(float(p[a]) / 1000)

        dis = np.array([abs(delta(t, "dbt_enh") - delta(t, "hd_enh")) for t in rows])
        ok = np.isfinite(dis)
        cuts = np.nanquantile(dis, [1 / 3, 2 / 3])
        groups = (("segmenters agree (lowest third)", ok & (dis <= cuts[0])),
                  ("middle third", ok & (dis > cuts[0]) & (dis <= cuts[1])),
                  ("segmenters disagree (highest third)", ok & (dis > cuts[1])))
        for m in MODELS:
            s = scores[m]
            miss, false_alarm, called = per_model[m]["_masks"]
            seg = {"cutoffs_abs_log_change_difference": [float(c) for c in cuts], "rows_with_both": int(ok.sum())}
            for name, mask in groups:
                b = auroc_boot({"m": s[mask]}, label[mask], pid[mask], rng, "m")["m"]
                seg[name] = {"scans": int(mask.sum()), "pd_rate": float(label[mask].mean()), "auroc": b["auroc"], "ci95": b["ci95"],
                             "error_rate": float((miss | false_alarm)[mask].mean())}
            seg["median_disagreement_errors"] = float(np.nanmedian(dis[ok & (miss | false_alarm)]))
            seg["median_disagreement_correct"] = float(np.nanmedian(dis[ok & ~(miss | false_alarm)]))
            per_model[m]["segmentation_quality"] = seg
    else:
        dis = np.full(len(rows), np.nan)
        print(f"{args.native_volumes} not found; run progression_robustness.py for the segmentation-quality analysis")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    miss, false_alarm, _ = per_model[MODELS[0]]["_masks"]
    with (args.output_dir / "errors.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["patient_id", "prev_source", "cur_source", "error", "expert_rating", "previous_rating", "probability_pd",
                    "reason_group", "rationale", "weeks_since_surgery", "enhancing_log_change_vs_nadir", "new_enhancing_ml",
                    "segmenter_disagreement", "next_scan", "weeks_to_next_scan"])
        for i in np.flatnonzero(miss | false_alarm):
            w.writerow([*keys[i], "miss" if miss[i] else "false alarm", rating[i], norm_rating(rows[i].last.rating),
                        round(float(scores[MODELS[0]][i]), 3), reason[i], text[i], int(weeks[i]), round(float(enh_nadir[i]), 3),
                        round(float(new_et[i]), 2), "" if np.isnan(dis[i]) else round(float(dis[i]), 3), nxt_kind[i],
                        "" if np.isnan(nxt_gap[i]) else int(nxt_gap[i])])
    for m in MODELS:
        per_model[m].pop("_masks")
    res["models"] = per_model
    res["first_scan_after_surgery_rows"] = int(first_in_segment.sum())
    (args.output_dir / "metrics.json").write_text(json.dumps(res, indent=2, allow_nan=False) + "\n")
    write_markdown(res, args.output_dir / "metrics.md")
    print((args.output_dir / "metrics.md").read_text())


def pct(v) -> str:
    return "-" if v is None else f"{v:.0%}"


def write_markdown(res: dict, path: Path) -> None:
    o = res["models"]["released_volumes_masks"]
    v = res["models"]["released_volumes"]
    c = o["confusion"]
    L = ["# Where the automated progression call disagrees with the expert", "",
         f"Held-out probabilities from patient-level CV; {res['rows']} rated follow-ups, {res['patients']} patients, "
         f"{res['pd_scans']} rated PD. Model: released volumes + masks; operating point: the 'likely PD' threshold "
         f"({o['threshold_likely_pd']:.3f}, 80% specificity). The expert's rationale, the LessThan3Months flag and the next "
         "scan are used only to explain errors.", "",
         f"Hits {c['hits']}, misses {c['misses']}, false alarms {c['false_alarms']}, true negatives {c['true_negatives']}. "
         f"{o['confident_misses_pd_in_lowest_band']} PD scans fell in the lowest band ('PD less likely').", "",
         "## A. PD scans: found or missed, by the expert's reason", "",
         "| expert's reason for PD | PD scans | misses | found (masks model) | found (volumes model) | in lowest band | median new enhancing (mL) | median enhancing change vs nadir (log) |",
         "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for g, x in o["pd_by_reason"].items():
        L.append(f"| {g} | {x['pd_scans']} | {x['misses']} | {pct(x['sensitivity'])} | {pct(v['pd_by_reason'][g]['sensitivity'])} | "
                 f"{pct(x['in_lowest_band'])} | {x['median_new_enhancing_ml']:.2f} | {x['median_enhancing_log_change_vs_nadir']:+.2f} |")
    L += ["", f"Previous scan already PD: {pct(o['misses_share_previous_pd'])} of misses, {pct(o['hits_share_previous_pd'])} of hits.", "",
          "## B. Non-PD scans: false alarms, by the expert's reason", "",
          "| expert's reason for not calling PD | scans | false-alarm rate | false alarms | mean P(PD) |", "| :--- | ---: | ---: | ---: | ---: |"]
    for g, x in o["non_pd_by_reason"].items():
        L.append(f"| {g} | {x['scans']} | {pct(x['false_alarm_rate'])} | {x['false_alarms']} | {x['mean_probability']:.2f} |")
    L += ["", "By rating: " + "; ".join(f"{r} {x['scans']} scans, {pct(x['false_alarm_rate'])} false alarms" for r, x in o["non_pd_by_rating"].items())
          + f". {o['false_alarms_flagged_less_than_3_months']} false alarms carry the LessThan3Months flag.", "",
          "By previous rating: " + "; ".join(f"{n}: {x['scans']} non-PD scans, {pct(x['false_alarm_rate'])} false alarms ({x['false_alarms']})"
                                             for n, x in o["non_pd_by_previous_rating"].items()) + ".", "",
          (lambda a: f"Of the {a['false_alarms']} false alarms after a PD rating, {a['of_them_pr']} were rated PR; the enhancing volume "
                     f"fell since the PD scan in {a['enhancing_fell_since_previous']}, and "
                     f"{a['no_enhancing_tumor_on_either_scan']} had no enhancing tumor on either scan "
                     f"({a['enhancing_pct_change_since_previous_range'][0]:+.0f}% to {a['enhancing_pct_change_since_previous_range'][1]:+.0f}%). "
                     f"The fully automated volumes + masks model, which does not see the rating, also calls {a['pr_also_called_without_rating']} "
                     f"of the {a['of_them_pr']} PR scans and {a['not_pr_also_called_without_rating']} of the others at its own 80%-specificity threshold.")(
              o["false_alarms_after_pd"]), "",
          "## C. Was the model early? The next scan", "",
          "| non-PD scans | scans | with a rated next scan | next scan PD or re-operation | share | median weeks to it |",
          "| :--- | ---: | ---: | ---: | ---: | ---: |"]
    for name, x in o["next_scan"].items():
        wk = "-" if x["median_weeks_to_that_scan"] is None else f"{x['median_weeks_to_that_scan']:.0f}"
        L.append(f"| {name} | {x['scans']} | {x['with_rated_next_scan']} | {x['next_pd_or_reoperation']} | "
                 f"{pct(x['share_next_pd_or_reoperation'])} | {wk} |")
    if "segmentation_quality" in o:
        s = o["segmentation_quality"]
        L += ["", "## D. Segmentation quality", "",
              "Disagreement = |difference in enhancing log-change since the previous scan| between DeepBraTumIA and HD-GLIO on the "
              f"native grid ({s['rows_with_both']} scans with both).", "",
              "| scans | n | PD rate | AUROC (masks model) [95% CI] | AUROC (volumes model) | error rate |", "| :--- | ---: | ---: | :---: | :---: | ---: |"]
        for name in ("segmenters agree (lowest third)", "middle third", "segmenters disagree (highest third)"):
            x, y = s[name], v["segmentation_quality"][name]
            L.append(f"| {name} | {x['scans']} | {x['pd_rate']:.2f} | {x['auroc']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}] | "
                     f"{y['auroc']:.3f} | {pct(x['error_rate'])} |")
        L += ["", f"Median disagreement: {s['median_disagreement_errors']:.2f} on errors, {s['median_disagreement_correct']:.2f} on correct calls."]
    L += ["", "## E. Time since surgery", "",
          "| weeks since surgery | scans | PD rate | AUROC [95% CI] | sensitivity | false-alarm rate |", "| :--- | ---: | ---: | :---: | ---: | ---: |"]
    for name, x in o["by_weeks_since_surgery"].items():
        a = "-" if x["auroc"] is None else f"{x['auroc']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]"
        L.append(f"| {name} | {x['scans']} | {x['pd_rate']:.2f} | {a} | {pct(x['sensitivity'])} | {pct(x['false_alarm_rate'])} |")
    k = o["concentration"]
    L += ["", "## F. Concentration", "",
          f"{k['errors']} errors in {k['patients_with_an_error']} of {k['patients']} patients; the 10 patients with most errors hold "
          f"{k['share_of_errors_in_top_10_patients']:.0%} of them. Every error is listed in errors.csv.", "",
          f"CIs: {N_BOOT} patient resamples."]
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
