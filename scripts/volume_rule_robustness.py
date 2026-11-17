#!/usr/bin/env python3
"""Is the volume rule's survival advantage robust? (follow-up to progression_survival.py)

progression_survival.py found that a simple volumetric definition of progression
(enhancing tumor >= 40% above the segment nadir and >= 1 mL above it) was more prognostic
than the expert's RANO call: time-dependent C-index 0.607 vs 0.539. Two checks:

1. Segmenter. The same rule computed from three volume sources:
     DeepBraTumIA, atlas space (the original analysis)
     DeepBraTumIA, native grid
     HD-GLIO-AUTO, native grid
   Native-grid volumes come from results/progression_robustness/native_volumes.csv
   (written by progression_robustness.py).
2. Thresholds. A grid of relative (25/40/65/100% above nadir) and absolute (0/0.5/1/2 mL)
   thresholds for DeepBraTumIA and HD-GLIO. 40% is RANO 2.0's volumetric PD threshold; the
   1 mL floor is the original analysis's choice. If the advantage holds only near
   40% / 1 mL it is fragile.
3. A second measure of surrogacy: the patient-level Spearman correlation between
   progression-free survival (progression or death) and overall survival, both from the
   segment start.

Same patients, time origin, re-operation rule, Cox model, time-dependent C-index and
landmarks as progression_survival.py, whose functions are reused unchanged. CPU, ~2 min.
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
from progression_survival import LANDMARKS, concordance, cox_fit, progression_times, z_matrix  # noqa: E402

N_BOOT = 2000
SOURCES = {"DeepBraTumIA (atlas)": "dbt_atlas", "DeepBraTumIA (native)": "dbt_native", "HD-GLIO (native)": "hd_native"}
REL = (0.25, 0.40, 0.65, 1.00)
ABS_ML = (0.0, 0.5, 1.0, 2.0)


def load_survival(cohort, clinical) -> tuple[dict[str, float], list[str]]:
    """As in progression_survival.main: drop patients whose recorded death precedes their last scan."""
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
    return survival, inconsistent


def rule_week(cohort, pid: str, enhancing, rel: float, abs_mm3: float) -> float | None:
    """First rated follow-up of the first segment with enhancing >= (1+rel) x nadir and >= nadir + abs; else the re-operation."""
    scans = cohort.scans[pid]
    seg0 = [s for s in scans if s.segment == 0]
    reop = next((s.week for s in scans if s.segment == 1), None)
    nadir = enhancing(seg0[0])
    for s in seg0[1:]:
        enh = enhancing(s)
        if norm_rating(s.rating) in ("PD", "SD", "PR", "CR") and enh >= (1 + rel) * nadir and enh - nadir >= abs_mm3:
            return s.week
        nadir = min(nadir, enh)
    return reop


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data/lumiere/extracted/Imaging")
    ap.add_argument("--clinical-dir", type=Path, default=REPO / "data/lumiere/clinical")
    ap.add_argument("--native-volumes", type=Path, default=REPO / "results/progression_robustness/native_volumes.csv")
    ap.add_argument("--output-dir", type=Path, default=REPO / "results/volume_rule_robustness")
    args = ap.parse_args()

    cohort = build_cohort(args.data_dir, args.clinical_dir / "LUMIERE-ExpertRating-v202211.csv")
    survival, _ = load_survival(cohort, load_clinical(args.clinical_dir / "LUMIERE-Demographics_Pathology.csv"))
    with args.native_volumes.open() as h:
        nv = {(r["patient_id"], r["source"]): r for r in csv.DictReader(h)}
    enhancing = {"dbt_atlas": lambda s: s.volumes_mm3["Enhancing_Core"],
                 "dbt_native": lambda s: float(nv[(s.patient_id, s.source_name)]["dbt_enh"]),
                 "hd_native": lambda s: float(nv[(s.patient_id, s.source_name)]["hd_enh"])}
    base = progression_times(cohort, survival, [], {"automated (+masks)": {}, "automated (volumes)": {}})
    pids = [p["patient_id"] for p in base]
    n = len(pids)
    entry = np.array([p["entry"] for p in base])
    death = np.array([p["death"] for p in base])

    def weeks(values) -> np.ndarray:
        return np.array([np.nan if v is None else v for v in values], dtype=float)

    prog = {"expert": weeks(p["progression"]["expert"] for p in base)}
    reference = weeks(p["progression"]["volume rule"] for p in base)
    for name, src in SOURCES.items():
        prog[f"volume rule, {name}"] = weeks(rule_week(cohort, pid, enhancing[src], 0.40, 1000.0) for pid in pids)
    same = np.array_equal(np.nan_to_num(prog["volume rule, DeepBraTumIA (atlas)"], nan=-1), np.nan_to_num(reference, nan=-1))
    assert same, "rule re-implementation must reproduce progression_survival.py"
    grid = {}
    for name in ("DeepBraTumIA (atlas)", "HD-GLIO (native)"):
        for rel in REL:
            for a in ABS_ML:
                grid[(name, rel, a)] = weeks(rule_week(cohort, pid, enhancing[SOURCES[name]], rel, a * 1000.0) for pid in pids)

    Z = {k: z_matrix(v, death) for k, v in prog.items()}
    ZG = {k: z_matrix(v, death) for k, v in grid.items()}
    rng = np.random.default_rng(20261116)
    draws_idx = [rng.integers(0, n, n) for _ in range(N_BOOT)]
    weights = [np.bincount(i, minlength=n).astype(np.float64) for i in draws_idx]
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    c_exp = np.array([concordance(entry, death, Z["expert"], weights=w) for w in weights])

    res: dict = {"patients": n, "bootstrap": N_BOOT, "main": {}, "grid": {}, "landmarks": {}}
    for k, p in prog.items():
        done = np.isfinite(p)
        hr_draws = [np.exp(cox_fit(entry, death, Z[k][None], weights=w)[0]) for w in weights]
        c_draws = np.array([concordance(entry, death, Z[k], weights=w) for w in weights])
        c = concordance(entry, death, Z[k])
        res["main"][k] = {
            "progressed": int(done.sum()), "median_weeks_start_to_call": float(np.median((p - entry)[done])),
            "median_weeks_call_to_death": float(np.median((death - p)[done])),
            "hr": float(np.exp(cox_fit(entry, death, Z[k][None])[0])), "hr_ci95": ci(hr_draws),
            "c_index": c, "c_index_ci95": ci(c_draws),
            "c_minus_expert": c - concordance(entry, death, Z["expert"]), "c_minus_expert_ci95": ci(c_draws - c_exp)}
    # Surrogacy: Spearman correlation of progression-free survival with overall survival.
    from scipy.stats import spearmanr

    os_weeks = death - entry
    pfs = {k: np.where(np.isfinite(p), np.minimum(p, death), death) - entry for k, p in prog.items()}
    rho = {k: float(spearmanr(v, os_weeks)[0]) for k, v in pfs.items()}
    rho_draws = {k: np.array([spearmanr(v[i], os_weeks[i])[0] for i in draws_idx]) for k, v in pfs.items()}
    for k in prog:
        res["main"][k].update({"median_pfs_weeks": float(np.median(pfs[k])), "spearman_pfs_os": rho[k],
                               "spearman_pfs_os_ci95": ci(rho_draws[k]), "spearman_minus_expert": rho[k] - rho["expert"],
                               "spearman_minus_expert_ci95": ci(rho_draws[k] - rho_draws["expert"])})
    # Agreement between the segmenters' rule calls.
    a, b = prog["volume rule, DeepBraTumIA (atlas)"], prog["volume rule, HD-GLIO (native)"]
    both = np.isfinite(a) & np.isfinite(b)
    res["agreement_dbt_vs_hd"] = {"both_progressed": int(both.sum()), "same_scan": int((a[both] == b[both]).sum()),
                                  "hd_earlier": int((b[both] < a[both]).sum()), "hd_later": int((b[both] > a[both]).sum())}
    for (name, rel, a_ml), p in grid.items():
        c_draws = np.array([concordance(entry, death, ZG[(name, rel, a_ml)], weights=w) for w in weights])
        c = concordance(entry, death, ZG[(name, rel, a_ml)])
        res["grid"].setdefault(name, []).append({
            "relative": rel, "absolute_ml": a_ml, "progressed": int(np.isfinite(p).sum()),
            "median_weeks_start_to_call": float(np.median((p - entry)[np.isfinite(p)])),
            "c_index": c, "c_minus_expert": c - concordance(entry, death, Z["expert"]), "c_minus_expert_ci95": ci(c_draws - c_exp)})
    # Landmarks, as in progression_survival.analyse.
    for L in LANDMARKS:
        alive = death > entry + L
        e, dd = (entry + L)[alive], death[alive]
        lw = [np.bincount(rng.integers(0, len(dd), len(dd)), minlength=len(dd)).astype(np.float64) for _ in range(N_BOOT)]
        out = {"alive_patients": int(alive.sum())}
        for k, p in prog.items():
            g = (p[alive] <= e).astype(np.float64)
            zz = np.repeat(g[:, None], len(dd), axis=1)[None]
            ok = 0 < g.sum() < len(g)
            draws = [np.exp(cox_fit(e, dd, zz, weights=w)[0]) for w in lw if 0 < (w * g).sum() < w.sum()] if ok else []
            out[k] = {"progressed": int(g.sum()), "hr": float(np.exp(cox_fit(e, dd, zz)[0])) if ok else None,
                      "hr_ci95": ci(draws) if draws else None}
        res["landmarks"][str(L)] = out

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(res, indent=2, allow_nan=False) + "\n")
    write_markdown(res, args.output_dir / "metrics.md")
    print((args.output_dir / "metrics.md").read_text())


def write_markdown(res: dict, path: Path) -> None:
    f = lambda v, d=3: f"{v:.{d}f}"  # noqa: E731
    L = ["# Volume rule vs expert: robustness of the survival result", "",
         f"{res['patients']} patients, all deaths observed; time-dependent Cox and C-index as in progression_survival.py; "
         f"95% CIs from {res['bootstrap']} patient resamples.", "",
         "## 1. Rule (>= 40% and >= 1 mL above nadir) from three volume sources", "",
         "| definition | progressed | median weeks start -> call | median weeks call -> death | HR [95% CI] | C-index [95% CI] | C minus expert [95% CI] |",
         "| :--- | ---: | ---: | ---: | :---: | :---: | :---: |"]
    for k, v in res["main"].items():
        L.append(f"| {k} | {v['progressed']} | {v['median_weeks_start_to_call']:.0f} | {v['median_weeks_call_to_death']:.0f} | "
                 f"{v['hr']:.2f} [{v['hr_ci95'][0]:.2f}, {v['hr_ci95'][1]:.2f}] | {f(v['c_index'])} [{f(v['c_index_ci95'][0])}, "
                 f"{f(v['c_index_ci95'][1])}] | {v['c_minus_expert']:+.3f} [{v['c_minus_expert_ci95'][0]:+.3f}, {v['c_minus_expert_ci95'][1]:+.3f}] |")
    L += ["", "Patient-level surrogacy: Spearman correlation of progression-free survival (progression or death) with overall survival.", "",
          "| definition | median PFS (weeks) | Spearman PFS-OS [95% CI] | minus expert [95% CI] |", "| :--- | ---: | :---: | :---: |"]
    for k, v in res["main"].items():
        L.append(f"| {k} | {v['median_pfs_weeks']:.0f} | {v['spearman_pfs_os']:.3f} [{v['spearman_pfs_os_ci95'][0]:.3f}, "
                 f"{v['spearman_pfs_os_ci95'][1]:.3f}] | {v['spearman_minus_expert']:+.3f} [{v['spearman_minus_expert_ci95'][0]:+.3f}, "
                 f"{v['spearman_minus_expert_ci95'][1]:+.3f}] |")
    g = res["agreement_dbt_vs_hd"]
    L += ["", f"DeepBraTumIA vs HD-GLIO rule: both called progression in {g['both_progressed']} patients; same scan in "
          f"{g['same_scan']}, HD-GLIO earlier in {g['hd_earlier']}, later in {g['hd_later']}.", "",
          "## 2. Threshold grid: C-index minus expert [95% CI]", ""]
    for name, rows in res["grid"].items():
        L += [f"### {name}", "", "| above nadir | " + " | ".join(f">= {a:g} mL" for a in ABS_ML) + " |",
              "| :--- | " + " | ".join(":---:" for _ in ABS_ML) + " |"]
        for rel in REL:
            cells = [next(r for r in rows if r["relative"] == rel and r["absolute_ml"] == a) for a in ABS_ML]
            L.append(f"| {rel:.0%} | " + " | ".join(
                f"{c['c_minus_expert']:+.3f} [{c['c_minus_expert_ci95'][0]:+.3f}, {c['c_minus_expert_ci95'][1]:+.3f}]" for c in cells) + " |")
        L.append("")
    L += ["## 3. Landmarks (progressed by the landmark; fixed covariate)", "",
          "| definition | " + " | ".join(f"{k} weeks: progressed, HR [95% CI]" for k in res["landmarks"]) + " |",
          "| :--- | " + " | ".join(":---:" for _ in res["landmarks"]) + " |"]
    for k in res["main"]:
        cells = []
        for lm in res["landmarks"].values():
            x = lm[k]
            cells.append("-" if x["hr"] is None else f"{x['progressed']}/{lm['alive_patients']}, {x['hr']:.2f} "
                         f"[{x['hr_ci95'][0]:.2f}, {x['hr_ci95'][1]:.2f}]")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
