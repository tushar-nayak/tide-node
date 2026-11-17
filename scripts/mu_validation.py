#!/usr/bin/env python3
"""External validation of the fully automated progression call on MU-Glioma-Post.

Pre-specified in results/mu_validation/PLAN.md. The model and thresholds were frozen on
LUMIERE by scripts/mu_freeze_model.py; this script refuses to run if the plan or the model
file has changed since the freeze. Steps:
  1. Per-scan volumes from the released tumor masks. They are cached in
     results/mu_validation/mask_volumes.csv, so the 11 GB package is read once.
  2. Labels for each scan from the documented first-progression day.
  3. The frozen models and simple comparators applied to every evaluable scan.
  4. Reports of discrimination, operating points, sensitivity analyses, the radiotherapy
     window, a censoring-aware survival test of the volume rule, and a check of the TCIA
     volume spreadsheet against the masks.
Data: MU-Glioma-Post (TCIA, CC BY 4.0; Yaseen et al., Sci Data 2025). CPU, about 2 min
(the first run also reads ~600 masks).
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
REPO = SCRIPTS.parent

from approach_screen import N_BOOT, auroc_boot  # noqa: E402
from mu_freeze_model import PRIMARY, rule_fires, sha256  # noqa: E402
from progression_robustness import two_compartment_features  # noqa: E402
from progression_survival import z_matrix  # noqa: E402
from xlsx_stdlib import read, records  # noqa: E402

WINDOW_DAYS = 14  # a scan this close to the documented first-progression day is the PD scan
RT_WINDOW_DAYS = 84  # RANO: 12 weeks after the end of radiotherapy
CLINICAL_SHEET = "MU Glioma Post"
VOLUME_SHEETS = {"Necrotic Tumor Core (Label1)": "NETC", "Tumor Infiltration and Edema": "SNFH",
                 "Enhancing Tumor Core (Label3)": "ET", "Resection Cavity (Label4)": "RC"}
ORDINAL = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th"}
SURGERY = "Number of days from Diagnosis to First surgery or procedure"
FIRST_PROGRESSION = "Number of days from Diagnosis to date of First Progression"
DEATH = "Number of days from Diagnosis to death (Days)"
RT_END = "Number of days from Diagnosis to Radiation Therapy end date"
MODEL_NAMES = {"dbt_atlas": "model trained on DeepBraTumIA volumes (primary)", "hd_native": "model trained on HD-GLIO volumes"}
REF = MODEL_NAMES[PRIMARY]
DOC = "documented progression (at scans)"


def tp_column(k: int) -> str:
    return f"Number of Days from Diagnosis to {ORDINAL[k]} MRI (Timepoint_{k})"


def num(x) -> float | None:
    try:
        v = float(str(x).strip())
    except ValueError:
        return None
    return v if np.isfinite(v) else None


# ----------------------------------------------------------------------------- volumes


def mask_volumes(path: str) -> tuple[float, float, float, float]:
    """(NETC, SNFH, ET, RC) in mm3 from one tumorMask.nii.gz."""
    import nibabel as nib

    img = nib.load(path)
    seg = np.asarray(img.dataobj).astype(np.int64)
    voxel = float(np.prod(img.header.get_zooms()[:3]))
    counts = np.bincount(seg.ravel(), minlength=5)
    return tuple(float(counts[i]) * voxel for i in (1, 2, 3, 4))


def load_mask_volumes(package_dir: Path, cache: Path, workers: int) -> dict[tuple[str, int], tuple[float, ...]]:
    if cache.exists():
        with cache.open() as h:
            return {(r["patient_id"], int(r["timepoint"])): tuple(float(r[c]) for c in ("netc", "snfh", "et", "rc"))
                    for r in csv.DictReader(h)}
    if any(package_dir.rglob("*.partial")):
        raise RuntimeError(f"{package_dir} is still downloading")
    found = {}
    for f in package_dir.rglob("*_tumorMask.nii.gz"):
        m = re.fullmatch(r"(PatientID_\d+)_Timepoint_(\d+)_tumorMask\.nii\.gz", f.name)
        if m:
            found[(m.group(1), int(m.group(2)))] = str(f)
    keys = sorted(found)
    print(f"Reading {len(keys)} tumor masks", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        vols = dict(zip(keys, pool.map(mask_volumes, [found[k] for k in keys], chunksize=4)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["patient_id", "timepoint", "netc", "snfh", "et", "rc"])
        for k in keys:
            w.writerow([*k, *vols[k]])
    return vols


def spreadsheet_volumes(path: Path, timepoints: dict[str, list[int]]) -> tuple[dict, dict]:
    """Map the TCIA volume spreadsheet to scans, by the rule in PLAN.md (volume source check)."""
    explicit: dict[tuple[str, str], dict[int, float]] = {}
    plain: dict[tuple[str, str], list[float]] = {}
    sheets = read(path)
    for sheet, label in VOLUME_SHEETS.items():
        rows = sheets[sheet]
        col = [h.strip() for h in rows[0]].index("Volume (mm^3)")
        for r in rows[1:]:
            m = re.fullmatch(r"(PatientID_\d+)(?:-Post-treatment_(\d+))?", r[0].strip())
            if not m:
                continue
            if m.group(2):
                explicit.setdefault((m.group(1), label), {})[int(m.group(2))] = float(r[col])
            else:
                plain.setdefault((m.group(1), label), []).append(float(r[col]))
    labels = list(VOLUME_SHEETS.values())
    out, status = {}, {"explicit": 0, "in_order": 0, "skipped_ambiguous": 0, "skipped_mixed": 0}
    for pid, tps in timepoints.items():
        has_explicit = any((pid, lab) in explicit for lab in labels)
        has_plain = any((pid, lab) in plain for lab in labels)
        if has_explicit and has_plain:
            status["skipped_mixed"] += 1
            continue
        if has_explicit:
            present = sorted(set().union(*(explicit.get((pid, lab), {}).keys() for lab in labels)))
            for k in present:
                out[(pid, k)] = tuple(explicit.get((pid, lab), {}).get(k, 0.0) for lab in labels)
            status["explicit"] += 1
            continue
        cols = []
        for lab in labels:
            v = plain.get((pid, lab), [])
            cols.append(v if len(v) == len(tps) else [0.0] * len(tps) if not v else None)
        if any(c is None for c in cols):
            status["skipped_ambiguous"] += 1
            continue
        for i, k in enumerate(tps):
            out[(pid, k)] = tuple(c[i] for c in cols)
        status["in_order"] += 1
    return out, status


# ----------------------------------------------------------------------------- cohort


@dataclass
class Scan:
    k: int
    day: float
    enh: float
    non: float


@dataclass
class Patient:
    pid: str
    diagnosis: str
    surgery_day: float | None
    progressed: bool
    fp_day: float | None
    prog_type: str
    reoperated: bool
    died: bool
    death_day: float | None
    last_day: float
    rt_end: float | None
    scans: list[Scan]


def build_patients(clinical: list[dict], volumes: dict) -> list[Patient]:
    day_cols = [c for c in clinical[0] if re.match(r"(number of )?days from di?agnosis", c, re.I)]
    out = []
    for r in clinical:
        pid = r["Patient_ID"].strip()
        if not pid.startswith("PatientID"):
            continue
        surgery = num(r[SURGERY])
        scans = []
        for k in ORDINAL:
            day = num(r[tp_column(k)])
            if day is None or (pid, k) not in volumes or (surgery is not None and day < surgery):
                continue
            netc, snfh, et, _ = volumes[(pid, k)]
            scans.append(Scan(k, day, et, netc + snfh))
        scans.sort(key=lambda s: (s.day, s.k))
        scans = [s for i, s in enumerate(scans) if i == 0 or s.day != scans[i - 1].day]
        known = [d for d in (num(r[c]) for c in day_cols) if d is not None] + [s.day for s in scans]
        progressed = r["Progression"].strip() == "1"
        out.append(Patient(pid, r["Primary Diagnosis"].strip(), surgery, progressed,
                           num(r[FIRST_PROGRESSION]) if progressed else None, r["Type of 1st Progression"].strip(),
                           r["Multiple surgeries"].strip() == "1", r["Overall Survival (Death)"].strip() == "1",
                           num(r[DEATH]), max(known) if known else float("nan"), num(r[RT_END]), scans))
    return out


def label_scans(days: list[float], fp: float | None, window: float = WINDOW_DAYS) -> list[str]:
    """PD: the scan closest to the first-progression day within the window (earlier on ties);
    non-PD: scans before the window, or every scan if no progression; excluded: the rest."""
    if fp is None:
        return ["non-PD"] * len(days)
    near = [i for i, d in enumerate(days) if abs(d - fp) <= window]
    pd_i = min(near, key=lambda i: (abs(days[i] - fp), days[i])) if near else None
    return ["PD" if i == pd_i else "non-PD" if d < fp - window else "excluded" for i, d in enumerate(days)]


def features(p: Patient, i: int) -> np.ndarray:
    """Two-compartment features of scan i given the patient's earlier MU scans; the segment
    starts at the first surgery (LUMIERE's segments start at the post-operative scan)."""
    ns = [SimpleNamespace(patient_id=p.pid, source_name=f"tp{s.k}", week=s.day / 7.0) for s in p.scans[:i + 1]]
    vol = {(p.pid, f"tp{s.k}"): (s.enh, s.non) for s in p.scans[:i + 1]}
    x = two_compartment_features(ns[:i], ns[i], vol)
    if p.surgery_day is not None:
        x[11] = (p.scans[i].day - p.surgery_day) / 7.0 / 52.0
    return x


def eligible(p: Patient, diagnosis: str | None) -> bool:
    return (diagnosis is None or p.diagnosis == diagnosis) and len(p.scans) >= 2 and not (p.progressed and p.fp_day is None)


def evaluable_calls(patients: list[Patient], models: dict) -> list[dict]:
    rows = []
    for p in patients:
        labels = label_scans([s.day for s in p.scans], p.fp_day)
        for i in range(1, len(p.scans)):
            if labels[i] == "excluded":
                continue
            x = features(p, i)
            s = p.scans[i]
            rows.append({"patient": p, "patient_id": p.pid, "timepoint": s.k, "day": s.day, "label": labels[i], "x": x,
                         **{src: float(m.predict_proba(x[None])[0, 1]) for src, m in models.items()},
                         "rule": rule_fires([h.enh for h in p.scans[:i]], s.enh),
                         "in_rt_window": None if p.rt_end is None else bool(s.day <= p.rt_end + RT_WINDOW_DAYS)})
    return rows


# ----------------------------------------------------------------------------- metrics


def discrimination(rows: list[dict], rng) -> dict:
    y = np.array([r["label"] == "PD" for r in rows], dtype=int)
    pid = np.array([r["patient_id"] for r in rows])
    x = np.stack([r["x"] for r in rows])
    cands = {**{MODEL_NAMES[src]: np.array([r[src] for r in rows]) for src in MODEL_NAMES},
             "enhancing change vs nadir": x[:, 4], "enhancing change vs previous": x[:, 2]}
    out = {"calls": len(rows), "pd": int(y.sum()), "patients": len(set(pid))}
    if 0 < y.sum() < len(y):
        out["auroc"] = auroc_boot(cands, y, pid, rng, REF)
    return out


def operating_points(rows: list[dict], frozen: dict) -> dict:
    y = np.array([r["label"] == "PD" for r in rows], dtype=bool)
    out = {}
    for src in MODEL_NAMES:
        p = np.array([r[src] for r in rows])
        thr = frozen["sources"][src]["thresholds_previous_not_pd"]
        t80, t90 = thr["specificity_80"]["threshold"], thr["sensitivity_90"]["threshold"]
        out[MODEL_NAMES[src]] = {
            "threshold_80pct_specificity": t80, "specificity_there": float((p[~y] < t80).mean()),
            "sensitivity_there": float((p[y] >= t80).mean()),
            "threshold_90pct_sensitivity": t90, "sensitivity_at_90pct_threshold": float((p[y] >= t90).mean()),
            "specificity_at_90pct_threshold": float((p[~y] < t90).mean()),
            "brier": float(np.mean((p - y) ** 2)), "mean_probability": float(p.mean())}
    fires = np.array([r["rule"] for r in rows])
    out["volume rule (>= 40% and >= 1 mL above nadir)"] = {"sensitivity": float(fires[y].mean()),
                                                           "specificity": float((~fires[~y]).mean())}
    out["pd_rate"] = float(y.mean())
    return out


def rt_window(rows: list[dict], frozen: dict, rng) -> dict:
    t80 = frozen["sources"][PRIMARY]["thresholds_previous_not_pd"]["specificity_80"]["threshold"]
    out = {}
    for name, flag in (("within 12 weeks after radiotherapy", True), ("later", False)):
        sub = [r for r in rows if r["in_rt_window"] is flag]
        y = np.array([r["label"] == "PD" for r in sub], dtype=bool)
        p = np.array([r[PRIMARY] for r in sub])
        res = {"calls": len(sub), "pd": int(y.sum()),
               "specificity_at_80pct_threshold": float((p[~y] < t80).mean()) if (~y).any() else None,
               "sensitivity_at_80pct_threshold": float((p[y] >= t80).mean()) if y.any() else None}
        if 0 < y.sum() < len(y):
            res["auroc"] = auroc_boot({REF: p}, y.astype(int), np.array([r["patient_id"] for r in sub]), rng, REF)[REF]
        out[name] = res
    out["no_radiotherapy_dates"] = sum(r["in_rt_window"] is None for r in rows)
    return out


def cox_fit_censored(entry, exit_, event, z_at_exits, weights=None, iters=50):
    """progression_survival.cox_fit with right censoring: only events enter the likelihood."""
    n = len(exit_)
    w = np.ones(n) if weights is None else weights
    at_risk = (entry[:, None] < exit_[None, :]) & (exit_[:, None] >= exit_[None, :])  # [j, i]
    z = np.asarray(z_at_exits, dtype=np.float64)
    k = z.shape[0]
    z_self = z[:, np.arange(n), np.arange(n)]
    ev = np.asarray(event, dtype=bool) & (w > 0)
    z, z_self, at_risk, w_ev = z[:, :, ev], z_self[:, ev], at_risk[:, ev], w[ev]

    def evaluate(beta):
        r = at_risk * w[:, None] * np.exp(np.einsum("k,kji->ji", beta, z))
        s0 = r.sum(axis=0)
        mean = np.einsum("kji,ji->ki", z, r) / s0
        ll = float((w_ev * ((beta @ z_self) - np.log(s0))).sum())
        grad = ((z_self - mean) * w_ev).sum(axis=1)
        info = ((np.einsum("kji,lji,ji->kli", z, z, r) / s0 - np.einsum("ki,li->kli", mean, mean)) * w_ev).sum(axis=2)
        return ll, grad, info

    beta = np.zeros(k)
    ll, grad, info = evaluate(beta)
    for _ in range(iters):
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


def concordance_censored(entry, exit_, event, z_at_exits, weights=None) -> float:
    """progression_survival.concordance with right censoring: pairs (subject i who dies, subject
    j still at risk after i's death); marker = progression status at i's death."""
    n = len(exit_)
    w = np.ones(n) if weights is None else weights
    later = (entry[:, None] < exit_[None, :]) & (exit_[:, None] > exit_[None, :])  # [j, i]
    zi = z_at_exits[np.arange(n), np.arange(n)][None, :]
    pair_w = later * w[:, None] * (w * np.asarray(event, dtype=float))[None, :]
    total = pair_w.sum()
    conc = ((zi > z_at_exits) * pair_w).sum() + 0.5 * ((zi == z_at_exits) * pair_w).sum()
    return float(conc / total) if total > 0 else float("nan")


def survival(patients: list[Patient], models: dict, rng) -> dict:
    keep, dropped = [], 0
    for p in patients:
        if p.died and (p.death_day is None or p.death_day < p.scans[-1].day):
            dropped += 1
            continue
        exit_ = p.death_day if p.died else p.last_day
        if exit_ > p.scans[0].day:
            keep.append(p)
        else:
            dropped += 1
    entry = np.array([p.scans[0].day for p in keep])
    exit_ = np.array([p.death_day if p.died else p.last_day for p in keep])
    event = np.array([p.died for p in keep])
    prog: dict[str, list] = {DOC: [], "volume rule": [], "automated call (P >= 0.5)": [], "documented progression (recorded day)": []}
    for p in keep:
        later = range(1, len(p.scans))
        prob = {i: float(models[PRIMARY].predict_proba(features(p, i)[None])[0, 1]) for i in later}
        prog[DOC].append(next((p.scans[i].day for i in later if p.fp_day is not None and p.scans[i].day >= p.fp_day - WINDOW_DAYS), None))
        prog["volume rule"].append(next((p.scans[i].day for i in later if rule_fires([s.enh for s in p.scans[:i]], p.scans[i].enh)), None))
        prog["automated call (P >= 0.5)"].append(next((p.scans[i].day for i in later if prob[i] >= 0.5), None))
        prog["documented progression (recorded day)"].append(p.fp_day)
    prog_arr = {d: np.array([np.nan if v is None else v for v in vals], dtype=float) for d, vals in prog.items()}
    Z = {d: z_matrix(v, exit_) for d, v in prog_arr.items()}
    n = len(keep)
    weights = [np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64) for _ in range(N_BOOT)]
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]  # noqa: E731
    c_ref = np.array([concordance_censored(entry, exit_, event, Z[DOC], w) for w in weights])
    c_ref_point = concordance_censored(entry, exit_, event, Z[DOC])
    out = {"patients": n, "deaths": int(event.sum()), "excluded_inconsistent_or_no_follow_up": dropped,
           "median_follow_up_days_censored": float(np.median((exit_ - entry)[~event])) if (~event).any() else None,
           "definitions": {}}
    for d, v in prog_arr.items():
        done = np.isfinite(v)
        c_draws = np.array([concordance_censored(entry, exit_, event, Z[d], w) for w in weights])
        c = concordance_censored(entry, exit_, event, Z[d])
        hr = float(np.exp(cox_fit_censored(entry, exit_, event, Z[d][None])[0]))
        hr_draws = [np.exp(cox_fit_censored(entry, exit_, event, Z[d][None], w)[0]) for w in weights]
        out["definitions"][d] = {
            "progressed": int(done.sum()), "median_days_entry_to_call": float(np.median((v - entry)[done])) if done.any() else None,
            "hr": hr, "hr_ci95": ci(hr_draws), "c_index": c, "c_index_ci95": ci(c_draws),
            "c_minus_documented": c - c_ref_point, "c_minus_documented_ci95": ci(c_draws - c_ref)}
    return out


def analyse(clinical: list[dict], volumes: dict, models: dict, frozen: dict, seed: int, full: bool = True) -> dict:
    patients = build_patients(clinical, volumes)
    gbm = [p for p in patients if eligible(p, "GBM")]
    rows = evaluable_calls(gbm, models)
    res = {"cohort": {"patients_in_table": len(patients), "gbm_patients": sum(p.diagnosis == "GBM" for p in patients),
                      "gbm_with_two_scans": len(gbm),
                      "gbm_progressed_without_day": sum(p.diagnosis == "GBM" and len(p.scans) >= 2 and p.progressed
                                                        and p.fp_day is None for p in patients)},
           "discrimination": discrimination(rows, np.random.default_rng(seed))}
    if not full:
        return res
    res["operating_points"] = operating_points(rows, frozen)
    rng = np.random.default_rng(seed + 1)
    subsets = {
        "imaging-documented progression only (types 1, 3)": [r for r in rows if r["patient"].prog_type != "2"],
        "no re-operation": [r for r in rows if not r["patient"].reoperated],
        "PD scans on the exact progression day only": [r for r in rows if r["label"] != "PD" or r["day"] == r["patient"].fp_day],
        "all glioma types": evaluable_calls([p for p in patients if eligible(p, None)], models),
    }
    res["sensitivity_analyses"] = {k: discrimination(v, rng) for k, v in subsets.items()}
    res["radiotherapy_window"] = rt_window(rows, frozen, np.random.default_rng(seed + 2))
    res["survival"] = survival(gbm, models, np.random.default_rng(seed + 3))
    res["_rows"] = rows
    return res


# ----------------------------------------------------------------------------- report


def fmt(v: dict) -> str:
    return f"{v['auroc']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"


def write_markdown(res: dict, frozen: dict, path: Path) -> None:
    d, c = res["discrimination"], res["cohort"]
    bench = frozen["sources"][PRIMARY]["held_out_auroc_previous_not_pd"]["model"]
    L = ["# External validation on MU-Glioma-Post", "",
         "Pre-specified in PLAN.md; model and thresholds frozen on LUMIERE (frozen.json). "
         f"95% CIs from {N_BOOT} patient resamples. PD = the scan at the documented first progression; "
         "non-PD = earlier scans and all scans of patients without progression.", "",
         f"Cohort: {c['gbm_patients']} GBM patients, {c['gbm_with_two_scans']} with at least two scans; "
         f"{d['calls']} evaluable calls from {d['patients']} patients, {d['pd']} PD "
         f"({c['gbm_progressed_without_day']} progressed patients without a recorded day excluded).", "",
         "## 1. Discrimination", "",
         "| score | AUROC [95% CI] | minus primary [95% CI] |", "| :--- | :---: | :---: |"]
    for k, v in d["auroc"].items():
        L.append(f"| {k} | {fmt(v)} | {v[f'minus_{REF}']:+.3f} [{v[f'minus_{REF}_ci95'][0]:+.3f}, {v[f'minus_{REF}_ci95'][1]:+.3f}] |")
    L += ["", f"LUMIERE benchmark (same model, held out, previous scan not PD): {fmt(bench)}.", "",
          "## 2. Operating points at the frozen LUMIERE thresholds", "",
          f"PD rate {res['operating_points']['pd_rate']:.2f}.", "",
          "| model | specificity at the 80%-spec threshold | sensitivity there | sensitivity at the 90%-sens threshold | specificity there | Brier | mean P(PD) |",
          "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in MODEL_NAMES.values():
        o = res["operating_points"][name]
        L.append(f"| {name} | {o['specificity_there']:.2f} | {o['sensitivity_there']:.2f} | {o['sensitivity_at_90pct_threshold']:.2f} | "
                 f"{o['specificity_at_90pct_threshold']:.2f} | {o['brier']:.3f} | {o['mean_probability']:.2f} |")
    vr = res["operating_points"]["volume rule (>= 40% and >= 1 mL above nadir)"]
    L += ["", f"Volume rule as a binary call: sensitivity {vr['sensitivity']:.2f}, specificity {vr['specificity']:.2f}.", "",
          "## 3. Sensitivity analyses", "",
          f"| subset | calls (PD) | {MODEL_NAMES['dbt_atlas']} | {MODEL_NAMES['hd_native']} | enhancing change vs nadir |",
          "| :--- | ---: | :---: | :---: | :---: |"]
    for k, v in res["sensitivity_analyses"].items():
        a = v.get("auroc")
        cells = [fmt(a[n]) if a else "n/a" for n in (MODEL_NAMES["dbt_atlas"], MODEL_NAMES["hd_native"], "enhancing change vs nadir")]
        L.append(f"| {k} | {v['calls']} ({v['pd']}) | " + " | ".join(cells) + " |")
    rt = res["radiotherapy_window"]
    L += ["", "## 4. Radiotherapy window (primary model)", "",
          "| scans | calls (PD) | specificity at the 80%-spec threshold | sensitivity there | AUROC [95% CI] |",
          "| :--- | ---: | ---: | ---: | :---: |"]
    for k in ("within 12 weeks after radiotherapy", "later"):
        v = rt[k]
        sp = "n/a" if v["specificity_at_80pct_threshold"] is None else f"{v['specificity_at_80pct_threshold']:.2f}"
        se = "n/a" if v["sensitivity_at_80pct_threshold"] is None else f"{v['sensitivity_at_80pct_threshold']:.2f}"
        L.append(f"| {k} | {v['calls']} ({v['pd']}) | {sp} | {se} | {fmt(v['auroc']) if 'auroc' in v else 'n/a'} |")
    L.append(f"\n{rt['no_radiotherapy_dates']} calls have no radiotherapy dates.")
    s = res["survival"]
    L += ["", "## 5. Survival: does the volume rule beat documented progression? (time-dependent, censoring-aware)", "",
          f"{s['patients']} patients, {s['deaths']} deaths ({s['excluded_inconsistent_or_no_follow_up']} excluded: death before "
          f"last scan or no follow-up). Time origin diagnosis; entry at the first MU scan; censored at the last documented day.", "",
          "| definition | progressed | median days entry -> call | HR [95% CI] | C-index [95% CI] | C minus documented (at scans) [95% CI] |",
          "| :--- | ---: | ---: | :---: | :---: | :---: |"]
    for k, v in s["definitions"].items():
        md = "n/a" if v["median_days_entry_to_call"] is None else f"{v['median_days_entry_to_call']:.0f}"
        L.append(f"| {k} | {v['progressed']} | {md} | {v['hr']:.2f} [{v['hr_ci95'][0]:.2f}, {v['hr_ci95'][1]:.2f}] | "
                 f"{v['c_index']:.3f} [{v['c_index_ci95'][0]:.3f}, {v['c_index_ci95'][1]:.3f}] | "
                 f"{v['c_minus_documented']:+.3f} [{v['c_minus_documented_ci95'][0]:+.3f}, {v['c_minus_documented_ci95'][1]:+.3f}] |")
    chk = res.get("volume_source_check")
    if chk:
        L += ["", "## 6. Volume source check (TCIA spreadsheet vs masks)", "",
              f"Spreadsheet rows mapped to scans: {chk['mapping']}. Of {chk['compared_scans']} scans in both sources, "
              f"all four volumes agree exactly for {chk['exact_match_scans']}.",
              f"Endpoints on spreadsheet volumes: {chk['discrimination']['calls']} calls ({chk['discrimination']['pd']} PD); "
              + ("primary AUROC " + fmt(chk['discrimination']['auroc'][REF]) if "auroc" in chk["discrimination"] else "too few calls") + "."]
    path.write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = REPO / "data/mu_glioma_post"
    ap.add_argument("--clinical", type=Path, default=base / "MU-Glioma-Post_ClinicalData-July2025.xlsx")
    ap.add_argument("--spreadsheet", type=Path, default=base / "MU-Glioma-Post_Segmentation_Volumes.xlsx")
    ap.add_argument("--package-dir", type=Path, default=base / "package")
    ap.add_argument("--output-dir", type=Path, default=REPO / "results/mu_validation")
    ap.add_argument("--volumes", choices=("masks", "spreadsheet"), default="masks")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    frozen = json.loads((args.output_dir / "frozen.json").read_text())
    if sha256(args.output_dir / "PLAN.md") != frozen["plan_sha256"]:
        raise RuntimeError("PLAN.md changed after the freeze")
    model_file = REPO / frozen["model_file"]
    if sha256(model_file) != frozen["model_sha256"]:
        raise RuntimeError(f"{model_file} is not the frozen model")
    with model_file.open("rb") as h:  # our own file, verified by hash above
        models = pickle.load(h)["models"]
    clinical = [r for r in records(args.clinical, CLINICAL_SHEET) if r["Patient_ID"].strip()]

    masks = None
    if (args.output_dir / "mask_volumes.csv").exists() or args.package_dir.exists():
        masks = load_mask_volumes(args.package_dir, args.output_dir / "mask_volumes.csv", args.workers)
    if masks is not None:
        tps = {}
        for pid, k in sorted(masks):
            tps.setdefault(pid, []).append(k)
    else:
        tps = {r["Patient_ID"].strip(): [k for k in ORDINAL if num(r[tp_column(k)]) is not None] for r in clinical}
    sheet, mapping = spreadsheet_volumes(args.spreadsheet, tps)
    if args.volumes == "masks" and masks is None:
        raise FileNotFoundError("no mask package or mask_volumes.csv; use --volumes spreadsheet")
    main_vols = masks if args.volumes == "masks" else sheet

    res = analyse(clinical, main_vols, models, frozen, seed=20261117)
    res["volume_source"] = args.volumes
    if masks is not None:
        both = sorted(set(masks) & set(sheet))
        res["volume_source_check"] = {
            "mapping": mapping, "compared_scans": len(both),
            "exact_match_scans": sum(np.allclose(masks[k], sheet[k], atol=0.5) for k in both),
            "discrimination": analyse(clinical, sheet, models, frozen, seed=20261117, full=False)["discrimination"]}
    rows = res.pop("_rows")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if args.volumes == "masks" else "_spreadsheet"
    with (args.output_dir / f"predictions{tag}.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["patient_id", "timepoint", "day", "label", "p_primary", "p_hd_trained", "enhancing_ml", "nonenhancing_ml",
                    "enhancing_change_vs_nadir", "volume_rule", "within_12_weeks_of_radiotherapy"])
        for r in rows:
            w.writerow([r["patient_id"], r["timepoint"], r["day"], r["label"], f"{r['dbt_atlas']:.4f}", f"{r['hd_native']:.4f}",
                        f"{np.expm1(r['x'][0]):.3f}", f"{np.expm1(r['x'][1]):.3f}", f"{r['x'][4]:.4f}", int(r["rule"]), r["in_rt_window"]])
    (args.output_dir / f"metrics{tag}.json").write_text(json.dumps(res, indent=2, allow_nan=False) + "\n")
    write_markdown(res, frozen, args.output_dir / f"metrics{tag}.md")
    print((args.output_dir / f"metrics{tag}.md").read_text())


if __name__ == "__main__":
    main()
