"""Two follow-up analyses of step 2 (no training, nothing re-registered).

ceiling  How good can any next-visit mask forecast be, given how noisy the target masks are?
         (a) Same scan, two segmenters: every step-2 target scan's DeepBraTumIA atlas mask
             (the label step 2 is scored against) vs LUMIERE's HD-GLIO-AUTO mask of the same
             scan, scored on the same 2 mm atlas grid and > 0.5 rule as step 2.
         (b) Stable short intervals: copy-last-mask Dice on transitions the expert rated SD,
             from the registered-frame test predictions (computed in `report`).
         LUMIERE's same-week duplicates are pre-op / post-op pairs, so there is no true
         test-retest scan pair.

         Getting HD-GLIO into the atlas: LUMIERE's HD-GLIO native-space files are the
         dataset authors' own back-transform and are mis-oriented for many scans (the mask
         lands outside the brain or on the wrong side). So the native files are not used.
         HD-GLIO's ``registered/`` mask shares a grid with HD-GLIO's own CT1, and that CT1 is
         registered to DeepBraTumIA's atlas CT1 (Mattes MI; rigid from 4 starting
         orientations, lowest metric wins; then affine). The registration only sees the two
         CT1 images, never a tumor mask, so it cannot inflate agreement. QC: brain-mask Dice.

report   Writes results/step2_followups/: noise ceiling table, and step 2 results split by
         how much the tumor actually changed (expert RANO rating, volume change, overlap
         change, interval), with patient-bootstrap CIs. Reads the test predictions in
         runs/mask_forecast{,_registered}/ and ceiling.csv.

  .venv/bin/python scripts/step2_followup_analysis.py ceiling   # ~10 min, 10 CPU workers (tmux)
  .venv/bin/python scripts/step2_followup_analysis.py report
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lumiere_cohort import build_cohort  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/step2_followups"
REGIONS = ("wt", "tc", "et")


# ----------------------------------------------------------------------------- ceiling


def pool2_zyx(v: np.ndarray) -> np.ndarray:
    z, y, x = (s // 2 * 2 for s in v.shape)
    v = v[:z, :y, :x]
    return v.reshape(z // 2, 2, y // 2, 2, x // 2, 2).mean(axis=(1, 3, 5))


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * (a & b).sum() / s)


def register_to_atlas(sitk, moving, fixed):
    """Rigid (4 starting orientations, best Mattes MI wins) then affine; image-only, no tumor labels."""
    import math

    fix = sitk.Cast(fixed, sitk.sitkFloat32)
    mov = sitk.Cast(moving, sitk.sitkFloat32)
    base = sitk.CenteredTransformInitializer(fix, mov, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)

    def run(init, iters, shrink, sigma):
        r = sitk.ImageRegistrationMethod()
        r.SetMetricAsMattesMutualInformation(32)
        r.SetMetricSamplingStrategy(r.RANDOM)
        r.SetMetricSamplingPercentage(0.2, seed=1)
        r.SetInterpolator(sitk.sitkLinear)
        r.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, iters, relaxationFactor=0.6)
        r.SetOptimizerScalesFromPhysicalShift()
        r.SetShrinkFactorsPerLevel(shrink)
        r.SetSmoothingSigmasPerLevel(sigma)
        r.SetInitialTransform(init, inPlace=False)
        t = r.Execute(fix, mov)
        return t, r.GetMetricValue()

    best = None
    for name, rot in (("as_header", (0, 0, 0)), ("rot_x", (math.pi, 0, 0)), ("rot_y", (0, math.pi, 0)), ("rot_z", (0, 0, math.pi))):
        init = sitk.Euler3DTransform(base)
        init.SetRotation(*rot)
        t, m = run(init, 150, [4, 2], [2, 1])
        if best is None or m < best[2]:
            best = (name, t, m)
    name, rigid, _ = best
    aff = sitk.AffineTransform(3)
    flat = rigid.GetNthTransform(0) if isinstance(rigid, sitk.CompositeTransform) else rigid
    flat = sitk.Euler3DTransform(flat)
    aff.SetCenter(flat.GetCenter())
    aff.SetMatrix(flat.GetMatrix())
    aff.SetTranslation(flat.GetTranslation())
    final, metric = run(aff, 100, [2], [1])
    return name, final, metric


def ceiling_one(job: dict) -> dict:
    import SimpleITK as sitk

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)  # one thread per worker process
    d = Path(job["dir"])
    row = {k: job[k] for k in ("patient_id", "source", "rating")}
    p_atlas = d / "DeepBraTumIA-segmentation/atlas/segmentation/seg_mask.nii.gz"
    p_atlas_ct1 = d / "DeepBraTumIA-segmentation/atlas/skull_strip/ct1_skull_strip.nii.gz"
    p_atlas_brain = d / "DeepBraTumIA-segmentation/atlas/skull_strip/brain_mask.nii.gz"
    p_dbt_nat = d / "DeepBraTumIA-segmentation/native/segmentation/ct1_seg_mask.nii.gz"
    p_tfm = d / "DeepBraTumIA-segmentation/native/transformation/ct1.tfm"
    p_hdg_ct1 = d / "HD-GLIO-AUTO-segmentation/registered/CT1_r2s_bet_reg.nii.gz"
    p_hdg_seg = d / "HD-GLIO-AUTO-segmentation/registered/segmentation.nii.gz"
    missing = [p.name for p in (p_atlas, p_atlas_ct1, p_atlas_brain, p_dbt_nat, p_tfm, p_hdg_ct1, p_hdg_seg) if not p.exists()]
    if missing:
        return {**row, "status": "missing:" + ",".join(missing)}

    atlas = sitk.ReadImage(str(p_atlas))
    nn = sitk.sitkNearestNeighbor
    # resampling floor: DeepBraTumIA's own native mask pushed through its native->atlas transform
    to_atlas = sitk.ReadTransform(str(p_tfm)).GetInverse()  # verified: maps atlas points to native
    dbt_round = sitk.GetArrayFromImage(sitk.Resample(sitk.ReadImage(str(p_dbt_nat)), atlas, to_atlas, nn, 0)).astype(np.int16)

    # HD-GLIO: its segmentation shares a grid with its own CT1, so register that CT1 to the atlas CT1
    hdg_ct1 = sitk.ReadImage(str(p_hdg_ct1))
    hdg_seg = sitk.ReadImage(str(p_hdg_seg))
    init_name, tf, metric = register_to_atlas(sitk, hdg_ct1, sitk.ReadImage(str(p_atlas_ct1)))
    hdg_atlas = sitk.GetArrayFromImage(sitk.Resample(hdg_seg, atlas, tf, nn, 0)).astype(np.int16)
    hdg_brain = sitk.GetArrayFromImage(sitk.Resample(sitk.Cast(hdg_ct1 > 0, sitk.sitkUInt8), atlas, tf, nn, 0)) > 0
    atlas_brain = sitk.GetArrayFromImage(sitk.ReadImage(str(p_atlas_brain))) > 0

    dbt_atlas = sitk.GetArrayFromImage(atlas).astype(np.int16)

    def masks2mm(seg, labels):
        return pool2_zyx(np.isin(seg, labels).astype(np.float32)) > 0.5

    ref_wt, ref_wt_noncr, ref_et = masks2mm(dbt_atlas, (1, 2, 3)), masks2mm(dbt_atlas, (1, 3)), masks2mm(dbt_atlas, (1,))
    h_wt, h_et = masks2mm(hdg_atlas, (1, 2)), masks2mm(hdg_atlas, (2,))
    r_wt, r_et = masks2mm(dbt_round, (1, 2, 3)), masks2mm(dbt_round, (1,))
    ml = 8.0 / 1000.0
    row.update(
        status="ok",
        hdg_init=init_name,
        hdg_metric=float(metric),
        hdg_brain_dice=dice(atlas_brain, hdg_brain),
        dice_wt_dbt_vs_hdg=dice(ref_wt, h_wt),
        dice_wt_nonecrosis_dbt_vs_hdg=dice(ref_wt_noncr, h_wt),
        dice_et_dbt_vs_hdg=dice(ref_et, h_et),
        dice_wt_dbt_roundtrip=dice(ref_wt, r_wt),
        dice_et_dbt_roundtrip=dice(ref_et, r_et),
        vol_wt_dbt_ml=float(ref_wt.sum() * ml),
        vol_wt_hdg_ml=float(h_wt.sum() * ml),
        vol_et_dbt_ml=float(ref_et.sum() * ml),
        vol_et_hdg_ml=float(h_et.sum() * ml),
    )
    return row


def ceiling(args) -> None:
    cohort = build_cohort(args.data_dir, args.clinical_dir / "LUMIERE-ExpertRating-v202211.csv", require_seg_mask=True)
    seen, jobs = set(), []
    for t in cohort.transitions:
        key = (t.patient_id, t.target.source_name)
        if key not in seen:
            seen.add(key)
            jobs.append({"patient_id": t.patient_id, "source": t.target.source_name,
                         "rating": t.target.rating or "", "dir": str(t.target.scan.directory)})
    print(f"{len(jobs)} step-2 target scans", flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(ceiling_one, jobs, chunksize=4)):
            rows.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("patient_id", "source", "rating", "status"), k))
    with (OUT / "ceiling.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"ok {len(ok)}/{len(rows)}; HD-GLIO brain Dice median {np.median([r['hdg_brain_dice'] for r in ok]):.3f}")


# ----------------------------------------------------------------------------- report


def load_preds(frame: str, variant: str) -> list[dict]:
    sub = "mask_forecast" if frame == "atlas" else "mask_forecast_registered"
    rows = []
    for f in range(5):
        p = REPO / "runs" / sub / variant / f"fold{f}" / "test_predictions.csv"
        for r in csv.DictReader(p.open()):
            r["fold"] = f
            rows.append(r)
    return rows


def key(r: dict) -> tuple:
    return (r["patient_id"], r["last_source"], r["target_source"])


def patient_boot(vals: np.ndarray, pids: np.ndarray, rng, n: int = 2000) -> tuple[float, float, float, int]:
    """Mean over patients of per-patient mean; 95% bootstrap CI over patients."""
    ps = sorted(set(pids))
    d = np.array([vals[pids == p].mean() for p in ps])
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), len(ps)


def summ(x: np.ndarray) -> dict:
    x = x[~np.isnan(x)]
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
            "q25": float(np.percentile(x, 25)), "q75": float(np.percentile(x, 75))}


def report(args) -> None:
    rng = np.random.default_rng(0)
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    md: list[str] = []

    # --- stratifiers, from the registered frame (copy-last-mask in the aligned frame)
    reg = {key(r): r for r in load_preds("registered", "ode")}
    f = lambda r, c: float(r[c])  # noqa: E731
    strat: dict[tuple, dict] = {}
    pdice = np.array([f(r, "persistence_dice_wt") for r in reg.values()])
    q1, q2 = np.percentile(pdice, [100 / 3, 200 / 3])
    for k, r in reg.items():
        last, true = f(r, "persistence_vol_wt_ml"), f(r, "model_true_vol_wt_ml")
        ratio = (true + 1) / (last + 1)
        pd_ = f(r, "persistence_dice_wt")
        dt = int(r["dt_weeks"])
        strat[k] = {
            "RANO rating of target": r["target_rating"] or "unrated",
            "whole-tumor volume change": "shrank >20%" if ratio < 0.8 else ("grew >25%" if ratio > 1.25 else "within -20%/+25%"),
            "overlap with last mask": (f"low (<{q1:.2f})" if pd_ < q1 else (f"high (>={q2:.2f})" if pd_ >= q2 else "middle")),
            "interval": "<=8 wk" if dt <= 8 else ("9-16 wk" if dt <= 16 else ">16 wk"),
        }
    order = {
        "RANO rating of target": ["PD", "SD", "PR", "CR", "unrated"],
        "whole-tumor volume change": ["shrank >20%", "within -20%/+25%", "grew >25%"],
        "overlap with last mask": [f"low (<{q1:.2f})", "middle", f"high (>={q2:.2f})"],
        "interval": ["<=8 wk", "9-16 wk", ">16 wk"],
    }

    # --- (1) noise ceiling
    cpath = OUT / "ceiling.csv"
    ceil = [r for r in csv.DictReader(cpath.open()) if r["status"] == "ok"] if cpath.exists() else []
    md += ["# Step 2 follow-ups: noise ceiling and stratified results", "",
           "Generated by `scripts/step2_followup_analysis.py report`.", "",
           "## 1. Noise ceiling: how well can the target masks be predicted at all?", ""]
    if ceil:
        arr = lambda c: np.array([float(r[c]) for r in ceil])  # noqa: E731
        pid_c = np.array([r["patient_id"] for r in ceil])
        both_et = np.array([(float(r["vol_et_dbt_ml"]) > 0) or (float(r["vol_et_hdg_ml"]) > 0) for r in ceil])
        rows_c = []
        for label, col, mask in (
            ("DeepBraTumIA vs its own native mask resampled to atlas (resampling floor)", "dice_wt_dbt_roundtrip", None),
            ("DeepBraTumIA vs HD-GLIO, same scan, whole tumor", "dice_wt_dbt_vs_hdg", None),
            ("  same, DeepBraTumIA whole tumor without necrosis/cavity", "dice_wt_nonecrosis_dbt_vs_hdg", None),
            ("DeepBraTumIA vs HD-GLIO, same scan, enhancing (either has ET)", "dice_et_dbt_vs_hdg", both_et),
        ):
            x = arr(col) if mask is None else arr(col)[mask]
            p = pid_c if mask is None else pid_c[mask]
            m, lo, hi, npat = patient_boot(x, p, rng)
            s = summ(x)
            rows_c.append((label, s, (m, lo, hi, npat)))
            res.setdefault("ceiling", {})[col + ("" if mask is None else "_either_et")] = {**s, "patient_mean": m, "ci95": [lo, hi], "patients": npat}
        qc = arr("hdg_brain_dice")
        good = qc >= 0.9
        for col in ("dice_wt_dbt_vs_hdg", "dice_wt_nonecrosis_dbt_vs_hdg", "dice_et_dbt_vs_hdg"):
            msk = good & (both_et if "et_dbt" in col else True)
            m, lo, hi, npat = patient_boot(arr(col)[msk], pid_c[msk], rng)
            res["ceiling"][col + "_qc_pass"] = {**summ(arr(col)[msk]), "patient_mean": m, "ci95": [lo, hi], "patients": npat}
            rows_c.append(("  robustness: only scans with HD-GLIO brain Dice >= 0.9, " + col.split("_")[1].replace("wt", "whole tumor").replace("et", "enhancing")
                           + (" (no necrosis)" if "nonecrosis" in col else ""), summ(arr(col)[msk]), (m, lo, hi, npat)))
        inits = {k: int(sum(r["hdg_init"] == k for r in ceil)) for k in sorted({r["hdg_init"] for r in ceil})}
        res["ceiling"]["hdg_registration"] = {"brain_dice_median": float(np.median(qc)), "brain_dice_p10": float(np.percentile(qc, 10)),
                                              "below_0.9": int((~good).sum()), "chosen_init": inits}
        vol = np.array([[float(r["vol_wt_dbt_ml"]), float(r["vol_wt_hdg_ml"])] for r in ceil])
        res["ceiling"]["median_wt_volume_ml"] = {"deepbratumia": float(np.median(vol[:, 0])), "hdglio": float(np.median(vol[:, 1]))}
        md += [f"(a) Same scan, two segmenters. {len(ceil)} step-2 target scans; 2 mm atlas grid, > 0.5 rule, as in step 2.",
               f"HD-GLIO -> atlas registration QC: brain-mask Dice median {np.median(qc):.3f} (10th pct {np.percentile(qc, 10):.3f}; "
               f"{int((~good).sum())} scans below 0.9); starting orientation chosen: {inits}.",
               f"Median whole-tumor volume: DeepBraTumIA {np.median(vol[:, 0]):.1f} mL, HD-GLIO {np.median(vol[:, 1]):.1f} mL "
               "(HD-GLIO has no necrosis/cavity label).", "",
               "| comparison | scans | median Dice [IQR] | patient mean [95% CI] |", "| :--- | ---: | :---: | :---: |"]
        for label, s, (m, lo, hi, npat) in rows_c:
            md.append(f"| {label} | {s['n']} | {s['median']:.3f} [{s['q25']:.3f}, {s['q75']:.3f}] | {m:.3f} [{lo:.3f}, {hi:.3f}] |")
        md.append("")
    # (b) stable short intervals, registered frame
    md += ["(b) Copy-last-mask Dice in the registered frame, by expert rating and interval.", "",
           "| subset | transitions | patients | WT median [IQR] | WT mean | TC mean | ET mean |", "| :--- | ---: | ---: | :---: | ---: | ---: | ---: |"]
    regl = list(reg.values())
    subsets = [
        ("SD, interval <= 12 wk", lambda r: r["target_rating"] == "SD" and int(r["dt_weeks"]) <= 12),
        ("SD, any interval", lambda r: r["target_rating"] == "SD"),
        ("PD", lambda r: r["target_rating"] == "PD"),
        ("all step-2 transitions", lambda r: True),
    ]
    for label, cond in subsets:
        rs = [r for r in regl if cond(r)]
        wt = np.array([f(r, "persistence_dice_wt") for r in rs])
        s = summ(wt)
        means = {g: float(np.mean([f(r, f"persistence_dice_{g}") for r in rs])) for g in REGIONS}
        res.setdefault("stable_pairs", {})[label] = {**s, **{f"mean_{g}": v for g, v in means.items()},
                                                     "patients": len({r["patient_id"] for r in rs})}
        md.append(f"| {label} | {len(rs)} | {len({r['patient_id'] for r in rs})} | {s['median']:.3f} [{s['q25']:.3f}, {s['q75']:.3f}] | "
                  f"{means['wt']:.3f} | {means['tc']:.3f} | {means['et']:.3f} |")
    md.append("")

    # --- (2) stratified step 2 results
    md += ["## 2. Step 2 split by how much the tumor actually changed", "",
           "Strata are defined once, from the registered frame, and applied to every frame/variant.",
           "Dice diff = model minus copy-last-mask, mean of WT/TC/ET (as in the main table); change-Dice diff likewise.",
           "Per-patient means, 95% bootstrap CI over patients in the stratum. 'pred / true dlogV' = mean",
           "log((V+1)/(V_last+1)) of the forecast vs of the real next scan, whole tumor.", ""]
    res["strata"] = {}
    for frame in ("registered", "atlas"):
        for variant in ("ode", "static"):
            rows = load_preds(frame, variant)
            name = f"{frame} frame, {'Neural ODE' if variant == 'ode' else 'no-ODE'}"
            res["strata"][f"{frame}_{variant}"] = {}
            md += [f"### {name}", "",
                   "| stratifier | stratum | trans. | pat. | WT Dice model / copy | Dice diff [95% CI] | change-Dice diff [95% CI] | pred / true dlogV |",
                   "| :--- | :--- | ---: | ---: | :---: | :---: | :---: | :---: |"]
            for sname, levels in order.items():
                for lev in levels:
                    rs = [r for r in rows if strat[key(r)][sname] == lev]
                    if len(rs) < 3:
                        continue
                    pids = np.array([r["patient_id"] for r in rs])
                    dd = np.array([np.mean([f(r, f"model_dice_{g}") - f(r, f"persistence_dice_{g}") for g in REGIONS]) for r in rs])
                    cd = np.array([np.mean([f(r, f"model_change_dice_{g}") - f(r, f"persistence_change_dice_{g}") for g in REGIONS]) for r in rs])
                    m1 = patient_boot(dd, pids, rng)
                    m2 = patient_boot(cd, pids, rng)
                    last = np.array([f(r, "persistence_vol_wt_ml") for r in rs])
                    pdv = np.mean(np.log((np.array([f(r, "model_vol_wt_ml") for r in rs]) + 1) / (last + 1)))
                    tdv = np.mean(np.log((np.array([f(r, "model_true_vol_wt_ml") for r in rs]) + 1) / (last + 1)))
                    mw = np.mean([f(r, "model_dice_wt") for r in rs])
                    pw = np.mean([f(r, "persistence_dice_wt") for r in rs])
                    res["strata"][f"{frame}_{variant}"][f"{sname}|{lev}"] = {
                        "transitions": len(rs), "patients": m1[3], "model_dice_wt": float(mw), "persistence_dice_wt": float(pw),
                        "dice_diff": m1[0], "dice_diff_ci95": [m1[1], m1[2]],
                        "change_dice_diff": m2[0], "change_dice_diff_ci95": [m2[1], m2[2]],
                        "pred_dlogv": float(pdv), "true_dlogv": float(tdv)}
                    md.append(f"| {sname} | {lev} | {len(rs)} | {m1[3]} | {mw:.3f} / {pw:.3f} | {m1[0]:+.3f} [{m1[1]:+.3f}, {m1[2]:+.3f}] | "
                              f"{m2[0]:+.3f} [{m2[1]:+.3f}, {m2[2]:+.3f}] | {pdv:+.2f} / {tdv:+.2f} |")
            # does the forecast know the direction of change? (rows where the model is not a copy)
            moved = [r for r in rows if abs(f(r, "model_vol_wt_ml") - f(r, "persistence_vol_wt_ml")) > 1e-6]
            if len(moved) > 10:
                from scipy.stats import rankdata, spearmanr

                last = np.array([f(r, "persistence_vol_wt_ml") for r in moved])
                pdv = np.log((np.array([f(r, "model_vol_wt_ml") for r in moved]) + 1) / (last + 1))
                tdv = np.log((np.array([f(r, "model_true_vol_wt_ml") for r in moved]) + 1) / (last + 1))
                rho = float(spearmanr(pdv, tdv).statistic)
                small = -np.log1p(last)  # trivial rule: the smaller the last tumor, the more it grows
                rho_small = float(spearmanr(small, tdv).statistic)
                rk = lambda y: rankdata(y) - rankdata(y).mean()  # noqa: E731
                resid = lambda y: rk(y) - np.polyval(np.polyfit(rk(small), rk(y), 1), rk(small))  # noqa: E731
                partial = float(np.corrcoef(resid(pdv), resid(tdv))[0, 1])
                res["strata"][f"{frame}_{variant}"]["direction"] = {
                    "transitions": len(moved), "spearman_pred_vs_true_dlogv": rho,
                    "spearman_small_last_volume_vs_true_dlogv": rho_small,
                    "partial_spearman_given_last_volume": partial,
                    "frac_pred_growth": float(np.mean(pdv > 0)), "frac_true_growth": float(np.mean(tdv > 0))}
                md += ["", f"Direction of whole-tumor volume change, {len(moved)} transitions where the forecast differs from the last mask: "
                       f"forecast grows in {np.mean(pdv > 0):.0%} (truth: {np.mean(tdv > 0):.0%}). "
                       f"Spearman(forecast, true change) = {rho:+.2f}, but the last volume alone (smaller grows more) gives {rho_small:+.2f}; "
                       f"forecast vs truth after removing last volume: {partial:+.2f}."]
            md.append("")
    (OUT / "summary.json").write_text(json.dumps(res, indent=2))
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    figure(res, order)


def figure(res: dict, order: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    series = (("registered_ode", "Neural ODE", "#2a78d6", -0.14), ("registered_static", "no-ODE", "#8a8984", 0.14))
    labels, ys, y = [], [], 0.0
    for sname, levels in order.items():
        for lev in levels:
            if f"{sname}|{lev}" in res["strata"]["registered_ode"]:
                labels.append((sname, lev))
                ys.append(y)
                y += 1
        y += 0.6
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.36 * len(labels) + 1.6), dpi=200, sharey=True)
    for ax, metric, title in ((axes[0], "dice_diff", "Dice, model minus copy-last-mask"),
                              (axes[1], "change_dice_diff", "change-Dice, model minus copy-last-mask")):
        for k, name, color, off in series:
            for (sname, lev), yy in zip(labels, ys):
                v = res["strata"][k][f"{sname}|{lev}"]
                lo, hi = v[f"{metric}_ci95"]
                ax.plot([lo, hi], [yy + off] * 2, color=color, linewidth=2, solid_capstyle="round")
                ax.plot(v[metric], yy + off, "o", color=color, markersize=6, markeredgecolor="white", markeredgewidth=1.5,
                        label=name if (sname, lev) == labels[0] else None)
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    n = {lab: res["strata"]["registered_ode"][f"{lab[0]}|{lab[1]}"]["transitions"] for lab in labels}
    axes[0].set_yticks(ys, [f"{lev}  (n={n[(s, lev)]})" for s, lev in labels], color=INK)
    axes[0].invert_yaxis()
    prev = None
    for (sname, _), yy in zip(labels, ys):
        if sname != prev:
            axes[0].annotate(sname, xy=(0, yy - 0.55), xycoords=("axes fraction", "data"), xytext=(-4, 0),
                             textcoords="offset points", ha="right", fontsize=8, color=MUTED, style="italic")
            prev = sname
    axes[1].legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("Step 2, registered frame - gain over copying the last mask, by how the tumor changed (95% CI over patients)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = REPO / "results/figures/step2_strata_registered.png"
    fig.savefig(out)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("ceiling", "report"))
    p.add_argument("--data-dir", type=Path, default=REPO / "data/lumiere/extracted/Imaging")
    p.add_argument("--clinical-dir", type=Path, default=REPO / "data/lumiere/clinical")
    p.add_argument("--workers", type=int, default=10)
    a = p.parse_args()
    ceiling(a) if a.command == "ceiling" else report(a)
