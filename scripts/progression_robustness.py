#!/usr/bin/env python3
"""Two robustness checks of the automated progression call, using only LUMIERE.

1. Segmenter swap (a stand-in for external validation). LUMIERE ships a second automatic
   segmentation of every scan, HD-GLIO-AUTO, next to the DeepBraTumIA one the released model
   was trained on. A model is trained on one segmenter's volumes and tested, on held-out
   patients, on the other's volumes of the same scans. HD-GLIO has no necrosis label, so all
   swap models use two compartments: enhancing tumor, and non-enhancing abnormality
   (DeepBraTumIA necrosis + edema; HD-GLIO label 1). Three volume sources:
     dbt_atlas   DeepBraTumIA, measured in atlas space (what the released model uses)
     dbt_native  DeepBraTumIA, voxel count on the scan's own CT1 grid
     hd_native   HD-GLIO-AUTO, voxel count on the same CT1 grid (registered/segmentation.nii.gz)
   dbt_atlas -> dbt_native isolates the change of measurement space; dbt_native -> hd_native
   isolates the change of segmenter. Thresholds fixed on the training source's held-out scores
   are applied unchanged to the other source, as a deployment would.

2. Dependence on the expert. The released model reads the previous scan's expert rating,
   which other datasets rarely provide. The model is refitted without it, and without the
   clinical factors, to measure how much accuracy rests on the expert.

Same rows, folds and bootstrap as scripts/progression_call_experiment.py. CPU, about 3 min
(the first run also reads ~1,000 segmentations and caches their volumes).
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

import progression_model as pm  # noqa: E402
from approach_screen import N_BOOT, auroc_boot, norm_rating, oof  # noqa: E402
from lumiere_cohort import build_cohort, load_clinical  # noqa: E402
from volume_trajectory_experiment import static_features  # noqa: E402

SOURCES = ("dbt_atlas", "dbt_native", "hd_native")
SOURCE_LABEL = {"dbt_atlas": "DeepBraTumIA (atlas)", "dbt_native": "DeepBraTumIA (native)", "hd_native": "HD-GLIO (native)"}
V_COLS, R_COLS, S_COLS = list(range(0, 15)), list(range(15, 21)), list(range(21, 26))
TWO_FEATURES = (["log1p_ml_enhancing", "log1p_ml_nonenhancing", "change_vs_previous_enhancing", "change_vs_previous_nonenhancing",
                 "change_vs_nadir_enhancing", "change_vs_nadir_nonenhancing", "total_log1p_ml", "total_change_vs_previous",
                 "total_change_vs_nadir", "interval_years", "history_count", "years_since_segment_start"])
TWO_MONOTONE = [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0]
N_TWO = len(TWO_FEATURES)


def native_volumes(scan_dir: str) -> tuple[float, ...]:
    """(DBT enhancing, DBT non-enhancing, HD enhancing, HD non-enhancing) in mm3 on the native CT1 grid; NaN if missing."""
    import nibabel as nib

    def counts(path: Path, enhancing: tuple[int, ...], other: tuple[int, ...]) -> tuple[float, float]:
        if not path.exists():
            return float("nan"), float("nan")
        img = nib.load(str(path))
        seg = np.asarray(img.dataobj)
        voxel = float(np.prod(img.header.get_zooms()[:3]))
        return float(np.isin(seg, enhancing).sum()) * voxel, float(np.isin(seg, other).sum()) * voxel

    d = Path(scan_dir)
    dbt = counts(d / "DeepBraTumIA-segmentation/native/segmentation/ct1_seg_mask.nii.gz", (1,), (2, 3))
    hd = counts(d / "HD-GLIO-AUTO-segmentation/registered/segmentation.nii.gz", (2,), (1,))
    return (*dbt, *hd)


def load_volumes(cohort, cache: Path, workers: int) -> dict[str, dict[tuple[str, str], tuple[float, float]]]:
    stored: dict[tuple[str, str], tuple[float, ...]] = {}
    if cache.exists():
        with cache.open() as h:
            stored = {(r["patient_id"], r["source"]): tuple(float(r[c]) for c in ("dbt_enh", "dbt_non", "hd_enh", "hd_non"))
                      for r in csv.DictReader(h)}
    scans = [s for pid in cohort.scans for s in cohort.scans[pid]]
    missing = [s for s in scans if (s.patient_id, s.source_name) not in stored]
    if missing:
        print(f"Reading native segmentations of {len(missing)} scans", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for s, v in zip(missing, pool.map(native_volumes, [str(s.scan.directory) for s in missing], chunksize=4)):
                stored[(s.patient_id, s.source_name)] = tuple(round(x, 3) for x in v)  # as cached, so reruns match
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("w", newline="") as h:
            w = csv.writer(h)
            w.writerow(["patient_id", "source", "dbt_enh", "dbt_non", "hd_enh", "hd_non"])
            for k in sorted(stored):
                w.writerow([*k, *stored[k]])
    vols = {"dbt_atlas": {}, "dbt_native": {}, "hd_native": {}}
    for s in scans:
        k = (s.patient_id, s.source_name)
        v = s.volumes_mm3
        vols["dbt_atlas"][k] = (v["Enhancing_Core"], v["Necrotic_NonEnhancing"] + v["Edema_Compartment"])
        vols["dbt_native"][k] = stored[k][0:2]
        vols["hd_native"][k] = stored[k][2:4]
    return vols


def two_compartment_features(history: list, current, vol: dict) -> np.ndarray:
    """Volume block of the released features, rebuilt on (enhancing, non-enhancing)."""
    def y(s):
        return np.log1p(np.asarray(vol[(s.patient_id, s.source_name)], dtype=np.float64) / 1000.0)

    def tot(v):
        return float(np.log1p(np.expm1(v).sum()))

    y_c, y_p = y(current), y(history[-1])
    past = [y(s) for s in history]
    return np.concatenate([y_c, y_c - y_p, y_c - np.min(past, axis=0),
                           [tot(y_c), tot(y_c) - tot(y_p), tot(y_c) - min(tot(p) for p in past),
                            (current.week - history[-1].week) / 52.0, len(history), (current.week - history[0].week) / 52.0]])


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data/lumiere/extracted/Imaging")
    ap.add_argument("--clinical-dir", type=Path, default=REPO / "data/lumiere/clinical")
    ap.add_argument("--mask-features", type=Path, default=REPO / "results/progression_call/mask_features_atlas_8mm.csv")
    ap.add_argument("--output-dir", type=Path, default=REPO / "results/progression_robustness")
    ap.add_argument("--workers", type=int, default=11)
    args = ap.parse_args()

    cohort = build_cohort(args.data_dir, args.clinical_dir / "LUMIERE-ExpertRating-v202211.csv")
    clinical = load_clinical(args.clinical_dir / "LUMIERE-Demographics_Pathology.csv")
    statics = {p: static_features(p, clinical) for p in cohort.patients}
    rows = [t for t in cohort.transitions if norm_rating(t.target.rating) in ("PD", "SD", "PR", "CR")]
    pid = np.array([t.patient_id for t in rows])
    label = np.array([norm_rating(t.target.rating) == "PD" for t in rows], dtype=int)
    full = np.stack([pm.features(t.history, t.target, statics[t.patient_id]) for t in rows])
    with args.mask_features.open() as h:
        masks = {(r["patient_id"], r["prev_source"], r["cur_source"]): [float(r[f]) for f in pm.MASK_FEATURES] for r in csv.DictReader(h)}
    mfeat = np.array([masks[(t.patient_id, t.last.source_name, t.target.source_name)] for t in rows])
    rng = np.random.default_rng(20261116)
    res: dict = {"rows": len(rows), "patients": len(set(pid)), "pd_rate": float(label.mean())}
    prev_pd = full[:, R_COLS[0]] == 1

    # ---- 2. Dependence on the expert's previous rating ------------------------------------------------
    cst = pm.GBM_PARAMS["monotonic_cst"]
    plain = {k: v for k, v in pm.GBM_PARAMS.items() if k != "monotonic_cst"}
    xm = np.hstack([full, mfeat])

    def gbm_fit(x, cols, monotone):
        def fp(tr, te):
            model = HistGradientBoostingClassifier(**plain, monotonic_cst=monotone)
            return model.fit(x[tr][:, cols], label[tr]).predict_proba(x[te][:, cols])[:, 1]
        return fp

    m_cst = list(pm.MASK_MONOTONE)
    m_cols = list(range(full.shape[1], xm.shape[1]))
    ablation = {
        "volumes + rating + clinical (released)": (V_COLS + R_COLS + S_COLS, cst),
        "volumes + clinical, no rating": (V_COLS + S_COLS, [cst[i] for i in V_COLS + S_COLS]),
        "volumes only": (V_COLS, [cst[i] for i in V_COLS]),
        "volumes + masks + rating + clinical (released)": (V_COLS + R_COLS + S_COLS + m_cols, cst + m_cst),
        "volumes + masks + clinical, no rating": (V_COLS + S_COLS + m_cols, [cst[i] for i in V_COLS + S_COLS] + m_cst),
        "volumes + masks only": (V_COLS + m_cols, [cst[i] for i in V_COLS] + m_cst),
    }
    abl = {k: oof(pid, cohort.patients, gbm_fit(xm, cols, mono)) for k, (cols, mono) in ablation.items()}
    ref = "volumes + rating + clinical (released)"
    res["expert_dependence"] = {
        "all": auroc_boot(abl, label, pid, rng, ref),
        "previous_not_pd": {"rows": int((~prev_pd).sum()),
                            **auroc_boot({k: v[~prev_pd] for k, v in abl.items()}, label[~prev_pd], pid[~prev_pd], rng, ref)},
        "masks_vs_masks_released": auroc_boot({k: v for k, v in abl.items() if "masks" in k}, label, pid, rng,
                                              "volumes + masks + rating + clinical (released)"),
    }
    # Separate generator so that adding this comparison left every other interval unchanged.
    res["expert_dependence"]["masks_vs_masks_released_previous_not_pd"] = auroc_boot(
        {k: v[~prev_pd] for k, v in abl.items() if "masks" in k}, label[~prev_pd], pid[~prev_pd], np.random.default_rng(20261117),
        "volumes + masks + rating + clinical (released)")

    # ---- 1. Segmenter swap ---------------------------------------------------------------------------
    vols = load_volumes(cohort, args.output_dir / "native_volumes.csv", args.workers)
    keys_needed = [[(s.patient_id, s.source_name) for s in t.history + [t.target]] for t in rows]
    ok = np.array([all(np.isfinite(vols[src][k]).all() for src in SOURCES for k in ks) for ks in keys_needed])
    sub = np.flatnonzero(ok)
    rows_s, pid_s, y_s = [rows[i] for i in sub], pid[sub], label[sub]
    pats_s = sorted(set(pid_s))
    feats = {src: np.stack([two_compartment_features(t.history, t.target, vols[src]) for t in rows_s]) for src in SOURCES}
    extra = full[sub][:, R_COLS + S_COLS]
    agreement = {}
    for a, b in (("dbt_atlas", "dbt_native"), ("dbt_native", "hd_native"), ("dbt_atlas", "hd_native")):
        agreement[f"{a} vs {b}"] = {
            name: float(np.corrcoef(feats[a][:, j], feats[b][:, j])[0, 1])
            for j, name in ((0, "log1p_ml_enhancing"), (1, "log1p_ml_nonenhancing"), (2, "change_vs_previous_enhancing"),
                            (4, "change_vs_nadir_enhancing"), (5, "change_vs_nadir_nonenhancing"))}
    res["segmenter_swap"] = {"rows": int(ok.sum()), "patients": len(pats_s), "pd_rate": float(y_s.mean()),
                             "excluded_rows_missing_native_segmentation": int((~ok).sum()), "feature_correlation": agreement}

    def swap_fit(train_src, test_src, with_expert):
        xtr = np.hstack([feats[train_src], extra]) if with_expert else feats[train_src]
        xte = np.hstack([feats[test_src], extra]) if with_expert else feats[test_src]
        mono = TWO_MONOTONE + [0] * (extra.shape[1] if with_expert else 0)

        def fp(tr, te):
            model = HistGradientBoostingClassifier(**plain, monotonic_cst=mono)
            return model.fit(xtr[tr], y_s[tr]).predict_proba(xte[te])[:, 1]
        return fp

    for with_expert, tag in ((True, "with_rating_clinical"), (False, "volumes_only")):
        scores = {f"{tr} -> {te}": oof(pid_s, pats_s, swap_fit(tr, te, with_expert))
                  for tr in SOURCES for te in SOURCES if tr == te or tr == "dbt_atlas" or te == "dbt_atlas"}
        boot = auroc_boot(scores, y_s, pid_s, rng, "dbt_atlas -> dbt_atlas")
        # Thresholds fixed on the training source's own held-out scores, then applied to the swapped source.
        op = {}
        for k, s in scores.items():
            tr = k.split(" -> ")[0]
            thr = pm.thresholds(scores[f"{tr} -> {tr}"], y_s)
            t80, t90 = thr["specificity_80"]["threshold"], thr["sensitivity_90"]["threshold"]
            op[k] = {"specificity_at_80pct_specificity_threshold": float((s[y_s == 0] < t80).mean()),
                     "sensitivity_at_80pct_specificity_threshold": float((s[y_s == 1] >= t80).mean()),
                     "sensitivity_at_90pct_sensitivity_threshold": float((s[y_s == 1] >= t90).mean()),
                     "specificity_at_90pct_sensitivity_threshold": float((s[y_s == 0] < t90).mean()),
                     "brier": float(np.mean((s - y_s) ** 2)), "mean_probability": float(s.mean()),
                     "bands": pm.tier_rates(s, y_s, thr)}
        res["segmenter_swap"][tag] = {"auroc": boot, "fixed_thresholds": op}
    # Reference on the same subset: the released three-compartment DeepBraTumIA model.
    ref3 = oof(pid_s, pats_s, lambda tr, te: pm.new_model(False).fit(full[sub][tr], y_s[tr]).predict_proba(full[sub][te])[:, 1])
    res["segmenter_swap"]["released_three_compartment_auroc_same_rows"] = auroc_boot({"released": ref3}, y_s, pid_s, rng, "released")["released"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(res, indent=2, allow_nan=False) + "\n")
    write_markdown(res, args.output_dir / "metrics.md")
    print((args.output_dir / "metrics.md").read_text())


def ci(v: dict, key: str = "auroc") -> str:
    return f"{v[key]:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"


def diff(v: dict, ref: str) -> str:
    lo, hi = v[f"minus_{ref}_ci95"]
    return f"{v[f'minus_{ref}']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def write_markdown(res: dict, path: Path) -> None:
    e = res["expert_dependence"]
    ref = "volumes + rating + clinical (released)"
    L = ["# Progression call: robustness checks on LUMIERE", "",
         f"Held-out AUROC (patient-level 5-fold CV x 3 seeds, {N_BOOT} patient resamples). {res['rows']} rated follow-ups, "
         f"{res['patients']} patients, PD rate {res['pd_rate']:.2f}.", "",
         "## 1. How much does the model rest on the expert?", "",
         "| inputs | AUROC, all scans | minus released volumes model | AUROC, previous scan not PD | minus released |",
         "| :--- | :---: | :---: | :---: | :---: |"]
    for k, v in e["all"].items():
        p = e["previous_not_pd"][k]
        L.append(f"| {k} | {ci(v)} | {diff(v, ref)} | {ci(p)} | {diff(p, ref)} |")
    mref = "volumes + masks + rating + clinical (released)"
    L += ["", "Mask models against the released mask model, all scans: " + "; ".join(
        f"{k}: {diff(v, mref)}" for k, v in e["masks_vs_masks_released"].items() if k != mref) + ". Previous scan not PD: " + "; ".join(
        f"{k}: {diff(v, mref)}" for k, v in e["masks_vs_masks_released_previous_not_pd"].items() if k != mref) + ".", ""]
    s = res["segmenter_swap"]
    L += ["## 2. Segmenter swap", "",
          f"{s['rows']} scans, {s['patients']} patients, PD rate {s['pd_rate']:.2f} ({s['excluded_rows_missing_native_segmentation']} "
          "scans dropped because a native segmentation was missing). Two compartments (enhancing, non-enhancing) for every "
          f"model. The released three-compartment model scores {ci(s['released_three_compartment_auroc_same_rows'])} on these rows.", "",
          "Agreement of the features between sources (Pearson r):", "",
          "| sources | enhancing volume | non-enhancing volume | enhancing change vs previous | enhancing change vs nadir | non-enhancing change vs nadir |",
          "| :--- | ---: | ---: | ---: | ---: | ---: |"]
    for k, v in s["feature_correlation"].items():
        L.append(f"| {k} | " + " | ".join(f"{x:.2f}" for x in v.values()) + " |")
    for tag, title in (("with_rating_clinical", "with previous rating and clinical factors"), ("volumes_only", "volumes only")):
        t = s[tag]
        L += ["", f"### Train -> test, {title}", "",
              "Thresholds are fixed on the training source's own held-out scores (80% specificity and 90% sensitivity "
              "there) and applied unchanged.", "",
              "| train -> test | AUROC | minus DBT atlas -> DBT atlas | specificity at the 80%-spec threshold | sensitivity there "
              "| sensitivity at the 90%-sens threshold | Brier | mean P(PD) |",
              "| :--- | :---: | :---: | ---: | ---: | ---: | ---: | ---: |"]
        for k, v in t["auroc"].items():
            o = t["fixed_thresholds"][k]
            L.append(f"| {k} | {ci(v)} | {diff(v, 'dbt_atlas -> dbt_atlas')} | {o['specificity_at_80pct_specificity_threshold']:.2f} | "
                     f"{o['sensitivity_at_80pct_specificity_threshold']:.2f} | {o['sensitivity_at_90pct_sensitivity_threshold']:.2f} | "
                     f"{o['brier']:.3f} | {o['mean_probability']:.2f} |")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
