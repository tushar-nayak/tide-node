#!/usr/bin/env python3
"""Exploratory follow-up to the MU-Glioma-Post validation (not pre-specified; written after
seeing results/mu_validation/metrics.md, where the frozen call scored AUROC 0.56).

Two questions about why it failed:
  1. Does enhancing growth come before documented progression? The script compares the
     volume rule and the frozen model on three groups of scans: progression scans, earlier
     scans of the same patients, and scans of patients who never progressed.
  2. Can volumes predict MU's labels at all? The same two-compartment GBM is trained and
     tested within MU (patient-level 5-fold CV x 3 seeds), with and without time since the
     end of radiotherapy, which LUMIERE lacks. If even an MU-trained model is near chance,
     MU's documented progression is a different target from LUMIERE's RANO reads, not a
     target the model failed to transfer to.
Reads results/mu_validation/mask_volumes.csv; writes results/mu_validation/exploratory.{md,json}.
CPU, under a minute.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
REPO = SCRIPTS.parent

from approach_screen import auroc_boot, oof  # noqa: E402
from mu_freeze_model import gbm_params, sha256  # noqa: E402
from mu_validation import CLINICAL_SHEET, build_patients, eligible, evaluable_calls, load_mask_volumes  # noqa: E402
from xlsx_stdlib import records  # noqa: E402


OUT_DIR = REPO / "results/mu_validation"


def compute() -> tuple[dict, np.ndarray, dict]:
    """Results dict, MU labels, and per-call scores (frozen and within-MU), in call order."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    out_dir = OUT_DIR
    frozen = json.loads((out_dir / "frozen.json").read_text())
    model_file = REPO / frozen["model_file"]
    assert sha256(model_file) == frozen["model_sha256"]
    with model_file.open("rb") as h:  # our own file, verified by hash above
        models = pickle.load(h)["models"]
    clinical = [r for r in records(REPO / "data/mu_glioma_post/MU-Glioma-Post_ClinicalData-July2025.xlsx", CLINICAL_SHEET)
                if r["Patient_ID"].strip()]
    vols = load_mask_volumes(REPO / "data/mu_glioma_post/package", out_dir / "mask_volumes.csv", workers=1)
    patients = [p for p in build_patients(clinical, vols) if eligible(p, "GBM")]
    rows = evaluable_calls(patients, models)
    y = np.array([r["label"] == "PD" for r in rows], dtype=int)
    pid = np.array([r["patient_id"] for r in rows])
    res: dict = {"calls": len(rows), "pd": int(y.sum()), "patients": len(set(pid))}

    # 1. Growth before documented progression
    progressors = {r["patient_id"] for r in rows if r["label"] == "PD"}
    groups = {"progression scans": [r for r in rows if r["label"] == "PD"],
              "earlier scans of patients who progressed": [r for r in rows if r["label"] == "non-PD" and r["patient_id"] in progressors],
              "scans of patients who never progressed": [r for r in rows if r["label"] == "non-PD" and r["patient_id"] not in progressors]}
    res["groups"] = {k: {"scans": len(v), "volume_rule_positive": float(np.mean([r["rule"] for r in v])),
                         "median_enhancing_change_vs_nadir": float(np.median([r["x"][4] for r in v])),
                         "median_probability": float(np.median([r["dbt_atlas"] for r in v]))} for k, v in groups.items()}

    # 2. Within-MU learnability
    x = np.stack([r["x"] for r in rows])
    since_rt = np.array([np.nan if r["patient"].rt_end is None else (r["day"] - r["patient"].rt_end) / 7 / 52 for r in rows])
    xr = np.hstack([x, since_rt[:, None]])
    pats = sorted(set(pid))

    def cv(features, monotone):
        def fp(tr, te):
            params = dict(gbm_params(), monotonic_cst=monotone)
            return HistGradientBoostingClassifier(**params).fit(features[tr], y[tr]).predict_proba(features[te])[:, 1]
        return oof(pid, pats, fp)

    mono = list(gbm_params()["monotonic_cst"])
    scores = {"frozen LUMIERE model": np.array([r["dbt_atlas"] for r in rows]),
              "MU-trained, same features": cv(x, mono),
              "MU-trained, + time since radiotherapy": cv(xr, mono + [0])}
    res["within_mu"] = auroc_boot(scores, y, pid, np.random.default_rng(20261118), "frozen LUMIERE model")
    return res, y, scores


def main() -> None:
    res, _, _ = compute()
    out_dir = OUT_DIR
    out_dir.joinpath("exploratory.json").write_text(json.dumps(res, indent=2, allow_nan=False) + "\n")

    L = ["# MU-Glioma-Post validation: exploratory follow-up", "",
         "Not pre-specified; written after the frozen call scored AUROC 0.56 (metrics.md). "
         f"{res['calls']} evaluable calls, {res['pd']} PD, {res['patients']} GBM patients.", "",
         "## 1. Does enhancing growth come before documented progression?", "",
         "| scans | n | volume rule positive | median enhancing change vs nadir (log1p mL) | median P(PD), frozen model |",
         "| :--- | ---: | ---: | ---: | ---: |"]
    for k, v in res["groups"].items():
        L.append(f"| {k} | {v['scans']} | {v['volume_rule_positive']:.2f} | {v['median_enhancing_change_vs_nadir']:.2f} | {v['median_probability']:.2f} |")
    L += ["", "## 2. Can volumes predict MU's labels at all? (patient-level 5-fold CV x 3 seeds within MU)", "",
          "| score | AUROC [95% CI] |", "| :--- | :---: |"]
    for k, v in res["within_mu"].items():
        L.append(f"| {k} | {v['auroc']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] |")
    out_dir.joinpath("exploratory.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
