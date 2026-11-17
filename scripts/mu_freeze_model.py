#!/usr/bin/env python3
"""Freeze the fully automated progression call before external validation on MU-Glioma-Post.

Trains, on all of LUMIERE, the two-compartment volumes-only model of progression_robustness.py:
enhancing and non-enhancing volumes, their changes against the previous scan and the nadir,
and timing. It uses no expert rating, no clinical factors and no registration. It then fixes
the operating thresholds from LUMIERE held-out scores. Nothing from MU is read.

Two training sources:
  dbt_atlas  DeepBraTumIA atlas volumes (primary; the released model's source)
  hd_native  HD-GLIO-AUTO native volumes (secondary)

Thresholds come from held-out scores on follow-ups whose previous scan was not rated PD,
because MU's labels describe first progression. All-scan thresholds are recorded for
reference. The script also records LUMIERE reference values for the simple comparators that
the validation reports: enhancing change against the nadir and against the previous scan,
and the volume rule (>= 40% and >= 1 mL above nadir).

Writes runs/mu_validation/frozen_models.pkl and results/mu_validation/frozen.json, with the
pickle's and the analysis plan's SHA-256. It refuses to overwrite an existing freeze.
Runs on CPU in under a minute; native volumes come from the progression_robustness.py cache.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
REPO = SCRIPTS.parent

import progression_model as pm  # noqa: E402
from approach_screen import auroc_boot, norm_rating, oof  # noqa: E402
from lumiere_cohort import build_cohort  # noqa: E402
from progression_robustness import TWO_FEATURES, TWO_MONOTONE, load_volumes, two_compartment_features  # noqa: E402

TRAIN_SOURCES = ("dbt_atlas", "hd_native")
PRIMARY = "dbt_atlas"
RULE_REL, RULE_ABS_MM3 = 0.40, 1000.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gbm_params() -> dict:
    return dict({k: v for k, v in pm.GBM_PARAMS.items() if k != "monotonic_cst"}, monotonic_cst=list(TWO_MONOTONE))


def rule_fires(enh_history: list[float], enh_current: float, rel: float = RULE_REL, abs_mm3: float = RULE_ABS_MM3) -> bool:
    """Volume rule at one scan: enhancing >= (1+rel) x nadir of the earlier scans and >= nadir + abs."""
    nadir = min(enh_history)
    return enh_current >= (1 + rel) * nadir and enh_current - nadir >= abs_mm3


def operating_point(fires: np.ndarray, label: np.ndarray) -> dict:
    return {"sensitivity": float(fires[label == 1].mean()), "specificity": float((~fires[label == 0]).mean()),
            "called": int(fires.sum())}


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    import sklearn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data/lumiere/extracted/Imaging")
    ap.add_argument("--clinical-dir", type=Path, default=REPO / "data/lumiere/clinical")
    ap.add_argument("--native-volumes", type=Path, default=REPO / "results/progression_robustness/native_volumes.csv")
    ap.add_argument("--plan", type=Path, default=REPO / "results/mu_validation/PLAN.md")
    ap.add_argument("--model-out", type=Path, default=REPO / "runs/mu_validation/frozen_models.pkl")
    ap.add_argument("--json-out", type=Path, default=REPO / "results/mu_validation/frozen.json")
    args = ap.parse_args()
    for p in (args.model_out, args.json_out):
        if p.exists():
            raise FileExistsError(f"{p} exists: a freeze is never overwritten")
    if not args.plan.exists():
        raise FileNotFoundError(f"write the analysis plan first: {args.plan}")

    cohort = build_cohort(args.data_dir, args.clinical_dir / "LUMIERE-ExpertRating-v202211.csv")
    rows = [t for t in cohort.transitions if norm_rating(t.target.rating) in ("PD", "SD", "PR", "CR")]
    pid = np.array([t.patient_id for t in rows])
    label = np.array([norm_rating(t.target.rating) == "PD" for t in rows], dtype=int)
    new = np.array([norm_rating(t.history[-1].rating) != "PD" for t in rows])
    vols = load_volumes(cohort, args.native_volumes, workers=1)
    rng = np.random.default_rng(20261117)

    out: dict = {"created": dt.datetime.now().isoformat(timespec="seconds"), "plan_sha256": sha256(args.plan),
                 "features": TWO_FEATURES, "monotone": TWO_MONOTONE, "params": gbm_params(), "primary": PRIMARY,
                 "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__},
                 "training": {"rows": len(rows), "patients": len(set(pid)), "pd_rate": float(label.mean()),
                              "previous_not_pd_rows": int(new.sum()), "previous_not_pd_pd_rate": float(label[new].mean())},
                 "sources": {}}
    models = {}
    for src in TRAIN_SOURCES:
        x = np.stack([two_compartment_features(t.history, t.target, vols[src]) for t in rows])
        assert np.isfinite(x).all(), src

        def fit_predict(tr, te, x=x):
            return HistGradientBoostingClassifier(**gbm_params()).fit(x[tr], label[tr]).predict_proba(x[te])[:, 1]

        scores = oof(pid, cohort.patients, fit_predict)
        enh = lambda s, src=src: vols[src][(s.patient_id, s.source_name)][0]  # noqa: E731
        fires = np.array([rule_fires([enh(s) for s in t.history], enh(t.target)) for t in rows])
        cands = {"model": scores, "enhancing change vs nadir": x[:, 4], "enhancing change vs previous": x[:, 2]}
        out["sources"][src] = {
            "held_out_auroc_all": auroc_boot(cands, label, pid, rng, "model"),
            "held_out_auroc_previous_not_pd": auroc_boot({k: v[new] for k, v in cands.items()}, label[new], pid[new], rng, "model"),
            "thresholds_previous_not_pd": pm.thresholds(scores[new], label[new]),
            "thresholds_all": pm.thresholds(scores, label),
            "volume_rule_all": operating_point(fires, label),
            "volume_rule_previous_not_pd": operating_point(fires[new], label[new]),
        }
        models[src] = HistGradientBoostingClassifier(**gbm_params()).fit(x, label)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    with args.model_out.open("wb") as h:
        pickle.dump({"models": models, "features": TWO_FEATURES}, h)
    out["model_file"] = str(args.model_out.relative_to(REPO))
    out["model_sha256"] = sha256(args.model_out)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    for src, r in out["sources"].items():
        a, b = r["held_out_auroc_all"]["model"], r["held_out_auroc_previous_not_pd"]["model"]
        t = r["thresholds_previous_not_pd"]
        print(f"{src}: LUMIERE held-out AUROC all {a['auroc']:.3f} {a['ci95']}, previous not PD {b['auroc']:.3f} {b['ci95']}; "
              f"thresholds (previous not PD) 80% spec {t['specificity_80']['threshold']:.3f}, "
              f"90% sens {t['sensitivity_90']['threshold']:.3f}")
    print(f"frozen: {args.model_out} sha256 {out['model_sha256']}")


if __name__ == "__main__":
    main()
