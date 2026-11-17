#!/usr/bin/env python3
"""Would more patients improve the progression call? A learning curve on LUMIERE.

Within every outer fold of the usual patient-level CV (5 folds x 3 seeds), the model is
trained on a random subset of the training patients (25/50/75/100%; 10 random subsets per
fraction below 100%) and scored on the full held-out fold. Out-of-fold probabilities are
averaged over seeds as elsewhere, so the 100% point reproduces the published AUROCs.

If held-out AUROC is still rising between 75% and 100% of the training patients, more data
of the same kind should help; if it is flat, gains must come from new information
(radiotherapy dates, T2/FLAIR measures, other readers), not more of the same.

Models: the released volumes model (V+R+S), the released mask model (V+R+S+M), and the
fully automated versions (V, V+M). CPU, about 6 min.
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

import progression_model as pm  # noqa: E402
from approach_screen import FOLDS, SEEDS, fast_auroc, norm_rating  # noqa: E402
from lumiere_cohort import build_cohort, load_clinical, patient_folds  # noqa: E402
from volume_trajectory_experiment import static_features  # noqa: E402

FRACTIONS = (0.25, 0.5, 0.75, 1.0)
REPEATS = 10
V_COLS, R_COLS, S_COLS = list(range(0, 15)), list(range(15, 21)), list(range(21, 26))


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data/lumiere/extracted/Imaging")
    ap.add_argument("--clinical-dir", type=Path, default=REPO / "data/lumiere/clinical")
    ap.add_argument("--mask-features", type=Path, default=REPO / "results/progression_call/mask_features_atlas_8mm.csv")
    ap.add_argument("--output-dir", type=Path, default=REPO / "results/progression_learning_curve")
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
    x = np.hstack([full, [masks[(t.patient_id, t.last.source_name, t.target.source_name)] for t in rows]])
    m_cols = list(range(full.shape[1], x.shape[1]))
    cst, m_cst = pm.GBM_PARAMS["monotonic_cst"], list(pm.MASK_MONOTONE)
    plain = {k: v for k, v in pm.GBM_PARAMS.items() if k != "monotonic_cst"}
    models = {
        "released: V+R+S": (V_COLS + R_COLS + S_COLS, cst),
        "released: V+R+S+M": (V_COLS + R_COLS + S_COLS + m_cols, cst + m_cst),
        "fully automated: V": (V_COLS, [cst[i] for i in V_COLS]),
        "fully automated: V+M": (V_COLS + m_cols, [cst[i] for i in V_COLS] + m_cst),
    }
    prev_pd = full[:, R_COLS[0]] == 1
    patients = cohort.patients

    def oof_subsampled(cols, mono, fraction, repeat) -> tuple[np.ndarray, float]:
        total = np.zeros(len(rows))
        n_train = []
        for seed in SEEDS:
            pred = np.zeros(len(rows))
            for k, test_patients in enumerate(patient_folds(patients, FOLDS, seed)):
                train_patients = sorted(set(patients) - set(test_patients))
                if fraction < 1.0:
                    rng = np.random.default_rng([seed, k, repeat, int(round(fraction * 1000))])
                    keep = max(2, int(round(fraction * len(train_patients))))
                    train_patients = sorted(rng.choice(train_patients, size=keep, replace=False))
                tr, te = np.flatnonzero(np.isin(pid, train_patients)), np.flatnonzero(np.isin(pid, test_patients))
                n_train.append(len(train_patients))
                model = HistGradientBoostingClassifier(**plain, monotonic_cst=mono)
                pred[te] = model.fit(x[tr][:, cols], label[tr]).predict_proba(x[te][:, cols])[:, 1]
            total += pred
        return total / len(SEEDS), float(np.mean(n_train))

    res: dict = {"rows": len(rows), "patients": len(patients), "fractions": list(FRACTIONS), "repeats": REPEATS, "models": {}}
    for name, (cols, mono) in models.items():
        out = []
        for fraction in FRACTIONS:
            reps = 1 if fraction == 1.0 else REPEATS
            aucs, aucs_new, n_tr = [], [], []
            for r in range(reps):
                p, n = oof_subsampled(cols, mono, fraction, r)
                aucs.append(fast_auroc(p, label))
                aucs_new.append(fast_auroc(p[~prev_pd], label[~prev_pd]))
                n_tr.append(n)
            out.append({"fraction": fraction, "training_patients": float(np.mean(n_tr)),
                        "auroc_mean": float(np.mean(aucs)), "auroc_sd": float(np.std(aucs, ddof=1)) if reps > 1 else 0.0,
                        "auroc_min": float(np.min(aucs)), "auroc_max": float(np.max(aucs)),
                        "auroc_new_progression_mean": float(np.mean(aucs_new)),
                        "auroc_new_progression_sd": float(np.std(aucs_new, ddof=1)) if reps > 1 else 0.0})
            print(name, out[-1], flush=True)
        res["models"][name] = out
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(res, indent=2, allow_nan=False) + "\n")
    write_markdown(res, args.output_dir / "metrics.md")
    print((args.output_dir / "metrics.md").read_text())


def write_markdown(res: dict, path: Path) -> None:
    L = ["# Progression call: learning curve", "",
         f"{res['rows']} rated follow-ups, {res['patients']} patients. Each outer fold trains on a random subset of its training "
         f"patients ({res['repeats']} subsets per fraction below 100%) and is scored on the whole held-out fold; out-of-fold "
         "probabilities averaged over 3 fold seeds. Mean AUROC across subsets (SD).", "",
         "| model | " + " | ".join(f"{f:.0%} of training patients" for f in res["fractions"]) + " | gain 75% -> 100% |",
         "| :--- | " + " | ".join(":---:" for _ in res["fractions"]) + " | :---: |"]
    for kind, key in (("all scans", "auroc"), ("previous scan not PD", "auroc_new_progression")):
        for name, rows in res["models"].items():
            cells = [f"{r[f'{key}_mean']:.3f} ({r[f'{key}_sd']:.3f})" if r["fraction"] < 1 else f"{r[f'{key}_mean']:.3f}" for r in rows]
            gain = rows[-1][f"{key}_mean"] - next(r for r in rows if r["fraction"] == 0.75)[f"{key}_mean"]
            L.append(f"| {name}, {kind} | " + " | ".join(cells) + f" | {gain:+.3f} |")
    n = [f"{r['training_patients']:.0f}" for r in next(iter(res["models"].values()))]
    L += ["", "Training patients per fold at each fraction: " + ", ".join(n) + "."]
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
