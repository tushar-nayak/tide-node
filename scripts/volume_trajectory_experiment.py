"""Step 1: forecast tumor compartment volumes at the next visit from irregular history.

Data: LUMIERE DeepBraTumIA volumes (necrosis / enhancing / edema, mm^3) per scan, cohort
rules from ``lumiere_cohort.py`` (pre-op scans dropped, no forecasting across surgery).

Target: y = log1p(volume in mL) per compartment at the next scan, given all earlier scans
of the same post-surgical segment, the interval length, and baseline clinical covariates.

Evaluation: patient-level K-fold cross-validation, repeated over several fold seeds. Every
method sees exactly the same information. Reported: MAE of y per compartment, skill vs
persistence, patient-level paired bootstrap CIs, and AUROC of the predicted change in
total tumor volume against the expert RANO "PD" label of the target scan.

Methods:
  persistence   y_next = y_last
  loglinear     per-patient exponential extrapolation from the last two scans
  gompertz      population Gompertz (mean reversion in log volume), fit per compartment
  ridge         linear model on hand-crafted history features -> delta y
  gbm           gradient boosting on the same features -> delta y
  neural_ode    ODE-RNN: GRU updates at scans, learned latent ODE between scans,
                residual decoder (zero-initialized => starts exactly at persistence)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lumiere_cohort import COMPARTMENT_SHORT, COMPARTMENTS, Transition, build_cohort, load_clinical, patient_folds  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MAX_ODE_HISTORY = 8  # ODE-RNN reads at most the last 8 scans of a segment
METHODS = ("persistence", "loglinear", "gompertz", "ridge", "gbm", "neural_ode")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=REPO / "data/lumiere/extracted/Imaging")
    p.add_argument("--clinical-dir", type=Path, default=REPO / "data/lumiere/clinical")
    p.add_argument("--output-dir", type=Path, default=REPO / "results/volume_trajectory")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3, help="Number of fold-assignment seeds.")
    p.add_argument("--ensemble", type=int, default=5, help="Neural ODE seeds per fold.")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--quick", action="store_true", help="1 repeat, 2-member ensemble, 100 epochs (smoke test).")
    return p.parse_args()


# ----------------------------------------------------------------------------- data


def y_of(scan) -> np.ndarray:
    return np.array([math.log1p(scan.volumes_mm3[c] / 1000.0) for c in COMPARTMENTS], dtype=np.float64)


def static_features(pid: str, clinical: dict[str, dict[str, str]]) -> np.ndarray:
    row = clinical.get(pid, {})
    try:
        age = (float(row.get("Age at surgery (years)", "nan")) - 60.0) / 12.0
    except ValueError:
        age = float("nan")
    age_missing = float(math.isnan(age))
    age = 0.0 if math.isnan(age) else age
    sex = 1.0 if row.get("Sex", "").strip().lower() == "male" else 0.0
    mgmt = row.get("MGMT qualitative", "").strip().lower()
    mgmt_meth = 1.0 if mgmt == "methylated" else 0.0
    mgmt_known = 1.0 if mgmt in ("methylated", "not methylated") else 0.0
    return np.array([age, age_missing, sex, mgmt_meth, mgmt_known], dtype=np.float64)


def history_features(t: Transition, static: np.ndarray) -> np.ndarray:
    """Fixed-length summary for the tabular baselines (same raw info the ODE-RNN sees)."""
    y_last = y_of(t.last)
    if len(t.history) >= 2:
        prev = t.history[-2]
        gap = max(t.last.week - prev.week, 1)
        slope = (y_last - y_of(prev)) / (gap / 52.0)
    else:
        gap, slope = 0, np.zeros(3)
    y_first = y_of(t.history[0])
    dt = t.dt_weeks / 52.0
    since_start = (t.last.week - t.history[0].week) / 52.0
    return np.concatenate(
        [y_last, slope, slope * dt, y_last - y_first, [dt, math.log1p(t.dt_weeks), since_start, len(t.history), gap / 52.0], static]
    )


# ----------------------------------------------------------------------------- baselines


def predict_loglinear(t: Transition) -> np.ndarray:
    y_last = y_of(t.last)
    if len(t.history) < 2:
        return y_last
    prev = t.history[-2]
    rate = (y_last - y_of(prev)) / max(t.last.week - prev.week, 1)
    return np.maximum(y_last + rate * t.dt_weeks, 0.0)


def fit_gompertz(train: list[Transition]) -> np.ndarray:
    """Per compartment: y_next = c + (y_last - c) * exp(-a * dt). Returns (3, 2) of (a, c)."""
    from scipy.optimize import least_squares

    y0 = np.stack([y_of(t.last) for t in train])
    y1 = np.stack([y_of(t.target) for t in train])
    dt = np.array([t.dt_weeks / 52.0 for t in train])
    params = []
    for k in range(3):
        def resid(p, k=k):
            a, c = math.exp(p[0]), p[1]
            return c + (y0[:, k] - c) * np.exp(-a * dt) - y1[:, k]

        res = least_squares(resid, x0=[math.log(0.5), float(y1[:, k].mean())], loss="soft_l1")
        params.append((math.exp(res.x[0]), res.x[1]))
    return np.array(params)


def predict_gompertz(params: np.ndarray, t: Transition) -> np.ndarray:
    y_last = y_of(t.last)
    dt = t.dt_weeks / 52.0
    return params[:, 1] + (y_last - params[:, 1]) * np.exp(-params[:, 0] * dt)


# ----------------------------------------------------------------------------- neural ODE


def build_ode_tensors(transitions: list[Transition], statics: dict[str, np.ndarray], mu: np.ndarray, sd: np.ndarray):
    import torch

    n = len(transitions)
    max_h = min(MAX_ODE_HISTORY, max(len(t.history) for t in transitions))
    obs = np.zeros((n, max_h, 3))
    gaps = np.zeros((n, max_h))  # gap (years) before observation i (0 for the first)
    mask = np.zeros((n, max_h))
    target_dt = np.zeros(n)
    y_last = np.zeros((n, 3))
    y_next = np.zeros((n, 3))
    stat = np.zeros((n, len(next(iter(statics.values())))))
    for i, t in enumerate(transitions):
        # right-align so every sample's last observation sits at index max_h - 1
        hist = t.history[-max_h:]
        offset = max_h - len(hist)
        for j, s in enumerate(hist):
            obs[i, offset + j] = (y_of(s) - mu) / sd
            mask[i, offset + j] = 1.0
            if j > 0:
                gaps[i, offset + j] = (s.week - hist[j - 1].week) / 52.0
        target_dt[i] = t.dt_weeks / 52.0
        y_last[i] = y_of(t.last)
        y_next[i] = y_of(t.target)
        stat[i] = statics[t.patient_id]
    f = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    return f(obs), f(gaps), f(mask), f(target_dt), f(y_last), f(y_next), f(stat)


def make_ode_model(hidden: int, n_static: int):
    import torch
    import torch.nn as nn
    class ScaledODE(nn.Module):
        """dh/ds = dt * f(h) on s in [0, 1]: one batched solve with a per-sample interval."""

        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
            self.dt = None

        def forward(self, s, h):
            return self.dt[:, None] * self.net(h)

    class ODERNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.init = nn.Sequential(nn.Linear(n_static, hidden), nn.Tanh())
            self.ode = ScaledODE()
            self.gru = nn.GRUCell(3, hidden)
            self.dec = nn.Sequential(nn.Linear(hidden + 3, hidden), nn.Tanh(), nn.Linear(hidden, 3))
            nn.init.zeros_(self.dec[-1].weight)
            nn.init.zeros_(self.dec[-1].bias)

        def evolve(self, h, dt, steps: int = 4):
            # fixed-step RK4 on s in [0, 1] (same scheme as torchdiffeq rk4, minus per-call overhead)
            self.ode.dt = dt
            step = 1.0 / steps
            for _ in range(steps):
                k1 = self.ode(None, h)
                k2 = self.ode(None, h + 0.5 * step * k1)
                k3 = self.ode(None, h + 0.5 * step * k2)
                k4 = self.ode(None, h + step * k3)
                h = h + step / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            return h

        def forward(self, obs, gaps, mask, target_dt, stat):
            h = self.init(stat)
            for i in range(obs.shape[1]):
                m = mask[:, i : i + 1]
                h_evolved = self.evolve(h, gaps[:, i])
                h_new = self.gru(obs[:, i], h_evolved)
                h = m * h_new + (1 - m) * h
            h_t = self.evolve(h, target_dt)
            return self.dec(torch.cat([h_t, obs[:, -1]], dim=1))  # standardized delta

    return ODERNN()


def train_neural_ode(train, val, statics, args, seed):
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu = np.stack([y_of(t.target) for t in train]).mean(0)
    sd = np.stack([y_of(t.target) for t in train]).std(0) + 1e-6
    tr = build_ode_tensors(train, statics, mu, sd)
    va = build_ode_tensors(val, statics, mu, sd) if val else None
    model = make_ode_model(args.hidden, tr[-1].shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    sd_t = torch.tensor(sd, dtype=torch.float32)

    def loss_on(tensors):
        obs, gaps, mask, tdt, y_last, y_next, stat = tensors
        pred = model(obs, gaps, mask, tdt, stat)
        return torch.nn.functional.smooth_l1_loss(pred, (y_next - y_last) / sd_t, beta=0.5)

    best, best_state, patience = float("inf"), None, 0
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss = loss_on(tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if va is not None and epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                v = loss_on(va).item()
            if v < best - 1e-5:
                best, best_state, patience = v, {k: x.clone() for k, x in model.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 12:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def predict(transitions):
        tensors = build_ode_tensors(transitions, statics, mu, sd)
        obs, gaps, mask, tdt, y_last, _, stat = tensors
        with torch.no_grad():
            delta = model(obs, gaps, mask, tdt, stat) * sd_t
        return np.maximum((y_last + delta).numpy(), 0.0)

    return predict


# ----------------------------------------------------------------------------- metrics


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):  # average ranks for ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def total_log(y: np.ndarray) -> np.ndarray:
    return np.log1p(np.expm1(y).sum(axis=-1))


def summarize(rows: list[dict], n_boot: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out: dict[str, dict] = {}
    patients = sorted({r["patient_id"] for r in rows})
    for method in METHODS:
        err = np.array([[abs(r[f"{method}_{c}"] - r[f"true_{c}"]) for c in COMPARTMENT_SHORT] for r in rows])
        base = np.array([[abs(r[f"persistence_{c}"] - r[f"true_{c}"]) for c in COMPARTMENT_SHORT] for r in rows])
        pid = np.array([r["patient_id"] for r in rows])
        per_patient = np.array([err[pid == p].mean() for p in patients])
        per_patient_base = np.array([base[pid == p].mean() for p in patients])
        diffs = per_patient - per_patient_base
        boot = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(n_boot)])
        tot_pred = np.array([r[f"{method}_total"] for r in rows])
        tot_true = np.array([r["true_total"] for r in rows])
        tot_last = np.array([r["persistence_total"] for r in rows])
        labels = np.array([r["target_pd"] for r in rows])
        rated = labels >= 0
        out[method] = {
            "mae_mean": float(err.mean()),
            **{f"mae_{c}": float(err[:, k].mean()) for k, c in enumerate(COMPARTMENT_SHORT)},
            "mae_total": float(np.abs(tot_pred - tot_true).mean()),
            "skill_vs_persistence": float(1 - err.mean() / base.mean()),
            "patient_mae_mean": float(per_patient.mean()),
            "patient_mae_diff_vs_persistence": float(diffs.mean()),
            "patient_mae_diff_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "patients_better_than_persistence": int((diffs < 0).sum()),
            "auroc_pd_from_predicted_change": auroc((tot_pred - tot_last)[rated], labels[rated]),
            # persistence predicts exactly zero change, so direction is undefined for it
            "growth_direction_accuracy": float("nan") if method == "persistence" else float(
                (np.sign(tot_pred - tot_last) == np.sign(tot_true - tot_last))[np.abs(tot_true - tot_last) > 0.05].mean()
            ),
        }
    labels = np.array([r["target_pd"] for r in rows])
    rated = labels >= 0
    tot_true = np.array([r["true_total"] for r in rows])
    tot_last = np.array([r["persistence_total"] for r in rows])
    out["_reference"] = {
        "auroc_pd_from_true_volume_change": auroc((tot_true - tot_last)[rated], labels[rated]),
        "pd_rate": float(labels[rated].mean()),
        "n_rated": int(rated.sum()),
    }
    return out


# ----------------------------------------------------------------------------- main


def main() -> None:
    args = parse_args()
    if args.quick:
        args.repeats, args.ensemble, args.epochs, args.bootstrap = 1, 2, 100, 200
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cohort = build_cohort(args.data_dir, args.clinical_dir / "LUMIERE-ExpertRating-v202211.csv")
    clinical = load_clinical(args.clinical_dir / "LUMIERE-Demographics_Pathology.csv")
    patients = cohort.patients
    statics = {p: static_features(p, clinical) for p in patients}
    print(f"Cohort: {len(patients)} patients, {len(cohort.transitions)} transitions, exclusions {cohort.excluded}")

    rows: list[dict] = []
    for repeat in range(args.repeats):
        folds = patient_folds(patients, args.folds, seed=repeat)
        for f_idx, test_patients in enumerate(folds):
            test_set = set(test_patients)
            train_all = [t for t in cohort.transitions if t.patient_id not in test_set]
            test = [t for t in cohort.transitions if t.patient_id in test_set]
            train_patients = sorted({t.patient_id for t in train_all})
            rng = np.random.default_rng(1000 * repeat + f_idx)
            val_patients = set(rng.choice(train_patients, size=max(3, len(train_patients) // 8), replace=False))
            fit = [t for t in train_all if t.patient_id not in val_patients]
            val = [t for t in train_all if t.patient_id in val_patients]

            preds: dict[str, np.ndarray] = {}
            preds["persistence"] = np.stack([y_of(t.last) for t in test])
            preds["loglinear"] = np.stack([predict_loglinear(t) for t in test])
            g = fit_gompertz(train_all)
            preds["gompertz"] = np.stack([predict_gompertz(g, t) for t in test])

            X_tr = np.stack([history_features(t, statics[t.patient_id]) for t in train_all])
            D_tr = np.stack([y_of(t.target) - y_of(t.last) for t in train_all])
            X_te = np.stack([history_features(t, statics[t.patient_id]) for t in test])
            ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12))).fit(X_tr, D_tr)
            preds["ridge"] = np.maximum(preds["persistence"] + ridge.predict(X_te), 0.0)
            gbm = np.stack(
                [
                    HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, loss="absolute_error", random_state=repeat)
                    .fit(X_tr, D_tr[:, k])
                    .predict(X_te)
                    for k in range(3)
                ],
                axis=1,
            )
            preds["gbm"] = np.maximum(preds["persistence"] + gbm, 0.0)

            members = [train_neural_ode(fit, val, statics, args, seed=100 * repeat + 10 * f_idx + e) for e in range(args.ensemble)]
            preds["neural_ode"] = np.mean([m(test) for m in members], axis=0)

            for i, t in enumerate(test):
                rating = (t.target.rating or "").replace(" ", "")
                row = {
                    "repeat": repeat,
                    "fold": f_idx,
                    "patient_id": t.patient_id,
                    "last_week": t.last.week,
                    "target_week": t.target.week,
                    "target_source": t.target.source_name,
                    "dt_weeks": t.dt_weeks,
                    "n_history": len(t.history),
                    "target_rating": t.target.rating or "",
                    "target_pd": 1 if rating.startswith("PD") else (0 if rating in ("SD", "PR", "CR") else -1),
                }
                y_true = y_of(t.target)
                for k, c in enumerate(COMPARTMENT_SHORT):
                    row[f"true_{c}"] = float(y_true[k])
                row["true_total"] = float(total_log(y_true))
                for m, p in preds.items():
                    for k, c in enumerate(COMPARTMENT_SHORT):
                        row[f"{m}_{c}"] = float(p[i, k])
                    row[f"{m}_total"] = float(total_log(p[i]))
                rows.append(row)
            print(f"repeat {repeat} fold {f_idx}: test patients {len(test_patients)}, transitions {len(test)}")

    summary = summarize(rows, args.bootstrap)
    per_repeat = {
        str(r): {m: v["mae_mean"] for m, v in summarize([x for x in rows if x["repeat"] == r], 200).items() if m in METHODS}
        for r in range(args.repeats)
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = {
        "cohort": {
            "patients": len(patients),
            "transitions": len(cohort.transitions),
            "exclusions": cohort.excluded,
            "rules": "see scripts/lumiere_cohort.py",
        },
        "protocol": {
            "folds": args.folds,
            "repeats": args.repeats,
            "ensemble": args.ensemble,
            "epochs_max": args.epochs,
            "hidden": args.hidden,
            "split": "patient-level; neural ODE early-stops on an inner patient-level validation split",
            "target": "log1p(volume mL) per compartment at next scan",
        },
        "summary": summary,
        "mae_mean_by_repeat": per_repeat,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(meta, indent=2))
    write_markdown(meta, args.output_dir / "metrics.md")
    print((args.output_dir / "metrics.md").read_text())


def write_markdown(meta: dict, path: Path) -> None:
    s = meta["summary"]
    lines = [
        "# Step 1: tumor volume trajectory forecasting (LUMIERE)",
        "",
        f"- Patients: {meta['cohort']['patients']}, forecast transitions: {meta['cohort']['transitions']}",
        f"- Exclusions: {meta['cohort']['exclusions']}",
        f"- Protocol: {meta['protocol']['folds']}-fold patient-level CV x {meta['protocol']['repeats']} repeats; "
        f"neural ODE = {meta['protocol']['ensemble']}-seed ensemble",
        "- Error unit: absolute error in log1p(volume, mL), averaged over necrosis / enhancing / edema",
        "- Patient-level diff: mean over patients of (method MAE - persistence MAE); negative = better than persistence; 95% bootstrap CI over patients",
        "",
        "| Method | MAE (mean) | necrosis | enhancing | edema | skill vs persistence | patient diff [95% CI] | patients better | AUROC (PD) | direction acc. |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: |",
    ]
    n_pat = meta["cohort"]["patients"]
    for m in METHODS:
        v = s[m]
        ci = v["patient_mae_diff_ci95"]
        lines.append(
            f"| {m} | {v['mae_mean']:.4f} | {v['mae_necrosis']:.4f} | {v['mae_enhancing']:.4f} | {v['mae_edema']:.4f} | "
            f"{v['skill_vs_persistence']:+.1%} | {v['patient_mae_diff_vs_persistence']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
            f"{v['patients_better_than_persistence']}/{n_pat} | {v['auroc_pd_from_predicted_change']:.3f} | {v['growth_direction_accuracy']:.3f} |"
        )
    r = s["_reference"]
    lines += [
        "",
        f"Reference: AUROC for RANO PD using the *true* next-scan volume change = {r['auroc_pd_from_true_volume_change']:.3f} "
        f"(upper bound for any volume-change score; PD rate {r['pd_rate']:.2f}, n={r['n_rated']} rated targets). "
        "Persistence predicts zero change, so its AUROC is 0.5 by construction.",
        "",
        "Rows are pooled over repeats (each transition appears once per repeat).",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
