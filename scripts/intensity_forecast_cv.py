"""Honest re-test of the original claim: next-scan MRI intensity forecasting, held-out patients.

The original result (+6.7% mean relative MSE improvement over persistence, README /
MANUSCRIPT) was scored on each patient's own training pairs (metric_split=all). This script
keeps the original model and data path but evaluates it properly:

* Same network, loss and inputs as ``run_neural_ode_pipeline.py``: attention U-Net encoder
  per history week, time-aware Neural ODE, zero-initialised warp + residual decoder
  (so the untrained model equals persistence), 5 centre axial slices (z = 89..93) of all
  four modalities, every history scan registered to the target scan via the
  ``data/lumiere_registered/`` cache.
* Same cohort as steps 1/2 (``lumiere_cohort.py``: no pre-op scans, no surgery targets,
  history within one treatment segment, last ``MAX_HISTORY`` scans).
* Patient-level 5-fold CV with the step-1/2 folds; a test patient is never trained on.
  Checkpoint chosen by MSE on held-out *training* patients (init = persistence).

Metrics vs persistence (latest history scan, registered to the target): whole-slice MSE
(the original metric), the original per-patient "relative improvement"
(baseline_mse - model_mse) / baseline_mse averaged over patients, and focal MSE inside
the FLAIR-hyperintense region (where the tumor is).

Usage (each step resumable):
  python scripts/intensity_forecast_cv.py build-cache          # CPU, once
  python scripts/intensity_forecast_cv.py train --fold 0       # GPU
  python scripts/intensity_forecast_cv.py summarize
  python scripts/intensity_forecast_cv.py blur-baseline      # CPU control, see below

blur-baseline: persistence smoothed in-plane with a Gaussian whose width is tuned per fold on
the training patients only. Under uncertainty an MSE-trained forecaster gains simply by
blurring; if this trivial control matches the network, the network's MSE gain is not
evidence of learned tumor dynamics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_neural_ode_pipeline as P  # noqa: E402
from lumiere_cohort import build_cohort  # noqa: E402
from lumiere_weeks import registered_cache_matches_sources  # noqa: E402
from mask_forecast_experiment import fold_split  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODS = ("FLAIR", "T1", "T2", "CT1")  # same order as run_neural_ode_pipeline.MODALITIES
OFFSETS = (-2, -1, 0, 1, 2)
MAX_HISTORY = 4
MAX_WEEKS = 400  # temporal embedding size (largest LUMIERE week is < 300)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("build-cache", "train", "summarize", "blur-baseline"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--model-size", choices=("standard", "tiny"), default="standard")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-steps", type=int, default=0, help="Debug: stop each epoch after N steps.")
    p.add_argument("--data-dir", type=Path, default=REPO / "data/lumiere/extracted/Imaging")
    p.add_argument("--clinical-dir", type=Path, default=REPO / "data/lumiere/clinical")
    p.add_argument("--registered-dir", type=Path, default=REPO / "data/lumiere_registered")
    p.add_argument("--cache-dir", type=Path, default=REPO / "data/lumiere_slices_registered")
    p.add_argument("--runs-dir", type=Path, default=REPO / "runs/intensity_forecast")
    p.add_argument("--results-dir", type=Path, default=REPO / "results/intensity_forecast")
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args()


# ----------------------------------------------------------------------------- data


def load_transitions(args):
    cohort = build_cohort(args.data_dir, args.clinical_dir / "LUMIERE-ExpertRating-v202211.csv", require_seg_mask=True)
    return cohort.transitions


def registered_dir(args, t, scan) -> Path:
    return (args.registered_dir / t.patient_id / f"target-{t.target.week:03d}" / f"week-{scan.week:03d}"
            / "DeepBraTumIA-segmentation/atlas/skull_strip")


def slice_path(cache_dir: Path, t, scan) -> Path:
    """History scan registered to t's target, or the target itself."""
    name = "target" if scan is t.target else f"{scan.source_name}"
    return cache_dir / t.patient_id / f"target-{t.target.source_name}" / f"{name}.npy"


def history_of(t):
    return t.history[-MAX_HISTORY:]


def extract_slices(paths: list[str]) -> np.ndarray:
    """(4, 5, 176, 208) float16, exactly as HistoryForecastDataset._load_week_slices."""
    out = []
    for path in paths:
        v = np.asarray(nib.load(path).dataobj, dtype=np.float32)
        v = (v - v.min()) / max(float(v.max() - v.min()), 1e-8)
        h, w = (v.shape[0] // 16) * 16, (v.shape[1] // 16) * 16
        v = v[:h, :w, :]
        c = v.shape[2] // 2
        out.append(np.stack([v[:, :, min(max(c + o, 0), v.shape[2] - 1)] for o in OFFSETS]))
    return np.stack(out).astype(np.float16)


def _build(job) -> str:
    out, paths = Path(job[0]), job[1]
    if out.exists():
        return "cached"
    arr = extract_slices(paths)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npy")
    np.save(tmp, arr)
    tmp.rename(out)
    return "built"


def build_cache(args) -> None:
    jobs, missing = {}, []
    for t in load_transitions(args):
        jobs[str(slice_path(args.cache_dir, t, t.target))] = [str(t.target.scan.modality_paths[m]) for m in MODS]
        for h in history_of(t):
            d = registered_dir(args, t, h)
            files = [d / f"{m.lower()}_skull_strip_registered.nii.gz" for m in MODS]
            if not all(f.exists() for f in files) or not registered_cache_matches_sources(d, h.scan, t.target.scan):
                missing.append(str(d))
                continue
            jobs[str(slice_path(args.cache_dir, t, h))] = [str(f) for f in files]
    if missing:
        raise SystemExit(f"{len(missing)} registered pairs missing or stale, e.g. {missing[0]}")
    print(f"{len(jobs)} slice stacks -> {args.cache_dir}", flush=True)
    counts: dict[str, int] = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for i, s in enumerate(ex.map(_build, list(jobs.items()), chunksize=4)):
            counts[s] = counts.get(s, 0) + 1
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(jobs)} {counts}", flush=True)
    print("done", counts)


class SliceDataset(Dataset):
    """Yields the same dict layout as HistoryForecastDataset, so P.collate_history_batch works."""

    def __init__(self, transitions, cache_dir: Path) -> None:
        self.t = transitions
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.t)

    def _load(self, t, scan) -> torch.Tensor:
        return torch.from_numpy(np.load(slice_path(self.cache_dir, t, scan)).astype(np.float32))

    def __getitem__(self, idx: int):
        t = self.t[idx]
        hist = history_of(t)
        weeks = [self._load(t, h) for h in hist]  # each (4, 5, H, W)
        history = torch.stack([w.reshape(-1, *w.shape[2:]) for w in weeks])  # (n, 20, H, W)
        target = self._load(t, t.target)
        return {
            "history": history,
            "target": target.reshape(-1, *target.shape[2:]),
            "latest_history_target": history[-1],
            "time_delta": torch.tensor(t.dt_weeks / 52.0, dtype=torch.float32),
            "history_weeks": torch.tensor([h.week for h in hist], dtype=torch.long),
            "patient_id": t.patient_id,
            "target_week": t.target.week,
            "idx": idx,
        }


def collate(batch):
    out = P.collate_history_batch(batch)
    out["idx"] = [b["idx"] for b in batch]
    return out


# ----------------------------------------------------------------------------- train / eval


def forward(model, batch, device):
    return model(batch["history"].to(device), batch["time_delta"].to(device), batch["history_weeks"].to(device),
                 batch["history_mask"].to(device), batch["latest_history_target"].to(device))


def per_sample_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """pred/target: (20, H, W) for one sample."""
    n = len(OFFSETS)
    focal_mse, focal_mae = P.focal_error_metrics(pred[None], target[None], n)
    m = {"mse": float(((pred - target) ** 2).mean()), "mae": float((pred - target).abs().mean()),
         "focal_mse": focal_mse, "focal_mae": focal_mae}
    for k, mod in enumerate(MODS):
        m[f"mse_{mod}"] = float(((pred[k * n:(k + 1) * n] - target[k * n:(k + 1) * n]) ** 2).mean())
    return m


@torch.no_grad()
def evaluate(model, loader, device, transitions, with_rows: bool):
    model.eval()
    mses, rows = [], []
    for batch in loader:
        pred, _ = forward(model, batch, device)
        target = batch["target"].to(device)
        base = batch["latest_history_target"].to(device)
        for b, idx in enumerate(batch["idx"]):
            mm = per_sample_metrics(pred[b], target[b])
            mses.append(mm["mse"])
            if with_rows:
                t = transitions[idx]
                pm = per_sample_metrics(base[b], target[b])
                row = {"patient_id": t.patient_id, "last_source": t.last.source_name, "target_source": t.target.source_name,
                       "dt_weeks": t.dt_weeks, "n_history": len(history_of(t)), "target_rating": t.target.rating or ""}
                row.update({f"model_{k}": v for k, v in mm.items()})
                row.update({f"persistence_{k}": v for k, v in pm.items()})
                rows.append(row)
    return float(np.mean(mses)), rows


def train(args) -> None:
    out_dir = args.runs_dir / f"fold{args.fold}"
    pred_path = out_dir / "test_predictions.csv"
    if pred_path.exists():
        print(f"{pred_path} exists; fold already finished, skipping.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    P.set_seed(args.seed + args.fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit, val, test = fold_split(load_transitions(args), args.folds, args.fold)
    print(f"fold={args.fold}: fit {len(fit)}, val {len(val)}, test {len(test)} transitions", flush=True)
    mk = lambda ts, shuf: DataLoader(SliceDataset(ts, args.cache_dir), batch_size=args.batch_size, shuffle=shuf,  # noqa: E731
                                     num_workers=args.workers, collate_fn=collate, persistent_workers=args.workers > 0)
    fit_dl, val_dl, test_dl = mk(fit, True), mk(val, False), mk(test, False)

    model = P.build_model(week_input_channels=4 * len(OFFSETS), output_channels=4 * len(OFFSETS),
                          max_weeks=MAX_WEEKS, model_size=args.model_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ckpt, state_path = out_dir / "best.pt", out_dir / "last_state.pt"
    if state_path.exists():
        s = torch.load(state_path, map_location=device)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        start, best = s["epoch"] + 1, s["best"]
        print(f"resumed at epoch {start}, best val mse {best:.6f}")
    else:
        start = 0
        best, _ = evaluate(model, val_dl, device, val, False)  # zero-init heads: = persistence
        torch.save(model.state_dict(), ckpt)
        print(f"epoch -1 (persistence init) val mse {best:.6f}", flush=True)
    log = (out_dir / "train_log.csv").open("a")
    n_slices = len(OFFSETS)
    for epoch in range(start, args.epochs):
        model.train()
        t0, losses = time.time(), []
        for step, batch in enumerate(fit_dl):
            if args.max_steps and step >= args.max_steps:
                break
            pred, smooth = forward(model, batch, device)
            loss = P.compute_losses(pred, batch["target"].to(device), n_slices, smooth)["total_loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        val_mse, _ = evaluate(model, val_dl, device, val, False)
        improved = val_mse < best
        if improved:
            best = val_mse
            torch.save(model.state_dict(), ckpt)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "best": best}, state_path)
        print(f"epoch {epoch} loss {np.mean(losses):.5f} val_mse {val_mse:.6f} best {best:.6f} "
              f"{'*' if improved else ''} {time.time() - t0:.0f}s", flush=True)
        log.write(f"{epoch},{np.mean(losses):.6f},{val_mse:.6f}\n"); log.flush()
    torch.save(model.state_dict(), out_dir / "final.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    _, rows = evaluate(model, test_dl, device, test, True)
    with pred_path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    state_path.unlink(missing_ok=True)
    print(f"wrote {pred_path}")


# ----------------------------------------------------------------------------- summarize


def summarize(args) -> None:
    rows, done = [], 0
    for f in range(args.folds):
        p = args.runs_dir / f"fold{f}" / "test_predictions.csv"
        if p.exists():
            rows += list(csv.DictReader(p.open()))
            done += 1
    if not rows:
        raise SystemExit("no finished folds")
    rng = np.random.default_rng(0)
    pid = np.array([r["patient_id"] for r in rows])
    patients = sorted(set(pid))
    get = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731

    def boot(d):
        b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(args.bootstrap)]
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

    res = {"folds_done": done, "transitions": len(rows), "patients": len(patients), "metrics": {}}
    lines = ["# Original intensity-forecast model, re-tested on held-out patients",
             "",
             f"Patient-level 5-fold CV ({done} folds done): {len(rows)} transitions, {len(patients)} patients.",
             "Persistence = latest history scan registered to the target. Slices: 5 centre axial slices, 4 modalities.",
             "rel. improvement = per-patient mean of (persistence - model) / persistence, then averaged over patients "
             "(the statistic behind the original +6.7%); positive = model better. 95% bootstrap CI over patients.",
             "",
             "| metric | model | persistence | rel. improvement [95% CI] | patients better |",
             "| :--- | ---: | ---: | :---: | ---: |"]
    for k in ("mse", "focal_mse", "mae", "mse_FLAIR", "mse_T1", "mse_T2", "mse_CT1"):
        m, b = get(f"model_{k}"), get(f"persistence_{k}")
        rel = np.array([((b[pid == p] - m[pid == p]) / np.maximum(b[pid == p], 1e-12)).mean() for p in patients])
        ci = boot(rel)
        better = int(sum(m[pid == p].mean() < b[pid == p].mean() for p in patients))
        res["metrics"][k] = {"model": float(m.mean()), "persistence": float(b.mean()), "rel_improvement": float(rel.mean()),
                             "rel_improvement_ci95": ci, "patients_better": better}
        lines.append(f"| {k} | {m.mean():.6f} | {b.mean():.6f} | {rel.mean():+.1%} [{ci[0]:+.1%}, {ci[1]:+.1%}] | {better}/{len(patients)} |")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    (args.results_dir / "metrics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _blur_rows(job):
    """Metrics of sigma-blurred persistence for one transition, for every sigma."""
    from scipy.ndimage import gaussian_filter

    cache_dir, t, sigmas = job
    ds = SliceDataset([t], cache_dir)
    item = ds[0]
    base, target = item["latest_history_target"], item["target"]
    out = {}
    for sg in sigmas:
        b = base if sg == 0 else torch.from_numpy(gaussian_filter(base.numpy(), sigma=(0, sg, sg)))
        out[sg] = per_sample_metrics(b, target)
    return out


def blur_baseline(args) -> None:
    sigmas = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
    transitions = load_transitions(args)
    with ProcessPoolExecutor(args.workers) as ex:
        per = list(ex.map(_blur_rows, [(args.cache_dir, t, sigmas) for t in transitions], chunksize=8))
    key = {id(t): i for i, t in enumerate(transitions)}
    model_rows = {}
    for f in range(args.folds):
        path = args.runs_dir / f"fold{f}" / "test_predictions.csv"
        if path.exists():
            for r in csv.DictReader(path.open()):
                model_rows[(r["patient_id"], r["last_source"], r["target_source"])] = r
    rows = []
    chosen = {}
    for f in range(args.folds):
        fit, val, test = fold_split(transitions, args.folds, f)
        train = fit + val
        sg = min(sigmas, key=lambda s_: np.mean([per[key[id(t)]][s_]["mse"] for t in train]))
        chosen[f] = sg
        for t in test:
            r = model_rows.get((t.patient_id, t.last.source_name, t.target.source_name))
            m0, mb = per[key[id(t)]][0.0], per[key[id(t)]][sg]
            rows.append({"patient_id": t.patient_id, "persist": m0, "blur": mb,
                         "model": None if r is None else {k: float(r[f"model_{k}"]) for k in m0}})
    pid = np.array([r["patient_id"] for r in rows])
    patients = sorted(set(pid))
    rng = np.random.default_rng(0)
    lines = ["# Control: Gaussian-blurred persistence vs the network (held-out patients)", "",
             f"Blur sigma (pixels, in-plane) tuned on each fold's training patients: {chosen}.",
             "rel. improvement over plain persistence, same statistic as metrics.md; 95% bootstrap CI over patients.", "",
             "| metric | network | blurred persistence | network minus blur, patient-level [95% CI] |",
             "| :--- | :---: | :---: | :---: |"]
    res = {"sigma_per_fold": chosen, "metrics": {}}
    for k in ("mse", "focal_mse", "mae", "mse_FLAIR", "mse_T1", "mse_T2", "mse_CT1"):
        rel = {}
        for who in ("model", "blur"):
            v = np.array([(r["persist"][k] - r[who][k]) / max(r["persist"][k], 1e-12) for r in rows])
            rel[who] = np.array([v[pid == p].mean() for p in patients])
        d = rel["model"] - rel["blur"]
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(args.bootstrap)]
        ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        res["metrics"][k] = {"model_rel": float(rel["model"].mean()), "blur_rel": float(rel["blur"].mean()),
                             "diff": float(d.mean()), "diff_ci95": ci}
        lines.append(f"| {k} | {rel['model'].mean():+.1%} | {rel['blur'].mean():+.1%} | {d.mean():+.1%} [{ci[0]:+.1%}, {ci[1]:+.1%}] |")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "blur_baseline.md").write_text("\n".join(lines) + "\n")
    (args.results_dir / "blur_baseline.json").write_text(json.dumps(res, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    a = parse_args()
    {"build-cache": build_cache, "train": train, "summarize": summarize, "blur-baseline": blur_baseline}[a.command](a)
