"""Step 2: forecast the next-visit tumor segmentation (WT / TC / ET) in 3D.

Inputs come from ``build_atlas_cache.py`` (atlas space, 2 mm, no registration needed).
Cohort and patient folds are identical to step 1 (``lumiere_cohort.py``, fold seed 0).

Per transition the network sees the last scan (4 modalities + 3 region masks), the
previous scan's masks if the segment has one (plus its time gap), and the forecast
interval dt. It predicts a displacement field that warps the last mask (mass effect /
growth front motion) plus residual logits (new or vanished tumor). Both heads are
zero-initialized, so an untrained model reproduces the persistence prediction exactly.

Variants:
  ode     bottleneck features evolved by a learned Neural ODE over [0, dt]
  static  identical network, dt injected by an additive embedding instead of an ODE
          (ablation: does the ODE itself help?)

Metrics on held-out patients, model vs persistence (copy last mask):
  Dice per region, change-Dice (Dice of predicted change vs real change, relative to the
  last mask), and region volume error in log1p(mL).

Frames (--frame):
  atlas       every scan as LUMIERE ships it in atlas space (original step 2). alignment_check.py
              showed consecutive scans are then a median 2.5 mm apart, so part of the
              "change" is misalignment.
  registered  target and second-to-last scan registered into the last scan's frame with one
              transform per pair, masks included (register_step2_pairs.py). The last scan is
              untouched, so inputs carry no target information. Outputs go to
              runs/mask_forecast_registered/ and results/mask_forecast_registered/.

Usage (each fold is resumable; finished folds are skipped):
  python scripts/mask_forecast_experiment.py train --variant ode --fold 0 [--frame registered]
  python scripts/mask_forecast_experiment.py summarize [--frame registered]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lumiere_cohort import Transition, build_cohort, patient_folds  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
REGIONS = ("wt", "tc", "et")
PAD_SHAPE = (96, 112, 96)
VOXEL_ML = 8.0 / 1000.0  # 2 mm isotropic
PERSIST_LOGIT_SCALE = 8.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("train", "summarize"))
    p.add_argument("--variant", choices=("ode", "static"), default="ode")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-steps", type=int, default=0, help="Debug: stop each epoch after N steps.")
    p.add_argument("--data-dir", type=Path, default=REPO / "data/lumiere/extracted/Imaging")
    p.add_argument("--clinical-dir", type=Path, default=REPO / "data/lumiere/clinical")
    p.add_argument("--frame", choices=("atlas", "registered"), default="atlas")
    p.add_argument("--cache-dir", type=Path, default=REPO / "data/lumiere_atlas2mm")
    p.add_argument("--aligned-dir", type=Path, default=REPO / "data/lumiere_step2_aligned")
    p.add_argument("--runs-dir", type=Path, default=None, help="Default: runs/mask_forecast[_registered]")
    p.add_argument("--results-dir", type=Path, default=None, help="Default: results/mask_forecast[_registered]")
    p.add_argument("--bootstrap", type=int, default=2000)
    a = p.parse_args()
    suffix = "" if a.frame == "atlas" else "_registered"
    a.runs_dir = a.runs_dir or REPO / f"runs/mask_forecast{suffix}"
    a.results_dir = a.results_dir or REPO / f"results/mask_forecast{suffix}"
    return a


# ----------------------------------------------------------------------------- data


def load_cohort(args) -> list[Transition]:
    cohort = build_cohort(args.data_dir, args.clinical_dir / "LUMIERE-ExpertRating-v202211.csv", require_seg_mask=True)
    return cohort.transitions, cohort


def pad(x: np.ndarray) -> np.ndarray:
    out = np.zeros(x.shape[:-3] + PAD_SHAPE, dtype=np.float32)
    s = x.shape[-3:]
    out[..., : s[0], : s[1], : s[2]] = x
    return out


def scan_path(cache_dir: Path, scan) -> Path:
    return cache_dir / scan.patient_id / f"{scan.source_name}.npy"


def in_last_frame_path(frame: str, cache_dir: Path, aligned_dir: Path, t: Transition, scan) -> Path:
    """Cache file holding `scan` expressed in the frame of the transition's last scan."""
    if frame == "atlas" or scan is t.last:
        return scan_path(cache_dir, scan)
    return aligned_dir / scan.patient_id / f"{scan.source_name}__in__{t.last.source_name}.npy"


class MaskForecastDataset(Dataset):
    def __init__(self, transitions: list[Transition], cache_dir: Path, augment: bool,
                 frame: str = "atlas", aligned_dir: Path | None = None) -> None:
        self.t = transitions
        self.cache_dir = cache_dir
        self.augment = augment
        self.frame = frame
        self.aligned_dir = aligned_dir
        if frame == "registered":
            missing = [str(self._path(t, s)) for t in transitions for s in ([t.target] + t.history[-2:-1])
                       if not self._path(t, s).exists()]
            if missing:
                raise FileNotFoundError(f"{len(missing)} registered pairs missing (run register_step2_pairs.py), e.g. {missing[0]}")

    def __len__(self) -> int:
        return len(self.t)

    def _path(self, t: Transition, scan) -> Path:
        return in_last_frame_path(self.frame, self.cache_dir, self.aligned_dir, t, scan)

    def _load(self, scan, t: Transition | None = None) -> np.ndarray:
        path = scan_path(self.cache_dir, scan) if t is None else self._path(t, scan)
        return np.load(path, mmap_mode="r")

    def __getitem__(self, idx: int):
        t = self.t[idx]
        last = np.asarray(self._load(t.last), dtype=np.float32)
        if len(t.history) >= 2:
            prev = np.asarray(self._load(t.history[-2], t)[4:7], dtype=np.float32)
            prev_flag = 1.0
            prev_gap = (t.last.week - t.history[-2].week) / 52.0
        else:
            prev = np.zeros_like(last[4:7])
            prev_flag, prev_gap = 0.0, 0.0
        tgt = np.asarray(self._load(t.target, t)[4:7], dtype=np.float32)
        const = np.ones((2,) + last.shape[1:], dtype=np.float32) * np.array([prev_flag, prev_gap], dtype=np.float32)[:, None, None, None]
        x = pad(np.concatenate([last, prev, const]))
        last_mask = pad(last[4:7])
        y = pad((tgt > 0.5).astype(np.float32))
        if self.augment and np.random.rand() < 0.5:  # left-right flip (atlas x axis)
            x, last_mask, y = x[:, ::-1].copy(), last_mask[:, ::-1].copy(), y[:, ::-1].copy()
        return {
            "x": torch.from_numpy(x),
            "last_mask": torch.from_numpy(last_mask),
            "y": torch.from_numpy(y),
            "dt": torch.tensor(t.dt_weeks / 52.0, dtype=torch.float32),
            "idx": idx,
        }


# ----------------------------------------------------------------------------- model


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(min(8, cout), cout),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv3d(cout, cout, 3, padding=1, bias=False),
        nn.GroupNorm(min(8, cout), cout),
        nn.LeakyReLU(0.1, inplace=True),
    )


class BottleneckODE(nn.Module):
    """dz/ds = dt * f(z), s in [0, 1]; fixed-step RK4 (batched per-sample dt), fp32."""

    def __init__(self, ch: int, steps: int = 2) -> None:
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.Tanh(), nn.Conv3d(ch, ch, 3, padding=1)
        )
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)
        self.steps = steps

    def forward(self, z: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            scale = dt.float().view(-1, 1, 1, 1, 1)
            h = 1.0 / self.steps
            for _ in range(self.steps):
                k1 = scale * self.f(z)
                k2 = scale * self.f(z + 0.5 * h * k1)
                k3 = scale * self.f(z + 0.5 * h * k2)
                k4 = scale * self.f(z + h * k3)
                z = z + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return z


class DtEmbedding(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2, ch), nn.Tanh(), nn.Linear(ch, ch))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        e = self.mlp(torch.stack([dt, torch.log1p(dt * 52.0)], dim=1).to(z.dtype))
        return z + e[:, :, None, None, None]


class MaskForecastNet(nn.Module):
    def __init__(self, in_ch: int, base: int, variant: str) -> None:
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList([block(in_ch, ch[0]), block(ch[0], ch[1]), block(ch[1], ch[2]), block(ch[2], ch[3])])
        self.temporal = BottleneckODE(ch[3]) if variant == "ode" else DtEmbedding(ch[3])
        self.up = nn.ModuleList([nn.ConvTranspose3d(ch[i + 1], ch[i], 2, stride=2) for i in range(3)])
        self.dec = nn.ModuleList([block(ch[i] * 2, ch[i]) for i in range(3)])
        self.flow = nn.Conv3d(ch[0], 3, 1)
        self.resid = nn.Conv3d(ch[0], 3, 1)
        for head in (self.flow, self.resid):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x, last_mask, dt):
        skips = []
        h = x
        for i, enc in enumerate(self.enc):
            h = enc(h)
            if i < 3:
                skips.append(h)
                h = F.max_pool3d(h, 2)
        h = self.temporal(h, dt).to(skips[-1].dtype)
        for i in reversed(range(3)):
            h = self.up[i](h)
            h = self.dec[i](torch.cat([h, skips[i]], dim=1))
        flow = 0.1 * torch.tanh(self.flow(h).float())  # normalized coords; <=10% of FOV
        # binarize with the same > 0.5 rule the evaluation uses: soft 0.5 voxels would give
        # logit 0 and flip under float noise in the identity warp
        persist = PERSIST_LOGIT_SCALE * ((last_mask.float() > 0.5).float() - 0.5)
        b = x.shape[0]
        theta = torch.eye(3, 4, device=x.device).unsqueeze(0).expand(b, 3, 4)
        grid = F.affine_grid(theta, list(last_mask.shape), align_corners=True)
        # grid last dim is (x, y, z) = (W, H, D) order; flow channels map to the same order
        warped = F.grid_sample(persist, grid + flow.permute(0, 2, 3, 4, 1), mode="bilinear", padding_mode="border", align_corners=True)
        return warped + self.resid(h).float(), flow


def flow_smoothness(flow: torch.Tensor) -> torch.Tensor:
    return sum((flow.diff(dim=d) ** 2).mean() for d in (2, 3, 4))


def soft_dice_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * y).sum(dim=(2, 3, 4))
    denom = p.sum(dim=(2, 3, 4)) + y.sum(dim=(2, 3, 4))
    return (1 - (2 * inter + 1.0) / (denom + 1.0)).mean()


# ----------------------------------------------------------------------------- metrics


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * (a & b).sum() / s)


def transition_metrics(pred: np.ndarray, last: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """pred/last/truth: bool (3, D, H, W)."""
    out = {}
    for k, r in enumerate(REGIONS):
        out[f"dice_{r}"] = dice(pred[k], truth[k])
        out[f"change_dice_{r}"] = dice(pred[k] ^ last[k], truth[k] ^ last[k])
        out[f"vol_{r}_ml"] = float(pred[k].sum() * VOXEL_ML)
        out[f"true_vol_{r}_ml"] = float(truth[k].sum() * VOXEL_ML)
        out[f"logvol_err_{r}"] = abs(math.log1p(out[f"vol_{r}_ml"]) - math.log1p(out[f"true_vol_{r}_ml"]))
    return out


# ----------------------------------------------------------------------------- train


def fold_split(transitions, n_folds, fold):
    patients = sorted({t.patient_id for t in transitions})
    folds = patient_folds(patients, n_folds, seed=0)  # same assignment as step 1, repeat 0
    test_p = set(folds[fold])
    train_all = [t for t in transitions if t.patient_id not in test_p]
    rng = np.random.default_rng(1000 + fold)
    train_patients = sorted({t.patient_id for t in train_all})
    val_p = set(rng.choice(train_patients, size=max(3, len(train_patients) // 8), replace=False))
    fit = [t for t in train_all if t.patient_id not in val_p]
    val = [t for t in train_all if t.patient_id in val_p]
    test = [t for t in transitions if t.patient_id in test_p]
    return fit, val, test


@torch.no_grad()
def evaluate(model, loader, device, transitions, with_rows: bool):
    model.eval()
    rows, dices = [], []
    for batch in loader:
        x, lm, y, dt = (batch[k].to(device, non_blocking=True) for k in ("x", "last_mask", "y", "dt"))
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits, _ = model(x, lm, dt)
        pred = (logits > 0).cpu().numpy()
        last = (lm > 0.5).cpu().numpy()
        truth = (y > 0.5).cpu().numpy()
        for b, idx in enumerate(batch["idx"].tolist()):
            m = transition_metrics(pred[b], last[b], truth[b])
            dices.append(np.mean([m[f"dice_{r}"] for r in REGIONS]))
            if with_rows:
                t = transitions[idx]
                p = transition_metrics(last[b], last[b], truth[b])
                row = {
                    "patient_id": t.patient_id,
                    "last_source": t.last.source_name,
                    "target_source": t.target.source_name,
                    "dt_weeks": t.dt_weeks,
                    "n_history": len(t.history),
                    "target_rating": t.target.rating or "",
                }
                row.update({f"model_{k}": v for k, v in m.items()})
                row.update({f"persistence_{k}": v for k, v in p.items() if not k.startswith("true_")})
                rows.append(row)
    return float(np.mean(dices)), rows


def train(args) -> None:
    out_dir = args.runs_dir / args.variant / f"fold{args.fold}"
    pred_path = out_dir / "test_predictions.csv"
    if pred_path.exists():
        print(f"{pred_path} exists; fold already finished, skipping.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed + args.fold)
    np.random.seed(args.seed + args.fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transitions, _ = load_cohort(args)
    fit, val, test = fold_split(transitions, args.folds, args.fold)
    print(f"frame={args.frame} variant={args.variant} fold={args.fold}: fit {len(fit)}, val {len(val)}, test {len(test)} transitions", flush=True)

    mk = lambda ts, aug, shuf: DataLoader(  # noqa: E731
        MaskForecastDataset(ts, args.cache_dir, aug, args.frame, args.aligned_dir), batch_size=args.batch_size, shuffle=shuf,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
    )
    fit_dl, val_dl, test_dl = mk(fit, True, True), mk(val, False, False), mk(test, False, False)

    model = MaskForecastNet(in_ch=12, base=args.base, variant=args.variant).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    ckpt = out_dir / "best.pt"
    state_path = out_dir / "last_state.pt"
    start_epoch, best = 0, -1.0
    if state_path.exists():  # resume an interrupted fold
        s = torch.load(state_path, map_location=device)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        start_epoch, best = s["epoch"] + 1, s["best"]
        print(f"resumed at epoch {start_epoch}, best val dice {best:.4f}")
    else:
        best, _ = evaluate(model, val_dl, device, val, False)  # = persistence (zero-init)
        torch.save(model.state_dict(), ckpt)
        print(f"epoch -1 (persistence init) val dice {best:.4f}", flush=True)

    log = (out_dir / "train_log.csv").open("a")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, losses = time.time(), []
        for step, batch in enumerate(fit_dl):
            if args.max_steps and step >= args.max_steps:
                break
            x, lm, y, dt = (batch[k].to(device, non_blocking=True) for k in ("x", "last_mask", "y", "dt"))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, flow = model(x, lm, dt)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y) + soft_dice_loss(logits.float(), y) + 1.0 * flow_smoothness(flow)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()
        val_dice, _ = evaluate(model, val_dl, device, val, False)
        improved = val_dice > best
        if improved:
            best = val_dice
            torch.save(model.state_dict(), ckpt)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch, "best": best}, state_path)
        msg = f"epoch {epoch} loss {np.mean(losses):.4f} val_dice {val_dice:.4f} best {best:.4f} {'*' if improved else ''} {time.time() - t0:.0f}s"
        print(msg, flush=True)
        log.write(f"{epoch},{np.mean(losses):.5f},{val_dice:.5f}\n"); log.flush()

    # keep the final-epoch weights too: when early stopping falls back to the persistence
    # init, these still show what the network learned (used by visualize_mask_forecasts.py)
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
    rng = np.random.default_rng(0)
    out: dict[str, dict] = {}
    for variant in ("ode", "static"):
        rows = []
        for f in range(args.folds):
            p = args.runs_dir / variant / f"fold{f}" / "test_predictions.csv"
            if p.exists():
                rows += list(csv.DictReader(p.open()))
        if not rows:
            continue
        done = sum((args.runs_dir / variant / f"fold{f}" / "test_predictions.csv").exists() for f in range(args.folds))
        pid = np.array([r["patient_id"] for r in rows])
        patients = sorted(set(pid))
        res = {"folds_done": done, "transitions": len(rows), "patients": len(patients)}
        for who in ("model", "persistence"):
            for metric in ("dice", "change_dice", "logvol_err"):
                for r in REGIONS:
                    res[f"{who}_{metric}_{r}"] = float(np.mean([float(x[f"{who}_{metric}_{r}"]) for x in rows]))
        for metric in ("dice", "change_dice", "logvol_err"):
            m = np.array([[float(x[f"model_{metric}_{r}"]) for r in REGIONS] for x in rows]).mean(1)
            b = np.array([[float(x[f"persistence_{metric}_{r}"]) for r in REGIONS] for x in rows]).mean(1)
            d = np.array([(m[pid == p] - b[pid == p]).mean() for p in patients])
            boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(args.bootstrap)]
            res[f"patient_diff_{metric}"] = float(d.mean())
            res[f"patient_diff_{metric}_ci95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
            res[f"patients_better_{metric}"] = int((d < 0).sum() if metric == "logvol_err" else (d > 0).sum())
        out[variant] = res
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    lines = [
        f"# Step 2: 3D next-visit tumor mask forecasting (LUMIERE, 2 mm, frame: {args.frame})",
        "",
        "Patient-level 5-fold CV (same folds as step 1). Persistence = copy the last mask.",
        "change-Dice: Dice between predicted and real change relative to the last mask (persistence scores 0 unless nothing changed).",
        "Patient diff = mean over patients of (model - persistence); 95% bootstrap CI over patients.",
        "",
        "| Variant | folds | transitions | metric | WT model / persist | TC model / persist | ET model / persist | patient diff [95% CI] | patients better |",
        "| :--- | ---: | ---: | :--- | :---: | :---: | :---: | :---: | ---: |",
    ]
    for v, res in out.items():
        for metric in ("dice", "change_dice", "logvol_err"):
            cells = " | ".join(f"{res[f'model_{metric}_{r}']:.3f} / {res[f'persistence_{metric}_{r}']:.3f}" for r in REGIONS)
            ci = res[f"patient_diff_{metric}_ci95"]
            lines.append(
                f"| {v} | {res['folds_done']} | {res['transitions']} | {metric} | {cells} | "
                f"{res[f'patient_diff_{metric}']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {res[f'patients_better_{metric}']}/{res['patients']} |"
            )
    lines += ["", "logvol_err: |log1p(predicted mL) - log1p(true mL)|, lower is better."]
    (args.results_dir / "metrics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    a = parse_args()
    train(a) if a.command == "train" else summarize(a)
