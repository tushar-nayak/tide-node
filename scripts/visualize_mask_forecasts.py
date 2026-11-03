"""Render step-2 forecasts: what the tumor looked like, what it became, what the model predicted.

For held-out test transitions of one fold, one row per transition (axial slice through the
tumor), columns:
  1. last scan FLAIR + last whole-tumor outline      (what the model sees)
  2. real next scan FLAIR + real outline             (ground truth)
  3. real next FLAIR + predicted outline             (forecast)
  4. growth map vs the last mask: correctly predicted change, missed change, false change

Usage:
  python scripts/visualize_mask_forecasts.py --variant ode --fold 1 --weights final
  (--weights best = early-stopped checkpoint used for the metrics; final = last epoch)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mask_forecast_experiment as M  # noqa: E402

LAST_C, TRUE_C, PRED_C = "#eda100", "#1baf7a", "#2a78d6"
HIT_C, MISS_C, FALSE_C = "#2a78d6", "#eb6834", "#e87ba4"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=("ode", "static"), default="ode")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--weights", choices=("best", "final"), default="final")
    p.add_argument("--n", type=int, default=6, help="Transitions to show (largest real change first).")
    p.add_argument("--frame", choices=("atlas", "registered"), default="atlas")
    p.add_argument("--region", type=int, default=0, help="0=whole tumor, 1=tumor core, 2=enhancing")
    p.add_argument("--out", type=Path, default=M.REPO / "results/figures")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    ns = argparse.Namespace(data_dir=M.REPO / "data/lumiere/extracted/Imaging", clinical_dir=M.REPO / "data/lumiere/clinical")
    transitions, _ = M.load_cohort(ns)
    _, _, test = M.fold_split(transitions, 5, a.fold)
    runs = M.REPO / ("runs/mask_forecast" if a.frame == "atlas" else "runs/mask_forecast_registered")
    weights = runs / a.variant / f"fold{a.fold}" / f"{a.weights}.pt"
    if not weights.exists():
        raise SystemExit(f"missing {weights} (fold not trained yet, or trained before final.pt was saved)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = M.MaskForecastNet(12, 16, a.variant).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    cache, aligned = M.REPO / "data/lumiere_atlas2mm", M.REPO / "data/lumiere_step2_aligned"
    ds = M.MaskForecastDataset(test, cache, augment=False, frame=a.frame, aligned_dir=aligned)
    tpath = lambda t: M.in_last_frame_path(a.frame, cache, aligned, t, t.target)  # noqa: E731
    k = a.region
    items = []
    for i, t in enumerate(test):
        last = np.load(cache / t.patient_id / f"{t.last.source_name}.npy", mmap_mode="r")
        tgt = np.load(tpath(t), mmap_mode="r")
        change = ((last[4 + k] > 0.5) ^ (tgt[4 + k] > 0.5)).sum()
        items.append((int(change), i))
    chosen = [i for _, i in sorted(items, reverse=True)[: a.n]]

    fig, axes = plt.subplots(len(chosen), 4, figsize=(11, 2.9 * len(chosen)), dpi=150, squeeze=False)
    for row, i in enumerate(chosen):
        t = test[i]
        b = ds[i]
        with torch.no_grad():
            logits, _ = model(b["x"][None].to(device), b["last_mask"][None].to(device), b["dt"][None].to(device))
        d, h, w = M.MaskForecastDataset._load(ds, t.last).shape[1:]
        pred = (logits[0, k] > 0).cpu().numpy()[:d, :h, :w]
        last_np = np.load(cache / t.patient_id / f"{t.last.source_name}.npy")
        tgt_np = np.load(tpath(t))
        last_m, true_m = last_np[4 + k] > 0.5, tgt_np[4 + k] > 0.5
        union = last_m | true_m | pred
        z = int(np.argmax(union.sum(axis=(0, 1)))) if union.any() else union.shape[2] // 2

        def sl(v):
            return np.rot90(v[:, :, z])

        flair_last, flair_next = sl(last_np[0].astype(np.float32)), sl(tgt_np[0].astype(np.float32))
        panels = [
            (flair_last, [(sl(last_m), LAST_C)], f"{t.patient_id} {t.last.source_name}\nlast scan (input)"),
            (flair_next, [(sl(true_m), TRUE_C)], f"{t.target.source_name} (+{t.dt_weeks} wk)\nreal next scan"),
            (flair_next, [(sl(pred), PRED_C), (sl(true_m), TRUE_C)], "forecast (blue) vs real (green)"),
        ]
        for col, (img, contours, title) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=-2, vmax=4)
            for mask, color in contours:
                if mask.any():
                    ax.contour(mask, levels=[0.5], colors=color, linewidths=1.2)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        real_change, pred_change = sl(true_m ^ last_m), sl(pred ^ last_m)
        cat = np.zeros(real_change.shape, dtype=int)
        cat[real_change & pred_change] = 1
        cat[real_change & ~pred_change] = 2
        cat[~real_change & pred_change] = 3
        ax = axes[row, 3]
        ax.imshow(flair_next, cmap="gray", vmin=-2, vmax=4)
        ax.imshow(np.ma.masked_equal(cat, 0), cmap=ListedColormap([HIT_C, MISS_C, FALSE_C]), vmin=1, vmax=3, interpolation="nearest")
        n_real = int((true_m ^ last_m).sum())
        n_hit = int(((true_m ^ last_m) & (pred ^ last_m)).sum())
        ax.set_title(f"change vs last scan\ncaught {n_hit}/{n_real} changed voxels (3D)", fontsize=8)
        ax.axis("off")
    fig.legend(
        handles=[Patch(color=HIT_C, label="change predicted correctly"), Patch(color=MISS_C, label="real change missed"),
                 Patch(color=FALSE_C, label="predicted change that did not happen")],
        loc="lower center", ncol=3, fontsize=8, frameon=False,
    )
    region = ("whole tumor", "tumor core", "enhancing")[k]
    fig.suptitle(f"Step 2 forecasts - {a.frame} frame, {a.variant} variant, fold {a.fold}, {a.weights} weights, {region} (held-out patients)", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    a.out.mkdir(parents=True, exist_ok=True)
    tag = "" if a.frame == "atlas" else "registered_"
    out = a.out / f"step2_forecast_{tag}{a.variant}_fold{a.fold}_{a.weights}.png"
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
