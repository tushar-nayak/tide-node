"""Figures for step 1 (volume trajectory) and step 2 (mask forecast) from their metrics.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
HIGHLIGHT, OTHER = "#2a78d6", "#8a8984"  # neural ODE highlighted; every other method recessive
PERSIST, MODEL = "#8a8984", "#2a78d6"


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def step1(out: Path) -> None:
    path = REPO / "results/volume_trajectory/metrics.json"
    if not path.exists():
        return
    s = json.loads(path.read_text())["summary"]
    methods = [m for m in ("loglinear", "gompertz", "ridge", "gbm", "neural_ode")]
    fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=200)
    for i, m in enumerate(methods):
        v = s[m]
        lo, hi = v["patient_mae_diff_ci95"]
        c = HIGHLIGHT if m == "neural_ode" else OTHER
        ax.plot([lo, hi], [i, i], color=c, linewidth=2, solid_capstyle="round")
        ax.plot(v["patient_mae_diff_vs_persistence"], i, "o", color=c, markersize=8, markeredgecolor="white", markeredgewidth=2)
        ax.text(hi + 0.01, i, f"{v['patient_mae_diff_vs_persistence']:+.3f}", va="center", fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(range(len(methods)), [m.replace("_", " ") for m in methods], color=INK)
    ax.set_xlabel("Patient-level MAE minus persistence, log1p(mL)  (left = better)", color=MUTED, fontsize=9)
    ax.set_title("Step 1 - next-visit tumor volume error vs persistence (95% CI)", loc="left", fontsize=10, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "step1_volume_vs_persistence.png")
    plt.close(fig)


def step2(out: Path) -> None:
    path = REPO / "results/mask_forecast/metrics.json"
    if not path.exists():
        return
    s = json.loads(path.read_text())
    variants = [v for v in ("ode", "static") if v in s]
    regions = ("wt", "tc", "et")
    fig, axes = plt.subplots(1, len(variants), figsize=(3.4 * len(variants), 2.8), dpi=200, sharey=True, squeeze=False)
    for ax, v in zip(axes[0], variants):
        r = s[v]
        y = list(range(len(regions)))
        for i, reg in enumerate(regions):
            p, m = r[f"persistence_dice_{reg}"], r[f"model_dice_{reg}"]
            ax.plot([p, m], [i, i], color=GRID, linewidth=2)
            # persistence as a tall tick so it stays visible when the model dot sits on top of it
            ax.plot(p, i, "|", color=PERSIST, markersize=16, markeredgewidth=2.5)
            ax.plot(m, i, "o", color=MODEL, markersize=7, markeredgecolor="white", markeredgewidth=1.5)
            ax.text(max(p, m) + 0.02, i, f"{m:.3f} vs {p:.3f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(y, ["whole tumor", "tumor core", "enhancing"], color=INK)
        ax.set_xlim(0, 1)
        ax.set_title(f"{'Neural ODE' if v == 'ode' else 'No-ODE ablation'} ({r['folds_done']}/5 folds)", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Mean Dice", color=MUTED, fontsize=9)
        style(ax)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=MODEL, markersize=7, label="model"),
        plt.Line2D([], [], marker="|", linestyle="", color=PERSIST, markersize=12, markeredgewidth=2.5, label="persistence"),
    ]
    fig.legend(handles=handles, loc="lower right", frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Step 2 - next-visit 3D tumor mask Dice", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "step2_mask_dice.png")
    plt.close(fig)


def step1_trajectories(out: Path, n: int = 6) -> None:
    """Per-patient total tumor volume over time: real scans vs each held-out forecast."""
    import csv
    import math

    path = REPO / "results/volume_trajectory/predictions.csv"
    if not path.exists():
        return
    rows = [r for r in csv.DictReader(path.open()) if r["repeat"] == "0"]
    by_p: dict[str, list[dict]] = {}
    for r in rows:
        by_p.setdefault(r["patient_id"], []).append(r)
    chosen = sorted(by_p, key=lambda p: -len(by_p[p]))[:n]
    ml = lambda v: math.expm1(float(v))  # noqa: E731
    series = (("persistence", PERSIST, "s"), ("gbm", "#eb6834", "^"), ("neural_ode", HIGHLIGHT, "o"))
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(11, 5.6), dpi=200, squeeze=False)
    for ax, pid in zip(axes.flat, chosen):
        rs = sorted(by_p[pid], key=lambda r: int(r["target_week"]))
        weeks = sorted({int(r["last_week"]) for r in rs} | {int(r["target_week"]) for r in rs})
        true_at = {int(r["target_week"]): ml(r["true_total"]) for r in rs}
        true_at.update({int(r["last_week"]): ml(r["persistence_total"]) for r in rs})
        ax.plot(weeks, [true_at[w] for w in weeks], color=INK, linewidth=2, marker="o", markersize=4, label="real (DeepBraTumIA)", zorder=3)
        for r in rs:
            x0, x1 = int(r["last_week"]), int(r["target_week"])
            for name, color, marker in series:
                ax.plot([x0, x1], [ml(r["persistence_total"]), ml(r[f"{name}_total"])], color=color, linewidth=1, alpha=0.6)
                ax.plot(x1, ml(r[f"{name}_total"]), marker, color=color, markersize=5, markeredgecolor="white", markeredgewidth=0.8)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(pid, loc="left", fontsize=9, color=INK)
        ax.set_xlabel("week", fontsize=8, color=MUTED)
        ax.set_ylabel("total tumor (mL)", fontsize=8, color=MUTED)
        style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    for ax in list(axes.flat)[len(chosen):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=INK, marker="o", linewidth=2, markersize=4, label="real volume")] + [
        plt.Line2D([], [], color=c, marker=m, linestyle="-", linewidth=1, markersize=5, label=f"{nm.replace('_', ' ')} forecast")
        for nm, c, m in series
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Step 1 - next-visit forecasts on held-out patients (each line starts at the last real scan)", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out / "step1_patient_trajectories.png")
    plt.close(fig)


if __name__ == "__main__":
    out = REPO / "results/figures"
    out.mkdir(parents=True, exist_ok=True)
    step1(out)
    step1_trajectories(out)
    step2(out)
    print(sorted(p.name for p in out.glob("*.png")))
