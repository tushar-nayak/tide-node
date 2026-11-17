#!/usr/bin/env python3
"""Figure for the external test (MANUSCRIPT Figure 3): results/figures/mu_validation.png.

A  ROC curves of the frozen fully automated call. One is on LUMIERE held-out follow-ups
   whose previous scan was not PD; another is on MU-Glioma-Post. An exploratory third is
   the same model trained within MU.
B  Share of scans meeting the volume rule (enhancing >= 40% and >= 1 mL above nadir), by
   label group in each dataset.

Recomputes the held-out and within-MU scores. Both are deterministic, and the script checks
that they give the AUROCs in frozen.json and exploratory.json. CPU, about 30 s.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
REPO = SCRIPTS.parent

from approach_screen import fast_auroc, norm_rating, oof  # noqa: E402
from lumiere_cohort import build_cohort  # noqa: E402
from mu_freeze_model import PRIMARY, gbm_params, rule_fires  # noqa: E402
from mu_validation_explore import compute  # noqa: E402
from progression_robustness import load_volumes, two_compartment_features  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LUMIERE, MU, MU_TRAINED = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical order (dataviz check)


def roc(scores: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
    thr = np.unique(scores)[::-1]
    return ([0.0] + [float((scores[y == 0] >= t).mean()) for t in thr],
            [0.0] + [float((scores[y == 1] >= t).mean()) for t in thr])


def style(ax, grid_axis: str = "both") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def lumiere_scores() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Held-out scores, labels and volume-rule calls on LUMIERE follow-ups whose previous scan was not PD."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    cohort = build_cohort(REPO / "data/lumiere/extracted/Imaging", REPO / "data/lumiere/clinical/LUMIERE-ExpertRating-v202211.csv")
    rows = [t for t in cohort.transitions if norm_rating(t.target.rating) in ("PD", "SD", "PR", "CR")]
    pid = np.array([t.patient_id for t in rows])
    label = np.array([norm_rating(t.target.rating) == "PD" for t in rows], dtype=int)
    new = np.array([norm_rating(t.history[-1].rating) != "PD" for t in rows])
    vols = load_volumes(cohort, REPO / "results/progression_robustness/native_volumes.csv", workers=1)[PRIMARY]
    x = np.stack([two_compartment_features(t.history, t.target, vols) for t in rows])
    scores = oof(pid, cohort.patients,
                 lambda tr, te: HistGradientBoostingClassifier(**gbm_params()).fit(x[tr], label[tr]).predict_proba(x[te])[:, 1])
    enh = lambda s: vols[(s.patient_id, s.source_name)][0]  # noqa: E731
    fires = np.array([rule_fires([enh(s) for s in t.history], enh(t.target)) for t in rows])
    return scores[new], label[new], fires[new]


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frozen = json.loads((REPO / "results/mu_validation/frozen.json").read_text())
    lum_s, lum_y, lum_rule = lumiere_scores()
    lum_auc = fast_auroc(lum_s, lum_y)
    assert abs(lum_auc - frozen["sources"][PRIMARY]["held_out_auroc_previous_not_pd"]["model"]["auroc"]) < 1e-9
    res, mu_y, mu_scores = compute()
    series = ((lum_s, lum_y, LUMIERE, "-", f"LUMIERE, held out (n={len(lum_y)})"),
              (mu_scores["frozen LUMIERE model"], mu_y, MU, "--", f"MU, frozen call (n={len(mu_y)})"),
              (mu_scores["MU-trained, same features"], mu_y, MU_TRAINED, ":", "MU, trained within MU (exploratory)"))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 4.4), dpi=200)
    for s, y, color, ls, name in series:
        fpr, tpr = roc(s, y)
        ax.plot(fpr, tpr, color=color, linewidth=2, linestyle=ls, label=f"{name}: AUROC {fast_auroc(s, y):.2f}")
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False-positive rate (non-PD scans called PD)", color=MUTED, fontsize=8)
    ax.set_ylabel("Sensitivity (PD scans found)", color=MUTED, fontsize=8)
    ax.set_title("A  Frozen call: held-out LUMIERE vs MU-Glioma-Post", loc="left", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=7, loc="lower right", labelcolor=INK, handlelength=2.6)
    style(ax)

    g = res["groups"]
    bars = [("LUMIERE: PD", float(lum_rule[lum_y == 1].mean()), int((lum_y == 1).sum()), LUMIERE),
            ("LUMIERE: non-PD", float(lum_rule[lum_y == 0].mean()), int((lum_y == 0).sum()), LUMIERE),
            ("MU: at documented progression", g["progression scans"]["volume_rule_positive"], g["progression scans"]["scans"], MU),
            ("MU: before it (progressed later)", g["earlier scans of patients who progressed"]["volume_rule_positive"],
             g["earlier scans of patients who progressed"]["scans"], MU),
            ("MU: never progressed", g["scans of patients who never progressed"]["volume_rule_positive"],
             g["scans of patients who never progressed"]["scans"], MU)]
    ypos = np.arange(len(bars))[::-1]
    for yy, (name, share, n, color) in zip(ypos, bars):
        bx.barh(yy, share, height=0.62, color=color, edgecolor="white", linewidth=2)
        bx.text(share + 0.015, yy, f"{share:.0%}", va="center", fontsize=8, color=INK)
    bx.set_yticks(ypos, [f"{name} (n={n})" for name, _, n, _ in bars], color=INK, fontsize=8)
    bx.set_xlim(0, 1)
    bx.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    bx.set_xlabel("Scans meeting the volume rule (≥ 40% and ≥ 1 mL above nadir)", color=MUTED, fontsize=8)
    bx.set_title("B  Growth before documented progression", loc="left", fontsize=9, color=INK)
    bx.legend(handles=[matplotlib.patches.Patch(color=LUMIERE, label="LUMIERE: expert read per scan"),
                       matplotlib.patches.Patch(color=MU, label="MU: documented progression")],
              frameon=False, fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.70), labelcolor=INK)
    style(bx, "x")
    fig.suptitle("External test of the fully automated progression call", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    out = REPO / "results/figures/mu_validation.png"
    fig.savefig(out)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
