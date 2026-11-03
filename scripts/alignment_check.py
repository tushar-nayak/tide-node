"""Is LUMIERE's atlas-space alignment between visits good enough for voxel-wise forecasting?

Step 2 assumes two scans of one patient are voxel-aligned because DeepBraTumIA resampled
both into the same atlas space. This script tests that assumption on every cohort
transition (last scan -> target scan) by comparing, on the SAME 1 mm grid:

  atlas      the moving scan exactly as LUMIERE ships it (what step 2 uses)
  affine     the same scan after the project's affine registration to the target
             (data/lumiere_registered/, i.e. atlas + an extra per-pair affine refinement)

Alignment is scored only where nothing should change: inside the target brain, outside a
dilated union of both scans' tumor masks. Metrics per transition and modality:
  ncc         normalized cross-correlation with the target's same modality (1 = identical)
  brain_dice  overlap of the two brain masks (skull-stripped foreground)
  shift_mm    distance between the two brain-mask centroids

If the affine refinement improves these a lot, part of the "real change" that step 2
counts is misalignment that no forecasting model could predict.

Outputs: results/alignment/{alignment.csv,summary.md,summary.json} and a figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lumiere_cohort import build_cohort  # noqa: E402
from lumiere_weeks import registered_cache_matches_sources  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODS = {"T1": "t1", "FLAIR": "flair", "T2": "t2"}  # CT1 excluded: it is the registration's fixed image
TUMOR_MARGIN_MM = 5


def load(path) -> np.ndarray:
    v = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / max(hi - lo, 1e-8)


def ncc(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    x, y = a[m], b[m]
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-8))


def dice(a: np.ndarray, b: np.ndarray) -> float:
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def centroid(m: np.ndarray) -> np.ndarray:
    return np.array(ndimage.center_of_mass(m))


def score(job: dict) -> dict:
    out = {k: job[k] for k in ("patient_id", "last_source", "target_source", "dt_weeks")}
    reg_dir = Path(job["reg_dir"])
    if not job["reg_valid"]:
        out["status"] = "no_valid_registration"
        return out
    seg_t = np.asarray(nib.load(job["seg_target"]).dataobj) > 0
    seg_m = np.asarray(nib.load(job["seg_last"]).dataobj) > 0
    tumor = ndimage.binary_dilation(seg_t | seg_m, iterations=TUMOR_MARGIN_MM)
    t1_target = load(job["target"]["T1"])
    brain_t = t1_target > 1e-3
    region = ndimage.binary_erosion(brain_t, iterations=2) & ~tumor
    out["region_ml"] = round(region.sum() / 1000.0, 1)
    out["tumor_ml"] = round((seg_t | seg_m).sum() / 1000.0, 1)
    for mod, prefix in MODS.items():
        tgt = t1_target if mod == "T1" else load(job["target"][mod])
        variants = {"atlas": load(job["last"][mod]), "affine": load(reg_dir / f"{prefix}_skull_strip_registered.nii.gz")}
        for name, mov in variants.items():
            out[f"ncc_{mod}_{name}"] = ncc(mov, tgt, region)
            if mod == "T1":
                brain_m = mov > 1e-3
                out[f"brain_dice_{name}"] = dice(brain_m, brain_t)
                out[f"shift_mm_{name}"] = float(np.linalg.norm(centroid(brain_m) - centroid(brain_t)))
    out["status"] = "ok"
    return out


def boot_ci(x: np.ndarray, groups: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean of per-patient means, resampling patients."""
    ids = np.unique(groups)
    per = np.array([x[groups == g].mean() for g in ids])
    rng = np.random.default_rng(seed)
    means = per[rng.integers(0, len(per), (n, len(per)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def link_step2(rows: list[dict]) -> dict:
    """Does worse atlas alignment go with lower persistence Dice / more 'real change' in step 2?"""
    preds = {}
    for path in (REPO / "runs/mask_forecast/ode").glob("fold*/test_predictions.csv"):
        for r in csv.DictReader(path.open()):
            preds[(r["patient_id"], r["last_source"], r["target_source"])] = r
    xs, ys, zs = [], [], []
    for r in rows:
        p = preds.get((r["patient_id"], r["last_source"], r["target_source"]))
        if p is None:
            continue
        xs.append(r["ncc_T1_atlas"])
        ys.append(float(p["persistence_dice_wt"]))
        zs.append(r["ncc_T1_affine"] - r["ncc_T1_atlas"])
    if len(xs) < 10:
        return {}
    from scipy.stats import spearmanr

    return {
        "n": len(xs),
        "spearman_atlas_ncc_vs_persistence_dice_wt": float(spearmanr(xs, ys).statistic),
        "spearman_affine_gain_vs_persistence_dice_wt": float(spearmanr(zs, ys).statistic),
    }


def figure(rows: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, c = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    panels = [("ncc_T1", "T1 correlation, healthy tissue"), ("ncc_FLAIR", "FLAIR correlation, healthy tissue"),
              ("brain_dice", "brain-mask Dice")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=200)
    for ax, (key, title) in zip(axes, panels):
        a = np.array([r[f"{key}_atlas"] for r in rows])
        b = np.array([r[f"{key}_affine"] for r in rows])
        lo = min(a.min(), b.min())
        lo = max(lo - 0.02, 0.0)
        ax.plot([lo, 1], [lo, 1], color=muted, linewidth=1, zorder=1)
        ax.scatter(a, b, s=12, color=c, alpha=0.6, edgecolors="white", linewidths=0.4, zorder=2)
        ax.set_xlim(lo, 1)
        ax.set_ylim(lo, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("atlas only (step 2 input)", fontsize=8, color=muted)
        ax.set_ylabel("after affine registration", fontsize=8, color=muted)
        above = float((b > a).mean())
        ax.set_title(f"{title}\n{above:.0%} of transitions improved", loc="left", fontsize=9, color=ink)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=muted, labelsize=7)
        ax.grid(color=grid, linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Alignment of last scan to next scan, per cohort transition (points above the line = registration helps)",
                 x=0.01, ha="left", fontsize=10, color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=REPO / "data/lumiere/extracted/Imaging")
    p.add_argument("--clinical-dir", type=Path, default=REPO / "data/lumiere/clinical")
    p.add_argument("--registered-dir", type=Path, default=REPO / "data/lumiere_registered")
    p.add_argument("--out", type=Path, default=REPO / "results/alignment")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="Debug: score only the first N transitions.")
    args = p.parse_args()

    cohort = build_cohort(args.data_dir, args.clinical_dir / "LUMIERE-ExpertRating-v202211.csv", require_seg_mask=True)
    jobs = []
    for t in cohort.transitions:
        last, tgt = t.last.scan, t.target.scan
        reg_dir = (args.registered_dir / t.patient_id / f"target-{t.target.week:03d}" / f"week-{t.last.week:03d}"
                   / "DeepBraTumIA-segmentation/atlas/skull_strip")
        valid = all((reg_dir / f"{p}_skull_strip_registered.nii.gz").exists() for p in MODS.values()) and \
            registered_cache_matches_sources(reg_dir, last, tgt)
        jobs.append({"reg_valid": valid,
            "patient_id": t.patient_id, "last_source": last.source_name, "target_source": tgt.source_name,
            "dt_weeks": t.dt_weeks,
            "last": {m: str(last.modality_paths[m]) for m in MODS}, "target": {m: str(tgt.modality_paths[m]) for m in MODS},
            "seg_last": str(last.seg_mask_path), "seg_target": str(tgt.seg_mask_path),
            "reg_dir": str(reg_dir),
        })
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} transitions", flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(score, jobs, chunksize=2)):
            rows.append(r)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in jobs[0], k))
    with (args.out / "alignment.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    status = {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})}
    groups = np.array([r["patient_id"] for r in ok])
    summary = {"transitions": len(rows), "status": status, "patients": len(set(groups)), "metrics": {}}
    lines = [
        "# Alignment check: atlas space vs affine registration",
        "",
        f"{len(ok)} cohort transitions ({len(set(groups))} patients) scored; status counts {status}.",
        f"Region: target brain (eroded 2 mm) minus both tumor masks dilated by {TUMOR_MARGIN_MM} mm.",
        "Patient diff = mean over patients of (affine - atlas), 95% bootstrap CI over patients.",
        "",
        "| metric | atlas (step 2) median | affine median | patient diff [95% CI] | transitions improved | atlas < 0.9 / affine < 0.9 |",
        "| :--- | ---: | ---: | :---: | ---: | :---: |",
    ]
    for key, better in (("ncc_T1", 1), ("ncc_FLAIR", 1), ("ncc_T2", 1), ("brain_dice", 1), ("shift_mm", -1)):
        a = np.array([r[f"{key}_atlas"] for r in ok])
        b = np.array([r[f"{key}_affine"] for r in ok])
        d = b - a
        lo, hi = boot_ci(d, groups)
        per = float(np.mean([d[groups == g].mean() for g in np.unique(groups)]))
        improved = float(((d * better) > 0).mean())
        low = f"{int((a < 0.9).sum())} / {int((b < 0.9).sum())}" if key != "shift_mm" else "-"
        summary["metrics"][key] = {
            "atlas_median": float(np.median(a)), "affine_median": float(np.median(b)),
            "atlas_p10": float(np.percentile(a, 10)), "affine_p10": float(np.percentile(b, 10)),
            "patient_diff": per, "ci95": [lo, hi], "frac_improved": improved,
        }
        lines.append(f"| {key} | {np.median(a):.3f} | {np.median(b):.3f} | {per:+.3f} [{lo:+.3f}, {hi:+.3f}] | {improved:.0%} | {low} |")
    link = link_step2(ok)
    summary["step2_link"] = link
    if link:
        lines += ["", f"Link to step 2 ({link['n']} transitions, Spearman):",
                  f"* atlas T1 NCC vs persistence WT Dice: {link['spearman_atlas_ncc_vs_persistence_dice_wt']:+.2f}",
                  f"* registration gain in T1 NCC vs persistence WT Dice: {link['spearman_affine_gain_vs_persistence_dice_wt']:+.2f}"]
    worst = sorted(ok, key=lambda r: r["ncc_T1_atlas"])[:10]
    lines += ["", "Worst atlas alignment (T1 NCC):", "", "| patient | last -> target | atlas | affine |", "| :--- | :--- | ---: | ---: |"]
    lines += [f"| {r['patient_id']} | {r['last_source']} -> {r['target_source']} | {r['ncc_T1_atlas']:.3f} | {r['ncc_T1_affine']:.3f} |" for r in worst]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    figure(ok, REPO / "results/figures/alignment_atlas_vs_affine.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
