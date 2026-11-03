"""Register step-2 scan pairs into the LAST scan's frame, masks included. Run once.

``alignment_check.py`` showed that LUMIERE's per-scan atlas alignment leaves consecutive
scans of one patient several mm apart, so step 2's "real change" partly measures
misalignment. ``data/lumiere_registered/`` cannot fix this: it registers each modality
separately, only intensity volumes (no masks), into the *target* frame, and kept no
transforms.

For every cohort transition this script registers, into the frame of its last history
scan (the scan the forecast is made from, so no target information enters the inputs):
  * the target scan          (the ground truth the forecast is scored against)
  * the second-to-last scan  (whose masks are a model input), if the segment has one

Per pair: ONE transform (rigid, then affine refinement), estimated on T1 (pyramid
4 mm -> 2 mm, the cache resolution) with both tumors (dilated) masked out of the metric, so
it is driven by healthy anatomy; initialised at identity (both scans are already roughly in
atlas space). Accepted only if its brain-mask overlap is not worse than the untouched atlas
alignment; otherwise rigid-only, then a second attempt without tumor masks, else identity
(recorded as qc.method). The transform is then applied to all four modalities
(linear) and the segmentation (nearest neighbour), and stored exactly like
``build_atlas_cache.py`` (2 mm, 7 channels), so step 2 can switch frames with one flag.

Outputs (existing pairs are skipped; delete a pair's files to redo it):
  data/lumiere_step2_aligned/<Patient>/<moving>__in__<fixed>.npy   7x91x109x91 float16
  data/lumiere_step2_aligned/<Patient>/<moving>__in__<fixed>.h5    the SimpleITK transform
  data/lumiere_step2_aligned/<Patient>/<moving>__in__<fixed>.json  sources + alignment QC
  data/lumiere_step2_aligned/manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_atlas_cache import CACHE_SHAPE, MODS, pool2  # noqa: E402
from lumiere_cohort import build_cohort  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TUMOR_MARGIN_MM = 5


def pair_stem(moving: str, fixed: str) -> str:
    return f"{moving}__in__{fixed}"


def _np(path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)


def _ncc(a, b, m) -> float:
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-8))


def _cache_channels(mods: dict[str, np.ndarray], seg: np.ndarray) -> np.ndarray:
    """Same normalisation and pooling as build_atlas_cache.build_one."""
    chans = []
    for mod in MODS:
        v = mods[mod]
        m = v > 0
        mu, sd = v[m].mean(), v[m].std() + 1e-6
        chans.append(pool2(np.where(m, np.clip((v - mu) / sd, -5, 5), 0.0)))
    for labels in ((1, 2, 3), (1, 2), (1,)):
        chans.append(pool2(np.isin(seg, labels).astype(np.float32)))
    arr = np.stack(chans).astype(np.float16)
    assert arr.shape == CACHE_SHAPE, arr.shape
    return arr


def as_single_affine(t):
    """Collapse a (composite of) affine-family transform(s) into one AffineTransform.

    A composition of affine maps is affine, so fitting it from 4+ non-coplanar points is
    exact; this keeps the saved file a single plain transform whatever the stacking order.
    """
    import SimpleITK as sitk

    pts = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100], [50, 60, 70], [180, 210, 180]], dtype=float)
    mapped = np.array([t.TransformPoint(tuple(p)) for p in pts])
    A = np.hstack([pts, np.ones((len(pts), 1))])
    sol, *_ = np.linalg.lstsq(A, mapped, rcond=None)  # mapped = pts @ M.T + b
    out = sitk.AffineTransform(3)
    out.SetMatrix(sol[:3].T.ravel().tolist())
    out.SetTranslation(sol[3].tolist())
    check = np.array([out.TransformPoint(tuple(p)) for p in pts])
    assert np.abs(check - mapped).max() < 1e-3, "transform is not affine"
    return out


def register_pair(job: dict) -> dict:
    import SimpleITK as sitk

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(job["threads"]))
    out = Path(job["out_npy"])
    if out.exists() and out.with_suffix(".json").exists():
        return {"stem": out.stem, "status": "cached"}
    t0 = time.time()
    fixed_t1, moving_t1 = _np(job["fixed"]["T1"]), _np(job["moving"]["T1"])
    seg_f = np.asarray(nib.load(job["fixed"]["SEG"]).dataobj).astype(np.int16)
    seg_m = np.asarray(nib.load(job["moving"]["SEG"]).dataobj).astype(np.int16)
    mask_f = (fixed_t1 > 0) & ~ndimage.binary_dilation(seg_f > 0, iterations=TUMOR_MARGIN_MM)
    mask_m = (moving_t1 > 0) & ~ndimage.binary_dilation(seg_m > 0, iterations=TUMOR_MARGIN_MM)

    def img(a, dtype=sitk.sitkFloat32):
        return sitk.Cast(sitk.GetImageFromArray(a), dtype)  # unit spacing, zero origin: same 1 mm grid

    F_img, M_img = img(fixed_t1), img(moving_t1)
    fmask, mmask = img(mask_f.astype(np.uint8), sitk.sitkUInt8), img(mask_m.astype(np.uint8), sitk.sitkUInt8)
    center = F_img.TransformContinuousIndexToPhysicalPoint([(s - 1) / 2 for s in F_img.GetSize()])

    def run(stage, lr, init=None, masked=True):
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.25, seed=1)
        if masked:
            reg.SetMetricFixedMask(fmask)
            reg.SetMetricMovingMask(mmask)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsRegularStepGradientDescent(learningRate=lr, minStep=1e-4, numberOfIterations=300,
                                                     gradientMagnitudeTolerance=1e-8)
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel([4, 2])  # stop at 2 mm: the step-2 cache is 2 mm
        reg.SetSmoothingSigmasPerLevel([2, 1])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        if init is not None:
            reg.SetMovingInitialTransform(init)
        stage.SetCenter(center)
        reg.SetInitialTransform(stage, inPlace=False)
        t = reg.Execute(F_img, M_img)
        return sitk.CompositeTransform([init, t]) if init is not None else t

    M_brain = img((moving_t1 > 0).astype(np.uint8), sitk.sitkUInt8)
    f_brain = fixed_t1 > 0

    def warped_brain(t) -> np.ndarray:
        # nearest neighbour: linear interpolation would smear the brain edge by ~1 voxel and
        # bias every non-identity transform against the identity
        return sitk.GetArrayFromImage(sitk.Resample(M_brain, F_img, t, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)) > 0

    def brain_dice(t) -> float:
        w = warped_brain(t)
        return float(2 * (w & f_brain).sum() / (w.sum() + f_brain.sum()))

    # rigid first (identity init: both scans are roughly in atlas space), then affine on top.
    # Masked affine-only diverges on some post-op pairs; rigid->affine does not. Never accept a
    # result whose brain overlap is worse than the untouched atlas alignment.
    identity = sitk.Euler3DTransform()
    atlas_dice = brain_dice(identity)
    rigid = run(sitk.Euler3DTransform(), 1.0)
    final, method = rigid, "rigid"
    try:
        affine = run(sitk.AffineTransform(3), 0.25, init=rigid)
        if brain_dice(affine) >= brain_dice(rigid) - 0.005:
            final, method = affine, "rigid+affine"
    except RuntimeError:
        pass
    if brain_dice(final) < atlas_dice - 0.002:
        # second attempt without tumor masks: with big lesions / resection cavities the masks
        # can leave too little tissue for the metric. Keep whichever candidate overlaps best.
        cands = []
        try:
            r2 = run(sitk.Euler3DTransform(), 1.0, masked=False)
            cands.append((brain_dice(r2), r2, "rigid_unmasked"))
            a2 = run(sitk.AffineTransform(3), 0.25, init=r2, masked=False)
            cands.append((brain_dice(a2), a2, "rigid+affine_unmasked"))
        except RuntimeError:
            pass
        best = max(cands, key=lambda c: c[0], default=None)
        if best is not None and best[0] >= atlas_dice - 0.002:
            final, method = best[1], best[2]
        else:
            final, method = identity, "atlas_fallback"
    final = as_single_affine(final)

    def warp(a, interp):
        return sitk.GetArrayFromImage(sitk.Resample(img(a), F_img, final, interp, 0.0, sitk.sitkFloat32))

    if method == "atlas_fallback":  # untouched atlas data, no resampling at all
        mods = {m: (moving_t1 if m == "T1" else _np(job["moving"][m])) for m in MODS}
        seg_w, brain_w = seg_m, moving_t1 > 0
    else:
        brain_w = warped_brain(final)
        mods = {m: warp(moving_t1 if m == "T1" else _np(job["moving"][m]), sitk.sitkLinear) * brain_w for m in MODS}
        seg_w = np.rint(warp(seg_m.astype(np.float32), sitk.sitkNearestNeighbor)).astype(np.int16)

    region = ndimage.binary_erosion(fixed_t1 > 0, iterations=2) & mask_f & ~ndimage.binary_dilation(seg_w > 0, iterations=TUMOR_MARGIN_MM)
    qc = {
        "ncc_T1_atlas": _ncc(moving_t1, fixed_t1, region),
        "ncc_T1_registered": _ncc(mods["T1"], fixed_t1, region),
        "brain_dice_atlas": atlas_dice,
        "brain_dice_registered": float(2 * (brain_w & f_brain).sum() / (brain_w.sum() + f_brain.sum())),
        "method": method,
        "seconds": round(time.time() - t0, 1),
    }
    arr = _cache_channels(mods, seg_w)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npy")
    np.save(tmp, arr)
    sitk.WriteTransform(final, str(out.with_suffix(".h5")))
    meta = {"patient_id": job["patient_id"], "moving_source": job["moving_source"], "fixed_source": job["fixed_source"],
            "role": job["role"], "qc": qc}
    tmp.rename(out)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))  # written last = pair complete
    return {"stem": out.stem, "status": "built", **qc}


def needed_pairs(cohort) -> list[tuple[str, object, object, str]]:
    """(patient, moving scan, fixed scan, role) for every transition, deduplicated."""
    seen, pairs = set(), []
    for t in cohort.transitions:
        wanted = [(t.target, "target")]
        if len(t.history) >= 2:
            wanted.append((t.history[-2], "previous"))
        for mov, role in wanted:
            key = (t.patient_id, mov.source_name, t.last.source_name)
            if key not in seen:
                seen.add(key)
                pairs.append((t.patient_id, mov, t.last, role))
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=REPO / "data/lumiere/extracted/Imaging")
    p.add_argument("--clinical-dir", type=Path, default=REPO / "data/lumiere/clinical")
    p.add_argument("--output-dir", type=Path, default=REPO / "data/lumiere_step2_aligned")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--threads", type=int, default=1, help="ITK threads per worker.")
    p.add_argument("--limit", type=int, default=0, help="Debug: only the first N pairs.")
    args = p.parse_args()

    cohort = build_cohort(args.data_dir, args.clinical_dir / "LUMIERE-ExpertRating-v202211.csv", require_seg_mask=True)
    jobs = []
    for pid, mov, fix, role in needed_pairs(cohort):
        paths = lambda s: {**{m: str(s.scan.modality_paths[m]) for m in MODS}, "SEG": str(s.scan.seg_mask_path)}  # noqa: E731
        jobs.append({"patient_id": pid, "moving_source": mov.source_name, "fixed_source": fix.source_name, "role": role,
                     "moving": paths(mov), "fixed": paths(fix), "threads": args.threads,
                     "out_npy": str(args.output_dir / pid / f"{pair_stem(mov.source_name, fix.source_name)}.npy")})
    if args.limit:
        jobs = jobs[: args.limit]
    todo = sum(not (Path(j["out_npy"]).exists() and Path(j["out_npy"]).with_suffix(".json").exists()) for j in jobs)
    print(f"{len(jobs)} pairs ({todo} to register) -> {args.output_dir}", flush=True)
    counts: dict[str, int] = {}
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(register_pair, jobs)):
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            if r["status"] == "built":
                print(f"  [{i + 1}/{len(jobs)}] {r['stem']}: T1 NCC {r['ncc_T1_atlas']:.3f} -> {r['ncc_T1_registered']:.3f}, "
                      f"brain Dice {r['brain_dice_atlas']:.3f} -> {r['brain_dice_registered']:.3f} [{r['method']}] ({r['seconds']}s, "
                      f"elapsed {(time.time() - t0) / 60:.0f} min)", flush=True)
    print("done", counts)
    (args.output_dir / "manifest.json").write_text(json.dumps({"pairs": len(jobs), "counts": counts, "source": str(args.data_dir)}, indent=2))


if __name__ == "__main__":
    main()
