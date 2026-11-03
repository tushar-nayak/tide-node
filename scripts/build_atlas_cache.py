"""Build the step-2 input cache once: every cohort scan, atlas space, downsampled to 2 mm.

LUMIERE's DeepBraTumIA ``atlas/`` outputs are already resampled into one common 1 mm
atlas space (182x218x182) per scan, so scans of the same patient are voxel-aligned
without any further registration. This script therefore does NOT register anything;
it only normalizes and downsamples, and skips scans already cached (re-runs are free).

Per scan it writes ``<out>/<Patient>/<week-dir-name>.npy`` (float16, shape 7x91x109x91):
  channels 0-3: FLAIR, T1, T2, CT1, z-scored inside the brain, clipped to [-5, 5]
  channels 4-6: fraction of each 2 mm voxel inside whole tumor / tumor core / enhancing
                (WT = labels {1,2,3}, TC = {1,2}, ET = {1}; 1=enhancing, 2=necrosis, 3=edema)
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lumiere_cohort import build_cohort  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODS = ("FLAIR", "T1", "T2", "CT1")
CACHE_SHAPE = (7, 91, 109, 91)


def pool2(vol: np.ndarray) -> np.ndarray:
    x, y, z = (s // 2 * 2 for s in vol.shape)
    v = vol[:x, :y, :z]
    return v.reshape(x // 2, 2, y // 2, 2, z // 2, 2).mean(axis=(1, 3, 5))


def build_one(job: tuple[str, dict, str]) -> str:
    out_path, paths, _ = job
    out = Path(out_path)
    if out.exists():
        return "cached"
    chans = []
    brain = None
    for mod in MODS:
        v = np.asarray(nib.load(paths[mod]).dataobj, dtype=np.float32)
        m = v > 0
        brain = m if brain is None else brain | m
        mu, sd = v[m].mean(), v[m].std() + 1e-6
        z = np.where(m, np.clip((v - mu) / sd, -5, 5), 0.0)
        chans.append(pool2(z))
    seg = np.asarray(nib.load(paths["SEG"]).dataobj).astype(np.int8)
    for labels in ((1, 2, 3), (1, 2), (1,)):
        chans.append(pool2(np.isin(seg, labels).astype(np.float32)))
    arr = np.stack(chans).astype(np.float16)
    assert arr.shape == CACHE_SHAPE, arr.shape
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npy")
    np.save(tmp, arr)
    tmp.rename(out)
    return "built"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=REPO / "data/lumiere/extracted/Imaging")
    p.add_argument("--clinical-dir", type=Path, default=REPO / "data/lumiere/clinical")
    p.add_argument("--output-dir", type=Path, default=REPO / "data/lumiere_atlas2mm")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    cohort = build_cohort(args.data_dir, args.clinical_dir / "LUMIERE-ExpertRating-v202211.csv", require_seg_mask=True)
    jobs = []
    for pid, scans in cohort.scans.items():
        for s in scans:
            paths = {m: str(s.scan.modality_paths[m]) for m in MODS}
            paths["SEG"] = str(s.scan.seg_mask_path)
            jobs.append((str(args.output_dir / pid / f"{s.source_name}.npy"), paths, pid))
    print(f"{len(jobs)} scans to cache under {args.output_dir}")
    counts: dict[str, int] = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for i, status in enumerate(ex.map(build_one, jobs, chunksize=4)):
            counts[status] = counts.get(status, 0) + 1
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)} {counts}", flush=True)
    print("done", counts)
    (args.output_dir / "cache_manifest.json").write_text(
        json.dumps({"scans": len(jobs), "shape": CACHE_SHAPE, "counts": counts, "source": str(args.data_dir)}, indent=2)
    )


if __name__ == "__main__":
    main()
