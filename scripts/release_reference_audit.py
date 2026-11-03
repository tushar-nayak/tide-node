"""Recompute every saved mask reference/persistence metric from read-only caches."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from release_audit import read_csv, require
from mask_forecast_experiment import transition_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results/release_audit/references.json'))
    args = parser.parse_args()
    require(not args.output.resolve().is_relative_to(args.source.resolve()), 'output must be outside source')
    report = {'frames': {}, 'cache_mask_sha256': {}}
    for frame, run in [('atlas', 'mask_forecast'), ('registered', 'mask_forecast_registered')]:
        max_diff, count = 0., 0
        for f in range(5):
            for row in read_csv(args.source/'runs'/run/'ode'/f'fold{f}'/'test_predictions.csv'):
                pid = row['patient_id']
                last_path = args.source/'data/lumiere_atlas2mm'/pid/(row['last_source']+'.npy')
                target_path = (args.source/'data/lumiere_atlas2mm'/pid/(row['target_source']+'.npy') if frame == 'atlas' else
                               args.source/'data/lumiere_step2_aligned'/pid/(row['target_source']+'__in__'+row['last_source']+'.npy'))
                masks = []
                for path in [last_path, target_path]:
                    arr = np.load(path, mmap_mode='r')
                    require(arr.shape == (7, 91, 109, 91), f'cache shape mismatch {path}')
                    mask = np.asarray(arr[4:7])
                    require(np.isfinite(mask).all() and ((mask >= 0) & (mask <= 1)).all(), f'invalid mask {path}')
                    report['cache_mask_sha256'][str(path.relative_to(args.source))] = hashlib.sha256(mask.tobytes()).hexdigest()
                    masks.append(mask > .5)
                metrics = transition_metrics(masks[0], masks[0], masks[1])
                for name, value in metrics.items():
                    column = 'model_'+name if name.startswith('true_') else 'persistence_'+name
                    delta = abs(value - float(row[column]))
                    max_diff = max(max_diff, delta)
                    require(delta < 1e-9, f'reference mismatch {frame} {pid} {row["target_source"]} {column}: {delta}')
                count += 1
        report['frames'][frame] = {'transitions': count, 'max_absolute_metric_difference': max_diff}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report['frames'], indent=2))


if __name__ == '__main__':
    main()
