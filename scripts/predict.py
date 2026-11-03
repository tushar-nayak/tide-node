#!/usr/bin/env python3
"""Forecast masks or compartment volumes from history-only inputs."""
import argparse
from pathlib import Path

import torch
from prediction import export_prediction, infer, load_bundle, prepare_history, predict_volume_history, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, required=True, help='History-only JSON manifest')
    parser.add_argument('--horizon-weeks', type=float, required=True)
    parser.add_argument('--bundle', type=Path, required=True, help='Packaged directory or mask descriptor JSON')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory (must not exist)')
    parser.add_argument('--threads', type=int, default=4, help='CPU threads (default: 4)')
    parser.add_argument('--trust-volume-bundle', action='store_true', help='Explicitly trust local sklearn pickle')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.output_dir.exists():
        parser.error('--output-dir must not already exist')
    torch.set_num_threads(args.threads)
    try:
        if args.bundle.is_dir() and read_json(args.bundle / 'manifest.json').get('kind') == 'volume_gbm':
            predict_volume_history(args.history, args.horizon_weeks, args.bundle, args.output_dir,
                                   trusted=args.trust_volume_bundle)
            print(f'Wrote forecast to {args.output_dir}')
            return
        model, bundle = load_bundle(args.bundle)
        batch, last, metadata = prepare_history(args.history, args.horizon_weeks, bundle['frame'])
        masks = infer(model, batch)
        metadata['bundle'] = bundle
        export_prediction(args.output_dir, masks, last, metadata)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f'Prediction failed: {exc}\n')
    print(f'Wrote forecast to {args.output_dir}')


if __name__ == '__main__':
    main()
