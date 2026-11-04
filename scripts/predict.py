#!/usr/bin/env python3
"""Run a packaged TIDE model on history-only inputs.

Bundle kinds:
  progression_gbm  is the current scan RANO progression? (volumes + previous rating)
  volume_gbm       next-visit compartment volumes, optionally with calibrated ranges
  mask             3D next-visit mask forecast (directory or CV-ensemble descriptor JSON)
"""
import argparse
from pathlib import Path

import torch
from prediction import export_prediction, infer, load_bundle, prepare_history, predict_volume_history, read_json


def print_ranges(ranges):
    print(f"Forecast for week {ranges['forecast_week']:g} ({ranges['horizon_weeks']:g} weeks after the last scan), mL:")
    for name, v in ranges['volumes'].items():
        spans = '  '.join(f'{level} {lo:.1f}-{hi:.1f}' for level, (lo, hi) in v['ranges'].items())
        print(f"  {name:<22} last {v['last_observed']:7.1f}  forecast {v['forecast']:7.1f}  ranges {spans}")
    coverage = (ranges.get('held_out_patient_coverage') or {}).get('Total')
    if coverage:
        print('  Held-out coverage of the total-volume ranges (averaged over patients): '
              + ', '.join(f'{level} range -> {value:.0%}' for level, value in coverage.items()))
    if ranges.get('warning'):
        print(f"  Warning: {ranges['warning']}")


def print_progression(r):
    t80, t90 = r['thresholds']['specificity_80'], r['thresholds']['sensitivity_90']
    m = r['measurements_ml']

    def pct(v):
        return 'n/a' if v is None else f'{v:+.0f}%'
    band = r.get('held_out_pd_rate_in_this_band')
    seen = '' if band is None else f" (on held-out patients, {band:.0%} of scans in this band were rated PD)"
    print(f"Scan at week {r['assessed_week']:g}: probability of RANO progression {r['probability_pd']:.2f} -> {r['call']}{seen}")
    print(f"  'likely PD' at >= {t80['threshold']:.2f} (held-out specificity {t80['held_out_specificity']:.0%}, "
          f"sensitivity {t80['held_out_sensitivity']:.0%}); 'PD less likely' below {t90['threshold']:.2f} "
          f"(sensitivity {t90['held_out_sensitivity']:.0%}, specificity {t90['held_out_specificity']:.0%}); "
          f"held-out AUROC {r['held_out_auroc']:.2f}")
    print(f"  Enhancing tumor {m['enhancing_current']:.1f} mL: {pct(m['enhancing_change_vs_previous_pct'])} vs previous "
          f"({m['enhancing_previous']:.1f}), {pct(m['enhancing_change_vs_nadir_pct'])} vs nadir ({m['enhancing_nadir']:.1f}); "
          f"total {m['total_current']:.1f} mL ({pct(m['total_change_vs_previous_pct'])} vs previous). "
          f"Previous rating: {r['previous_rating']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--history', type=Path, required=True, help='History-only JSON manifest')
    parser.add_argument('--horizon-weeks', type=float, help='Forecast horizon; required for volume and mask bundles')
    parser.add_argument('--bundle', type=Path, required=True, help='Packaged directory or mask descriptor JSON')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory (must not exist)')
    parser.add_argument('--threads', type=int, default=4, help='CPU threads (default: 4)')
    parser.add_argument('--trust-volume-bundle', action='store_true', help='Explicitly trust local sklearn pickle')
    parser.add_argument('--trust-progression-bundle', action='store_true', help='Explicitly trust local sklearn pickle')
    parser.add_argument('--intervals', type=Path,
                        help='Volume bundles only: calibration JSON from volume_intervals.py calibrate; adds forecast ranges')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.output_dir.exists():
        parser.error('--output-dir must not already exist')
    torch.set_num_threads(args.threads)
    kind = read_json(args.bundle / 'manifest.json').get('kind') if (args.bundle / 'manifest.json').is_file() else None
    if kind == 'progression_gbm':
        if args.horizon_weeks is not None or args.intervals is not None:
            parser.error('--horizon-weeks and --intervals do not apply to a progression call')
    elif args.horizon_weeks is None:
        parser.error('--horizon-weeks is required for volume and mask forecasts')
    if args.intervals is not None and kind != 'volume_gbm':
        parser.error('--intervals applies only to volume bundles')
    try:
        if kind == 'progression_gbm':
            from progression_model import predict_progression
            result = predict_progression(args.history, args.bundle, args.output_dir, trusted=args.trust_progression_bundle)
            print(f'Wrote assessment to {args.output_dir}')
            print_progression(result)
            return
        if kind == 'volume_gbm':
            ranges = predict_volume_history(args.history, args.horizon_weeks, args.bundle, args.output_dir,
                                            trusted=args.trust_volume_bundle, intervals=args.intervals)
            print(f'Wrote forecast to {args.output_dir}')
            if ranges is not None:
                print_ranges(ranges)
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
