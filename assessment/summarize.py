"""Whole-patient uncertainty and readable plots; no fitting or test-selected thresholds."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .cohort import sha


def stats(frame, weights=None):
    y = frame.target.to_numpy(dtype=float)
    p = frame.probability.to_numpy(dtype=float)
    pred = frame.binary_prediction.to_numpy(dtype=float)
    w = np.ones(len(y)) if weights is None else weights
    total = w.sum(); pos = (w*y).sum(); neg = (w*(1-y)).sum()
    def ratio(a,b):
        return float(a/b) if b else None
    order = np.argsort(p, kind='stable')
    sp, sy, sw = p[order], y[order], w[order]
    # Collapse ties before ranking metrics (equivalent to sklearn AUROC and average precision).
    ends = np.r_[np.flatnonzero(np.diff(sp)), len(sp)-1]
    cp = np.cumsum(sw*sy)[ends]; cn = np.cumsum(sw*(1-sy))[ends]
    dp = np.diff(np.r_[0,cp]); dn = np.diff(np.r_[0,cn])
    auc = ratio(np.sum(dp*(cn-.5*dn)), pos*neg)
    above_pos = pos - cp + dp
    above_total = total - cp - cn + dp + dn
    ap = ratio(np.sum(dp*np.divide(above_pos,above_total,out=np.zeros_like(cp),where=above_total>0)),pos)
    covered = frame.band.to_numpy() != 'defer'
    higher = frame.band.to_numpy() == 'higher'
    lower = frame.band.to_numpy() == 'lower'
    ece = 0.
    bins = np.minimum((p*10).astype(int),9)
    for b in range(10):
        mask = bins == b
        ww = w[mask].sum()
        if ww:
            ece += abs((w[mask]*(p[mask]-y[mask])).sum())/total
    result = dict(auroc=auc, average_precision=ap, brier=ratio((w*(p-y)**2).sum(),total),
        sensitivity=ratio((w*y*pred).sum(),pos), specificity=ratio((w*(1-y)*(1-pred)).sum(),neg),
        band_coverage=ratio((w*covered).sum(),total), band_deferral=ratio((w*~covered).sum(),total),
        band_accuracy=ratio((w*(higher*y+lower*(1-y))).sum(),(w*covered).sum()),
        lower_exclusion_sensitivity=ratio((w*y*~lower).sum(),pos),
        higher_inclusion_specificity=ratio((w*(1-y)*~higher).sum(),neg),
        higher_precision=ratio((w*y*higher).sum(),(w*higher).sum()),
        prevalence=ratio(pos,total), mean_probability=ratio((w*p).sum(),total),
        calibration_gap=ratio((w*(p-y)).sum(),total), ece_10bins=float(ece),
        quality_admitted_coverage=0.)
    if frame.fixed_rule_prediction.notna().all():
        fixed = frame.fixed_rule_prediction.to_numpy(dtype=float)
        result.update(fixed_rule_sensitivity=ratio((w*y*fixed).sum(),pos),
                      fixed_rule_specificity=ratio((w*(1-y)*(1-fixed)).sum(),neg))
    return result


def interval_summary(frame):
    point = stats(frame)
    patients, inverse = np.unique(frame.patient_id, return_inverse=True)
    rng = np.random.default_rng(90210)
    samples = {k:[] for k in point}
    for _ in range(2000):
        multiplicity = np.bincount(rng.integers(0,len(patients),len(patients)), minlength=len(patients))
        result = stats(frame, multiplicity[inverse])
        for k,v in result.items():
            if v is not None and np.isfinite(v):
                samples[k].append(v)
    return dict(patients=len(patients), unique_assessments=len(frame[['patient_id','visit']].drop_duplicates()),
                positive_unique_assessments=len(frame[frame.target == 1][['patient_id','visit']].drop_duplicates()),
                prediction_rows=len(frame), metrics={k:dict(value=v,
                    ci95=np.quantile(samples[k],[.025,.975]).tolist() if samples[k] else None,
                    valid_bootstraps=len(samples[k])) for k,v in point.items()})


def summarize(out):
    out = Path(out)
    frame = pd.read_csv(out/'outer_predictions.csv')
    summaries, calibration = [], []
    for (variant,method), group in frame.groupby(['variant','method'],sort=True):
        print(f'Patient bootstrap {variant}/{method}', flush=True)
        item = dict(variant=variant,method=method,all=interval_summary(group),
                    new_progression=interval_summary(group[group.new_progression_eligible]),
                    by_seed={str(s):stats(g) for s,g in group.groupby('seed')})
        summaries.append(item)
        bins = pd.cut(group.probability, np.linspace(0,1,11), include_lowest=True, labels=False)
        for b, part in group.groupby(bins):
            calibration.append(dict(variant=variant, method=method, bin=int(b), count=len(part),
                mean_probability=float(part.probability.mean()), fraction_pd=float(part.target.mean())))
    (out/'metrics.json').write_text(json.dumps(summaries,indent=2,allow_nan=False)+'\n')
    pd.DataFrame(calibration).to_csv(out/'calibration_bins.csv',index=False)
    fig, axes = plt.subplots(1,3,figsize=(15,6), layout='constrained')
    selected = [s for s in summaries if s['variant'] != 'volumes_matched_previous']
    labels = [s['variant'].replace('previous_rating','prev rating')+'/'+s['method'].replace('enhancing_','ET ') for s in selected]
    for ax,metric,title in zip(axes,['auroc','brier','band_coverage'],['AUROC','Brier score','Research band coverage']):
        for i,s in enumerate(selected):
            m = s['all']['metrics'][metric]; v=m['value']; lo,hi=m['ci95']
            ax.errorbar(v,i,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',color='tab:blue')
        ax.set_yticks(range(len(labels)), labels if ax is axes[0] else [])
        ax.set_title(title); ax.grid(axis='x',alpha=.3)
    provenance = 'Frozen config '+sha(out/'freeze.json')[:16]+'; 3 × 5 patient folds; patient bootstrap n=2000'
    fig.suptitle('Exploratory LUMIERE current-rating assessment\n'+provenance)
    fig.savefig(out/'comparison.png',dpi=170,metadata={'Description':provenance})
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    cal = pd.DataFrame(calibration)
    for ax,variant in zip(axes,['volumes','previous_rating']):
        for method, group in cal[cal.variant == variant].groupby('method'):
            ax.plot(group.mean_probability,group.fraction_pd,'o-',label=method,markersize=4)
        ax.plot([0,1],[0,1],'k--',alpha=.5);ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean outer calibrated probability',
            ylabel='Observed expert PD fraction',title=variant)
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Outer predictions, fixed 10-bin calibration\n'+provenance)
    fig.savefig(out/'calibration.png',dpi=170,metadata={'Description':provenance});plt.close(fig)
    lines=['# Completed internal assessment benchmark','',
           'All LUMIERE results are exploratory internal estimates of expert current-visit PD ratings. '
           'No independent patients, anatomical certification, clinical readiness, or formal conformal guarantee.', '',
           'Intervals resample whole patients 2000 times, retaining all their visits and three repeat predictions. '
           'They describe patient sampling, not uncertainty from reselecting the pipeline or changing cohorts. '
           'Patients with many visits contribute more to visit-level metrics. Research bands defer between '
           'training-selected cutoffs. Quality-admitted coverage is **0%** because anatomy is unverified.', '',
           '| Variant / model | Patients / visits / PD | AUROC (95% CI) | AP | Brier | Sensitivity | Specificity | Band coverage |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |']
    def show(m, ci=False):
        if m['value'] is None:return 'undefined'
        return f"{m['value']:.3f}"+(f" ({m['ci95'][0]:.3f}–{m['ci95'][1]:.3f})" if ci else '')
    for s in summaries:
        a=s['all']; m=a['metrics']
        lines.append(f"| {s['variant']} / {s['method']} | {a['patients']} / {a['unique_assessments']} / {a['positive_unique_assessments']} | "+
            ' | '.join(show(m[k],k=='auroc') for k in ['auroc','average_precision','brier','sensitivity','specificity','band_coverage'])+' |')
    lines += ['', '## New-progression subset', '',
              'Previous visit had observed SD/PR/CR; current visits include both PD and non-PD. '
              'Denominators below count unique visits; repeat predictions stay together in patient bootstrap.', '',
              '| Variant / model | Patients / visits / PD | AUROC (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) |',
              '| --- | --- | --- | --- | --- |']
    for s in summaries:
        a=s['new_progression'];m=a['metrics']
        lines.append(f"| {s['variant']} / {s['method']} | {a['patients']} / {a['unique_assessments']} / {a['positive_unique_assessments']} | "+
                     ' | '.join(show(m[k],True) for k in ['auroc','sensitivity','specificity'])+' |')
    lines += ['', 'Machine-readable metrics contain intervals for every measure, calibration gaps/ECE, '
              'band operating performance, fixed heuristic performance, per-seed results and valid bootstrap counts. '
              'Platt calibration and cutoffs use selected-candidate inner OOF scores within each outer training set. '
              'The all-cohort logistic refit has no independent accuracy claim. Rule heuristics are not RANO criteria.', '',
              'Provenance: `freeze.json`, `config.json`, `input_manifest.json`, `cohort.json`, `splits.json`; '
              'local training audits and environment under `runs/assessment_v1/`.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
