"""Full-resolution CPU packaging equivalence, distinct from historical CSV replay."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from lumiere_cohort import build_cohort
from mask_forecast_experiment import MaskForecastDataset, MaskForecastNet, fold_split
from package_models import load_mask, sha256
from prediction import infer, load_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(2)
    cohort = build_cohort(a.source/'data/lumiere/extracted/Imaging',
        a.source/'data/lumiere/clinical/LUMIERE-ExpertRating-v202211.csv', require_seg_mask=True)
    t = fold_split(cohort.transitions,5,1)[2][0]
    report = {'case': dict(patient=t.patient_id, last=t.last.source_name, target=t.target.source_name),
              'scope': 'CPU fp32 export equivalence; not historical CSV reproduction or accuracy evaluation', 'checks': []}
    for frame, folder in [('atlas','mask_forecast'),('registered','mask_forecast_registered')]:
        item = MaskForecastDataset([t],a.source/'data/lumiere_atlas2mm',False,frame,a.source/'data/lumiere_step2_aligned')[0]
        batch = {k:item[k][None] for k in ('x','last_mask','dt')}
        for variant in ['static','ode']:
            bundle = a.release/'masks'/folder/variant/'fold1/best'
            exported, manifest = load_mask(bundle)
            original = MaskForecastNet(**manifest['architecture']['constructor']).eval()
            checkpoint = a.source/'runs'/folder/variant/'fold1/best.pt'
            original.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
            with torch.inference_mode():
                before = original(*(batch[k] for k in ('x','last_mask','dt')))[0]
                after = exported(*(batch[k] for k in ('x','last_mask','dt')))[0]
                torch.testing.assert_close(before,after,rtol=0,atol=0)
            np.testing.assert_array_equal(infer(exported,batch), (before[0,:,:91,:109,:91]>0).numpy())
            ensemble, _ = load_bundle(a.release/f'{frame}-{variant}-cv-ensemble.json')
            result = infer(ensemble,batch)
            assert result.shape == (3,91,109,91)
            report['checks'].append(dict(frame=frame,variant=variant,checkpoint_sha256=sha256(checkpoint),
                exact_full_resolution_logits=True,exact_cli_masks=True,cv_ensemble_forward=True))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
