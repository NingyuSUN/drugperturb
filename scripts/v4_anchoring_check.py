"""Validation-only check behind amendment A1: delta and vehicle-anchored baselines against sealed arms.

Usage: v4_anchoring_check.py PREPARED SEALED_ROOT DATASET (reads validation truth only)."""
import sys
import numpy as np
from pathlib import Path
import v4_fit, v4_diagnostics_report as rep
from drugperturb.v3 import runner
from drugperturb.v4 import counterfactual, anchoring, diagnostics
prep, sealed, dataset = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
settings = [s for s, v in counterfactual.SETTINGS.items() if v['dataset'] == dataset]
for setting in settings:
    spec = counterfactual.SETTINGS[setting]; ctx = spec['contexts']; R = spec['replicates']
    obs, counts, library, qt, train_rows, vc, truth, comp = v4_fit.load_setting(setting, prep)
    truth = {s: {k: v for k, v in t.items() if k[1] in ctx} for s, t in truth.items()}
    q = {'train': {k: v for k, v in counterfactual.build_queries(train_rows, ctx, ['train'], R).items() if k in truth['train']},
         'validation': {k: v for k, v in counterfactual.build_queries(qt, ctx, ['validation'], R).items() if k in truth['validation']}}
    tm, vm, _ = counterfactual.setting_cells(obs, ctx, vc)
    keep = np.flatnonzero(tm | vm); sub = obs.iloc[keep].reset_index(drop=True)
    wells = sorted({r['vehicle_well'] for x in q.values() for v in x.values() for r in v})
    veh = runner.vehicle_wells(sub, counts[keep], library[keep], wells)
    lev = anchoring.vehicle_levels(veh)
    vbar = anchoring.training_vehicle_means(q['train'], lev, sorted(q['train']))
    refs, _, val = rep.reference_predictions(setting, prep)
    ref = diagnostics.context_means(truth['train'])
    rows = {}
    for name in ['B1c', 'M1c']:
        rows[name] = refs[name]
        rows[name[:2]+'a'] = anchoring.anchor(refs[name], q['validation'], lev, vbar)
    for arm in v4_fit.ARMS:
        with np.load(sealed/f'{arm}_{setting}_predictions.npz') as z:
            keys, c, a = z['validation_keys'], z['validation_contrast'], z['validation_anchored']
        rows[arm] = {tuple(k.split('|')): c[i] for i, k in enumerate(keys)}
        rows[arm+'_anch'] = {tuple(k.split('|')): a[i] for i, k in enumerate(keys)}
    b1a = rows['B1a']
    print(setting, 'pairs', len(val))
    for name, pred in rows.items():
        s = diagnostics.summarize(pred, val, ref)
        keys = sorted(set(pred) & set(val))
        rb = np.mean([diagnostics.pearson(pred[k]-b1a[k], val[k]-b1a[k]) for k in keys])
        print('  %-38s rmse %.4f zero %.4f r %.3f r_vs_B1a %.3f' % (name, s['rmse'], s['rmse_zero'], s['mean_gene_pearson_r'], rb))
    for arm in v4_fit.ARMS:
        d = np.mean([np.sqrt(np.mean((rows[arm][k]-rows[arm+'_anch'][k])**2)) for k in rows[arm]])
        print('  %-38s vehicle-pass reproduction RMSE %.4f' % (arm, d))
