"""Tabulate training/validation collapse diagnostics from sealed v4 arms (no test truth is read).

For each setting and arm: seed-averaged effect size, gene-wise r, compound-specific
r, RMSE against zero change, latent ratio and the non-collapse verdict; per-seed
rows; and, as references, the context training mean (B1c) and context Ridge (M1c)
on the same validation pairs, refitted from training truth exactly as the sealed
baselines are.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_context_baselines as v3b  # noqa: E402
from v4_fit import ARMS, load_truth  # noqa: E402
from drugperturb.chemistry import morgan_fingerprint  # noqa: E402
from drugperturb.response import baseline_predictions  # noqa: E402
from drugperturb.v4 import anchoring, counterfactual, diagnostics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COMPOUNDS = {'sciplex3': 'manifests/split_manifest.tsv', 'op3': 'manifests/v4/op3_compounds.tsv'}


def reference_predictions(setting, prepared):
    """B1c and M1c validation predictions per context, from training truth only."""
    spec = counterfactual.SETTINGS[setting]
    table = pd.read_csv(ROOT/COMPOUNDS[spec['dataset']], sep='\t', keep_default_na=False)
    fps = {c: morgan_fingerprint(s).astype(float) for c, s in zip(table.compound_id, table.canonical_smiles)}
    train, val = (load_truth(prepared/f'{s}_truth.npz') for s in ['train', 'validation'])
    out = {'B1c': {}, 'M1c': {}}
    for context in spec['contexts']:
        t_ids = sorted(c for c, x in train if x == context)
        v_ids = sorted(c for c, x in val if x == context)
        ty = np.vstack([train[(c, context)] for c in t_ids])
        vy = np.vstack([val[(c, context)] for c in v_ids])
        tf, vf = (np.vstack([fps[c] for c in ids]) for ids in (t_ids, v_ids))
        ridge, _ = v3b.fit_context(ty, tf, vy, vf, vf)
        mean = baseline_predictions(ty, tf, vf)['B1']
        for i, c in enumerate(v_ids):
            out['B1c'][(c, context)], out['M1c'][(c, context)] = mean[i], ridge[i]
    return out, {k: v for k, v in train.items() if k[1] in spec['contexts']}, \
        {k: v for k, v in val.items() if k[1] in spec['contexts']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sealed-root', type=Path, action='append', required=True)
    parser.add_argument('--prepared', action='append', required=True, help='dataset=path, e.g. op3=/path/prepared')
    parser.add_argument('--setting', action='append', choices=list(counterfactual.SETTINGS),
                        help='settings to tabulate (default: every setting whose dataset is given)')
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    prepared = dict(p.split('=', 1) for p in args.prepared)
    rows, seeds = [], []
    for setting, spec in counterfactual.SETTINGS.items():
        if spec['dataset'] not in prepared or (args.setting and setting not in args.setting):
            continue
        import v4_anchored_baselines  # local import: that script imports this module
        refs, train, val = reference_predictions(setting, Path(prepared[spec['dataset']]))
        queries, levels, vbar, _, _ = v4_anchored_baselines.vehicle_context(setting, Path(prepared[spec['dataset']]))
        for name in ['B1c', 'M1c']:  # amendment A1: vehicle-anchored references
            refs[f'{name[:2]}a'] = anchoring.anchor(refs[name], queries['validation'], levels, vbar)
        b1a = refs['B1a']
        versus_b1a = lambda predicted: float(np.mean([diagnostics.pearson(predicted[k]-b1a[k], val[k]-b1a[k])
                                                      for k in sorted(set(predicted) & set(val))]))
        reference = diagnostics.context_means(train)
        for name, predicted in refs.items():
            s = diagnostics.summarize(predicted, val, reference)
            rows.append(dict(setting=setting, arm=name, split='validation', role='reference', **s,
                             **diagnostics.non_collapse(s), r_vs_B1a=versus_b1a(predicted)))
        for arm in ARMS:
            found = [r for r in args.sealed_root if (r/f'{arm}_{setting}_seal.json').exists()]
            if len(found) != 1:
                raise ValueError(f'{arm} {setting}: expected one seal, found {len(found)}')
            seal = json.loads((found[0]/f'{arm}_{setting}_seal.json').read_text())
            with np.load(found[0]/seal['predictions']['path'], allow_pickle=False) as z:
                validation = {tuple(k.split('|')): v for k, v in zip(z['validation_keys'], z['validation_contrast'])}
            # Seals from code 8eb216d8 name the criterion flags without the pass_ prefix.
            verdict = {k if k == 'non_collapsed' or k.startswith('pass_') else f'pass_{k}': v
                       for k, v in seal['non_collapse'].items()}
            for split in ['train', 'validation']:
                rows.append(dict(setting=setting, arm=arm, split=split, role='chemCPA arm',
                                 **seal['seed_averaged'][split], **{f'latent_{k}': v for k, v in seal['latent'][split].items()},
                                 **(verdict if split == 'validation' else {}),
                                 **(dict(r_vs_B1a=versus_b1a(validation)) if split == 'validation' else {}),
                                 mean_seed_validation_rmse=seal['mean_seed_validation_rmse']))
            for fit in seal['fits']:
                for split in ['train', 'validation']:
                    seeds.append(dict(setting=setting, arm=arm, seed=fit['seed'], split=split,
                                      best_epoch=fit['best_epoch'], epochs_executed=fit['epochs_executed'],
                                      stopped=fit['stopped'], minutes=fit['seconds']/60,
                                      **fit['diagnostics'][split],
                                      **{f'latent_{k}': fit['latent'][split][k] for k in
                                         ['basal_norm', 'drug_term_norm', 'drug_contrast_norm', 'latent_ratio']}))
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out/'diagnostics.tsv', sep='\t', index=False)
    pd.DataFrame(seeds).to_csv(out/'diagnostics_per_seed.tsv', sep='\t', index=False)
    columns = ['setting', 'arm', 'split', 'effect_size_ratio', 'mean_gene_pearson_r', 'compound_specific_r', 'rmse',
               'rmse_zero', 'r_vs_B1a', 'latent_latent_ratio', 'non_collapsed']
    print(table[[c for c in columns if c in table]].to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
