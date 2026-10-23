"""Tabulate v5 training/validation diagnostics from sealed arms and comparators; no test truth is read.

For each arm: seed-averaged effect size, gene-wise r, compound-specific r, RMSE
against zero change, latent ratio, the non-collapse verdict and r against B1a
on validation pairs; per-seed rows; and the comparators' validation predictions
(sealed by v5_baselines.py) as references on the same validation pairs.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_evaluate import by_key, load_sealed  # noqa: E402
from drugperturb.v4 import diagnostics  # noqa: E402
from drugperturb.v4.counterfactual import load_truth  # noqa: E402
from drugperturb.v5.selection import ARMS  # noqa: E402

REFERENCES = ['B1c', 'M1c', 'B1a', 'M1a', 'B1p']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sealed-root', type=Path, nargs='+', required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()

    def root_of(name):
        found = [r for r in args.sealed_root if (r/name).exists()]
        if len(found) != 1:
            raise ValueError(f'{name}: expected one copy, found {len(found)}')
        return found[0]
    train, val = (load_truth(args.prepared/f'{s}_truth.npz') for s in ['train', 'validation'])
    reference = diagnostics.context_means(train)
    base, _ = load_sealed(root_of('baselines_v5_seal.json'), 'baselines', 'v5')
    refs = {m: by_key(base['validation_keys'], base[f'validation_{m}']) for m in REFERENCES}
    b1a = refs['B1a']
    versus_b1a = lambda predicted: float(np.mean([diagnostics.pearson(predicted[k]-b1a[k], val[k]-b1a[k])
                                                  for k in sorted(set(predicted) & set(val))]))
    rows, seeds = [], []
    for name, predicted in refs.items():
        s = diagnostics.summarize(predicted, val, reference)
        rows.append(dict(arm=name, split='validation', role='comparator', **s, **diagnostics.non_collapse(s),
                         r_vs_B1a=versus_b1a(predicted)))
    for arm in ARMS:
        arrays, seal = load_sealed(root_of(f'{arm}_v5_seal.json'), arm, 'v5')
        validation = by_key(arrays['validation_keys'], arrays['validation_contrast'])
        for split in ['train', 'validation']:
            rows.append(dict(arm=arm, split=split, role='chemCPA arm', **seal['seed_averaged'][split],
                             **{f'latent_{k}': v for k, v in seal['latent'][split].items()},
                             **(seal['non_collapse'] if split == 'validation' else {}),
                             **(dict(r_vs_B1a=versus_b1a(validation)) if split == 'validation' else {}),
                             mean_seed_validation_rmse=seal['mean_seed_validation_rmse'],
                             seed_rmse_min=min(seal['seed_validation_rmse']), seed_rmse_max=max(seal['seed_validation_rmse'])))
        for fit in seal['fits']:
            for split in ['train', 'validation']:
                seeds.append(dict(arm=arm, seed=fit['seed'], split=split, best_epoch=fit['best_epoch'],
                                  epochs_executed=fit['epochs_executed'], stopped=fit['stopped'],
                                  minutes=fit['seconds']/60, **fit['diagnostics'][split],
                                  **{f'latent_{k}': fit['latent'][split][k] for k in
                                     ['basal_norm', 'drug_term_norm', 'drug_contrast_norm', 'latent_ratio']}))
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out/'diagnostics.tsv', sep='\t', index=False)
    pd.DataFrame(seeds).to_csv(out/'diagnostics_per_seed.tsv', sep='\t', index=False)
    selection = json.loads((root_of('selection_v5.json')/'selection_v5.json').read_text())
    (out/'selection_summary.json').write_text(json.dumps(diagnostics.finite(
        {k: selection[k] for k in ['selected', 'primary_arms', 'near_tie', 'bonferroni_k', 'interval_level',
                                   'passing', 'any_passing', 'mean_seed_validation_rmse', 'comparisons_to_leader',
                                   'descriptor_gate']}), indent=2)+'\n')
    columns = ['arm', 'split', 'effect_size_ratio', 'mean_gene_pearson_r', 'compound_specific_r', 'rmse', 'rmse_zero',
               'r_vs_B1a', 'latent_latent_ratio', 'non_collapsed', 'mean_seed_validation_rmse']
    print(table[[c for c in columns if c in table]].to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
