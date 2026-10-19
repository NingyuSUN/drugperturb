"""Select the v3 candidate on validation compounds, average seeds, seal test predictions."""
from pathlib import Path
import argparse
import json

import numpy as np
import yaml

from drugperturb.v3 import runner

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['chemcpa', 'biolord'], required=True)
    parser.add_argument('--protocol', choices=['Q1', 'Q2'], required=True)
    parser.add_argument('--fit-root', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true', help='Engineering runs only')
    args = parser.parse_args()
    protocol = yaml.safe_load((ROOT/'configs/protocol_v3.yaml').read_text())
    spec = protocol['models']['chemCPA' if args.method == 'chemcpa' else 'biolord']
    seeds = protocol['models']['seeds']
    fits, arrays = {}, {}
    for candidate in spec['candidates']:
        for seed in seeds:
            name = f'{args.method}_{args.protocol}_{candidate}_s{seed}'
            record = json.loads((args.fit_root/f'{name}.json').read_text())
            if record['engineering_smoke'] and not args.allow_smoke:
                raise ValueError(f'{name} is an engineering smoke fit')
            if record['test_outcomes_loaded'] is not False:
                raise ValueError(f'{name} does not certify test-outcome blindness')
            fits[(candidate, seed)] = record
            with np.load(args.fit_root/f'{name}.npz', allow_pickle=False) as z:
                arrays[(candidate, seed)] = {k: z[k] for k in z.files}
    reference = arrays[next(iter(arrays))]
    for value in arrays.values():
        for split in ['validation', 'test']:
            if not np.array_equal(value[f'{split}_keys'], reference[f'{split}_keys']):
                raise ValueError('fits disagree on query identities')
    scores = {c: float(np.mean([fits[(c, s)]['validation_rmse'] for s in seeds])) for c in spec['candidates']}
    chosen = min(spec['candidates'], key=lambda c: (scores[c], list(spec['candidates']).index(c)))
    final = {}
    for split in ['validation', 'test']:
        final[f'{split}_keys'] = reference[f'{split}_keys']
        for kind in ['contrast', 'anchored']:
            final[f'{split}_{kind}'] = np.mean([arrays[(chosen, s)][f'{split}_{kind}'] for s in seeds], axis=0)
            for s in seeds:
                final[f'{split}_{kind}_s{s}'] = arrays[(chosen, s)][f'{split}_{kind}']
    inputs = {'protocol': ROOT/'configs/protocol_v3.yaml', 'model_cells': args.prepared/'model_cells.npz',
              'queries': args.prepared/'queries.tsv', 'validation_truth': args.prepared/'validation_truth.npz',
              **{f'fit_{c}_s{s}': args.fit_root/f'{args.method}_{args.protocol}_{c}_s{s}.npz' for c, s in fits}}
    metadata = dict(selected_candidate=chosen, candidate_mean_validation_rmse=scores,
                    selection_metric=protocol['models']['selection_metric'],
                    final_prediction=protocol['models']['final_prediction'],
                    engineering_smoke=any(r['engineering_smoke'] for r in fits.values()),
                    fits=[{k: v for k, v in r.items() if k != 'history'} for r in fits.values()])
    seal = runner.write_seal(args.output_root, args.protocol, args.method, final, metadata, inputs)
    print(json.dumps({'method': args.method, 'protocol': args.protocol, 'selected': chosen,
                      'scores': scores, 'sha256': seal['predictions']['sha256']}, indent=2))


if __name__ == '__main__':
    main()
