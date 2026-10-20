"""Average seeds per arm, compute validation-only collapse diagnostics, select the OP3 primary arm, seal.

Reads fit outputs (training, validation and test predictions; no test truth
exists in their inputs). Writes one sealed archive per (setting, arm) and, for
OP3, the prespecified primary-arm selection, which uses validation compounds only.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4_fit import ARMS, load_truth  # noqa: E402
from drugperturb.v3 import runner  # noqa: E402
from drugperturb.v4 import counterfactual, diagnostics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def by_key(keys, values):
    return {tuple(k.split('|')): values[i] for i, k in enumerate(keys)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', choices=list(counterfactual.SETTINGS), required=True)
    parser.add_argument('--fit-root', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true', help='Engineering runs only')
    args = parser.parse_args()
    protocol = yaml.safe_load((ROOT/'configs/protocol_v4.yaml').read_text())
    seeds = protocol['training']['seeds'][args.setting]
    contexts = counterfactual.SETTINGS[args.setting]['contexts']
    truth = {s: {k: v for k, v in load_truth(args.prepared/f'{s}_truth.npz').items() if k[1] in contexts}
             for s in ['train', 'validation']}
    reference = diagnostics.context_means(truth['train'])
    arms = {}
    for arm in ARMS:
        records, arrays = {}, {}
        for seed in seeds:
            name = f'{arm}_{args.setting}_s{seed}'
            record = json.loads((args.fit_root/f'{name}.json').read_text())
            if record['engineering_smoke'] and not args.allow_smoke:
                raise ValueError(f'{name} is an engineering smoke fit')
            if record['test_outcomes_loaded'] is not False:
                raise ValueError(f'{name} does not certify test-outcome blindness')
            records[seed] = record
            with np.load(args.fit_root/f'{name}.npz', allow_pickle=False) as z:
                arrays[seed] = {k: z[k] for k in z.files}
        reference_keys = arrays[seeds[0]]
        for value in arrays.values():
            for split in ['train', 'validation', 'test']:
                if not np.array_equal(value[f'{split}_keys'], reference_keys[f'{split}_keys']):
                    raise ValueError(f'{arm}: seeds disagree on query identities')
        final = {}
        for split in ['train', 'validation', 'test']:
            final[f'{split}_keys'] = reference_keys[f'{split}_keys']
            for kind in ['contrast', 'anchored']:
                final[f'{split}_{kind}'] = np.mean([arrays[s][f'{split}_{kind}'] for s in seeds], axis=0)
                for s in seeds:
                    final[f'{split}_{kind}_s{s}'] = arrays[s][f'{split}_{kind}']
        predicted = {s: by_key(final[f'{s}_keys'], final[f'{s}_contrast']) for s in ['train', 'validation']}
        train_means = diagnostics.context_means(predicted['train'])
        summary = {s: diagnostics.summarize(predicted[s], truth[s], reference, train_means) for s in predicted}
        latent = {s: {k: float(np.mean([records[seed]['latent'][s][k] for seed in seeds]))
                      for k in ['basal_norm', 'drug_term_norm', 'drug_contrast_norm', 'latent_ratio']}
                  for s in ['train', 'validation']}
        validation_rmse = float(np.mean([records[s]['validation_rmse'] for s in seeds]))
        verdict = diagnostics.non_collapse(summary['validation'])
        metadata = dict(setting=args.setting, arm=arm, seeds=seeds, final_prediction=protocol['training']['final_prediction'],
                        mean_seed_validation_rmse=validation_rmse, seed_averaged=summary, latent=latent,
                        non_collapse=verdict, engineering_smoke=any(r['engineering_smoke'] for r in records.values()),
                        fits=[{k: v for k, v in r.items() if k not in ('history_tail',)} for r in records.values()])
        inputs = {'protocol': ROOT/'configs/protocol_v4.yaml', 'model_cells': args.prepared/'model_cells.npz',
                  'queries': args.prepared/'queries.tsv', 'validation_truth': args.prepared/'validation_truth.npz',
                  **{f'fit_s{s}': args.fit_root/f'{arm}_{args.setting}_s{s}.npz' for s in seeds}}
        seal = runner.write_seal(args.output_root, args.setting, arm, final, diagnostics.finite(metadata), inputs)
        arms[arm] = dict(validation_rmse=validation_rmse, non_collapse=verdict, sha256=seal['predictions']['sha256'])
        print(json.dumps(diagnostics.finite(dict(arm=arm, validation_rmse=validation_rmse, **verdict))), flush=True)
    if args.setting == 'op3':
        rule = protocol['primary_arm_selection']
        selection = diagnostics.select_primary(arms, rule['candidates'])
        record = dict(status='SELECTED_ON_VALIDATION_BEFORE_TEST_SCORING', test_outcomes_loaded=False,
                      rule=rule['rule'], arms=arms, **selection)
        (args.output_root/'selection_op3.json').write_text(json.dumps(diagnostics.finite(record), indent=2)+'\n')
        print(json.dumps(diagnostics.finite(selection), indent=2))


if __name__ == '__main__':
    main()
