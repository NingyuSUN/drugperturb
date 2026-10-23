"""Average seeds per arm, compute validation-only diagnostics, select the primary arm(s), seal; never reads test outcomes.

Reads v5 fit outputs (training, validation and test predictions; no test truth
exists in their inputs). Before writing anything, refuses fits whose scaled
descriptors exceed DESCRIPTOR_Z_LIMIT (decision D6) unless
--override-descriptor-gate is passed, which is recorded. Writes one sealed
archive per arm and the selection record (protocol_v5 near-tie rule,
drugperturb.v5.selection), which uses validation compounds only.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v5_fit import ARMS, fit_seeds  # noqa: E402
from drugperturb.v3 import runner  # noqa: E402
from drugperturb.v3.descriptors import descriptor_gate  # noqa: E402
from drugperturb.v4 import diagnostics  # noqa: E402
from drugperturb.v4.counterfactual import load_truth  # noqa: E402
from drugperturb.v5 import selection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def by_key(keys, values):
    return {tuple(k.split('|')): values[i] for i, k in enumerate(keys)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-root', type=Path, nargs='+', required=True, help='directories holding fits/')
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true', help='Engineering runs only')
    parser.add_argument('--override-descriptor-gate', action='store_true')
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/protocol_v5.yaml')
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol['status'] != 'PRESPECIFIED_BEFORE_V5_FITTING':
        raise ValueError('v5 protocol is not prespecified')
    truth = {s: load_truth(args.prepared/f'{s}_truth.npz') for s in ['train', 'validation']}
    reference = diagnostics.context_means(truth['train'])

    def locate(name):
        found = [r/'fits'/name for r in args.fit_root if (r/'fits'/name).exists()]
        if len(found) != 1:
            raise ValueError(f'{name}: expected exactly one copy, found {len(found)}')
        return found[0]
    fit_records = {}
    for arm in ARMS:
        for seed in fit_seeds(protocol, arm):
            name = f'{arm}_v5_s{seed}'
            record = json.loads(locate(f'{name}.json').read_text())
            if record['engineering_smoke'] and not args.allow_smoke:
                raise ValueError(f'{name} is an engineering smoke fit')
            if record['test_outcomes_loaded'] is not False or record.get('analysis') != 'v5':
                raise ValueError(f'{name} does not certify a test-blind v5 fit')
            fit_records[name] = record
    gate = descriptor_gate(fit_records, override=args.override_descriptor_gate)
    arms = {}
    for arm in ARMS:
        seeds = fit_seeds(protocol, arm)
        records = {s: fit_records[f'{arm}_v5_s{s}'] for s in seeds}
        arrays = {}
        for seed in seeds:
            with np.load(locate(f'{arm}_v5_s{seed}.npz'), allow_pickle=False) as z:
                arrays[seed] = {k: z[k] for k in z.files}
        first = arrays[seeds[0]]
        for value in arrays.values():
            for split in ['train', 'validation', 'test']:
                if not np.array_equal(value[f'{split}_keys'], first[f'{split}_keys']):
                    raise ValueError(f'{arm}: seeds disagree on query identities')
        final = {}
        for split in ['train', 'validation', 'test']:
            final[f'{split}_keys'] = first[f'{split}_keys']
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
        seed_rmse = [records[s]['validation_rmse'] for s in seeds]
        verdict = diagnostics.non_collapse(summary['validation'])
        metadata = dict(analysis='v5', arm=arm, seeds=seeds, final_prediction=protocol['training']['final_prediction'],
                        mean_seed_validation_rmse=float(np.mean(seed_rmse)), seed_validation_rmse=seed_rmse,
                        seed_averaged=summary, latent=latent, non_collapse=verdict, descriptor_gate=gate,
                        engineering_smoke=any(r['engineering_smoke'] for r in records.values()),
                        fits=[{k: v for k, v in r.items() if k != 'history_tail'} for r in records.values()])
        inputs = {'protocol': args.protocol, 'model_cells': args.prepared/'model_cells.npz',
                  'queries': args.prepared/'queries.tsv', 'validation_truth': args.prepared/'validation_truth.npz',
                  **{f'fit_s{s}': locate(f'{arm}_v5_s{s}.npz') for s in seeds}}
        seal = runner.write_seal(args.output_root, 'v5', arm, final, diagnostics.finite(metadata), inputs)
        arms[arm] = dict(seed_rmse=seed_rmse, validation_rmse=float(np.mean(seed_rmse)), non_collapse=verdict,
                         sha256=seal['predictions']['sha256'])
        print(json.dumps(diagnostics.finite(dict(arm=arm, validation_rmse=arms[arm]['validation_rmse'], **verdict))),
              flush=True)
    rule = protocol['primary_arm_selection']
    chosen = selection.select(arms, rule['candidates'])
    record = dict(status='SELECTED_ON_VALIDATION_BEFORE_TEST_SCORING', test_outcomes_loaded=False,
                  rule=rule['rule'], near_tie_rule=rule['near_tie_rule'], arms=arms, descriptor_gate=gate, **chosen)
    (args.output_root/'selection_v5.json').write_text(json.dumps(diagnostics.finite(record), indent=2)+'\n')
    print(json.dumps(diagnostics.finite(chosen), indent=2))


if __name__ == '__main__':
    main()
