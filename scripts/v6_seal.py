"""Average seeds per v6 arm, run the validation-only diagnostics, select the primary arm(s), seal.

Reads v6 fit outputs, whose inputs hold no evaluation-half test delta. Before
writing anything it refuses fits whose scaled descriptors exceed
DESCRIPTOR_Z_LIMIT unless --override-descriptor-gate is passed, which is
recorded. Writes one sealed archive per arm and the selection record.

The non-collapse criteria and the selection metric are v6's, not v5's: an arm
must beat chance on validation direction accuracy by at least
MIN_ACCURACY_MARGIN and produce a non-trivial effect, and the leader is the arm
with the HIGHEST validation direction accuracy over responder keys
(drugperturb.v6.selection).
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drugperturb.v3 import runner  # noqa: E402
from drugperturb.v3.descriptors import descriptor_gate  # noqa: E402
from drugperturb.v4 import diagnostics  # noqa: E402
from drugperturb.v4.counterfactual import load_truth  # noqa: E402
from drugperturb.v6 import selection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_STATUS = 'PRESPECIFIED_BEFORE_V6_FITTING'
CHANCE = 0.5
MIN_ACCURACY_MARGIN = 0.01  # protocol_v6 diagnostics_before_test_scoring non_collapse_criteria
MIN_EFFECT_SIZE_RATIO = 0.10
MIN_COMPLETED_SEEDS = 8  # protocol_v6 training.failed_fit


def non_collapse(endpoint, summary):
    """protocol_v6: beat chance on validation direction by the margin, with a non-trivial effect."""
    accuracy = endpoint['validation_direction_accuracy']
    ratio = summary['effect_size_ratio']
    checks = dict(direction_above_chance=bool(np.isfinite(accuracy) and accuracy >= CHANCE+MIN_ACCURACY_MARGIN),
                  effect_size_ratio=bool(np.isfinite(ratio) and ratio >= MIN_EFFECT_SIZE_RATIO))
    return dict(non_collapsed=all(checks.values()), checks=checks,
                validation_direction_accuracy=accuracy, effect_size_ratio=ratio,
                chance=CHANCE, margin=MIN_ACCURACY_MARGIN, min_effect_size_ratio=MIN_EFFECT_SIZE_RATIO)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-root', type=Path, nargs='+', required=True, help='directories holding fits/')
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true', help='Engineering runs only')
    parser.add_argument('--override-descriptor-gate', action='store_true')
    parser.add_argument('--arms', nargs='+', choices=selection.ARMS,
                        help='the arms this run fitted; fewer than all of them is recorded in the selection '
                             'record, and leaves endpoint S4 unestimable')
    parser.add_argument('--seeds', type=int, nargs='+',
                        help='the seeds the plan names for each arm; defaults to the protocol candidates')
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/protocol_v6.yaml')
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol['status'] != PROTOCOL_STATUS:
        raise ValueError('v6 protocol is not prespecified')
    planned = protocol['training']['seeds']['candidates']
    seeds_of = args.seeds or planned
    unknown = sorted(set(seeds_of)-set(planned))
    if unknown:
        raise ValueError(f'seeds {unknown} are not prespecified candidates')
    fitted_arms = args.arms or selection.ARMS
    reduced = sorted(set(selection.ARMS)-set(fitted_arms))
    truth = {s: load_truth(args.prepared/f'{s}_truth.npz') for s in ['train', 'validation']}
    reference = diagnostics.context_means(truth['train'])

    def locate(name):
        found = [r/'fits'/name for r in args.fit_root if (r/'fits'/name).exists()]
        if len(found) != 1:
            raise ValueError(f'{name}: expected exactly one copy, found {len(found)}')
        return found[0]

    fit_records, present = {}, {}
    for arm in fitted_arms:
        seeds = [s for s in seeds_of if (any((r/'fits'/f'{arm}_v6_s{s}.json').exists() for r in args.fit_root))]
        if not seeds:
            raise ValueError(f'{arm}: no fit found')
        if len(seeds) < min(MIN_COMPLETED_SEEDS, len(seeds_of)):
            # protocol_v6 training.failed_fit: an arm with fewer than 8 completed seeds is not a candidate.
            raise ValueError(f'{arm}: the plan names seeds {sorted(seeds_of)} but only {seeds} completed, '
                             f'fewer than the {MIN_COMPLETED_SEEDS} a candidate needs')
        present[arm] = seeds
        for seed in seeds:
            name = f'{arm}_v6_s{seed}'
            record = json.loads(locate(f'{name}.json').read_text())
            if record['engineering_smoke'] and not args.allow_smoke:
                raise ValueError(f'{name} is an engineering smoke fit')
            if record['test_outcomes_loaded'] is not False or record.get('analysis') != 'v6':
                raise ValueError(f'{name} does not certify a test-blind v6 fit')
            if record.get('evaluation_half_test_delta_loaded') is not False:
                raise ValueError(f'{name} does not certify that no evaluation-half test delta was read')
            fit_records[name] = record
    gate = descriptor_gate(fit_records, override=args.override_descriptor_gate)

    arms = {}
    for arm in fitted_arms:
        seeds = present[arm]
        records = {s: fit_records[f'{arm}_v6_s{s}'] for s in seeds}
        arrays = {}
        for seed in seeds:
            with np.load(locate(f'{arm}_v6_s{seed}.npz'), allow_pickle=False) as z:
                arrays[seed] = {k: z[k] for k in z.files}
        first = arrays[seeds[0]]
        for value in arrays.values():
            for split in ['train', 'validation', 'test']:
                if not np.array_equal(value[f'{split}_keys'], first[f'{split}_keys']):
                    raise ValueError(f'{arm}: seeds disagree on query identities')
        final = {}
        for split in ['train', 'validation', 'test']:
            final[f'{split}_keys'] = first[f'{split}_keys']
            for kind in ['contrast', 'anchored', 'raw']:
                if f'{split}_{kind}' not in first:
                    continue
                final[f'{split}_{kind}'] = np.mean([arrays[s][f'{split}_{kind}'] for s in seeds], axis=0)
                for s in seeds:
                    final[f'{split}_{kind}_s{s}'] = arrays[s][f'{split}_{kind}']
        predicted = {s: {tuple(k.split('|')): final[f'{s}_contrast'][i]
                         for i, k in enumerate(np.asarray(final[f'{s}_keys'], str))}
                     for s in ['train', 'validation']}
        train_means = diagnostics.context_means(predicted['train'])
        summary = {}
        for split in ['train', 'validation']:
            shared = {k: v for k, v in predicted[split].items() if k in truth[split]}
            summary[split] = diagnostics.summarize(shared, {k: truth[split][k] for k in shared},
                                                   reference, train_means)
        endpoint = {k: float(np.mean([records[s]['endpoint'][k] for s in seeds]))
                    for k in ['validation_direction_accuracy', 'validation_direction_accuracy_all_keys',
                              'b1p_direction_accuracy', 'split_noise_floor', 'residual_r', 'residual_share']}
        latent = {s: {k: float(np.mean([records[seed]['latent'][s][k] for seed in seeds]))
                      for k in ['basal_norm', 'drug_term_norm', 'drug_contrast_norm', 'latent_ratio']}
                  for s in ['train', 'validation']}
        seed_accuracy = [records[s]['endpoint']['validation_direction_accuracy'] for s in seeds]
        seed_rmse = [records[s]['validation_rmse'] for s in seeds]
        verdict = non_collapse(endpoint, summary['validation'])
        metadata = dict(analysis='v6', arm=arm, seeds=seeds, residual_target=records[seeds[0]]['residual_target'],
                        final_prediction=protocol['training']['final_prediction'],
                        mean_seed_validation_rmse=float(np.mean(seed_rmse)), seed_validation_rmse=seed_rmse,
                        mean_seed_validation_direction_accuracy=float(np.mean(seed_accuracy)),
                        seed_direction_accuracy=seed_accuracy, seed_averaged=summary, endpoint=endpoint,
                        latent=latent, non_collapse=verdict, descriptor_gate=gate,
                        engineering_smoke=any(r['engineering_smoke'] for r in records.values()),
                        fits=[{k: v for k, v in r.items() if k != 'history_tail'} for r in records.values()])
        inputs = {'protocol': args.protocol, 'model_cells': args.prepared/'model_cells.npz',
                  'queries': args.prepared/'queries.tsv', 'validation_truth': args.prepared/'validation_truth.npz',
                  'de_sets': args.prepared/'endpoint/de_sets.npz',
                  **{f'fit_s{s}': locate(f'{arm}_v6_s{s}.npz') for s in seeds}}
        seal = runner.write_seal(args.output_root, 'v6', arm, final, diagnostics.finite(metadata), inputs)
        arms[arm] = dict(seed_direction_accuracy=seed_accuracy, seed_rmse=seed_rmse,
                         validation_direction_accuracy=float(np.mean(seed_accuracy)),
                         validation_rmse=float(np.mean(seed_rmse)), non_collapse=verdict,
                         sha256=seal['predictions']['sha256'])
        print(json.dumps(diagnostics.finite(dict(arm=arm, **{k: arms[arm][k] for k in [
            'validation_direction_accuracy', 'validation_rmse']}, non_collapsed=verdict['non_collapsed']))),
              flush=True)

    rule = protocol['primary_arm_selection']
    candidates = [a for a in rule['candidates'] if a in arms]
    chosen = selection.select(arms, candidates)
    reduced_note = None
    if reduced:
        reduced_note = (f'{reduced} were not fitted, so selection is over the fitted candidates only and '
                        f'endpoint S4 (M2 minus M1) is not estimable in this study.')
    record = dict(status='SELECTED_ON_VALIDATION_BEFORE_TEST_SCORING', test_outcomes_loaded=False,
                  evaluation_half_test_delta_loaded=False, metric=rule['metric'], rule=rule['rule'],
                  near_tie_rule=rule['near_tie_rule'], arms=arms, descriptor_gate=gate,
                  planned_candidates=rule['candidates'], fitted_candidates=candidates,
                  arms_not_fitted=reduced, reduced_plan=reduced_note, seeds_planned=sorted(seeds_of), **chosen)
    (args.output_root/'selection_v6.json').write_text(json.dumps(diagnostics.finite(record), indent=2)+'\n')
    print(json.dumps(diagnostics.finite(chosen), indent=2))


if __name__ == '__main__':
    main()
