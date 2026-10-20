"""Information-matched per-context baselines (B0, B1c, B2c, M1c) for OP3; seals test predictions.

Same estimators and alpha grid as v3 (scripts/v3_context_baselines.py), applied
to each OP3 context's eligible training compounds. Validation predictions are
kept as a diagnostics reference. Test outcomes are never read.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_context_baselines as v3  # noqa: E402
from drugperturb.chemistry import morgan_fingerprint  # noqa: E402
from drugperturb.response import baseline_predictions  # noqa: E402
from drugperturb.v3 import runner  # noqa: E402
from drugperturb.v4.counterfactual import SETTINGS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    protocol = yaml.safe_load((ROOT/'configs/protocol_v4.yaml').read_text())
    alphas = [float(a) for a in protocol['comparators']['ridge_alphas']]
    if alphas != v3.ALPHAS:
        raise ValueError('v4 Ridge alpha grid must equal the v3 grid used by the shared estimator')
    compounds = pd.read_csv(ROOT/'manifests/v4/op3_compounds.tsv', sep='\t', keep_default_na=False)
    fps = {c: morgan_fingerprint(s).astype(float) for c, s in zip(compounds.compound_id, compounds.canonical_smiles)}
    train, val = v3.load_truth(args.prepared/'train_truth.npz'), v3.load_truth(args.prepared/'validation_truth.npz')
    queries = runner.read_query_table(args.prepared/'queries.tsv')
    tests = queries[queries.split.eq('test')]
    keys, val_keys, selection = [], [], {}
    arrays = {m: [] for m in ['B0', 'B1c', 'B2c', 'M1c']}
    val_arrays = {m: [] for m in ['B1c', 'M1c']}
    for context in SETTINGS['op3']['contexts']:
        t_ids = sorted(c for c, x in train if x == context)
        v_ids = sorted(c for c, x in val if x == context)
        q_ids = sorted(set(tests.loc[tests.context_id.eq(context), 'compound_id']))
        if len(t_ids) < 2 or not v_ids or not q_ids:
            raise ValueError(f'{context}: insufficient training/validation/query compounds')
        ty = np.vstack([train[(c, context)] for c in t_ids])
        vy = np.vstack([val[(c, context)] for c in v_ids])
        tf, vf, qf = (np.vstack([fps[c] for c in ids]) for ids in (t_ids, v_ids, q_ids))
        simple = baseline_predictions(ty, tf, qf)
        ridge, info = v3.fit_context(ty, tf, vy, vf, qf)
        ridge_val, _ = v3.fit_context(ty, tf, vy, vf, vf)
        val_simple = baseline_predictions(ty, tf, vf)
        info['validation_rmse_simple'] = {k: v3.rmse(vy, v) for k, v in val_simple.items()}
        info.update(n_validation=len(v_ids), n_test_queries=len(q_ids))
        selection[context] = info
        keys += [(c, context) for c in q_ids]
        val_keys += [(c, context) for c in v_ids]
        arrays['B0'].append(simple['B0']); arrays['B1c'].append(simple['B1'])
        arrays['B2c'].append(simple['B2']); arrays['M1c'].append(ridge)
        val_arrays['B1c'].append(val_simple['B1']); val_arrays['M1c'].append(ridge_val)
    final = dict(keys=np.asarray([f'{c}|{x}' for c, x in keys]), **{m: np.vstack(v) for m, v in arrays.items()},
                 validation_keys=np.asarray([f'{c}|{x}' for c, x in val_keys]),
                 **{f'validation_{m}': np.vstack(v) for m, v in val_arrays.items()})
    inputs = {'protocol': ROOT/'configs/protocol_v4.yaml', 'train_truth': args.prepared/'train_truth.npz',
              'validation_truth': args.prepared/'validation_truth.npz', 'queries': args.prepared/'queries.tsv'}
    seal = runner.write_seal(args.output_root, 'op3', 'context_baselines', final,
                             dict(selection=selection, alphas=alphas), inputs)
    print(json.dumps({c: v['alpha'] for c, v in selection.items()}, indent=2), seal['predictions']['sha256'])


if __name__ == '__main__':
    main()
