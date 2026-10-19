"""Information-matched simple baselines per context (B1c, B2c, M1c); seals test predictions.

Each context uses only its own eligible training-compound deltas: the training
mean, the nearest training compound by binary Tanimoto, and a fingerprint Ridge
fitted to 20 training-only principal components (alpha chosen on that context's
validation compounds), mirroring v1 M1. Test outcomes are never read.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import yaml
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from drugperturb.chemistry import morgan_fingerprint
from drugperturb.response import baseline_predictions
from drugperturb.v3 import runner

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = [float(a) for a in yaml.safe_load((ROOT/'configs/protocol_v3.yaml').read_text())['simple_context_baselines']['ridge_alphas']]


def load_truth(path):
    with np.load(path, allow_pickle=False) as z:
        return {tuple(k.split('|')): z['delta'][i] for i, k in enumerate(z['keys'])}


def rmse(truth, prediction):
    return float(np.sqrt(np.mean((truth-prediction)**2, axis=1)).mean())


def fit_context(train_y, train_fp, val_y, val_fp, query_fp):
    scaler = StandardScaler().fit(train_y)
    pca = PCA(n_components=min(20, len(train_y)-1), svd_solver='full').fit(scaler.transform(train_y))
    z = pca.transform(scaler.transform(train_y))
    inverse = lambda m, x: scaler.inverse_transform(pca.inverse_transform(m.predict(x)))
    scores = []
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, solver='cholesky').fit(train_fp, z)
        scores.append(rmse(val_y, inverse(model, val_fp)))
    chosen = int(np.argmin(scores))
    model = Ridge(alpha=ALPHAS[chosen], solver='cholesky').fit(train_fp, z)
    return inverse(model, query_fp), dict(alpha=ALPHAS[chosen], validation_rmse=scores,
                                          n_components=int(pca.n_components_), n_train=len(train_y))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    split = pd.read_csv(ROOT/'manifests/split_manifest.tsv', sep='\t', keep_default_na=False)
    fps = {c: morgan_fingerprint(s).astype(float) for c, s in zip(split.compound_id, split.canonical_smiles)}
    train, val = load_truth(args.prepared/'train_truth.npz'), load_truth(args.prepared/'validation_truth.npz')
    queries = runner.read_query_table(args.prepared/'queries.tsv')
    tests = queries[queries.split.eq('test')]
    keys, arrays, selection = [], {m: [] for m in ['B0', 'B1c', 'B2c', 'M1c']}, {}
    for context in runner.PROTOCOL_CONTEXTS['Q2']:
        t_ids = sorted(c for c, x in train if x == context)
        v_ids = sorted(c for c, x in val if x == context)
        q_ids = sorted(set(tests.loc[tests.context_id.eq(context), 'compound_id']))
        if len(t_ids) < 2 or not v_ids or not q_ids:
            raise ValueError(f'{context}: insufficient training/validation/query compounds')
        ty = np.vstack([train[(c, context)] for c in t_ids])
        vy = np.vstack([val[(c, context)] for c in v_ids])
        tf, vf, qf = (np.vstack([fps[c] for c in ids]) for ids in (t_ids, v_ids, q_ids))
        simple = baseline_predictions(ty, tf, qf)
        ridge, info = fit_context(ty, tf, vy, vf, qf)
        info['validation_rmse_simple'] = {k: rmse(vy, v) for k, v in baseline_predictions(ty, tf, vf).items()}
        selection[context] = info
        keys += [(c, context) for c in q_ids]
        arrays['B0'].append(simple['B0']); arrays['B1c'].append(simple['B1'])
        arrays['B2c'].append(simple['B2']); arrays['M1c'].append(ridge)
    final = dict(keys=np.asarray([f'{c}|{x}' for c, x in keys]), **{m: np.vstack(v) for m, v in arrays.items()})
    inputs = {'protocol': ROOT/'configs/protocol_v3.yaml', 'train_truth': args.prepared/'train_truth.npz',
              'validation_truth': args.prepared/'validation_truth.npz', 'queries': args.prepared/'queries.tsv'}
    seal = runner.write_seal(args.output_root, 'Q2', 'context_baselines', final,
                             dict(selection=selection, alphas=ALPHAS), inputs)
    print(json.dumps({c: v['alpha'] for c, v in selection.items()}, indent=2), seal['predictions']['sha256'])


if __name__ == '__main__':
    main()
