"""Per-seed validation RMSE of every (compound, context) pair, v4 and corrected seals (no test truth is read).

Shows how much of each arm's change in validation RMSE (the selection metric) comes
from CEP-37440, the one validation compound whose v4 descriptor z-scores were near 1e16.
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_evaluate import load_sealed  # noqa: E402
from v4_fit import ARMS, load_truth  # noqa: E402

AFFECTED_VALIDATION = 'LSM-45849'  # CEP-37440


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--v4-sealed', type=Path, required=True)
    parser.add_argument('--corrected-sealed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    truth = load_truth(args.prepared/'validation_truth.npz')
    rows = []
    for analysis, root in [('v4', args.v4_sealed), ('corrected', args.corrected_sealed)]:
        for arm in ARMS:
            arrays, seal = load_sealed(root, arm, 'op3')
            keys = [tuple(k.split('|')) for k in arrays['validation_keys']]
            for seed in seal['seeds']:
                for key, predicted in zip(keys, arrays[f'validation_contrast_s{seed}']):
                    rows.append(dict(analysis=analysis, arm=arm, seed=seed, compound_id=key[0], context_id=key[1],
                                     rmse=float(np.sqrt(np.mean((predicted - truth[key])**2)))))
    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, sep='\t', index=False)
    selection_metric = lambda t: t.groupby(['arm', 'analysis', 'seed']).rmse.mean().groupby(['arm', 'analysis']).mean()
    summary = pd.DataFrame({
        'all_pairs': selection_metric(table),
        'without_CEP-37440': selection_metric(table[table.compound_id.ne(AFFECTED_VALIDATION)]),
        'CEP-37440_pairs': table[table.compound_id.eq(AFFECTED_VALIDATION)].groupby(['arm', 'analysis']).rmse.mean()})
    print(summary.unstack('analysis').round(5).to_string())


if __name__ == '__main__':
    main()
