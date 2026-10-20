"""Post hoc OP3 sensitivity (not prespecified): pooled contrasts excluding compounds with descriptor z-scores > 1e6."""
import sys
import json
import pandas as pd
sys.path.insert(0, sys.argv[2])
from v3_evaluate import pooled_bootstrap
pc = pd.read_csv(sys.argv[1], sep='\t')
gate = ['Tcells_24h_1000nM', 'Bcells_24h_1000nM', 'Myeloid_24h_1000nM']
sets = {'all eligible (prespecified)': set(), 'excluding ixabepilone': {'LSM-43293'},
        'excluding the 3 test compounds with |z|>1e6': {'LSM-43293', 'LSM-1020', 'LSM-1011'}}
arms = ['A0_v3_reference', 'I1_nb_likelihood', 'I2_counterfactual_pseudobulk', 'I3_nb_plus_pseudobulk']
rows = []
for label, drop in sets.items():
    for arm in arms:
        for base in ['B0', 'B1c', 'M1c', 'B1a']:
            frames = {}
            for ctx in gate:
                a = pc[pc.method.eq(arm) & pc.context_id.eq(ctx) & ~pc.compound_id.isin(drop)].set_index('compound_id')
                b = pc[pc.method.eq(base) & pc.context_id.eq(ctx) & ~pc.compound_id.isin(drop)].set_index('compound_id')
                frames[ctx] = pd.DataFrame(dict(scaffold_group=a.scaffold_group, difference=a.rmse - b.loc[a.index, 'rmse'])).reset_index(drop=True)
            r = pooled_bootstrap(frames)
            n = sum(len(f) for f in frames.values())
            rows.append(dict(analysis=label, candidate=arm, baseline=base, point_estimate=r['point_estimate'],
                             ci_lower=r['ci_lower'], ci_upper=r['ci_upper'], compound_context_rows=n))
out = pd.DataFrame(rows)
out.to_csv(sys.argv[3], sep='\t', index=False)
print(out.round(4).to_string(index=False))
