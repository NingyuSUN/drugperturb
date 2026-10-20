"""Build response-free OP3 model inputs and training/validation truth; never test truth.

Reads the extraction outputs and writes, in the v3 prepared layout, the model
cell archive (training, validation and DMSO cells only; frozen genes), the
counterfactual query table for all splits, the vehicle-unit contexts, count QC
and strict deltas for training and validation compounds. GPU kernels mount only
these outputs.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy import sparse

from drugperturb.v3.cells import OBS_FIELDS, save_model_archive
from drugperturb.v3.runner import sha256
from drugperturb.v4 import design
from drugperturb.v4.diagnostics import finite

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return pd.read_csv(path, sep='\t', keep_default_na=False, dtype={'replicate': str})


def load_units(path):
    with np.load(path, allow_pickle=False) as z:
        counts = sparse.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))
        return counts, z['library'], z['unit_ids'].astype(str), z['gene_ids'].astype(str)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extraction-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    ext, out = args.extraction_root, args.output_root
    if (out/'prepare_receipt.json').exists():
        raise ValueError('prepare outputs already exist')
    (out/'count_qc').mkdir(parents=True, exist_ok=True)
    compounds = read(ROOT/'manifests/v4/op3_compounds.tsv')
    wells = read(ROOT/'manifests/v4/op3_wells.tsv')
    frozen = read(ext/'frozen_genes.tsv')
    unit_counts, unit_library, unit_ids, gene_ids = load_units(ext/'unit_counts.npz')
    if not np.array_equal(gene_ids[frozen.source_index.to_numpy()], frozen.gene_id.to_numpy()):
        raise ValueError('frozen gene list does not index the unit gene IDs')
    compound_qc = read(ext/'count_qc/compound_qc.tsv')
    for name in ['replicate_qc', 'compound_qc', 'context_summary']:
        read(ext/f'count_qc/{name}.tsv').to_csv(out/'count_qc'/f'{name}.tsv', sep='\t', index=False)
    frozen.to_csv(out/'frozen_genes.tsv', sep='\t', index=False)

    cells = pd.read_csv(ext/'cells/cells.tsv.gz', sep='\t', keep_default_na=False, dtype={'donor_id': str})
    with np.load(ext/'cells/frozen_counts.npz', allow_pickle=False) as z:
        counts = sparse.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))
    if not cells.role.isin(['train', 'validation', 'vehicle']).all() or counts.shape != (len(cells), len(frozen)):
        raise ValueError('extraction cell export is inconsistent or contains disallowed roles')
    obs = pd.DataFrame(dict(cell_id=cells.cell_id, Combo=cells.unit_id, cell_type=cells.cell_type,
                            replicate=cells.donor_id, plate_oligo=cells.library_id, compound_id=cells.compound_id,
                            role=cells.role, dose_nM=np.where(cells.role.eq('vehicle'), 0.0, cells.dose_nM)))
    save_model_archive(out/'model_cells.npz', obs[list(OBS_FIELDS)], counts, cells.library.to_numpy(np.float64))

    pairs = design.response_pairs(wells, compounds)
    eligible = compound_qc[compound_qc.eligible.astype(str).eq('True')]
    flags = set(map(tuple, eligible[['compound_id', 'context_id']].to_numpy()))
    queries = pairs.rename(columns=dict(treatment_unit='treatment_Combo', vehicle_unit='vehicle_Combo'))
    queries['count_eligible'] = [(c, x) in flags for c, x in zip(queries.compound_id, queries.context_id)]
    queries = queries[['compound_id', 'context_id', 'split', 'replicate', 'cell_type', 'dose_nM', 'treatment_Combo',
                       'vehicle_Combo', 'count_eligible', 'scaffold_group']]
    queries.to_csv(out/'queries.tsv', sep='\t', index=False)
    vehicle_units = sorted(set(queries.vehicle_Combo))
    pd.DataFrame(dict(Combo=vehicle_units, context_id=[design.context_id(u.rsplit('_', 1)[1]) for u in vehicle_units])
                 ).to_csv(out/'vehicle_contexts.tsv', sep='\t', index=False)

    columns = frozen.source_index.to_numpy()
    truth_counts = {}
    for part in ['train', 'validation']:
        keys = sorted(map(tuple, eligible.loc[eligible.split.eq(part), ['compound_id', 'context_id']].to_numpy()))
        needed = sorted(set(pairs.loc[pairs.split.eq(part), 'treatment_unit']) | set(pairs.vehicle_unit))
        position = {u: i for i, u in enumerate(unit_ids)}
        rows = [position[u] for u in needed]
        delta = design.strict_deltas(unit_counts[rows].toarray(), unit_library[rows], needed, pairs, keys, columns)
        np.savez_compressed(out/f'{part}_truth.npz', keys=np.asarray([f'{c}|{x}' for c, x in keys]), delta=delta)
        truth_counts[part] = len(keys)
    receipt = dict(status='PREPARED_RESPONSE_FREE_FOR_TEST', test_outcomes_computed=False, model_cells=len(obs),
                   model_cell_roles=obs.groupby(['cell_type', 'role']).size().rename('cells').reset_index().to_dict('records'),
                   training_truth_pairs=truth_counts['train'], validation_truth_pairs=truth_counts['validation'],
                   queries=len(queries), frozen_genes=len(frozen),
                   outputs={p.name: sha256(p) for p in sorted(out.glob('*')) if p.is_file()},
                   inputs={str(p.relative_to(ext)): sha256(p) for p in sorted(ext.rglob('*')) if p.is_file()})
    (out/'prepare_receipt.json').write_text(json.dumps(finite(receipt), indent=2)+'\n')
    print(json.dumps({k: receipt[k] for k in ['model_cells', 'training_truth_pairs', 'validation_truth_pairs', 'queries']}))


if __name__ == '__main__':
    main()
