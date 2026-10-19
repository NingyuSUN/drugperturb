"""Build response-free v3 model inputs and training/validation truth; never test truth.

Reads the three pseudobulk exports (export_context_rds.R) and cell exports
(v3_export_cells.R). Outputs count QC for all twelve contexts, cross-checks
against the committed v1/v2 records, the model-cell archive (training,
validation and vehicle cells only), counterfactual queries, vehicle-well
contexts, and strict deltas for training and validation compounds.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy import sparse

from drugperturb.context_qc import qualify_contexts
from drugperturb.transfer_v2 import alternate_controls, measured_responses, required_wells
from drugperturb.v3.cells import assign_roles, build_model_archive, read_cell_export, save_model_archive
from drugperturb.v3.runner import sha256

ROOT = Path(__file__).resolve().parents[1]
CELL_TYPES = ['A549', 'K562', 'MCF7']


def read(path):
    return pd.read_csv(path, sep='\t', keep_default_na=False, dtype={'dose_pattern': str, 'replicate': str})


def crosscheck_count_qc(compounds):
    """Recomputed QC must equal committed v2 (MCF7/K562) and v1 (A549 1 uM) records."""
    committed = read(ROOT/'manifests/v2/compound_count_qc.tsv')
    fields = ['eligible', 'count_eligible', 'rep1_cells', 'rep2_cells', 'rep1_control_cells', 'rep2_control_cells']
    key = ['context_id', 'compound_id']
    done = committed[committed.count_qc_done.astype(str).eq('True')]
    new = compounds.merge(done[key+fields], on=key, suffixes=('', '_v2'), validate='one_to_one')
    text = lambda s: s.astype('string').fillna('NA')
    mismatches = {f: int((text(new[f]) != text(new[f'{f}_v2'])).sum()) for f in fields}
    split = read(ROOT/'manifests/split_manifest.tsv')
    a549 = compounds[compounds.context_id.eq('A549_24h_1000nM')].set_index('compound_id').loc[split.compound_id]
    v1 = dict(all_split_eligible=bool(a549.eligible.astype(str).eq('True').all()),
              rep1_cells_equal=bool((a549.rep1_cells.astype(int).to_numpy() == split.rep1_cells.to_numpy()).all()),
              rep2_cells_equal=bool((a549.rep2_cells.astype(int).to_numpy() == split.rep2_cells.to_numpy()).all()))
    result = dict(v2_rows_compared=len(new), v2_field_mismatches=mismatches, v1_A549_1000nM=v1)
    if len(new) != len(done) or any(mismatches.values()) or not all(v1.values()):
        raise ValueError(f'recomputed count QC differs from committed records: {result}')
    return result


def crosscheck_cells(table, counts, pseudobulk, well_qc, gene_ids):
    """Per-well cell exports must aggregate exactly to the pseudobulk frozen-gene counts."""
    wells = sorted(set(table.Combo))
    codes = pd.Categorical(table.Combo, categories=wells).codes
    indicator = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(wells), len(codes)))
    summed = np.asarray((indicator @ counts).todense())
    expected = pseudobulk.loc[gene_ids, wells].to_numpy().T
    n_cells = table.Combo.value_counts().reindex(wells)
    qc_cells = well_qc.set_index('Combo').n_cells.astype(int).reindex(wells)
    return dict(wells=len(wells), frozen_gene_counts_equal=bool(np.array_equal(summed, expected)),
                cells_le_well_qc=bool((n_cells <= qc_cells).all()),
                cells_missing_vs_well_qc=int((qc_cells-n_cells).sum()))


def strict_truth(context, ids, wells, qualified, qc, counts, gene_ids):
    pairs = alternate_controls(wells, qualified, qc, ids)
    # Project to the wells v2 reads (treated, strict and qualified alternate vehicles)
    # first; unrelated, possibly empty, wells must not enter library checks.
    keys = required_wells(qualified, pairs, ids)
    strict, _ = measured_responses(counts.loc[:, keys], qualified, pairs, ids, gene_ids)
    return strict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extraction-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_root
    if (out/'prepare_receipt.json').exists():
        raise ValueError('prepare outputs already exist')
    (out/'count_qc').mkdir(parents=True, exist_ok=True)
    wells = read(ROOT/'manifests/v2/well_mapping.tsv')
    roles = read(ROOT/'manifests/v2/context_roles.tsv')
    split = read(ROOT/'manifests/split_manifest.tsv')
    gene_ids = read(ROOT/'manifests/selected_genes.tsv').gene_id.tolist()
    pb_root, cell_root = args.extraction_root/'pseudobulk', args.extraction_root/'cells'
    well_qc = {c: read(pb_root/f'{c}_24h/well_qc.tsv') for c in CELL_TYPES}
    qualification = qualify_contexts(wells, roles, pd.concat(well_qc.values(), ignore_index=True), minimum_cells=30)
    compounds, qualified = qualification['compound_qualification'], qualification['well_qualification']
    for name, frame in qualification.items():
        frame.to_csv(out/'count_qc'/f'{name}.tsv', sep='\t', index=False)
    checks = dict(count_qc=crosscheck_count_qc(compounds))

    eligible = compounds[compounds.eligible.astype(str).eq('True')]
    queries, parts, truth = [], [], {'train': {}, 'validation': {}}
    for cell in CELL_TYPES:
        cell_wells = wells[wells.cell_type.eq(cell)]
        table, counts, library = read_cell_export(cell_root/f'{cell}_24h', gene_ids)
        pseudobulk = pd.read_csv(pb_root/f'{cell}_24h/pseudobulk_counts.tsv.gz', sep='\t', index_col='gene_id')
        checks[f'cells_{cell}'] = crosscheck_cells(table, counts, pseudobulk, well_qc[cell], gene_ids)
        if not checks[f'cells_{cell}']['frozen_gene_counts_equal']:
            raise ValueError(f'{cell}: cell export does not aggregate to pseudobulk counts')
        parts.append((assign_roles(table, cell_wells, split), counts, library))
        for context in sorted(cell_wells.context_id.unique()):
            ctx_wells = cell_wells[cell_wells.context_id.eq(context)]
            ctx_qualified = qualified[qualified.context_id.eq(context)]
            dose = float(ctx_wells.context_dose_nM.iloc[0])
            ctx_eligible = set(eligible.loc[eligible.context_id.eq(context), 'compound_id'])
            drugs = ctx_wells[ctx_wells.row_kind.eq('drug') & ctx_wells.split.isin(['validation', 'test'])]
            for row in drugs.itertuples(index=False):
                queries.append(dict(compound_id=row.treatment, context_id=context, split=row.split,
                                    replicate=row.replicate, cell_type=cell, dose_nM=dose,
                                    treatment_Combo=row.Combo, vehicle_Combo=row.control_Combo,
                                    count_eligible=row.treatment in ctx_eligible, scaffold_group=row.scaffold_group))
            for part in ['train', 'validation']:
                ids = sorted(ctx_eligible & set(split.loc[split.split.eq(part), 'compound_id']))
                if not ids:
                    continue
                delta = strict_truth(context, ids, ctx_wells, ctx_qualified, well_qc[cell], pseudobulk, gene_ids)
                truth[part].update({(cid, context): delta[i] for i, cid in enumerate(ids)})
        del pseudobulk
    obs, counts, library, dropped = build_model_archive(parts)
    save_model_archive(out/'model_cells.npz', obs, counts, library)
    query_table = pd.DataFrame(queries).sort_values(['context_id', 'compound_id', 'replicate'])
    if query_table.duplicated(['context_id', 'compound_id', 'replicate']).any():
        raise ValueError('duplicate counterfactual query records')
    query_table.to_csv(out/'queries.tsv', sep='\t', index=False)
    vehicles = wells.loc[wells.row_kind.eq('vehicle'), ['Combo', 'context_id']].sort_values(['Combo', 'context_id'])
    vehicles.to_csv(out/'vehicle_contexts.tsv', sep='\t', index=False)
    for part, values in truth.items():
        keys = sorted(values)
        np.savez_compressed(out/f'{part}_truth.npz', keys=np.asarray([f'{c}|{x}' for c, x in keys]),
                            delta=np.vstack([values[k] for k in keys]))
    role_counts = obs.groupby(['cell_type', 'role']).size().rename('cells').reset_index()
    receipt = dict(status='PREPARED_RESPONSE_FREE_FOR_TEST', test_outcomes_computed=False,
                   test_compound_cells_dropped=int(dropped.get('test', 0)),
                   excluded_compound_cells_dropped=int(dropped.get('excluded', 0)),
                   model_cells=len(obs), model_cell_roles=role_counts.to_dict('records'),
                   training_truth_pairs=len(truth['train']), validation_truth_pairs=len(truth['validation']),
                   queries=len(query_table), checks=checks,
                   context_summary=qualification['context_summary'][['context_id', 'eligible_test_compounds',
                       'eligible_test_structure_groups', 'frozen_test_count_gate_met']].to_dict('records'),
                   outputs={p.name: sha256(p) for p in sorted(out.glob('*')) if p.is_file()},
                   inputs={str(p.relative_to(args.extraction_root)): sha256(p)
                           for p in sorted(args.extraction_root.rglob('*')) if p.is_file()})
    (out/'prepare_receipt.json').write_text(json.dumps(receipt, indent=2, default=str)+'\n')
    print(json.dumps({k: receipt[k] for k in ['model_cells', 'test_compound_cells_dropped',
                                              'validation_truth_pairs', 'queries']}, indent=2))


if __name__ == '__main__':
    main()
