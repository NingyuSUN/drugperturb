"""OP3 extraction: unit pseudobulk, count QC, training-only gene selection, model-cell export.

Input is the hash-verified GSE279945_sc_counts_processed.h5ad. Two passes over
raw/X: (1) all-gene counts summed per unit (cells of one type in one well),
per-cell libraries and DMSO-cell gene detection; (2) frozen-gene counts of
training-compound, validation-compound and DMSO cells only. Between them, count
QC uses cell numbers and gene selection uses training-compound deltas and DMSO
detection. No validation or test response is computed here; the unit
pseudobulk (all wells) is written for the later preparation and evaluation
kernels only.
"""
from pathlib import Path
import argparse
import hashlib
import json

import h5py
import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from drugperturb.v4 import design
from drugperturb.v4.diagnostics import finite
from drugperturb.v4.h5io import csr_shape, iter_csr_rows, read_obs, read_var_names

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROLES = ('train', 'validation', 'vehicle')


def read(path):
    return pd.read_csv(path, sep='\t', keep_default_na=False)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def cell_table(obs, wells, compounds):
    """Per-cell design labels: unit, compound, split role; expression is not touched."""
    table = pd.DataFrame(dict(cell_id=obs.index.to_numpy(), plate_name=obs.plate_name.astype(str).to_numpy(),
                              well=obs.well.astype(str).to_numpy(), donor_id=obs.donor_id.astype(str).to_numpy(),
                              library_id=obs.library_id.astype(str).to_numpy(),
                              cell_type=obs.cell_type.astype(str).map(design.CELL_CODES).to_numpy()))
    if table.cell_type.isna().any():
        raise ValueError('a cell has a cell type outside the four prespecified contexts')
    info = wells.set_index(['plate_name', 'well'])
    key = pd.MultiIndex.from_arrays([table.plate_name, table.well])
    table['compound_id'] = info.compound_id.reindex(key).to_numpy()
    table['well_role'] = info.role.reindex(key).to_numpy()
    table['dose_nM'] = info.dose_nM.reindex(key).to_numpy()
    split = compounds.set_index('compound_id').split
    table['role'] = np.where(table.well_role.eq('vehicle'), 'vehicle',
                             np.where(table.well_role.eq('control'), 'control', table.compound_id.map(split)))
    if table.role.isna().any():
        raise ValueError('a cell maps to no design well')
    table['unit_id'] = [design.unit_id(p, w, c) for p, w, c in zip(table.plate_name, table.well, table.cell_type)]
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h5ad', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--rows-per-block', type=int, default=20000)
    args = parser.parse_args()
    out = args.output_root
    if (out/'extract_receipt.json').exists():
        raise ValueError('extraction outputs already exist')
    (out/'count_qc').mkdir(parents=True, exist_ok=True)
    (out/'cells').mkdir(parents=True, exist_ok=True)
    protocol = yaml.safe_load((ROOT/'configs/protocol_v4.yaml').read_text())['datasets']['op3']
    compounds = read(ROOT/'manifests/v4/op3_compounds.tsv')
    wells = read(ROOT/'manifests/v4/op3_wells.tsv')
    checks = {}
    with h5py.File(args.h5ad, 'r') as h5:
        obs = read_obs(h5, [*design.DESIGN_COLUMNS, 'cell_type', 'cell_type_orig', 'cell_count_by_well_celltype'])
        import importlib.util
        spec = importlib.util.spec_from_file_location('v4_op3_design', ROOT/'scripts/v4_op3_design.py')
        design_script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(design_script)
        regenerated, regenerated_wells, _ = design_script.tables(obs, protocol['split'])
        checks['design_equals_committed_manifests'] = bool(
            design_script.read_roundtrip(regenerated).equals(compounds)
            and design_script.read_roundtrip(regenerated_wells).equals(wells))
        if not checks['design_equals_committed_manifests']:
            raise ValueError('regenerated design differs from the committed manifests')
        n_cells, n_genes = csr_shape(h5, 'raw/X')
        gene_ids = read_var_names(h5, 'raw/var')
        if n_cells != len(obs) or len(gene_ids) != n_genes or len(set(gene_ids)) != n_genes:
            raise ValueError('raw/X, obs and raw/var disagree or gene IDs are not unique')
        cells = cell_table(obs, wells, compounds)
        units = sorted(set(cells.unit_id) | {design.unit_id(p, w, c) for p, w in zip(wells.plate_name, wells.well)
                                             for c in design.CELL_CODES.values()})
        position = {u: i for i, u in enumerate(units)}
        codes = cells.unit_id.map(position).to_numpy()
        vehicle = cells.role.eq('vehicle').to_numpy()

        # Pass 1: unit sums over all genes, libraries, DMSO detection.
        unit_sum = sparse.csr_matrix((len(units), n_genes), dtype=np.int64)
        library = np.zeros(n_cells, dtype=np.int64)
        detected = np.zeros(n_genes, dtype=np.int64)
        for start, block in iter_csr_rows(h5, 'raw/X', args.rows_per_block):
            if (block.data < 0).any() or not np.array_equal(block.data, np.floor(block.data)):
                raise ValueError('raw/X must hold nonnegative integer counts')
            block = sparse.csr_matrix((block.data.astype(np.int64), block.indices, block.indptr), shape=block.shape)
            stop = start+block.shape[0]
            library[start:stop] = np.asarray(block.sum(axis=1)).ravel()
            indicator = sparse.csr_matrix((np.ones(block.shape[0], np.int64), (codes[start:stop], np.arange(block.shape[0]))),
                                          shape=(len(units), block.shape[0]))
            unit_sum = unit_sum+indicator@block
            v = vehicle[start:stop]
            if v.any():
                detected += np.asarray((block[v] > 0).sum(axis=0)).ravel()
        if (library <= 0).any():
            raise ValueError('a cell has an empty library')
        unit_cells = cells.unit_id.value_counts().reindex(units, fill_value=0)
        authors = obs.cell_count_by_well_celltype.to_numpy()
        # The authors' column may count reannotated or original labels; one must match exactly.
        per_label = {}
        for label in ['cell_type', 'cell_type_orig']:
            key = obs.plate_name.astype(str)+'_'+obs.well.astype(str)+'_'+obs[label].astype(str)
            per_label[label] = bool((key.map(key.value_counts()).to_numpy() == authors).all())
        checks['unit_cell_counts_equal_authors'] = dict(per_label, matched=[k for k, v in per_label.items() if v])
        if not any(per_label.values()):
            raise ValueError('cell counts differ from cell_count_by_well_celltype under both labelings')
        unit_library = np.asarray(unit_sum.sum(axis=1)).ravel()
        lib_by_unit = pd.Series(library).groupby(codes).sum().reindex(range(len(units)), fill_value=0).to_numpy()
        checks['unit_library_equals_cell_libraries'] = bool(np.array_equal(unit_library, lib_by_unit))

        # Count QC (cell numbers only) and training-only gene selection.
        pairs = design.response_pairs(wells, compounds)
        replicate_qc, compound_qc, summary = design.count_qc(
            pairs, unit_cells.to_dict(), minimum_cells=30,
            min_test_compounds=protocol['split']['min_test_compounds'], min_test_groups=protocol['split']['min_test_groups'])
        train_keys = [tuple(k) for k in compound_qc.loc[compound_qc.eligible & compound_qc.split.eq('train'),
                                                         ['compound_id', 'context_id']].itertuples(index=False)]
        dense = unit_sum.toarray()
        train_deltas = design.strict_deltas(dense, unit_library, units, pairs, train_keys)
        detection = detected/vehicle.sum()
        spec = protocol['gene_space']
        selected = design.select_genes(train_deltas, detection, n_genes=spec['n_genes'], min_detection=0.05)
        variance = train_deltas.var(axis=0)
        frozen = pd.DataFrame(dict(output_index=np.arange(len(selected)), gene_id=gene_ids[selected],
                                   source_index=selected, training_delta_variance=variance[selected],
                                   vehicle_detection=detection[selected]))
        del dense

        # Pass 2: frozen-gene counts of model cells (training, validation, vehicle) only.
        model_rows = cells.role.isin(MODEL_ROLES).to_numpy()
        blocks = []
        for start, block in iter_csr_rows(h5, 'raw/X', args.rows_per_block):
            keep = model_rows[start:start+block.shape[0]]
            part = block[keep][:, selected]
            blocks.append(sparse.csr_matrix((part.data.astype(np.int32), part.indices, part.indptr), shape=part.shape))
        model_counts = sparse.vstack(blocks).tocsr()
        model_counts.sort_indices()

        # Reported, not required: authors' normalized X versus log1p(raw / library * s) on DMSO cells.
        rng = np.random.default_rng(2026)
        sample = np.sort(rng.choice(np.flatnonzero(vehicle), size=min(500, int(vehicle.sum())), replace=False))
        X, raw = h5['X'], h5['raw/X']
        ratios, worst = [], 0.0
        for i in sample:
            lo, hi = int(X['indptr'][i]), int(X['indptr'][i+1])
            rlo, rhi = int(raw['indptr'][i]), int(raw['indptr'][i+1])
            xv = dict(zip(X['indices'][lo:hi], X['data'][lo:hi]))
            rv = dict(zip(raw['indices'][rlo:rhi], raw['data'][rlo:rhi]))
            shared = [g for g in rv if g in xv and rv[g] > 0]
            if shared:
                s = np.array([np.expm1(xv[g])*library[i]/rv[g] for g in shared])
                ratios.append(float(np.median(s)))
                worst = max(worst, float(np.max(np.abs(s/np.median(s)-1))))
        checks['authors_normalization'] = dict(cells=len(ratios), median_scale=float(np.median(ratios)) if ratios else None,
                                               scale_iqr=[float(np.quantile(ratios, 0.25)), float(np.quantile(ratios, 0.75))] if ratios else None,
                                               max_within_cell_relative_deviation=worst, required=False)

    model_cells = cells[model_rows].reset_index(drop=True)
    model_cells['library'] = library[model_rows]
    model_cells['frozen_gene_UMI'] = np.asarray(model_counts.sum(axis=1)).ravel()
    checks['model_cells_exclude_test_and_controls'] = bool(model_cells.role.isin(MODEL_ROLES).all())
    unit_frozen = unit_sum[:, selected].toarray()
    model_unit = pd.Categorical(model_cells.unit_id, categories=units).codes
    indicator = sparse.csr_matrix((np.ones(len(model_unit)), (model_unit, np.arange(len(model_unit)))),
                                  shape=(len(units), len(model_unit)))
    summed = np.asarray((indicator@model_counts).todense())
    present = np.unique(model_unit)
    checks['cell_frozen_counts_sum_to_unit_pseudobulk'] = bool(np.array_equal(summed[present], unit_frozen[present]))
    if not checks['cell_frozen_counts_sum_to_unit_pseudobulk']:
        raise ValueError('cell export does not aggregate to the unit pseudobulk')

    unit_table = pd.DataFrame(dict(unit_id=units, n_cells=unit_cells.to_numpy(), library=unit_library))
    unit_table.to_csv(out/'units.tsv', sep='\t', index=False)
    unit_sum = unit_sum.astype(np.int64).tocsr()
    if unit_sum.data.max(initial=0) > np.iinfo(np.int32).max:
        raise ValueError('unit counts overflow int32')
    np.savez_compressed(out/'unit_counts.npz', data=unit_sum.data.astype(np.int32), indices=unit_sum.indices.astype(np.int32),
                        indptr=unit_sum.indptr.astype(np.int64), shape=np.asarray(unit_sum.shape, np.int64),
                        unit_ids=np.asarray(units, dtype=str), gene_ids=np.asarray(gene_ids, dtype=str),
                        library=unit_library)
    pd.DataFrame(dict(gene_id=gene_ids, vehicle_detection=detection)).to_csv(out/'genes.tsv', sep='\t', index=False)
    frozen.to_csv(out/'frozen_genes.tsv', sep='\t', index=False)
    for name, frame in [('replicate_qc', replicate_qc), ('compound_qc', compound_qc), ('context_summary', summary)]:
        frame.to_csv(out/'count_qc'/f'{name}.tsv', sep='\t', index=False)
    model_cells.to_csv(out/'cells/cells.tsv.gz', sep='\t', index=False)
    np.savez_compressed(out/'cells/frozen_counts.npz', data=model_counts.data, indices=model_counts.indices.astype(np.int32),
                        indptr=model_counts.indptr.astype(np.int64), shape=np.asarray(model_counts.shape, np.int64))
    receipt = dict(status='EXTRACTED_TEST_RESPONSES_NOT_COMPUTED', test_outcomes_computed=False,
                   cells=int(n_cells), genes=int(n_genes), units=len(units),
                   model_cells=int(model_rows.sum()), model_cell_roles=model_cells.role.value_counts().to_dict(),
                   dropped_cells=cells.loc[~model_rows, 'role'].value_counts().to_dict(),
                   eligible_training_pairs=len(train_keys), vehicle_cells=int(vehicle.sum()),
                   context_summary=summary.to_dict('records'), checks=checks,
                   frozen_gene_detection_min=float(frozen.vehicle_detection.min()),
                   outputs={str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob('*')) if p.is_file()})
    (out/'extract_receipt.json').write_text(json.dumps(finite(receipt), indent=2)+'\n')
    print(json.dumps(finite({k: receipt[k] for k in ['cells', 'units', 'model_cells', 'eligible_training_pairs',
                                                     'context_summary', 'checks']}), indent=2))


if __name__ == '__main__':
    main()
