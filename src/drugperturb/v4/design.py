"""OP3 (GSE279945) design, count QC, measured responses and gene selection for v4.

Nothing here fits a model. The compound split uses structures only. Count QC
uses cell numbers only. Gene selection uses training-compound responses and
vehicle-cell detection only, so validation and test responses never shape the
gene space.

Layout facts the checks enforce: each plate holds one donor; each plate row is
one sequencing library with exactly one DMSO well; each compound has one well
per donor. A treated well's strict matched vehicle is the DMSO well in the same
plate row. The measurement unit of a context is the set of cells of that cell
type in one well.
"""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import pandas as pd

VEHICLE_NAME = 'Dimethyl Sulfoxide'
VEHICLE_ID = 'DMSO'
CONTROL_NAMES = ('Belinostat', 'Dabrafenib')
CELL_CODES = {'T cells': 'Tcells', 'NK cells': 'NKcells', 'B cells': 'Bcells', 'Myeloid cells': 'Myeloid'}
DOSE_NM = 1000.0
TIME_HOURS = 24.0
N_DONORS = 3
DESIGN_COLUMNS = ('plate_name', 'well', 'row', 'col', 'donor_id', 'library_id', 'sm_name', 'sm_lincs_id',
                  'SMILES', 'dose_uM', 'timepoint_hr', 'control')


def context_id(cell_code):
    return f'{cell_code}_24h_1000nM'


CONTEXTS = [context_id(c) for c in CELL_CODES.values()]


def unit_id(plate, well, cell_code):
    return f'{plate}_{well}_{cell_code}'


def split_by_structure_fractions(records, *, seed=42, test_fraction=0.25, validation_fraction=0.15,
                                 min_test_compounds=20, min_test_groups=10):
    """The v1 structure split with prespecified fractions instead of 15/15.

    Sorted scaffold keys are shuffled once with ``seed``. The two smallest groups
    are reserved so training and validation are nonempty; shuffled whole groups
    go to test until its target and minimums are met; validation takes its
    target from the remaining groups in order; training receives the rest. Only
    structures are read. Compound counts are unique canonical parents.
    """
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1 or test_fraction+validation_fraction >= 1:
        raise ValueError('fractions must be positive and leave a training share')
    rows = [dict(r) for r in records]
    structures, parent_groups = defaultdict(set), {}
    for row in rows:
        canonical, group = row['canonical_smiles'], row['scaffold_group']
        if parent_groups.setdefault(canonical, group) != group:
            raise ValueError('canonical duplicate assigned conflicting scaffold groups')
        structures[group].add(canonical)
    if len(structures) < 3:
        raise ValueError('at least three structure groups are required')
    keys = sorted(structures)
    random.Random(seed).shuffle(keys)
    sizes = {key: len(structures[key]) for key in keys}
    n_compounds = len(parent_groups)
    reserved = sorted(keys, key=lambda key: sizes[key])[:2]
    available = [key for key in keys if key not in reserved]
    if len(available) < min_test_groups or sum(sizes[k] for k in available) < min_test_compounds:
        raise ValueError('test minimums are infeasible for these structures')
    test_target = max(min_test_compounds, round(test_fraction*n_compounds))
    test, test_size = [], 0
    for key in available:
        test.append(key)
        test_size += sizes[key]
        if test_size >= test_target and len(test) >= min_test_groups:
            break
    remaining = [key for key in keys if key not in set(test)]
    validation, validation_size = [], 0
    for key in remaining[:-1]:
        validation.append(key)
        validation_size += sizes[key]
        if validation_size >= max(1, round(validation_fraction*n_compounds)):
            break
    assignment = {key: 'train' for key in keys}
    assignment.update({key: 'validation' for key in validation})
    assignment.update({key: 'test' for key in test})
    output = [{**row, 'split': assignment[row['scaffold_group']]} for row in rows]
    report = dict(seed=seed, test_fraction=test_fraction, validation_fraction=validation_fraction,
                  required_test_compounds=min_test_compounds, required_test_groups=min_test_groups,
                  compound_count_unit='unique_canonical_parent')
    for split in ('train', 'validation', 'test'):
        chosen = [r for r in output if r['split'] == split]
        report[split] = dict(n_compound_ids=len({r['compound_id'] for r in chosen}),
                             n_compounds=len({r['canonical_smiles'] for r in chosen}),
                             n_groups=len({r['scaffold_group'] for r in chosen}))
    report['test_minimum_met'] = (report['test']['n_compounds'] >= min_test_compounds
                                  and report['test']['n_groups'] >= min_test_groups)
    return output, report


def _single_valued(frame, keys, columns):
    counts = frame.groupby(list(keys), observed=True)[list(columns)].nunique(dropna=False)
    bad = counts[(counts > 1).any(axis=1)]
    if len(bad):
        raise ValueError(f'design fields vary within {keys}: {bad.head().to_dict()}')


def build_wells(design):
    """One row per (plate, well) with role, compound and same-row vehicle well.

    ``design`` is cell-level metadata with ``DESIGN_COLUMNS``. Raises if the
    layout differs from the one this protocol assumes.
    """
    missing = set(DESIGN_COLUMNS) - set(design.columns)
    if missing:
        raise ValueError(f'design metadata lacks {sorted(missing)}')
    frame = design[list(DESIGN_COLUMNS)].copy()
    for name in frame.columns:
        if isinstance(frame[name].dtype, pd.CategoricalDtype):
            frame[name] = frame[name].astype(object if name not in ('dose_uM', 'timepoint_hr', 'col') else float)
    _single_valued(frame, ['plate_name', 'well'], [c for c in DESIGN_COLUMNS if c not in ('plate_name', 'well')])
    wells = frame.drop_duplicates(['plate_name', 'well']).copy()
    wells['col'] = wells['col'].astype(int)
    wells['role'] = np.where(wells.sm_name.eq(VEHICLE_NAME), 'vehicle',
                             np.where(wells.sm_name.isin(CONTROL_NAMES), 'control', 'drug'))
    if not np.allclose(wells.timepoint_hr.astype(float), TIME_HOURS):
        raise ValueError('every well must be a 24 h exposure')
    drugs = wells[wells.role.eq('drug')]
    if not np.allclose(drugs.dose_uM.astype(float), DOSE_NM/1000):
        raise ValueError('every compound well must be dosed at 1 uM')
    if wells.groupby('plate_name').donor_id.nunique().max() != 1:
        raise ValueError('each plate must hold exactly one donor')
    per_row = wells.groupby(['plate_name', 'row'])
    if per_row.library_id.nunique().max() != 1 or wells.groupby('library_id')[['plate_name', 'row']].nunique().max().max() != 1:
        raise ValueError('each plate row must be exactly one library')
    vehicles = wells[wells.role.eq('vehicle')]
    if vehicles.groupby(['plate_name', 'row']).size().reindex(per_row.size().index).ne(1).any():
        raise ValueError('each plate row must contain exactly one vehicle well')
    vehicle_of = vehicles.set_index(['plate_name', 'row']).well
    wells['vehicle_well'] = [vehicle_of[(p, r)] for p, r in zip(wells.plate_name, wells.row)]
    per_compound = drugs.groupby('sm_lincs_id').agg(n=('well', 'size'), donors=('donor_id', 'nunique'))
    if not (per_compound.n.eq(N_DONORS) & per_compound.donors.eq(N_DONORS)).all():
        raise ValueError('each compound needs exactly one well per donor')
    if drugs.groupby('sm_lincs_id').sm_name.nunique().max() != 1 or drugs.groupby('sm_name').sm_lincs_id.nunique().max() != 1:
        raise ValueError('compound names and LINCS IDs must correspond one to one')
    wells['compound_id'] = np.where(wells.role.eq('vehicle'), VEHICLE_ID, wells.sm_lincs_id)
    wells['dose_nM'] = np.where(wells.role.eq('vehicle'), 0.0, wells.dose_uM.astype(float)*1000)
    columns = ['plate_name', 'well', 'row', 'col', 'donor_id', 'library_id', 'role', 'compound_id', 'sm_name',
               'dose_nM', 'vehicle_well']
    return wells.sort_values(['plate_name', 'row', 'col'])[columns].reset_index(drop=True)


def compound_records(design):
    """Unique (compound_id, name, SMILES) for drug wells; controls and vehicle excluded."""
    frame = design[~design.sm_name.isin([VEHICLE_NAME, *CONTROL_NAMES])]
    frame = frame[['sm_lincs_id', 'sm_name', 'SMILES']].astype(str).drop_duplicates()
    if frame.sm_lincs_id.duplicated().any():
        raise ValueError('a compound ID maps to more than one name or structure')
    return [dict(compound_id=r.sm_lincs_id, sm_name=r.sm_name, smiles=r.SMILES)
            for r in frame.sort_values('sm_lincs_id').itertuples(index=False)]


def response_pairs(wells, compounds):
    """One row per (compound, context, donor): treated and strict matched vehicle units."""
    info = compounds.set_index('compound_id')
    rows = []
    for well in wells[wells.role.eq('drug')].itertuples(index=False):
        for code in CELL_CODES.values():
            rows.append(dict(compound_id=well.compound_id, context_id=context_id(code), cell_type=code,
                             replicate=well.donor_id, plate_name=well.plate_name, library_id=well.library_id,
                             treatment_unit=unit_id(well.plate_name, well.well, code),
                             vehicle_unit=unit_id(well.plate_name, well.vehicle_well, code),
                             split=info.loc[well.compound_id, 'split'],
                             scaffold_group=info.loc[well.compound_id, 'scaffold_group'], dose_nM=well.dose_nM))
    return pd.DataFrame(rows).sort_values(['context_id', 'compound_id', 'replicate']).reset_index(drop=True)


def count_qc(pairs, unit_cells, *, minimum_cells=30, min_test_compounds=20, min_test_groups=10):
    """Replicate-, compound- and context-level count QC from cell numbers only.

    A (compound, context) is eligible when all three donor replicates have at
    least ``minimum_cells`` treated cells and as many in the matched vehicle unit.
    """
    table = pairs.copy()
    lookup = pd.Series(unit_cells, dtype=float)
    table['treatment_cells'] = table.treatment_unit.map(lookup).fillna(0).astype(int)
    table['vehicle_cells'] = table.vehicle_unit.map(lookup).fillna(0).astype(int)
    table['replicate_passes'] = (table.treatment_cells >= minimum_cells) & (table.vehicle_cells >= minimum_cells)
    compound = (table.groupby(['context_id', 'compound_id', 'split', 'scaffold_group'], as_index=False)
                .agg(replicates=('replicate', 'nunique'), passing=('replicate_passes', 'sum'),
                     min_treatment_cells=('treatment_cells', 'min'), min_vehicle_cells=('vehicle_cells', 'min')))
    compound['eligible'] = compound.replicates.eq(N_DONORS) & compound.passing.eq(N_DONORS)
    summary = []
    for context, group in compound.groupby('context_id'):
        row = dict(context_id=context)
        for split in ('train', 'validation', 'test'):
            chosen = group[group.split.eq(split) & group.eligible]
            row[f'eligible_{split}_compounds'] = len(chosen)
            row[f'eligible_{split}_groups'] = chosen.scaffold_group.nunique()
        row['test_count_gate_met'] = (row['eligible_test_compounds'] >= min_test_compounds
                                      and row['eligible_test_groups'] >= min_test_groups)
        summary.append(row)
    return table, compound, pd.DataFrame(summary)


def log_cpm(counts, library):
    """log1p CPM of unit rows with all-gene libraries."""
    library = np.asarray(library, dtype=np.float64)
    if (library <= 0).any() or not np.isfinite(library).all():
        raise ValueError('libraries must be finite and positive')
    return np.log1p(np.asarray(counts, dtype=np.float64)/library[:, None]*1e6)


def strict_deltas(unit_counts, unit_library, unit_index, pairs, keys, gene_columns=None):
    """Equal-weight donor mean of treated minus matched-vehicle log1p CPM per key.

    ``unit_counts`` is (units, genes) all-gene counts with rows named by
    ``unit_index``; ``unit_library`` holds each unit's all-gene total; ``keys``
    are (compound_id, context_id) pairs whose donor replicates are in ``pairs``.
    """
    position = {u: i for i, u in enumerate(unit_index)}
    columns = slice(None) if gene_columns is None else np.asarray(gene_columns)
    grouped = {k: g for k, g in pairs.groupby(['compound_id', 'context_id'])}
    out = []
    for key in keys:
        rows = grouped[key].sort_values('replicate')
        if rows.replicate.nunique() != N_DONORS or len(rows) != N_DONORS:
            raise ValueError(f'{key}: expected one record per donor')
        t = [position[u] for u in rows.treatment_unit]
        v = [position[u] for u in rows.vehicle_unit]
        treated = log_cpm(unit_counts[t][:, columns], unit_library[t])
        vehicle = log_cpm(unit_counts[v][:, columns], unit_library[v])
        out.append((treated-vehicle).mean(axis=0))
    return np.vstack(out) if out else np.zeros((0, 0))


def select_genes(train_deltas, vehicle_detection, *, n_genes=2000, min_detection=0.05):
    """Indices of the top-variance training-delta genes among vehicle-detected genes.

    The variance ranking is v1's rule (population variance over training rows,
    stable descending sort); the detection filter uses vehicle cells only.
    """
    deltas = np.asarray(train_deltas, dtype=np.float64)
    detection = np.asarray(vehicle_detection, dtype=np.float64)
    if deltas.ndim != 2 or deltas.shape[0] < 2 or deltas.shape[1] != detection.shape[0]:
        raise ValueError('need at least two training rows matching the detection vector')
    if not np.isfinite(deltas).all():
        raise ValueError('training deltas must be finite')
    variance = deltas.var(axis=0)
    candidates = np.flatnonzero((detection >= min_detection) & (variance > 0))
    if candidates.size < n_genes:
        raise ValueError(f'only {candidates.size} candidate genes for {n_genes} requested')
    return candidates[np.argsort(-variance[candidates], kind='stable')][:n_genes]
