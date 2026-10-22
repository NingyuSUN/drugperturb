"""Tahoe-100M design for v5: samples, vehicles, compounds, split and contexts from metadata only.

Nothing here reads expression, and nothing reads a cell count of a compound
sample. Samples come from the design columns of ``sample_metadata``; structures
from ``drug_metadata``; vehicle unit sizes from ``obs_metadata`` rows of vehicle
samples only. The compound split uses structures only.

Layout facts the checks enforce: every sample is one drug at one dose in uM;
every plate has exactly ``VEHICLE_WELLS_PER_PLATE`` DMSO_TF samples; every
compound dose is one of ``DOSES_UM``. A compound sample's strict matched vehicle
is all vehicle cells of the same cell line on the same plate. The measurement
unit of a context (cell line, dose) is the cells of that line in one sample.
"""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd

from drugperturb.chemistry import standardize_smiles
from drugperturb.v4.design import split_by_structure_fractions

VEHICLE_DRUG = 'DMSO_TF'
# Adagrasib at 0.05 uM is on every plate (an inter-plate reference). Like the OP3 positive
# controls in v4, it has a different replicate structure, so every Adagrasib sample is excluded.
REFERENCE_CONTROLS = ('Adagrasib',)
DOSES_UM = (0.05, 0.5, 5.0)
VEHICLE_WELLS_PER_PLATE = 2
SAMPLE_COLUMNS = ('sample', 'plate', 'drug', 'drugname_drugconc')
DRUG_COLUMNS = ('drug', 'canonical_smiles', 'pubchem_cid')
CELL_LINE_COLUMNS = ('cell_name', 'Cell_ID_Cellosaur', 'Organ')
DUPLICATE_PREFIX = 'duplicate_parent_represented_by:'


def _normalized(name):
    return ' '.join(str(name).split())


def plate_number(plate):
    if not str(plate).startswith('plate') or not str(plate)[5:].isdigit():
        raise ValueError(f'unexpected plate label {plate!r}')
    return int(str(plate)[5:])


def dose_nM(dose_uM):
    return int(round(float(dose_uM)*1000))


def context_id(cell_line, dose_uM):
    return f'{cell_line}_{dose_nM(dose_uM)}nM'


def parse_drugconc(value):
    """``"[('Name', 0.05, 'uM')]"`` -> ('Name', 0.05); a sample must hold exactly one agent in uM."""
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or len(parsed) != 1 or len(parsed[0]) != 3:
        raise ValueError(f'expected one (name, concentration, unit) entry, got {value!r}')
    name, concentration, unit = parsed[0]
    if unit != 'uM':
        raise ValueError(f'unexpected concentration unit in {value!r}')
    return _normalized(name), float(concentration)


def build_samples(sample_metadata):
    """One row per sample with role, dose and the plate's vehicle samples; raises on unexpected layouts."""
    missing = set(SAMPLE_COLUMNS) - set(sample_metadata.columns)
    if missing:
        raise ValueError(f'sample metadata lacks {sorted(missing)}')
    frame = sample_metadata[list(SAMPLE_COLUMNS)].copy()
    if frame['sample'].duplicated().any():
        raise ValueError('sample identifiers must be unique')
    parsed = [parse_drugconc(v) for v in frame.drugname_drugconc]
    names = [p[0] for p in parsed]
    if any(n != _normalized(d) for n, d in zip(names, frame.drug)):
        raise ValueError('drugname_drugconc names must match the drug column up to whitespace')
    # Some samples spell a drug with a trailing space ('Selinexor '); the source spelling is kept.
    frame['drug_source'] = frame.drug
    frame['drug'] = names
    frame['dose_uM'] = [p[1] for p in parsed]
    frame['role'] = np.where(frame.drug.eq(VEHICLE_DRUG), 'vehicle',
                             np.where(frame.drug.isin(REFERENCE_CONTROLS), 'reference_control', 'compound'))
    if not np.allclose(frame.loc[frame.role.eq('vehicle'), 'dose_uM'], 0.0):
        raise ValueError('vehicle samples must have concentration 0')
    doses = frame.loc[~frame.role.eq('vehicle'), 'dose_uM']
    if not doses.map(lambda d: any(np.isclose(d, x) for x in DOSES_UM)).all():
        raise ValueError(f'every compound dose must be one of {DOSES_UM} uM')
    frame['plate_number'] = frame.plate.map(plate_number)
    vehicles = frame[frame.role.eq('vehicle')]
    per_plate = vehicles.groupby('plate').size().reindex(frame.plate.unique(), fill_value=0)
    if per_plate.ne(VEHICLE_WELLS_PER_PLATE).any():
        raise ValueError(f'each plate needs exactly {VEHICLE_WELLS_PER_PLATE} vehicle samples: {per_plate.to_dict()}')
    vehicle_of = vehicles.groupby('plate')['sample'].agg(lambda s: ';'.join(sorted(s)))
    frame['vehicle_samples'] = frame.plate.map(vehicle_of)
    frame['dose_nM'] = np.where(frame.role.eq('vehicle'), 0, frame.dose_uM.map(dose_nM))
    columns = ['sample', 'plate', 'plate_number', 'role', 'drug', 'drug_source', 'dose_uM', 'dose_nM',
               'vehicle_samples']
    return frame.sort_values(['plate_number', 'sample'])[columns].reset_index(drop=True)


def compound_table(drug_metadata, samples):
    """v1 standardization of every drug used in compound samples, with v1 duplicate-parent representatives.

    Excluded drugs keep a row with the reason. Among accepted drugs sharing a
    canonical parent, the least modified formulation is kept, then the first
    name in sort order (the v1 rule in scripts/prepare_design.py); the others are
    excluded as represented by it. No response is read.
    """
    used = sorted(samples.loc[samples.role.eq('compound'), 'drug'].unique())
    meta = drug_metadata[list(DRUG_COLUMNS)].copy()
    if meta.drug.duplicated().any() or meta.drug.map(_normalized).ne(meta.drug).any():
        raise ValueError('drug metadata must have one row per drug, with whitespace-normalized names')
    meta = meta.set_index('drug')
    absent = sorted(set(used) - set(meta.index))
    if absent:
        raise ValueError(f'drugs without metadata: {absent}')
    rows = []
    for drug in used:
        smiles, cid = meta.loc[drug, 'canonical_smiles'], meta.loc[drug, 'pubchem_cid']
        row = dict(compound_id=drug, pubchem_cid='' if pd.isna(cid) else str(int(cid)),
                   original_smiles='' if pd.isna(smiles) else smiles, status='accepted', exclusion_reason='')
        if pd.isna(smiles) or not str(smiles).strip():
            row.update(status='excluded', exclusion_reason='no SMILES in drug_metadata')
        else:
            try:
                result = standardize_smiles(smiles)
                row.update({k: result[k] for k in ('canonical_smiles', 'scaffold_smiles', 'scaffold_group',
                                                   'n_original_fragments')})
                row['removed_fragments'] = ';'.join(result['removed_fragments'])
                row['n_removed_fragments'] = len(result['removed_fragments'])
            except ValueError as error:
                row.update(status='excluded', exclusion_reason=str(error))
        rows.append(row)
    table = pd.DataFrame(rows)
    for name in ('canonical_smiles', 'scaffold_smiles', 'scaffold_group', 'removed_fragments'):
        table[name] = table.get(name, pd.Series(dtype=object)).fillna('')
    for name in ('n_original_fragments', 'n_removed_fragments'):
        table[name] = table.get(name, pd.Series(dtype=float)).fillna(-1).astype(int)
    accepted = table[table.status.eq('accepted')].sort_values(['n_removed_fragments', 'compound_id'])
    for _, group in accepted.groupby('canonical_smiles', sort=True):
        for index in group.index[1:]:
            table.loc[index, 'status'] = 'excluded'
            table.loc[index, 'exclusion_reason'] = DUPLICATE_PREFIX+group.iloc[0].compound_id
    columns = ['compound_id', 'pubchem_cid', 'status', 'exclusion_reason', 'original_smiles', 'canonical_smiles',
               'scaffold_smiles', 'scaffold_group', 'n_original_fragments', 'removed_fragments', 'n_removed_fragments']
    return table[columns].sort_values('compound_id').reset_index(drop=True)


def split_compounds(compounds, spec):
    """Whole-scaffold-group split of accepted compounds (v4 procedure); excluded rows get an empty split."""
    accepted = compounds[compounds.status.eq('accepted')]
    records = accepted[['compound_id', 'canonical_smiles', 'scaffold_group']].to_dict('records')
    records, report = split_by_structure_fractions(
        records, seed=spec['seed'], test_fraction=spec['test_fraction'],
        validation_fraction=spec['validation_fraction'], min_test_compounds=spec['min_test_compounds'],
        min_test_groups=spec['min_test_groups'])
    split_of = {r['compound_id']: r['split'] for r in records}
    table = compounds.copy()
    table['split'] = table.compound_id.map(split_of).fillna('')
    return table, report


def replicate_table(samples, compounds):
    """One row per accepted (compound, dose): replicate samples and plates."""
    accepted = set(compounds.loc[compounds.status.eq('accepted'), 'compound_id'])
    chosen = samples[samples.role.eq('compound') & samples.drug.isin(accepted)]
    table = (chosen.sort_values(['plate_number', 'sample'])
             .groupby(['drug', 'dose_nM'], as_index=False)
             .agg(n_samples=('sample', 'size'), samples=('sample', ';'.join), plates=('plate', ';'.join)))
    return table.rename(columns={'drug': 'compound_id'})


def vehicle_units(vehicle_cells, samples, cell_lines):
    """Vehicle cells per (plate, cell line), pooled over the plate's vehicle samples; absent units are 0.

    ``vehicle_cells`` has one row per cell of a vehicle sample, with ``sample`` and
    ``cell_line_id``. Raises if it holds any cell of a non-vehicle sample.
    """
    vehicle = samples[samples.role.eq('vehicle')]
    if not vehicle_cells['sample'].isin(vehicle['sample']).all():
        raise ValueError('vehicle_cells must contain vehicle samples only')
    plate_of = vehicle.set_index('sample').plate
    counts = (vehicle_cells.assign(plate=vehicle_cells['sample'].map(plate_of))
              .groupby(['plate', 'cell_line_id']).size())
    plates = sorted(samples.plate.unique(), key=plate_number)
    index = pd.MultiIndex.from_product([plates, sorted(cell_lines)], names=['plate', 'cell_line_id'])
    unknown = set(counts.index.get_level_values('cell_line_id')) - set(cell_lines)
    if unknown:
        raise ValueError(f'vehicle cells with unknown cell lines: {sorted(unknown)}')
    return counts.reindex(index, fill_value=0).rename('vehicle_cells').reset_index()


def select_cell_lines(units, *, min_median_cells=200, n_lines=10, seed=42):
    """The v5 line rule: lines whose median vehicle unit over plates is >= min_median_cells,
    sorted by Cellosaurus ID, then ``n_lines`` drawn with numpy default_rng(seed). Vehicle cells only."""
    median = units.groupby('cell_line_id').vehicle_cells.median().sort_index()
    eligible = list(median.index[median >= min_median_cells])
    if len(eligible) < n_lines:
        raise ValueError(f'only {len(eligible)} lines meet the vehicle-size rule; {n_lines} are required')
    draw = np.random.default_rng(seed).choice(len(eligible), size=n_lines, replace=False)
    chosen = [eligible[i] for i in draw]
    table = median.rename('median_vehicle_cells').reset_index()
    table['eligible'] = table.cell_line_id.isin(eligible)
    table['draw_order'] = table.cell_line_id.map({c: i+1 for i, c in enumerate(chosen)}).fillna(0).astype(int)
    table['selected'] = table.draw_order.gt(0)
    return table, sorted(chosen)


def context_table(lines):
    rows = [dict(context_id=context_id(line, dose), cell_line_id=line, dose_uM=dose, dose_nM=dose_nM(dose))
            for line in sorted(lines) for dose in DOSES_UM]
    return pd.DataFrame(rows)
