"""Response-blind context inventory using immutable v1 chemistry and splits.

Only experimental design and chemical identity enter this module. A mapped well
is not a count-QC-qualified sample, and metadata capacity is not evaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .chemistry import standardize_smiles
from .design import REQUIRED, match_controls, select_design

CELL_TYPES = ('A549', 'MCF7', 'K562')
DOSES_NM = (10, 100, 1000, 10000)
IDENTITY_FIELDS = ('original_smiles', 'canonical_smiles', 'scaffold_group')


def _require(frame, columns, name):
    if not isinstance(frame, pd.DataFrame) or not set(columns).issubset(frame.columns):
        raise ValueError(f'{name} requires columns {list(columns)}')
    if frame.empty:
        raise ValueError(f'{name} is empty')


def _identities(chemistry, split):
    required = ['compound_id', 'status', 'exclusion_reason', *IDENTITY_FIELDS]
    _require(chemistry, required, 'chemistry identity')
    _require(split, ['compound_id', 'split', *IDENTITY_FIELDS], 'frozen split identity')
    chemistry = chemistry[required].fillna('').copy()
    split = split[['compound_id', 'split', *IDENTITY_FIELDS]].fillna('').copy()
    for frame in [chemistry, split]:
        if frame.compound_id.eq('').any() or frame.compound_id.duplicated().any():
            raise ValueError('compound identity keys must be unique and nonmissing')
    if not chemistry.status.isin(['accepted', 'excluded']).all():
        raise ValueError('chemistry identity status must be accepted or excluded')
    if chemistry.original_smiles.eq('').any():
        raise ValueError('chemistry original SMILES identity is missing')
    accepted = chemistry[chemistry.status.eq('accepted')]
    if accepted[['canonical_smiles', 'scaffold_group']].eq('').any().any():
        raise ValueError('accepted canonical/scaffold identity is missing')
    if accepted.canonical_smiles.duplicated().any():
        raise ValueError('duplicate accepted canonical parent identity')
    if chemistry.loc[chemistry.status.eq('excluded'), 'exclusion_reason'].eq('').any():
        raise ValueError('source exclusions require an explicit reason')
    for row in accepted.itertuples(index=False):
        actual = standardize_smiles(row.original_smiles)
        if any(actual[field] != getattr(row, field) for field in ['canonical_smiles', 'scaffold_group']):
            raise ValueError(f'chemistry identity differs from original SMILES: {row.compound_id}')
    reference = chemistry.set_index('compound_id')
    if not split.split.isin(['train', 'validation', 'test']).all():
        raise ValueError('invalid frozen split label')
    for row in split.itertuples(index=False):
        if row.compound_id not in reference.index or reference.loc[row.compound_id, 'status'] != 'accepted':
            raise ValueError('frozen split refers to missing or excluded chemistry identity')
        if any(reference.loc[row.compound_id, field] != getattr(row, field) for field in IDENTITY_FIELDS):
            raise ValueError(f'frozen split identity conflict: {row.compound_id}')
    for field in ['canonical_smiles', 'scaffold_group']:
        if split.groupby(field).split.nunique().gt(1).any():
            raise ValueError(f'{field} crosses frozen splits')
    representatives = {}
    for row in chemistry.itertuples(index=False):
        representative = row.compound_id if row.status == 'accepted' else ''
        prefix = 'duplicate_parent_represented_by:'
        if row.exclusion_reason.startswith(prefix):
            representative = row.exclusion_reason.removeprefix(prefix)
            if representative not in reference.index or reference.loc[representative, 'status'] != 'accepted':
                raise ValueError('duplicate parent representative identity is missing')
            if reference.loc[representative, 'canonical_smiles'] != row.canonical_smiles:
                raise ValueError('duplicate parent representative identity conflicts')
        representatives[row.compound_id] = representative
    chemistry['representative_compound_id'] = chemistry.compound_id.map(representatives)
    chemistry['split'] = chemistry.compound_id.map(split.set_index('compound_id').split).fillna('')
    return chemistry


def _role(cell, dose, split, exclusion):
    if exclusion:
        return 'excluded'
    if cell == 'A549' and dose == 1000:
        return {'train': 'source_train', 'validation': 'source_validation', 'test': 'source_test_locked'}[split]
    if split == 'test':
        if cell == 'MCF7':
            return 'primary_test' if dose == 1000 else 'secondary_test'
        if cell == 'K562' and dose == 1000:
            return 'secondary_test'
    return 'metadata_only'


def audit_contexts(metadata, chemistry, split):
    """Return deterministic context_roles, well_mapping and design_inventory tables.

    Includes all source labels, even excluded structures and parents outside the
    frozen v1 cohort. Vehicle rows include both dose patterns on drug-containing
    plates; deduplicate Combo across contexts before extracting raw counts.
    Every dose/context must preserve both experimental replicates for each label.
    """
    _require(metadata, [*REQUIRED, 'SMILES'], 'metadata')
    metadata = metadata.copy()
    for field in ['dose', 'time_point']:
        try:
            values = pd.to_numeric(metadata[field], errors='raise')
        except (TypeError, ValueError) as exc:
            raise ValueError(f'invalid metadata {field}') from exc
        if not np.isfinite(values).all() or values.lt(0).any() or not values.eq(np.floor(values)).all():
            raise ValueError(f'metadata {field} must be finite nonnegative integers')
        metadata[field] = values.astype(int)
    truth = metadata.vehicle.astype(str).str.lower()
    if not truth.isin(['true', 'false']).all():
        raise ValueError('vehicle must be explicit TRUE/FALSE')
    metadata['vehicle'] = truth.eq('true')
    scoped = metadata[metadata.cell_type.isin(CELL_TYPES) & metadata.time_point.eq(24)]
    drugs_all = scoped[~scoped.vehicle]
    if not drugs_all.dose.isin(DOSES_NM).all():
        raise ValueError('unsupported drug dose in v2 context scope')
    identities = _identities(chemistry, split)
    if set(drugs_all.treatment) != set(identities.compound_id):
        raise ValueError('source labels and chemistry identity labels differ')
    reference = identities.set_index('compound_id')
    for compound, rows in drugs_all.groupby('treatment'):
        if rows.SMILES.isna().any() or rows.SMILES.nunique() != 1 or rows.SMILES.iloc[0] != reference.loc[compound, 'original_smiles']:
            raise ValueError(f'source SMILES identity conflict: {compound}')
    roles, well_tables, inventory = [], [], []
    # Restrict output to scientific source fields; arbitrary input columns are not exported.
    well_fields = ['well_oligo', 'plate_oligo', 'Combo', 'cell_type', 'time_point',
                   'replicate', 'treatment', 'vehicle', 'dose', 'dose_pattern', 'SMILES']
    for cell in CELL_TYPES:
        for dose in DOSES_NM:
            context = f'{cell}_24h_{dose}nM'
            drugs, controls = select_design(metadata, cell_type=cell, time_hours=24, dose_nM=dose)
            mapping = match_controls(drugs, controls)
            for _, group in mapping.groupby('treatment'):
                if len(group) != 2 or set(group.replicate) != {'rep1', 'rep2'}:
                    raise ValueError(f'{context}: missing or duplicate treatment replicate')
            if set(mapping.treatment) != set(reference.index):
                raise ValueError(f'{context}: incomplete context treatments')
            context_roles = identities.copy()
            context_roles['exclusion_reason'] = [
                row.exclusion_reason if row.status == 'excluded' else
                ('' if row.split else 'not_in_frozen_v1_cohort')
                for row in context_roles.itertuples(index=False)]
            context_roles = context_roles.rename(columns={'status': 'source_status'})
            context_roles['context_id'] = context
            context_roles['cell_type'] = cell
            context_roles['time_hours'] = 24
            context_roles['dose_nM'] = dose
            context_roles['role'] = [_role(cell, dose, row.split, row.exclusion_reason)
                                     for row in context_roles.itertuples(index=False)]
            context_roles['count_qc_done'] = False
            roles.append(context_roles)
            role_keys = ['compound_id', 'canonical_smiles', 'scaffold_group', 'representative_compound_id',
                         'split', 'role', 'source_status', 'exclusion_reason']
            drug_rows = mapping[well_fields+['control_Combo']].merge(
                context_roles[role_keys], left_on='treatment', right_on='compound_id',
                how='left', validate='many_to_one')
            drug_rows['row_kind'] = 'drug'
            drug_rows['is_strict_control'] = False
            vehicles = controls[controls.plate_oligo.isin(set(mapping.plate_oligo))][well_fields].copy()
            vehicles['control_Combo'] = ''
            for field in role_keys:
                vehicles[field] = ''
            vehicles['role'] = 'vehicle'
            vehicles['row_kind'] = 'vehicle'
            vehicles['is_strict_control'] = vehicles.Combo.isin(set(mapping.control_Combo))
            wells = pd.concat([drug_rows, vehicles], ignore_index=True)
            wells['context_id'] = context
            wells['context_dose_nM'] = dose
            wells['count_qc_done'] = False
            if wells.duplicated(['context_id', 'Combo']).any():
                raise ValueError('context Combo duplicated during vehicle selection')
            well_tables.append(wells)
            frozen = context_roles[context_roles.split.ne('')]
            tests = frozen[frozen.split.eq('test')]
            row = dict(context_id=context, cell_type=cell, time_hours=24, dose_nM=dose,
                       source_treatment_labels=len(context_roles), source_drug_wells=len(mapping),
                       chemistry_accepted_parents=int(context_roles.source_status.eq('accepted').sum()),
                       source_structure_excluded=int(context_roles.source_status.eq('excluded').sum()),
                       not_in_frozen_v1_cohort=int(context_roles.exclusion_reason.eq('not_in_frozen_v1_cohort').sum()),
                       frozen_cohort_compounds=len(frozen), frozen_test_compounds=len(tests),
                       frozen_test_structure_groups=tests.scaffold_group.nunique(),
                       strict_control_wells=mapping.control_Combo.nunique(), same_plate_vehicle_wells=len(vehicles),
                       metadata_test_capacity_gate_met=len(tests) >= 20 and tests.scaffold_group.nunique() >= 10,
                       count_qc_done=False, evaluation_ready=False)
            row.update({f'role_{name}': int(context_roles.role.eq(name).sum())
                        for name in ['source_train', 'source_validation', 'source_test_locked',
                                     'primary_test', 'secondary_test', 'metadata_only', 'excluded']})
            inventory.append(row)
    return {
        'context_roles': pd.concat(roles, ignore_index=True).sort_values(
            ['cell_type', 'dose_nM', 'compound_id']).reset_index(drop=True),
        'well_mapping': pd.concat(well_tables, ignore_index=True).sort_values(
            ['cell_type', 'context_dose_nM', 'row_kind', 'Combo']).reset_index(drop=True),
        'design_inventory': pd.DataFrame(inventory).sort_values(
            ['cell_type', 'dose_nM']).reset_index(drop=True),
    }
