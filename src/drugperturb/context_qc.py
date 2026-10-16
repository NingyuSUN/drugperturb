"""Count-only qualification within each frozen experimental context.

This module reads no expression matrix and computes no treatment effects.
Absent source wells (known zero cells) and unavailable QC records are distinct.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .design import MATCH_KEYS, match_controls, qualify_wells


def _require(frame, columns, label):
    if not isinstance(frame, pd.DataFrame) or not set(columns).issubset(frame):
        raise ValueError(f'{label} requires columns {columns}')


def qualify_contexts(well_mapping, context_roles, well_qc, *, minimum_cells=30,
                     minimum_test_compounds=20, minimum_test_groups=10):
    """Qualify the original cohort per context, retaining all source exclusions.

    The formal CLI reads thresholds from the frozen protocol. Reduced values are
    useful only for synthetic engineering fixtures, never to rescue real gates.
    """
    _require(well_mapping, ['context_id', 'context_dose_nM', 'row_kind', 'treatment',
                           'replicate', 'Combo', 'control_Combo', *MATCH_KEYS], 'Well mapping')
    _require(context_roles, ['context_id', 'compound_id', 'cell_type', 'time_hours', 'dose_nM',
                            'split', 'role', 'scaffold_group', 'exclusion_reason'], 'Context roles')
    _require(well_qc, ['Combo', 'n_cells'], 'Well count QC')
    if any(isinstance(x, bool) or not isinstance(x, int) or x<1
           for x in [minimum_cells, minimum_test_compounds, minimum_test_groups]):
        raise ValueError('QC thresholds must be positive integers')
    wells = well_mapping.copy()
    roles = context_roles.fillna('').copy()
    if wells.empty or roles.empty or wells.duplicated(['context_id', 'Combo']).any():
        raise ValueError('Context well keys must be nonempty and unique')
    if roles.duplicated(['context_id', 'compound_id']).any():
        raise ValueError('Context compound keys must be unique')
    if not wells.row_kind.isin(['drug', 'vehicle']).all():
        raise ValueError('Unknown well row kind')
    if set(wells.context_id) != set(roles.context_id):
        raise ValueError('Mapping and role contexts differ')
    if well_qc.Combo.isna().any() or well_qc.Combo.duplicated().any():
        raise ValueError('Well count QC keys must be unique and nonmissing')
    try:
        values = pd.to_numeric(well_qc.n_cells, errors='raise').to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError('Cell counts must be numeric') from exc
    if not np.isfinite(values).all() or (values<0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError('Cell counts must be finite nonnegative integers')
    counts = pd.Series(values, index=well_qc.Combo)
    compound_tables, well_tables, summaries = [], [], []
    for context, selected in wells.groupby('context_id', sort=True):
        current_roles = roles[roles.context_id.eq(context)].copy()
        for source, role_field in [('cell_type', 'cell_type'), ('time_point', 'time_hours'),
                                   ('context_dose_nM', 'dose_nM')]:
            if selected[source].nunique()!=1 or current_roles[role_field].nunique()!=1 or selected[source].iloc[0]!=current_roles[role_field].iloc[0]:
                raise ValueError('Context physical metadata and frozen role definition differ')
        drugs = selected[selected.row_kind.eq('drug')].copy()
        controls = selected[selected.row_kind.eq('vehicle')].copy()
        if not set(drugs.treatment).issubset(set(current_roles.compound_id)):
            raise ValueError('Treatment lacks a frozen context role')
        # Recompute the strict key join instead of trusting an arbitrary control ID.
        expected = match_controls(drugs.drop(columns='control_Combo'), controls)
        declared = drugs.set_index('Combo').control_Combo
        if not expected.control_Combo.eq(expected.Combo.map(declared)).all():
            raise ValueError('Declared strict control does not match context/plate/pattern')
        qualified = qualify_wells(drugs, counts, minimum_cells)
        qualified['treatment_count_qc_available'] = qualified.Combo.isin(counts.index)
        qualified['control_count_qc_available'] = qualified.control_Combo.isin(counts.index)
        context_complete = bool(selected.Combo.isin(counts.index).all())
        results = []
        for role in current_roles.sort_values('compound_id').to_dict('records'):
            group = qualified[qualified.treatment.eq(role['compound_id'])]
            reasons = []
            if len(group)!=2 or set(group.replicate)!={'rep1','rep2'}:
                reasons.append('missing_or_duplicate_replicate')
            for name, count_column in [('treatment','treatment_cells'),('control','control_cells')]:
                available = group[f'{name}_count_qc_available']
                if group.empty or not available.all():
                    reasons.append(f'missing_{name}_count_qc')
                if group.loc[available, count_column].lt(minimum_cells).any():
                    reasons.append(f'low_{name}_cells')
            cell_eligible = not reasons
            source_reason = role['exclusion_reason']
            cohort_member = bool(role['split']) and not source_reason
            all_reasons = ([source_reason] if source_reason else [])+reasons
            if not role['split'] and not source_reason:
                all_reasons.insert(0, 'not_in_frozen_v1_cohort')
            record = dict(role, source_exclusion_reason=source_reason,
                          count_qc_exclusion_reason=';'.join(reasons),
                          exclusion_reason=';'.join(all_reasons),
                          metadata_replicates=len(group), count_eligible=cell_eligible,
                          eligible=cell_eligible and cohort_member,
                          count_qc_done=context_complete)
            for replicate in ['rep1','rep2']:
                row = group[group.replicate.eq(replicate)]
                record[f'{replicate}_cells'] = (int(row.treatment_cells.iloc[0])
                    if len(row)==1 and row.treatment_count_qc_available.iloc[0] else pd.NA)
                record[f'{replicate}_control_cells'] = (int(row.control_cells.iloc[0])
                    if len(row)==1 and row.control_count_qc_available.iloc[0] else pd.NA)
            results.append(record)
        compounds = pd.DataFrame(results)
        for field in ['rep1_cells','rep2_cells','rep1_control_cells','rep2_control_cells']:
            compounds[field] = compounds[field].astype('Int64')
        lookup = compounds.set_index('compound_id')
        for field in ['eligible','count_eligible','exclusion_reason','count_qc_exclusion_reason']:
            qualified[field] = qualified.treatment.map(lookup[field])
        qualified['count_qc_done'] = context_complete
        for name, column in [('treatment','treatment_cells'),('control','control_cells')]:
            qualified[column] = qualified[column].astype('Int64').mask(~qualified[f'{name}_count_qc_available'])
        # No source drugs are silently replaced when a source QC discrepancy appears.
        source_roles = compounds.role.isin(['source_train','source_validation'])
        if context_complete and not compounds.loc[source_roles,'eligible'].all():
            raise ValueError('Frozen source development cohort fails its original count QC')
        tests = compounds[compounds.split.eq('test')]
        eligible_tests = tests[tests.eligible]
        structure_groups = eligible_tests.scaffold_group.nunique()
        capacity_gate = len(eligible_tests)>=minimum_test_compounds and structure_groups>=minimum_test_groups
        evaluation_roles = compounds.role.isin(['primary_test','secondary_test'])
        predeclared = bool(evaluation_roles.any())
        summaries.append(dict(context_id=context,cell_type=current_roles.cell_type.iloc[0],
            time_hours=current_roles.time_hours.iloc[0],dose_nM=current_roles.dose_nM.iloc[0],
            source_treatment_labels=len(compounds),frozen_cohort_compounds=int(compounds.split.ne('').sum()),
            count_eligible_source_labels=int(compounds.count_eligible.sum()),
            eligible_frozen_compounds=int(compounds.eligible.sum()),
            frozen_test_compounds=len(tests),eligible_test_compounds=len(eligible_tests),
            eligible_test_structure_groups=structure_groups,
            primary_test_eligible_compounds=int((compounds.role.eq('primary_test') & compounds.eligible).sum()),
            secondary_test_eligible_compounds=int((compounds.role.eq('secondary_test') & compounds.eligible).sum()),
            required_test_compounds=minimum_test_compounds,required_test_structure_groups=minimum_test_groups,
            missing_well_qc_records=int((~selected.Combo.isin(counts.index)).sum()),
            count_qc_done=context_complete,predeclared_evaluation=predeclared,
            frozen_test_count_gate_met=context_complete and capacity_gate,
            evaluation_gate_met=context_complete and predeclared and capacity_gate,
            status=('COUNT_QC_UNAVAILABLE' if not context_complete else
                    ('COUNT_GATE_PASS' if predeclared and capacity_gate else
                     'COUNT_GATE_FAIL' if predeclared else 'COUNT_QC_ONLY'))))
        compound_tables.append(compounds)
        # Small operational fields suffice; do not duplicate SMILES/graphs into count reports.
        columns = ['context_id','cell_type','time_point','context_dose_nM','treatment','replicate','Combo',
                   'control_Combo','treatment_cells','control_cells','treatment_count_qc_available',
                   'control_count_qc_available','count_eligible','eligible','exclusion_reason',
                   'count_qc_exclusion_reason','count_qc_done']
        well_tables.append(qualified[columns])
    return {'compound_qualification': pd.concat(compound_tables,ignore_index=True).sort_values(
                ['context_id','compound_id']).reset_index(drop=True),
            'well_qualification': pd.concat(well_tables,ignore_index=True).sort_values(
                ['context_id','treatment','replicate','Combo']).reset_index(drop=True),
            'context_summary': pd.DataFrame(summaries).sort_values('context_id').reset_index(drop=True)}
