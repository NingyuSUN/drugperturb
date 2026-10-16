"""Exact-condition LINCS metadata audit; expression matrices never enter this API."""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation

import pandas as pd

from .chemistry import standardize_smiles


def _quantity(value, unit, factors):
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    key = str(unit).strip().replace('µ', 'u').replace('μ', 'u').lower()
    if not number.is_finite() or number < 0 or key not in factors:
        return None
    return number * factors[key]


def exact_condition(row):
    """Use numeric concentration/time, never rounded pert_idose/pert_itime."""
    dose = _quantity(row.get('pert_dose'), row.get('pert_dose_unit'),
                     {'um': Decimal(1), 'nm': Decimal('.001'), 'mm': Decimal(1000)})
    time = _quantity(row.get('pert_time'), row.get('pert_time_unit'),
                     {'h': Decimal(1), 'min': Decimal(1)/Decimal(60)})
    return bool(row.get('pert_type') == 'trt_cp' and row.get('cell_id') == 'A549'
                and dose == Decimal(1) and time == Decimal(24))


def _require(frame, columns, name):
    if not isinstance(frame, pd.DataFrame) or not set(columns).issubset(frame.columns):
        raise ValueError(f'{name}: missing required metadata columns')


def audit_lincs_frames(signatures, perturbations, lincs_genes, split, selected_genes):
    """Return explicit matches/exclusions and the pre-QC feasibility decision.

    A raw metadata pass would still require separately prespecified signature QC.
    Exact-symbol coverage is conservative, not a historical-alias upper bound.
    """
    _require(signatures, ['sig_id', 'pert_id', 'pert_type', 'cell_id', 'pert_dose',
                         'pert_dose_unit', 'pert_time', 'pert_time_unit'], 'signatures')
    _require(perturbations, ['pert_id', 'pert_iname', 'canonical_smiles'], 'perturbations')
    _require(lincs_genes, ['pr_gene_id', 'pr_gene_symbol', 'pr_is_lm'], 'LINCS genes')
    _require(split, ['compound_id', 'split', 'original_smiles', 'canonical_smiles',
                     'scaffold_group'], 'frozen split')
    _require(selected_genes, ['gene_id', 'gene_symbol'], 'output genes')
    if signatures.sig_id.isna().any() or signatures.sig_id.duplicated().any():
        raise ValueError('signature IDs must be unique and nonmissing')
    if split.compound_id.duplicated().any() or split.canonical_smiles.duplicated().any():
        raise ValueError('frozen compound IDs and parents must be unique')
    if not split.split.isin(['train', 'validation', 'test']).all():
        raise ValueError('invalid frozen split assignment')
    if selected_genes.gene_id.isna().any() or selected_genes.gene_id.duplicated().any():
        raise ValueError('selected gene IDs must be unique and nonmissing')
    for row in split.itertuples(index=False):
        actual = standardize_smiles(row.original_smiles)
        if any(actual[k] != getattr(row, k) for k in ['canonical_smiles', 'scaffold_group']):
            raise ValueError(f'frozen structure mismatch: {row.compound_id}')
    scoped = signatures.loc[signatures.pert_type.eq('trt_cp') & signatures.cell_id.eq('A549')]
    records = scoped.to_dict('records')
    exact = [r for r in records if exact_condition(r)]
    coarse = [r for r in records if r.get('pert_idose') == '1 µM' and r.get('pert_itime') == '24 h']
    source_by_id = {key: group for key, group in perturbations.groupby('pert_id', sort=False)}
    canonical_index, issues, eligible_structures = {}, [], []
    for pert_id in sorted({r['pert_id'] for r in exact}):
        rows = source_by_id.get(pert_id)
        if rows is None:
            issues.append({'pert_id': pert_id, 'reason': 'missing_pert_info'})
            continue
        if len(rows) != 1:
            issues.append({'pert_id': pert_id, 'reason': 'ambiguous_pert_info'})
            continue
        row = rows.iloc[0]
        try:
            identity = standardize_smiles(row.canonical_smiles)
        except (ValueError, TypeError) as error:
            issues.append({'pert_id': pert_id, 'reason': 'unsupported_structure', 'detail': str(error)})
            continue
        canonical_index.setdefault(identity['canonical_smiles'], []).append(pert_id)
        eligible_structures.append({'pert_id': pert_id, 'name': str(row.pert_iname),
                                    'original_smiles': str(row.canonical_smiles),
                                    'canonical_smiles': identity['canonical_smiles']})
    compound_rows = []
    for row in split.sort_values('compound_id').itertuples(index=False):
        matches = sorted(canonical_index.get(row.canonical_smiles, []))
        compound_rows.append({'compound_id': row.compound_id, 'split': row.split,
                              'scaffold_group': row.scaffold_group, 'matched_pert_ids': matches,
                              'matched_sig_ids': sorted(r['sig_id'] for r in exact if r['pert_id'] in matches),
                              'status': 'exact_parent_match' if matches else 'no_exact_parent_at_required_condition'})
    test_rows = [r for r in compound_rows if r['split'] == 'test']
    matched_test = [r for r in test_rows if r['matched_pert_ids']]
    landmarks = lincs_genes.loc[lincs_genes.pr_is_lm.astype(str).eq('1')]
    symbol_groups = {key: group for key, group in landmarks.groupby('pr_gene_symbol', sort=False)}
    symbol_counts = Counter(selected_genes.gene_symbol)
    gene_matches, ambiguities = [], []
    for row in selected_genes.itertuples(index=False):
        candidates = symbol_groups.get(row.gene_symbol)
        if candidates is None:
            continue
        if symbol_counts[row.gene_symbol] != 1 or len(candidates) != 1:
            ambiguities.append(row.gene_symbol)
            continue
        gene_matches.append({'gene_id': row.gene_id, 'gene_symbol': row.gene_symbol,
                             'lincs_gene_id': str(candidates.iloc[0].pr_gene_id)})
    count = {'source_signature_records': len(signatures),
             'coarse_label_signatures': len(coarse),
             'coarse_label_pert_ids': len({r['pert_id'] for r in coarse}),
             'exact_condition_signatures': len(exact),
             'exact_condition_pert_ids': len({r['pert_id'] for r in exact}),
             'structurally_resolved_exact_pert_ids': len(eligible_structures),
             'frozen_source_compounds': len(split), 'frozen_test_compounds': len(test_rows),
             'matched_test_compounds': len(matched_test),
             'matched_test_structure_groups': len({r['scaffold_group'] for r in matched_test}),
             'selected_output_genes': len(selected_genes), 'direct_landmark_records': len(landmarks),
             'exact_unambiguous_landmark_genes': len(gene_matches)}
    passed = (count['matched_test_compounds'] >= 20 and count['matched_test_structure_groups'] >= 10
              and count['exact_unambiguous_landmark_genes'] >= 100)
    if issues:
        decision = 'UNRESOLVED_METADATA_IDENTITY'
    elif passed:
        decision = 'METADATA_GO_PENDING_PRESPECIFIED_QUALITY_RULE'
    else:
        decision = 'NO_GO_EXACT_CONDITION_COHORT'
    return dict(schema_version=1, gate_decision=decision, counts=count,
                criteria={'minimum_test_compounds': 20, 'minimum_test_structure_groups': 10,
                          'minimum_direct_landmark_genes': 100},
                condition={'cell_type': 'A549', 'time_hours': 24, 'actual_dose_uM': 1.0},
                exact_condition_signature_metadata=exact, exact_condition_structures=eligible_structures,
                coarsened_label_actual_dose_counts=dict(Counter(r['pert_dose'] for r in coarse)),
                identity_issues=issues, test_compounds=test_rows,
                matched_compounds_by_split={key: sum(bool(r['matched_pert_ids']) for r in compound_rows if r['split'] == key)
                                           for key in ['train', 'validation', 'test']},
                direct_landmark_matches=gene_matches, gene_mapping_ambiguities=sorted(set(ambiguities)),
                gene_mapping_scope='exact symbols only; historical aliases not exhaustively resolved',
                expression_accessed=False, signature_quality_filter_applied=False,
                interpretation='Dataset compatibility only; no external model-performance estimate',
                caveats=['L1000 and sci-Plex differ in platform, reference and scale.',
                         'No overlap in this condition does not establish absence in other conditions.',
                         'A metadata pass would not establish biological or clinical validity.'])
