"""Scientific edge cases for exact-condition, response-blind LINCS qualification."""
import pandas as pd
import pytest

from drugperturb.chemistry import standardize_smiles
from drugperturb.external_v2 import audit_lincs_frames, exact_condition


def signature(**changes):
    row = dict(sig_id='sig1', pert_id='p1', pert_type='trt_cp', cell_id='A549',
               pert_dose='1', pert_dose_unit='µM', pert_time='24', pert_time_unit='h',
               pert_idose='1 µM', pert_itime='24 h')
    return dict(row, **changes)


@pytest.mark.parametrize('value,unit,expected', [('1','µM',True), ('1','um',True),
    ('1','μM',True), ('1000','nM',True), ('0.001','mM',True),
    ('1.11','µM',False), ('1.0001','µM',False), ('-666','µM',False),
    ('nan','µM',False), ('one','µM',False), ('1','unknown',False)])
def test_exact_numeric_dose_is_not_display_label(value, unit, expected):
    assert exact_condition(signature(pert_dose=value, pert_dose_unit=unit)) is expected


def test_exact_time_and_cell_not_rounded_display_fields():
    assert not exact_condition(signature(pert_time='23.5'))
    assert not exact_condition(signature(cell_id='MCF7'))
    assert exact_condition(signature(pert_time='1440', pert_time_unit='min'))
    assert not exact_condition(signature(pert_type='ctl_vehicle'))


def fixture():
    split = pd.DataFrame([dict(compound_id='d1', name='drug', split='test',
                              **standardize_smiles('CCO'))])
    sig = pd.DataFrame([signature()])
    pert = pd.DataFrame([dict(pert_id='p1', pert_iname='different-name', canonical_smiles='OCC')])
    genes = pd.DataFrame([dict(pr_gene_id='11', pr_gene_symbol='A', pr_is_lm='1'),
                          dict(pr_gene_id='12', pr_gene_symbol='B', pr_is_lm='0')])
    selected = pd.DataFrame([dict(gene_id='ENSG1', gene_symbol='A'),
                             dict(gene_id='ENSG2', gene_symbol='B')])
    return sig, pert, genes, split, selected


def test_identity_matches_parent_not_name_and_inferred_genes_are_excluded():
    result = audit_lincs_frames(*fixture())
    assert result['counts']['matched_test_compounds'] == 1
    assert result['counts']['matched_test_structure_groups'] == 1
    assert result['counts']['exact_unambiguous_landmark_genes'] == 1
    assert result['test_compounds'][0]['matched_pert_ids'] == ['p1']
    assert result['expression_accessed'] is False


def test_name_only_or_near_dose_cannot_rescue_overlap():
    sig, pert, genes, split, selected = fixture()
    pert.loc[0, 'pert_iname'] = 'drug'; pert.loc[0, 'canonical_smiles'] = 'CCN'
    assert audit_lincs_frames(sig, pert, genes, split, selected)['counts']['matched_test_compounds'] == 0
    pert.loc[0, 'canonical_smiles'] = 'CCO'; sig.loc[0, 'pert_dose'] = '1.11'
    result = audit_lincs_frames(sig, pert, genes, split, selected)
    assert result['counts']['coarse_label_signatures'] == 1
    assert result['counts']['exact_condition_signatures'] == 0


def test_missing_or_ambiguous_identity_is_explicit_not_silent_zero():
    sig, pert, genes, split, selected = fixture()
    result = audit_lincs_frames(sig, pert.iloc[:0], genes, split, selected)
    assert result['identity_issues'][0]['reason'] == 'missing_pert_info'
    duplicate = pd.concat([pert, pert.assign(canonical_smiles='CCN')], ignore_index=True)
    result = audit_lincs_frames(sig, duplicate, genes, split, selected)
    assert result['identity_issues'][0]['reason'] == 'ambiguous_pert_info'


def test_gene_symbol_ambiguity_is_reported_and_not_double_counted():
    sig, pert, genes, split, selected = fixture()
    genes = pd.concat([genes, genes.iloc[[0]].assign(pr_gene_id='99')], ignore_index=True)
    result = audit_lincs_frames(sig, pert, genes, split, selected)
    assert result['counts']['exact_unambiguous_landmark_genes'] == 0
    assert result['gene_mapping_ambiguities'] == ['A']


def test_changed_frozen_structure_or_duplicate_signature_fails():
    sig, pert, genes, split, selected = fixture()
    split.loc[0, 'canonical_smiles'] = 'CCN'
    with pytest.raises(ValueError, match='frozen'):
        audit_lincs_frames(sig, pert, genes, split, selected)
    sig, pert, genes, split, selected = fixture()
    with pytest.raises(ValueError, match='signature'):
        audit_lincs_frames(pd.concat([sig, sig]), pert, genes, split, selected)
