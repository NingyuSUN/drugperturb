"""Count-only context qualification must never combine wells across conditions."""
from importlib import import_module
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd
import pytest

from drugperturb.context_design import audit_contexts
from test_context_design import fixtures


def count_fixture():
    tables = audit_contexts(*fixtures())
    wells = tables['well_mapping']
    qc = wells[['Combo']].drop_duplicates().reset_index(drop=True)
    qc['n_cells'] = 40
    qc['human_UMI'] = 800
    qc['genes_detected'] = 20
    qc['source_observed'] = True
    return wells, tables['context_roles'], qc


def qualify(wells, roles, qc):
    return import_module('drugperturb.context_qc').qualify_contexts(
        wells, roles, qc, minimum_cells=30, minimum_test_compounds=1, minimum_test_groups=1)


def test_one_low_replicate_excludes_compound_only_in_that_context():
    wells, roles, qc = count_fixture()
    qc.loc[qc.Combo.eq('D2MCF7_1000_rep2'), 'n_cells'] = 29
    result = qualify(wells, roles, qc)
    compounds = result['compound_qualification']
    failed = compounds[compounds.context_id.eq('MCF7_24h_1000nM') & compounds.compound_id.eq('d3')].iloc[0]
    assert not failed.eligible
    assert failed.exclusion_reason == 'low_treatment_cells'
    comparison = compounds[compounds.context_id.eq('MCF7_24h_100nM') & compounds.compound_id.eq('d3')].iloc[0]
    assert comparison.eligible
    summary = result['context_summary'].set_index('context_id')
    assert not summary.loc['MCF7_24h_1000nM', 'evaluation_gate_met']
    assert summary.loc['MCF7_24h_100nM', 'evaluation_gate_met']
    assert len(compounds) == 48
    assert compounds.loc[compounds.compound_id.eq('bad'), 'exclusion_reason'].str.contains('undefined_atom').all()


def test_low_shared_strict_control_excludes_all_context_drugs_without_fallback():
    wells, roles, qc = count_fixture()
    qc.loc[qc.Combo.eq('C2MCF7_1000_rep1'), 'n_cells'] = 29
    result = qualify(wells, roles, qc)
    rows = result['well_qualification']
    failed = rows[rows.context_id.eq('MCF7_24h_1000nM')]
    assert not failed.eligible.any()
    assert failed.exclusion_reason.str.contains('low_control_cells').all()
    # There is a different same-plate vehicle with 40 cells; it must not rescue strict QC.
    assert qc.loc[qc.Combo.eq('C1MCF7_1000_rep1'), 'n_cells'].item() == 40
    assert result['compound_qualification'].query("context_id == 'MCF7_24h_100nM' and compound_id == 'd3'").eligible.item()


def test_low_non_strict_vehicle_does_not_change_the_strict_gate():
    wells, roles, qc = count_fixture()
    qc.loc[qc.Combo.eq('C1MCF7_1000_rep1'), 'n_cells'] = 1
    summary = qualify(wells, roles, qc)['context_summary'].set_index('context_id')
    assert summary.loc['MCF7_24h_1000nM', 'evaluation_gate_met']


def test_known_absent_well_is_zero_but_missing_qc_record_is_unavailable():
    wells, roles, qc = count_fixture()
    absent = 'D2MCF7_1000_rep2'
    qc.loc[qc.Combo.eq(absent), 'n_cells'] = 0
    qc.loc[qc.Combo.eq(absent), 'source_observed'] = False
    known = qualify(wells, roles, qc)['context_summary'].set_index('context_id')
    assert known.loc['MCF7_24h_1000nM', 'count_qc_done']
    incomplete = qualify(wells, roles, qc[qc.Combo.ne(absent)])
    summary = incomplete['context_summary'].set_index('context_id')
    assert not summary.loc['MCF7_24h_1000nM', 'count_qc_done']
    row = incomplete['compound_qualification'].query("context_id == 'MCF7_24h_1000nM' and compound_id == 'd3'").iloc[0]
    assert 'missing_treatment_count_qc' in row.exclusion_reason
    assert not row.eligible


def test_completely_unacquired_context_is_not_reported_as_failed_biology():
    wells, roles, qc = count_fixture()
    qc = qc[~qc.Combo.str.contains('K562')]
    result = qualify(wells, roles, qc)
    summary = result['context_summary']
    assert summary.loc[summary.cell_type.eq('K562'), 'status'].eq('COUNT_QC_UNAVAILABLE').all()
    assert not summary.loc[summary.cell_type.eq('K562'), 'evaluation_gate_met'].any()
    assert len(result['compound_qualification']) == 48


@pytest.mark.parametrize('change', ['duplicate_count_key', 'fractional_count', 'negative_count',
                                   'missing_vehicle_mapping', 'wrong_strict_control', 'duplicate_context_well'])
def test_invalid_count_or_control_mapping_is_rejected(change):
    wells, roles, qc = count_fixture()
    if change == 'duplicate_count_key':
        qc = pd.concat([qc, qc.iloc[[0]]], ignore_index=True)
    elif change == 'fractional_count':
        qc['n_cells'] = qc.n_cells.astype(float); qc.loc[0, 'n_cells'] = 30.5
    elif change == 'negative_count':
        qc.loc[0, 'n_cells'] = -1
    elif change == 'missing_vehicle_mapping':
        wells = wells[wells.Combo.ne('C2MCF7_1000_rep1')]
    elif change == 'wrong_strict_control':
        wells.loc[wells.context_id.eq('MCF7_24h_1000nM') & wells.row_kind.eq('drug'), 'control_Combo'] = 'C1MCF7_1000_rep1'
    elif change == 'duplicate_context_well':
        wells = pd.concat([wells, wells.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        qualify(wells, roles, qc)


def test_missing_treatment_replicate_is_excluded_without_dropping_source_label():
    wells, roles, qc = count_fixture()
    wells = wells[wells.Combo.ne('D2MCF7_1000_rep2')]
    result = qualify(wells, roles, qc)
    row = result['compound_qualification'].query("context_id == 'MCF7_24h_1000nM' and compound_id == 'd3'").iloc[0]
    assert not row.eligible
    assert 'missing_or_duplicate_replicate' in row.exclusion_reason
    assert len(result['compound_qualification']) == 48


def test_qualification_is_order_independent():
    wells, roles, qc = count_fixture()
    original = qualify(wells, roles, qc)
    reordered = qualify(wells.sample(frac=1, random_state=2), roles.sample(frac=1, random_state=3), qc.sample(frac=1, random_state=4))
    for name in original:
        pd.testing.assert_frame_equal(original[name], reordered[name])


@pytest.mark.parametrize('mutate_roles', [False, True])
def test_cli_requires_m0_lineage_and_only_count_summaries(tmp_path, mutate_roles):
    wells, roles, qc = count_fixture()
    inputs = tmp_path/'inputs';inputs.mkdir()
    sources = {'well_mapping': inputs/'wells.tsv', 'context_roles': inputs/'roles.tsv'}
    for name, frame in [('well_mapping',wells),('context_roles',roles)]:
        frame.to_csv(sources[name],sep='\t',index=False)
    qc.to_csv(inputs/'well_qc.tsv',sep='\t',index=False)
    root = Path(__file__).resolve().parents[1]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    metadata_receipt = dict(status='METADATA_ONLY_COMPLETE', count_qc_done=False,
        new_test_expression_read=False,models_trained=False,
        input_contract={'sha256':sha(root/'configs/v2_metadata_inputs.json')},
        code_hashes={'configs/protocol_v2.yaml':sha(root/'configs/protocol_v2.yaml')},
        output_hashes={'manifests/v2/well_mapping.tsv':sha(sources['well_mapping']),
                       'manifests/v2/context_roles.tsv':sha(sources['context_roles'])})
    (inputs/'metadata_audit.json').write_text(json.dumps(metadata_receipt))
    if mutate_roles:
        roles.loc[0,'role'] = 'changed'
        roles.to_csv(sources['context_roles'],sep='\t',index=False)
    output = tmp_path/'public'
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(root/'src') + os.pathsep + environment.get('PYTHONPATH', '')
    result = subprocess.run([sys.executable,str(root/'scripts/qualify_contexts_v2.py'),
        '--well-mapping',str(sources['well_mapping']),'--context-roles',str(sources['context_roles']),
        '--metadata-audit',str(inputs/'metadata_audit.json'),'--well-qc',str(inputs/'well_qc.tsv'),
        '--output-root',str(output)],capture_output=True,text=True,env=environment)
    if mutate_roles:
        assert result.returncode!=0
        assert 'M0 output hash mismatch' in result.stderr
        assert not (output/'receipts/v2_context_qc.json').exists()
    else:
        assert result.returncode==0,result.stderr
        receipt=json.loads((output/'receipts/v2_context_qc.json').read_text())
        assert receipt['expression_effects_read'] is False
        assert receipt['models_trained'] is False
        assert receipt['metadata_audit_sha256']==sha(inputs/'metadata_audit.json')
        summary=pd.read_csv(output/'reports/v2/context_count_qc.tsv',sep='\t')
        assert len(summary)==12 and summary.count_qc_done.all()
        assert not summary.evaluation_gate_met.any() # Real protocol requires 20 drugs/10 groups.
        for path in output.rglob('*'):
            if path.is_file(): assert str(tmp_path) not in path.read_text()
