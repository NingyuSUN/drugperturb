"""Metadata-only tests: identity, role isolation, controls and public receipts."""
from importlib import import_module
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from drugperturb.chemistry import standardize_smiles


def fixtures():
    chemistry = []
    for compound, smiles in [('d1', 'CCO'), ('d2', 'CCC'), ('d3', 'CCN')]:
        chemistry.append(dict(compound_id=compound,
                              status='accepted', exclusion_reason='', **standardize_smiles(smiles)))
    chemistry.append(dict(compound_id='bad', original_smiles='*', status='excluded',
                          exclusion_reason='undefined_atom', canonical_smiles='', scaffold_group=''))
    chemistry = pd.DataFrame(chemistry).fillna('')
    split = chemistry.iloc[:3].copy()
    split['split'] = ['train', 'validation', 'test']
    rows = []
    for cell in ['A549', 'MCF7', 'K562']:
        for dose in [10, 100, 1000, 10000]:
            for rep in ['rep1', 'rep2']:
                plate = f'{cell}_{dose}_{rep}'
                for i, item in enumerate(chemistry.to_dict('records')):
                    well = f'D{i}'
                    rows.append(dict(well_oligo=well, plate_oligo=plate, Combo=well+plate,
                                     cell_type=cell, time_point=24, replicate=rep,
                                     treatment=item['compound_id'], vehicle=False, dose=dose,
                                     dose_pattern='2', SMILES=item['original_smiles']))
                for pattern in ['1', '2']:
                    well = 'C'+pattern
                    rows.append(dict(well_oligo=well, plate_oligo=plate, Combo=well+plate,
                                     cell_type=cell, time_point=24, replicate=rep,
                                     treatment='vehicle', vehicle=True, dose=0,
                                     dose_pattern=pattern, SMILES=''))
    return pd.DataFrame(rows), chemistry, split


def audit(metadata, chemistry, split):
    return import_module('drugperturb.context_design').audit_contexts(metadata, chemistry, split)


def write_contract(inputs, protocol, path):
    sources = {name: inputs/(name+'.tsv'+('.gz' if name == 'metadata' else ''))
               for name in ['metadata', 'chemistry', 'split']}
    sources['protocol'] = protocol
    path.write_text(json.dumps({'schema_version': 1, 'expected_inputs': {
        name: {'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'bytes': source.stat().st_size}
        for name, source in sources.items()}}))


def test_contexts_keep_compound_units_separate_and_roles_locked():
    tables = audit(*fixtures())
    roles = tables['context_roles']
    assert len(roles) == 48  # Four source labels in every one of twelve contexts.
    def role(cell, dose, compound):
        return roles.loc[roles.cell_type.eq(cell) & roles.dose_nM.eq(dose)
                         & roles.compound_id.eq(compound), 'role'].item()
    assert role('A549', 1000, 'd1') == 'source_train'
    assert role('A549', 1000, 'd2') == 'source_validation'
    assert role('A549', 1000, 'd3') == 'source_test_locked'
    assert role('MCF7', 1000, 'd3') == 'primary_test'
    assert role('MCF7', 100, 'd3') == 'secondary_test'
    assert role('K562', 1000, 'd3') == 'secondary_test'
    assert role('K562', 100, 'd3') == 'metadata_only'
    assert role('MCF7', 1000, 'd1') == 'metadata_only'
    assert roles.loc[roles.compound_id.eq('bad'), 'exclusion_reason'].eq('undefined_atom').all()
    assert not roles.count_qc_done.any()


def test_vehicle_rows_retain_zero_dose_and_both_same_plate_patterns():
    tables = audit(*fixtures())
    wells = tables['well_mapping']
    context = wells[wells.context_id.eq('MCF7_24h_1000nM')]
    drugs = context[context.row_kind.eq('drug')]
    vehicles = context[context.row_kind.eq('vehicle')]
    assert len(drugs) == 8 and len(vehicles) == 4
    assert vehicles.dose.eq(0).all()
    assert vehicles.is_strict_control.sum() == 2
    assert set(drugs.control_Combo) == set(vehicles.loc[vehicles.is_strict_control, 'Combo'])
    assert not wells.duplicated(['context_id', 'Combo']).any()


@pytest.mark.parametrize('change,match', [
    ('duplicate_combo', 'Combo'), ('missing_control', 'control'),
    ('ambiguous_control', 'control'), ('invalid_vehicle', 'vehicle'),
    ('invalid_dose', 'dose'), ('negative_dose', 'dose'),
    ('missing_replicate', 'replicate'), ('missing_context', 'scope|context|drugs'),
    ('source_identity', 'SMILES|identity'), ('missing_identity', 'identity|chemistry'),
    ('empty_identity', 'identity|canonical'),
])
def test_metadata_defects_fail_instead_of_reducing_cohort(change, match):
    d, chemistry, split = fixtures()
    target = d.cell_type.eq('MCF7') & d.dose.eq(1000) & d.treatment.eq('d3')
    if change == 'duplicate_combo':
        d = pd.concat([d, d.iloc[[0]]], ignore_index=True)
    elif change == 'missing_control':
        d = d[~(d.Combo.eq('C2MCF7_1000_rep1'))]
    elif change == 'ambiguous_control':
        row = d[d.Combo.eq('C2MCF7_1000_rep1')].copy()
        row['well_oligo'] = 'C3'; row['Combo'] = 'C3MCF7_1000_rep1'
        d = pd.concat([d, row], ignore_index=True)
    elif change == 'invalid_vehicle':
        d['vehicle'] = d.vehicle.astype(str); d.loc[0, 'vehicle'] = 'perhaps'
    elif change == 'invalid_dose':
        d['dose'] = d.dose.astype(str); d.loc[0, 'dose'] = 'one'
    elif change == 'negative_dose':
        d.loc[0, 'dose'] = -10
    elif change == 'missing_replicate':
        d = d[~(target & d.replicate.eq('rep2'))]
    elif change == 'missing_context':
        d = d[~(d.cell_type.eq('MCF7') & d.dose.eq(1000))]
    elif change == 'source_identity':
        d.loc[target, 'SMILES'] = 'CCCC'
    elif change == 'missing_identity':
        chemistry = chemistry[chemistry.compound_id.ne('d3')]
    elif change == 'empty_identity':
        chemistry.loc[chemistry.compound_id.eq('d3'), 'canonical_smiles'] = ''
    with pytest.raises(ValueError, match=match):
        audit(d, chemistry, split)


@pytest.mark.parametrize('field', ['canonical_smiles', 'scaffold_group', 'original_smiles'])
def test_split_identity_mutation_is_rejected(field):
    d, chemistry, split = fixtures()
    split.loc[0, field] = 'invented'
    with pytest.raises(ValueError, match='identity|split'):
        audit(d, chemistry, split)


def test_scaffold_leakage_is_rejected_even_when_split_matches_chemistry():
    d, chemistry, split = fixtures()
    # Distinct ring substitutions share a real Murcko scaffold across two splits.
    for compound, smiles in [('d1', 'Cc1ccccc1'), ('d3', 'Oc1ccccc1')]:
        identities = standardize_smiles(smiles)
        for frame in [chemistry, split]:
            at = frame.compound_id.eq(compound)
            for key in ['canonical_smiles', 'scaffold_group']:
                frame.loc[at, key] = identities[key]
            frame.loc[at, 'original_smiles'] = smiles
        d.loc[d.treatment.eq(compound), 'SMILES'] = smiles
    with pytest.raises(ValueError, match='scaffold|split'):
        audit(d, chemistry, split)


def test_unlisted_accepted_parent_and_source_exclusions_stay_visible():
    d, chemistry, split = fixtures()
    split = split[split.compound_id.ne('d2')]
    tables = audit(d, chemistry, split)
    rows = tables['context_roles']
    assert len(rows) == 48
    assert rows.loc[rows.compound_id.eq('d2'), 'exclusion_reason'].eq('not_in_frozen_v1_cohort').all()
    assert rows.loc[rows.compound_id.eq('d2'), 'role'].eq('excluded').all()


def test_duplicate_parent_representative_is_not_silently_added_to_evaluation():
    d, chemistry, split = fixtures()
    alias = chemistry[chemistry.compound_id.eq('d1')].copy()
    alias['compound_id'] = 'alias'; alias['status'] = 'excluded'
    alias['exclusion_reason'] = 'duplicate_parent_represented_by:d1'
    chemistry = pd.concat([chemistry, alias], ignore_index=True)
    new = d[d.treatment.eq('d1')].copy()
    new['treatment'] = 'alias'; new['well_oligo'] = 'D4'; new['Combo'] = new.well_oligo+new.plate_oligo
    d = pd.concat([d, new], ignore_index=True)
    rows = audit(d, chemistry, split)['context_roles']
    assert rows.loc[rows.compound_id.eq('alias'), 'role'].eq('excluded').all()
    assert rows.loc[rows.compound_id.eq('alias'), 'representative_compound_id'].eq('d1').all()


def test_audit_is_order_independent():
    d, chemistry, split = fixtures()
    original = audit(d, chemistry, split)
    shuffled = audit(d.sample(frac=1, random_state=11), chemistry.sample(frac=1, random_state=2), split.sample(frac=1, random_state=3))
    for name in original:
        pd.testing.assert_frame_equal(original[name], shuffled[name])


def test_fixed_roles_agree_with_the_current_scientific_protocol():
    """Catch a role change or protocol edit that silently changes the test question."""
    root = Path(__file__).resolve().parents[1]
    protocol = yaml.safe_load((root/'configs/protocol_v2.yaml').read_text())
    roles = audit(*fixtures())['context_roles']
    def contexts(role):
        rows = roles[roles.role.eq(role)]
        return set(zip(rows.cell_type, rows.time_hours, rows.dose_nM))
    def key(context):
        return context['cell_type'], context['time_hours'], context['dose_nM']
    scope = protocol['scope']
    assert contexts('primary_test') == {key(scope['primary_target_context'])}
    assert contexts('secondary_test') == {key(row) for row in scope['secondary_target_contexts']}
    for role in ['source_train', 'source_validation', 'source_test_locked']:
        assert contexts(role) == {key(scope['source_context'])}
    assert set(roles.loc[roles.role.isin(['primary_test', 'secondary_test']), 'split']) == {'test'}


def test_cli_only_reads_declared_metadata_inputs_and_hides_absolute_paths(tmp_path):
    d, chemistry, split = fixtures()
    inputs = tmp_path/'private-inputs'; inputs.mkdir()
    d.to_csv(inputs/'metadata.tsv.gz', sep='\t', index=False)
    chemistry.to_csv(inputs/'chemistry.tsv', sep='\t', index=False)
    split.to_csv(inputs/'split.tsv', sep='\t', index=False)
    # No expression files exist in the fixture, so any implicit access fails.
    output = tmp_path/'public'; output.mkdir()
    script = Path(__file__).resolve().parents[1]/'scripts/audit_v2_metadata.py'
    contract = tmp_path/'fixture_contract.json'
    write_contract(inputs, script.parents[1]/'configs/protocol_v2.yaml', contract)
    result = subprocess.run([sys.executable, str(script), '--metadata', str(inputs/'metadata.tsv.gz'),
                             '--split', str(inputs/'split.tsv'), '--chemistry', str(inputs/'chemistry.tsv'),
                             '--output-root', str(output), '--input-contract', str(contract)],
                            text=True, capture_output=True, env=os.environ.copy())
    assert result.returncode == 0, result.stderr
    receipt = json.loads((output/'receipts/v2_metadata_audit.json').read_text())
    assert receipt['new_test_expression_read'] is False
    assert receipt['count_qc_done'] is False
    assert receipt['models_trained'] is False
    assert receipt['code_revision_is_base'] is True
    assert receipt['code_files_attested_by_hash'] is True
    protocol = script.parents[1]/'configs/protocol_v2.yaml'
    assert receipt['code_hashes']['configs/protocol_v2.yaml'] == hashlib.sha256(protocol.read_bytes()).hexdigest()
    assert receipt['input_contract']['sha256'] == hashlib.sha256(contract.read_bytes()).hexdigest()
    assert receipt['contexts'] == 12
    assert receipt['source_inputs']['metadata']['basename'] == 'metadata.tsv.gz'
    for path in output.rglob('*'):
        if path.is_file():
            assert str(tmp_path) not in path.read_text()
    assert len(pd.read_csv(output/'manifests/v2/context_roles.tsv', sep='\t')) == 48


def test_cli_rejects_structurally_valid_split_swap_against_frozen_input_hash(tmp_path):
    d, chemistry, split = fixtures()
    inputs = tmp_path/'inputs'; inputs.mkdir()
    d.to_csv(inputs/'metadata.tsv.gz', sep='\t', index=False)
    chemistry.to_csv(inputs/'chemistry.tsv', sep='\t', index=False)
    split.to_csv(inputs/'split.tsv', sep='\t', index=False)
    root = Path(__file__).resolve().parents[1]
    contract = tmp_path/'frozen.json'
    write_contract(inputs, root/'configs/protocol_v2.yaml', contract)
    swapped = split.copy()
    swapped['split'] = swapped.split.replace({'train': 'test', 'test': 'train'})
    # This is a legal grouped split with unchanged counts, but not the frozen question.
    assert set(audit(d, chemistry, swapped)['context_roles'].query("role == 'primary_test'").compound_id) == {'d1'}
    swapped.to_csv(inputs/'split.tsv', sep='\t', index=False)
    output = tmp_path/'outputs'
    result = subprocess.run([sys.executable, str(root/'scripts/audit_v2_metadata.py'),
                             '--metadata', str(inputs/'metadata.tsv.gz'), '--split', str(inputs/'split.tsv'),
                             '--chemistry', str(inputs/'chemistry.tsv'), '--input-contract', str(contract),
                             '--output-root', str(output)], capture_output=True, text=True, env=os.environ.copy())
    assert result.returncode != 0
    assert 'Frozen input hash/size mismatch: split' in result.stderr
    assert not (output/'receipts/v2_metadata_audit.json').exists()
