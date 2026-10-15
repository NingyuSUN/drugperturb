"""Chemical integrity tests use explicit structures; no response labels or real training."""
import numpy as np
import pytest

from drugperturb.chemistry import (
    molecular_graph,
    morgan_fingerprint,
    split_by_structure,
    standardize_compounds,
    standardize_smiles,
)


@pytest.mark.parametrize('smiles', ['', 'not_a_smiles', 'C1CC', 'Cl.[*].N[C@@H]1C[C@H]1C1=CC=CC=C1 |r,$;relative stereochemistry_R0;;;;;;;;;;$,c:7,9,t:5|'])
def test_invalid_or_undefined_atoms_are_rejected(smiles):
    with pytest.raises(ValueError):
        standardize_smiles(smiles)


def test_sodium_valproate_keeps_original_but_matches_parent_acid():
    original = '[Na+].CCCC(CCC)C([O-])=O'
    salt = standardize_smiles(original)
    acid = standardize_smiles('CCCC(CCC)C(O)=O')
    assert salt['original_smiles'] == original
    assert salt['canonical_smiles'] == acid['canonical_smiles']
    assert salt['scaffold_group'] == acid['scaffold_group']
    assert salt['n_original_fragments'] == 2


@pytest.mark.parametrize('smiles', [
    'CC.CN',  # Same-size organic mixture cannot be resolved by arbitrary tie breaking.
    'CCCCCC.CCO',  # Different size is insufficient evidence of a single active ingredient.
    '[Na+].CCCC(CCC)C(O)=O.CCCC(CCC)C([O-])=O',  # S1703 Divalproex.
])
def test_unresolved_active_mixtures_are_rejected(smiles):
    with pytest.raises(ValueError, match='mixture'):
        standardize_smiles(smiles)


def test_known_organic_counterion_is_removed_without_discarding_active_structure():
    result = standardize_smiles('CCN(CC)CC.CS(O)(=O)=O')
    assert result['canonical_smiles'] == 'CCN(CC)CC'
    assert result['n_original_fragments'] == 2


def test_same_compound_id_cannot_silently_point_to_different_structures():
    with pytest.raises(ValueError, match='conflicting'):
        standardize_compounds([
            {'compound_id': 'a', 'smiles': 'CCO'},
            {'compound_id': 'a', 'smiles': 'CCN'},
        ])


def test_parent_aliases_and_replicate_rows_are_preserved():
    records = [
        {'compound_id': 'a', 'smiles': 'CCO', 'replicate': '1'},
        {'compound_id': 'a', 'smiles': 'OCC', 'replicate': '2'},
        {'compound_id': 'alias', 'smiles': 'CCO', 'replicate': '1'},
    ]
    result = standardize_compounds(records)
    assert len(result) == 3
    assert [r['replicate'] for r in result] == ['1', '2', '1']
    assert {r['canonical_smiles'] for r in result} == {'CCO'}
    assert 'canonical_smiles' not in records[0]


def test_ring_substituents_group_together_but_acyclics_do_not_share_empty_scaffold():
    result = standardize_compounds([
        {'compound_id': 'benzene', 'smiles': 'c1ccccc1'},
        {'compound_id': 'toluene', 'smiles': 'Cc1ccccc1'},
        {'compound_id': 'ethanol', 'smiles': 'CCO'},
        {'compound_id': 'ethylamine', 'smiles': 'CCN'},
    ])
    assert result[0]['scaffold_group'] == result[1]['scaffold_group']
    assert result[2]['scaffold_group'] != result[3]['scaffold_group']
    assert result[2]['scaffold_group'].startswith('acyclic:')


def _split_input(n=100):
    return standardize_compounds([
        {'compound_id': f'alkane_{i}', 'smiles': 'C' * i, 'response': float(i)}
        for i in range(1, n + 1)
    ] + [
        {'compound_id': 'benzene', 'smiles': 'c1ccccc1'},
        {'compound_id': 'toluene', 'smiles': 'Cc1ccccc1'},
        {'compound_id': 'ethane_alias', 'smiles': 'CC'},
    ])


def test_split_is_deterministic_response_blind_and_isolates_groups_and_duplicates():
    rows = _split_input()
    first, report = split_by_structure(rows)
    altered = [dict(r, response=-999) for r in reversed(rows)]
    second, second_report = split_by_structure(altered)
    assignments = lambda rs: {r['compound_id']: r['split'] for r in rs}
    assert assignments(first) == assignments(second)
    assert report == second_report
    for key in ('canonical_smiles', 'scaffold_group'):
        seen = {}
        for r in first:
            seen.setdefault(r[key], set()).add(r['split'])
        assert all(len(splits) == 1 for splits in seen.values())
    assert len(first) == len(rows)  # Rows never silently collapsed.
    assert report['test']['n_compounds'] >= 20
    assert report['test']['n_groups'] >= 10
    assert report['test_minimum_met'] is True
    assert report['train']['n_compounds'] >= 50
    assert sum(report[s]['n_compounds'] for s in ('train', 'validation', 'test')) == 102


def test_small_split_reports_infeasible_minimum_without_fabricating_samples():
    rows = standardize_compounds([
        {'compound_id': str(i), 'smiles': 'C' * i} for i in range(1, 9)
    ])
    split, report = split_by_structure(rows)
    assert len(split) == 8
    assert report['test_minimum_feasible'] is False
    assert report['test_minimum_met'] is False
    assert all(report[s]['n_groups'] > 0 for s in ('train', 'validation', 'test'))


def test_morgan_fingerprint_has_binary_float32_values_and_salt_equivalence():
    fp = morgan_fingerprint('CCO')
    assert fp.shape == (2048,)
    assert fp.dtype == np.float32
    assert set(np.unique(fp)).issubset({0., 1.})
    assert 0 < fp.sum() < 2048
    np.testing.assert_array_equal(fp, morgan_fingerprint('OCC'))
    np.testing.assert_array_equal(
        morgan_fingerprint('[Na+].CCCC(CCC)C([O-])=O'),
        morgan_fingerprint('CCCC(CCC)C(O)=O'),
    )
    assert not np.array_equal(fp, morgan_fingerprint('CCN'))


def test_graph_preserves_directed_bond_topology_and_atom_identity():
    graph = molecular_graph('CCO')
    assert graph['x'].shape == (3, 10)
    assert graph['x'].dtype == np.float32
    assert graph['edge_index'].dtype == np.int64
    assert graph['edge_index'].shape == (2, 4)
    assert graph['edge_attr'].shape == (4, 6)
    assert graph['edge_attr'].dtype == np.float32
    edges = {tuple(e) for e in graph['edge_index'].T.tolist()}
    assert edges == {(0, 1), (1, 0), (1, 2), (2, 1)}
    np.testing.assert_allclose(graph['x'][:, 0], [0.06, 0.06, 0.08])
    np.testing.assert_array_equal(graph['edge_attr'][:, 0], np.ones(4))


def test_graph_handles_single_atoms_and_aromatic_bonds():
    isolated = molecular_graph('C')
    assert isolated['edge_index'].shape == (2, 0)
    assert isolated['edge_attr'].shape == (0, 6)
    aromatic = molecular_graph('c1ccccc1')
    assert aromatic['x'].shape == (6, 10)
    assert aromatic['edge_index'].shape == (2, 12)
    assert np.all(aromatic['x'][:, 4:6] == 1)
    assert np.all(aromatic['edge_attr'][:, 3] == 1)


def test_acyclic_stereoisomers_with_identical_model_features_stay_in_one_group():
    left = standardize_smiles("C[C@H](O)C(=O)O")
    right = standardize_smiles("C[C@@H](O)C(=O)O")
    assert left["canonical_smiles"] != right["canonical_smiles"]
    np.testing.assert_array_equal(morgan_fingerprint(left["canonical_smiles"]), morgan_fingerprint(right["canonical_smiles"]))
    assert left["scaffold_group"] == right["scaffold_group"]
