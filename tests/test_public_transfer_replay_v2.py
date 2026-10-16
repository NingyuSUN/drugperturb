"""Public prediction replay fails closed without using a training archive."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def api():
    spec = importlib.util.spec_from_file_location('public_replay_test', ROOT/'scripts/replay_public_transfer_v2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)+'\n')


@pytest.fixture
def bundle(tmp_path):
    m = api(); root = tmp_path/'public'; root.mkdir()
    for relative in m.SCIENTIFIC_INPUTS.values():
        path = root/relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('anchor\n')
    ids = np.array([f'd{i:02}' for i in range(26)])
    genes = [f'g{i}' for i in range(2000)]; sets = [f's{i}' for i in range(5)]
    pd.DataFrame({'compound_id': ids, 'split': 'test'}).to_csv(root/m.SCIENTIFIC_INPUTS['split'], sep='\t', index=False)
    pd.DataFrame({'gene_id': genes, 'output_index': range(2000)}).to_csv(root/m.SCIENTIFIC_INPUTS['genes'], sep='\t', index=False)
    directory = root/'reports/v2/predictions'; directory.mkdir(parents=True)
    a = {k: np.zeros((26, 2000)) for k in m.A_KEYS-{'ids'}}; a['ids'] = ids
    b = {k: np.zeros((26, 5)) for k in m.B_METHODS}; b.update(ids=ids, set_ids=np.array(sets))
    np.savez(directory/'gene_predictions.npz', **a); np.savez(directory/'biology_predictions.npz', **b)
    def reseal():
        v1 = {'test_outcomes_loaded': False, 'artifact_hashes': {
            'models/run_v1/test_predictions.npz': m.digest(directory/'gene_predictions.npz'),
            'manifests/selected_genes.tsv': m.digest(root/m.SCIENTIFIC_INPUTS['genes'])}}
        biology = {'status': 'SEALED_BEFORE_TARGET_SCORING', 'fit_count': 4,
            'test_outcomes_loaded': False, 'new_context_outcomes_loaded': False,
            'artifact_hashes': {'model:test_predictions.npz': m.digest(directory/'biology_predictions.npz'),
                'public:manifests/v2/biology_set_members.tsv': m.digest(root/m.SCIENTIFIC_INPUTS['members'])}}
        write_json(root/'receipts/prediction_seal.json', v1)
        write_json(root/'receipts/v2_biology_prediction_seal.json', biology)
        manifest = {'schema_version': 1, 'status': 'PUBLISHED_SEALED_PREDICTIONS',
            'files': {p.name: {k: v for k, v in m.file_record(p).items() if k != 'basename'}
                      for p in [directory/'gene_predictions.npz', directory/'biology_predictions.npz']},
            'source_seals': {name: dict(relative_path=relative, **{k: v for k, v in m.file_record(root/relative).items() if k != 'basename'})
                             for name, relative in m.SOURCE_SEALS.items()},
            'source_prediction_hashes': {'v1': v1['artifact_hashes']['models/run_v1/test_predictions.npz'],
                                         'biology': biology['artifact_hashes']['model:test_predictions.npz']},
            'scientific_inputs': {name: dict(relative_path=relative, **{k: v for k, v in m.file_record(root/relative).items() if k != 'basename'})
                                  for name, relative in m.SCIENTIFIC_INPUTS.items()}}
        write_json(directory/'manifest.json', manifest)
        return manifest
    manifest = reseal()
    return m, root, directory, ids.tolist(), genes, sets, a, b, reseal, manifest


def load(bundle):
    m, root, directory, ids, genes, sets, *_ = bundle
    return m.load_public_predictions(root, directory/'manifest.json', m.digest(directory/'manifest.json'), ids, genes, sets)


def test_original_bytes_load_without_private_archive(bundle):
    m, root, _, ids, *_ = bundle
    assert not (root/'models').exists()
    a, b, paths = load(bundle)
    assert a['ids'].tolist() == ids and b['Ridge'].shape == (26, 5)
    assert paths['gene_predictions'].name == 'gene_predictions.npz'


@pytest.mark.parametrize('change', ['reorder', 'duplicate', 'missing', 'shape', 'nonfinite', 'seed_mean', 'set_order', 'extra_key'])
def test_prediction_content_is_checked_even_after_coherent_rehash(bundle, change):
    m, root, directory, ids, genes, sets, a, b, reseal, _ = bundle
    if change == 'reorder': a['ids'] = a['ids'][::-1]
    if change == 'duplicate': a['ids'][1] = a['ids'][0]
    if change == 'missing': a['ids'] = a['ids'][:-1]
    if change == 'shape': a['M1'] = np.zeros((26, 1999))
    if change == 'nonfinite': b['Ridge'][0, 0] = np.inf
    if change == 'seed_mean': a['M3_s42'][0, 0] = 3
    if change == 'set_order': b['set_ids'] = b['set_ids'][::-1]
    if change == 'extra_key': a['unsealed'] = np.zeros(1)
    np.savez(directory/'gene_predictions.npz', **a); np.savez(directory/'biology_predictions.npz', **b); reseal()
    with pytest.raises(ValueError): load(bundle)


def test_prediction_bytes_must_equal_original_seal_not_repacked_claim(bundle):
    m, root, directory, *rest = bundle
    manifest = json.loads((directory/'manifest.json').read_text())
    manifest['source_prediction_hashes']['v1'] = '0'*64
    write_json(directory/'manifest.json', manifest)
    with pytest.raises(ValueError, match='original|seal'): load(bundle)


@pytest.mark.parametrize('anchor', ['source_seals', 'scientific_inputs'])
def test_anchor_descriptor_cannot_redirect_to_identical_arbitrary_file(bundle, anchor):
    m, root, directory, *rest = bundle
    manifest = json.loads((directory/'manifest.json').read_text())
    key = next(iter(manifest[anchor])); entry = manifest[anchor][key]
    clone = root/'clone'; clone.write_bytes((root/entry['relative_path']).read_bytes())
    entry['relative_path'] = 'clone'; write_json(directory/'manifest.json', manifest)
    with pytest.raises(ValueError, match='anchor'): load(bundle)


def test_manifest_hash_checked_before_opening_npz(bundle):
    m, root, directory, *_ = bundle
    expected = m.digest(directory/'manifest.json')
    (directory/'manifest.json').write_text('{}')
    (directory/'gene_predictions.npz').unlink()
    with pytest.raises(ValueError, match='manifest.*hash'): m.load_public_predictions(root, directory/'manifest.json', expected, [], [], [])


def test_reference_receipt_and_all_six_output_bytes_are_pinned(tmp_path):
    m = api(); root = tmp_path/'repo'; reference = root/'receipts/v2_transfer_evaluation.json'
    outputs = {}
    for name in m.EVALUATOR.OUTPUTS:
        path = root/'reports/v2/transfer'/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name+'\n')
        outputs['reports/v2/transfer/'+name] = m.digest(path)
    write_json(reference, {'schema_version': 2, 'status': 'SEALED_TRANSFER_EVALUATION',
                          'new_models_fitted': 0, 'test_outcomes_used_for_selection': False,
                          'source_inputs': {}, 'target_sources': {}, 'output_hashes': outputs})
    m.PUBLISHED_REFERENCE_SHA256 = m.digest(reference)
    receipt, paths = m.check_published_reference(root)
    assert len(paths) == 7
    (root/'reports/v2/transfer/summary.json').write_text('changed\n')
    with pytest.raises(ValueError, match='published.*hash'): m.check_published_reference(root)
    write_json(reference, {**receipt, 'new_models_fitted': 1})
    with pytest.raises(ValueError, match='published.*hash'): m.check_published_reference(root)


def test_new_coherent_outputs_are_rejected_before_publication(tmp_path):
    m = api(); stage = tmp_path/'stage'; stage.mkdir()
    expected = {}
    for name in m.EVALUATOR.OUTPUTS:
        (stage/name).write_text('frozen\n'); expected['reports/v2/transfer/'+name] = m.digest(stage/name)
    (stage/'per_set.tsv').write_text('new coherent analysis\n')
    with pytest.raises(ValueError, match='published.*per_set'): m.compare_outputs(stage, expected)


def test_five_context_scoring_uses_original_science_without_training_archive(tmp_path):
    """Hand-computed all-gene truth checks the bridge, not a mocked scorer."""
    m = api()
    spec = importlib.util.spec_from_file_location('transfer_fixtures', ROOT/'tests/test_transfer_v2.py')
    fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)
    counts, wells, qualified, qc = fixtures.fixture()
    contexts = ['MCF7_24h_1000nM', 'K562_24h_1000nM', 'MCF7_24h_10nM',
                'MCF7_24h_100nM', 'MCF7_24h_10000nM']
    targets = [dict(cell_type=c.split('_')[0], time_hours=24, dose_nM=int(c.split('_')[2][:-2])) for c in contexts]
    protocol = {'scope': {'primary_target_context': targets[0], 'secondary_target_contexts': targets[1:]}}
    all_wells, all_qualified, compounds, summaries = [], [], [], []
    for i, context in enumerate(contexts):
        w = wells.copy(); q = qualified.copy()
        w['context_id'] = context; q['context_id'] = context
        w['cell_type'] = context.split('_')[0]
        all_wells.append(w); all_qualified.append(q)
        role = 'primary_test' if i == 0 else 'secondary_test'
        for cid in ['a', 'b']:
            compounds.append(dict(context_id=context, compound_id=cid, split='test',
                                  scaffold_group=cid, canonical_smiles=cid, role=role, eligible=True))
        summaries.append(dict(context_id=context, count_qc_done=True, eligible_test_compounds=2,
                              eligible_test_structure_groups=2, evaluation_gate_met=False))
    counts.index.name = 'gene_id'
    counts.to_csv(tmp_path/'counts.tsv', sep='\t')
    pd.DataFrame({'gene_id': counts.index}).to_csv(tmp_path/'genes.tsv', sep='\t', index=False)
    qc.to_csv(tmp_path/'qc.tsv', sep='\t', index=False)
    source = {'files': {'counts': tmp_path/'counts.tsv', 'gene_metadata': tmp_path/'genes.tsv',
                        'well_qc': tmp_path/'qc.tsv'}, 'summary': {'source_human_genes': '3'}}
    data = dict(compounds=pd.DataFrame(compounds), qualified=pd.concat(all_qualified),
                summary=pd.DataFrame(summaries), wells=pd.concat(all_wells))
    split = pd.DataFrame({'compound_id': ['a', 'b'], 'split': 'test', 'canonical_smiles': ['a', 'b'],
                          'scaffold_group': ['a', 'b']})
    sets = list('12345'); members = pd.DataFrame({'set_id': sets, 'gene_id': ['g1', 'g2', 'g1', 'g2', 'zero']})
    a = {'ids': np.array(['a', 'b']), **{k: np.zeros((2, 3)) for k in m.A_METHODS}}
    b = {'ids': a['ids'], **{k: np.zeros((2, 5)) for k in m.B_METHODS}}
    stage = tmp_path/'stage'; stage.mkdir()
    m.score_to_stage(stage, protocol, {'MCF7': source, 'K562': source}, data, split, a, b,
                     ['g1', 'g2', 'zero'], members, sets)
    status = pd.read_csv(stage/'context_status.tsv', sep='\t')
    assert status.context_id.tolist() == contexts
    assert not status.gate_met.any() and status.sensitivity_compounds.tolist() == [2]*5
    per = pd.read_csv(stage/'per_compound.tsv', sep='\t')
    first = per[(per.context_id == contexts[0]) & (per.compound_id == 'a')
                & (per.method == 'B0') & (per.analysis == 'A') & (per.comparison == 'strict_full')]
    truth = np.array([((np.log1p(900000)-np.log1p(500000))+(np.log1p(100000)-np.log1p(500000)))/2,
                      ((np.log1p(100000)-np.log1p(500000))+(np.log1p(900000)-np.log1p(500000)))/2, 0])
    assert first.error.iloc[0] == pytest.approx(np.sqrt(np.mean(truth**2)))
    assert set(p.name for p in stage.iterdir()) == set(m.EVALUATOR.OUTPUTS)
    contrasts = pd.read_csv(stage/'contrasts.tsv', sep='\t')
    assert contrasts.ci_lower.isna().all()  # Failed gates do not acquire CIs in replay.
    assert not (tmp_path/'models').exists()


def test_relocated_target_manifest_is_allowed_but_source_changes_are_not(tmp_path):
    m = api(); source = tmp_path/'counts'; source.write_text('same extracted counts\n')
    records = {'MCF7': {'files': {'counts': m.file_record(source)}}}
    reference = {'target_sources': records, 'source_inputs': {'MCF7_counts': m.file_record(source)}}
    moved = tmp_path/'relocated'/'counts'; moved.parent.mkdir(); moved.write_bytes(source.read_bytes())
    m.check_reference_inputs(reference, {'MCF7_counts': moved}, records)
    moved.write_text('different but coherent counts\n')
    with pytest.raises(ValueError, match='published M3 input'): m.check_reference_inputs(reference, {'MCF7_counts': moved}, records)


@pytest.mark.parametrize('existing', ['context_status.tsv', 'summary.json', 'receipt'])
def test_cli_no_overwrite_is_checked_before_scoring(tmp_path, existing):
    m = api(); paths = []
    for name in ['prediction', 'target', 'qc', 'contract']:
        path = tmp_path/name; path.write_text('{}\n'); paths.append(path)
    output = tmp_path/'out'
    destination = output/('receipts/v2_public_replay.json' if existing == 'receipt'
                         else 'reports/v2/transfer/'+existing)
    destination.parent.mkdir(parents=True); destination.write_text('preserve\n')
    args = ['--output-root', str(output)]
    for name, path in zip(['prediction-manifest', 'target-manifest', 'qc-receipt', 'input-contract'], paths):
        args += ['--'+name, str(path), '--'+name+'-sha256', m.digest(path)]
    with pytest.raises(ValueError, match='overwrite'): m.main(args)
    assert destination.read_text() == 'preserve\n'


def test_corrupted_seal_and_manifest_npz_hash_are_rejected(bundle):
    m, root, directory, *_ = bundle
    (root/'receipts/prediction_seal.json').write_text('{}\n')
    with pytest.raises(ValueError, match='anchor.*hash'): load(bundle)


def test_repacked_bytes_with_changed_manifest_but_original_seal_are_rejected(bundle):
    m, root, directory, ids, genes, sets, a, b, _, _ = bundle
    a['M1'][0, 0] = 1
    np.savez(directory/'gene_predictions.npz', **a)
    manifest = json.loads((directory/'manifest.json').read_text())
    manifest['files']['gene_predictions.npz'] = {k: v for k, v in m.file_record(directory/'gene_predictions.npz').items() if k != 'basename'}
    write_json(directory/'manifest.json', manifest)
    with pytest.raises(ValueError, match='original sealed archive'): load(bundle)
