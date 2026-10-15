"""Cached entrypoints must bind current inputs without fitting or loading truth in train."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1]
TRAINING_FILES = [
    'configs/protocol_v1.yaml',
    'manifests/split_manifest.tsv',
    'manifests/selected_genes.tsv',
    'data/processed/development.npz',
    'data/processed/target_transform.joblib',
    'data/processed/test_features.npz',
    'data/processed/graphs_train.json',
    'data/processed/graphs_validation.json',
    'data/processed/graphs_test.json',
]
TRUTH = 'data/processed/sealed_test_truth.npz'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, dict) else value)
    return digest(path)


@pytest.fixture(scope='module')
def entrypoints():
    modules = {}
    for name in ('train', 'score'):
        spec = importlib.util.spec_from_file_location(f'cached_{name}', SOURCE / f'scripts/{name}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


@pytest.fixture
def completed(tmp_path):
    """Hash-bound bytes suffice: a valid reuse must never deserialize or fit them."""
    root = tmp_path
    frozen = {name: put(root, name, f'frozen {name}') for name in TRAINING_FILES}
    frozen['configs/protocol_v1.yaml'] = put(
        root, 'configs/protocol_v1.yaml', 'status: FROZEN_BEFORE_TRAINING\nmodels: {}\n')
    frozen[TRUTH] = put(root, TRUTH, 'sealed test truth')
    freeze = put(root, 'receipts/dataset_freeze.json', {'artifact_hashes': frozen})
    artifacts = {
        name: put(root, name, f'sealed {name}') for name in (
            'models/run_v1/M1_c0.joblib',
            'models/run_v1/test_predictions.npz',
            'models/run_v1/validation_predictions.npz',
            'receipts/model_selection.json',
            'receipts/training_runs.tsv',
        )
    }
    artifacts['configs/protocol_v1.yaml'] = frozen['configs/protocol_v1.yaml']
    artifacts['manifests/selected_genes.tsv'] = frozen['manifests/selected_genes.tsv']
    seal = put(root, 'receipts/prediction_seal.json', {
        'dataset_freeze_sha256': freeze, 'artifact_hashes': artifacts,
        'source_commit': 'synthetic-fixture', 'test_outcomes_loaded': False,
    })
    output = put(root, 'reports/final_test/summary.json', {'synthetic': True})
    put(root, 'receipts/final_evaluation.json', {
        'prediction_seal_sha256': seal, 'test_truth_sha256': frozen[TRUTH],
        'output_hashes': {'reports/final_test/summary.json': output},
    })
    return root


def invoke(entrypoints, name, root, monkeypatch):
    module = entrypoints[name]
    monkeypatch.setattr(module, 'ROOT', root)
    monkeypatch.setattr('sys.argv', [name, '--source-commit', 'synthetic-fixture'] if name == 'train' else [name])
    # Cache verification can hash files, but must not load arrays/checkpoints or fit.
    def forbidden(*args, **kwargs):
        pytest.fail('Cached entrypoint attempted array/model loading or fitting')
    monkeypatch.setattr(module.np, 'load', forbidden)
    if name == 'train':
        monkeypatch.setattr(module.joblib, 'load', forbidden)
        monkeypatch.setattr(module, 'fit_network', forbidden)
    module.main()


@pytest.mark.parametrize('name', ['train', 'score'])
def test_valid_completed_result_is_reused_without_loading_or_fitting(completed, entrypoints, monkeypatch, capsys, name):
    before = {p.relative_to(completed): p.read_bytes() for p in completed.rglob('*') if p.is_file()}
    invoke(entrypoints, name, completed, monkeypatch)
    assert 'verified' in capsys.readouterr().out
    assert before == {p.relative_to(completed): p.read_bytes() for p in completed.rglob('*') if p.is_file()}


@pytest.mark.parametrize('name', ['train', 'score'])
@pytest.mark.parametrize('relative', TRAINING_FILES)
@pytest.mark.parametrize('reuse', [True, False], ids=['cached', 'fresh'])
def test_fresh_or_cached_run_rejects_changed_frozen_training_input(completed, entrypoints, monkeypatch, name, relative, reuse):
    if not reuse:
        (completed / ('receipts/prediction_seal.json' if name == 'train' else 'receipts/final_evaluation.json')).unlink()
    (completed / relative).write_text('changed upstream bytes')
    with pytest.raises(ValueError, match='changed|mismatch|binding'):
        invoke(entrypoints, name, completed, monkeypatch)


@pytest.mark.parametrize('name', ['train', 'score'])
def test_reuse_rejects_replaced_dataset_freeze_receipt(completed, entrypoints, monkeypatch, name):
    path = completed / 'receipts/dataset_freeze.json'
    value = json.loads(path.read_text())
    value['replacement'] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='binding|mismatch'):
        invoke(entrypoints, name, completed, monkeypatch)


@pytest.mark.parametrize('name', ['train', 'score'])
@pytest.mark.parametrize('relative', ['models/run_v1/M1_c0.joblib', 'models/run_v1/test_predictions.npz'])
def test_reuse_rejects_changed_sealed_model_or_prediction(completed, entrypoints, monkeypatch, name, relative):
    (completed / relative).write_text('changed sealed bytes')
    with pytest.raises(ValueError, match='changed|mismatch'):
        invoke(entrypoints, name, completed, monkeypatch)


@pytest.mark.parametrize('relative', [TRUTH, 'receipts/prediction_seal.json', 'reports/final_test/summary.json'])
def test_score_reuse_rejects_changed_truth_seal_or_output(completed, entrypoints, monkeypatch, relative):
    path = completed / relative
    if path.suffix == '.json':
        value = json.loads(path.read_text())
        value['replacement'] = True
        path.write_text(json.dumps(value))
    else:
        path.write_text('changed truth')
    with pytest.raises(ValueError, match='changed|mismatch|binding'):
        invoke(entrypoints, 'score', completed, monkeypatch)


def test_train_reuse_does_not_read_final_test_truth(completed, entrypoints, monkeypatch):
    (completed / TRUTH).unlink()
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path == completed / TRUTH:
            pytest.fail('Training must not read or hash final-test truth')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    invoke(entrypoints, 'train', completed, monkeypatch)


def test_train_rejects_truth_in_seal_without_reading_it(completed, entrypoints, monkeypatch):
    seal = completed / 'receipts/prediction_seal.json'
    value = json.loads(seal.read_text())
    value['artifact_hashes'][TRUTH] = digest(completed / TRUTH)
    seal.write_text(json.dumps(value))
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path == completed / TRUTH:
            pytest.fail('Training must reject a truth entry before reading it')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    with pytest.raises(ValueError, match='final-test truth'):
        invoke(entrypoints, 'train', completed, monkeypatch)


@pytest.mark.parametrize('name', ['train', 'score'])
def test_empty_completion_artifact_map_is_rejected(completed, entrypoints, monkeypatch, name):
    relative, key = ('receipts/prediction_seal.json', 'artifact_hashes') if name == 'train' else ('receipts/final_evaluation.json', 'output_hashes')
    path = completed / relative
    value = json.loads(path.read_text())
    value[key] = {}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='incomplete|empty'):
        invoke(entrypoints, name, completed, monkeypatch)
