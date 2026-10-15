"""Verify completed result chains before presenting scientific output."""
from pathlib import Path
import hashlib,json

TRAINING_INPUTS = (
    'configs/protocol_v1.yaml', 'manifests/split_manifest.tsv',
    'manifests/selected_genes.tsv', 'data/processed/development.npz',
    'data/processed/target_transform.joblib', 'data/processed/test_features.npz',
    'data/processed/graphs_train.json', 'data/processed/graphs_validation.json',
    'data/processed/graphs_test.json',
)
TEST_TRUTH = 'data/processed/sealed_test_truth.npz'


def _project_file(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root):
        raise ValueError('Unsafe receipt path')
    return path


def _digest(root, relative):
    with _project_file(root, relative).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _load_receipt(root, relative):
    return json.loads(_project_file(root, relative).read_text())


def _verify_hashes(root, values, *, required=(), forbid_truth=False):
    if not isinstance(values, dict) or not values or not set(required).issubset(values):
        raise ValueError('Receipt artifact list is incomplete')
    if forbid_truth:
        truth = _project_file(root, TEST_TRUTH).resolve()
        if any(_project_file(root, name).resolve() == truth for name in values):
            raise ValueError('Training receipt must not include final-test truth')
    for relative, expected in values.items():
        if _digest(root, relative) != expected:
            raise ValueError(f'Frozen or sealed artifact changed: {relative}')


def verify_training_inputs(root, *, completed=False):
    """Check training dependencies and optional saved outputs without reading test truth.

    The freeze also names final-test outcomes. Only the training dependency
    allowlist is read here; test features contain structures, not outcomes.
    """
    root = Path(root).resolve()
    freeze = _load_receipt(root, 'receipts/dataset_freeze.json')
    values = freeze['artifact_hashes']
    if not set(TRAINING_INPUTS).issubset(values):
        raise ValueError('Frozen training artifact list is incomplete')
    _verify_hashes(root, {name: values[name] for name in TRAINING_INPUTS}, forbid_truth=True)
    if completed:
        seal = _load_receipt(root, 'receipts/prediction_seal.json')
        if _digest(root, 'receipts/dataset_freeze.json') != seal['dataset_freeze_sha256']:
            raise ValueError('Dataset freeze binding failed')
        _verify_hashes(root, seal['artifact_hashes'], forbid_truth=True)
    return freeze


def verify_evaluation_inputs(root, *, completed=False):
    """Bind current frozen inputs, predictions and truth before fresh or cached scoring."""
    root = Path(root).resolve()
    freeze = _load_receipt(root, 'receipts/dataset_freeze.json')
    seal = _load_receipt(root, 'receipts/prediction_seal.json')
    if _digest(root, 'receipts/dataset_freeze.json') != seal['dataset_freeze_sha256']:
        raise ValueError('Dataset freeze binding failed')
    if completed:
        final = _load_receipt(root, 'receipts/final_evaluation.json')
        if _digest(root, 'receipts/prediction_seal.json') != final['prediction_seal_sha256']:
            raise ValueError('Final evaluation binding failed')
        if _digest(root, TEST_TRUTH) != final['test_truth_sha256']:
            raise ValueError('Test truth binding failed')
    _verify_hashes(root, freeze['artifact_hashes'], required=(*TRAINING_INPUTS, TEST_TRUTH))
    _verify_hashes(root, seal['artifact_hashes'])
    if completed:
        _verify_hashes(root, final['output_hashes'])
    return seal, freeze


def verify_report_inputs(root):
    """Verify primary and secondary completed receipts; return checked hashes."""
    root=Path(root).resolve();checked={}
    def path(relative):
        p=root/relative
        if Path(relative).is_absolute() or not p.resolve().is_relative_to(root):raise ValueError('Unsafe receipt path')
        return p
    def digest(relative):
        with path(relative).open('rb') as f:value=hashlib.file_digest(f,'sha256').hexdigest()
        checked[relative]=value;return value
    def load(relative):
        digest(relative);return json.loads(path(relative).read_text())
    def verify_map(values):
        if not values:raise ValueError('Receipt artifact list is incomplete')
        for relative,expected in values.items():
            if digest(relative)!=expected:raise ValueError(f'Report input changed: {relative}')
    freeze=load('receipts/dataset_freeze.json');seal=load('receipts/prediction_seal.json');final=load('receipts/final_evaluation.json')
    if digest('receipts/dataset_freeze.json')!=seal['dataset_freeze_sha256']:raise ValueError('Dataset freeze binding failed')
    if digest('receipts/prediction_seal.json')!=final['prediction_seal_sha256']:raise ValueError('Final evaluation binding failed')
    if digest('data/processed/sealed_test_truth.npz')!=final['test_truth_sha256']:raise ValueError('Test truth binding failed')
    for values in [freeze['artifact_hashes'],seal['artifact_hashes'],final['output_hashes']]:verify_map(values)
    diagnostic=load('receipts/development_diagnostics.json')
    if diagnostic.get('status')!='COMPLETE_EXPLORATORY_DEVELOPMENT_ONLY':raise ValueError('Development diagnostics incomplete')
    verify_map(diagnostic['input_hashes']);verify_map(diagnostic['artifact_hashes'])
    sensitivity=load('receipts/sensitivity.json')
    if sensitivity.get('status')!='COMPLETE_FROZEN_PREDICTION_SENSITIVITY':raise ValueError('Sensitivity analysis incomplete')
    if sensitivity['prediction_seal_sha256']!=digest('receipts/prediction_seal.json') or sensitivity['final_evaluation_sha256']!=digest('receipts/final_evaluation.json'):raise ValueError('Sensitivity result binding failed')
    verify_map(sensitivity['input_hashes']);verify_map(sensitivity['output_hashes'])
    return checked
