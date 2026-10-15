"""Verify completed result chains before presenting scientific output."""
from pathlib import Path
import hashlib,json

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
