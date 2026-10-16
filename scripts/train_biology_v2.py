"""Fit four direct five-set Ridge candidates after training-only coverage QC."""
from pathlib import Path
import argparse
import datetime
import json
import os
import platform
import subprocess
import tempfile

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml

from drugperturb.biology_v2 import ALPHAS, develop_head, file_record, verify_seal

ROOT = Path(__file__).resolve().parents[1]
SOURCE_KEYS = {
    'counts': 'data/processed/pseudobulk_counts.tsv.gz',
    'mapping': 'manifests/qualified_well_mapping.tsv',
    'gene_metadata': 'data/processed/gene_metadata.tsv',
    'split': 'manifests/split_manifest.tsv',
    'biology': 'configs/biology_v1.json',
}
PUBLIC = {
    'members': 'manifests/v2/biology_set_members.tsv',
    'coverage': 'reports/v2/biology_coverage.tsv',
    'coverage_receipt': 'receipts/v2_biology_coverage.json',
    'selection': 'receipts/v2_biology_selection.json',
}
SEAL = 'receipts/v2_biology_prediction_seal.json'


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [*SOURCE_KEYS, 'protocol', 'input_contract', 'source_receipt']:
        parser.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--input-contract-sha256', required=True)
    parser.add_argument('--source-receipt-sha256', required=True)
    parser.add_argument('--source-commit', help='If supplied, must equal actual current Git HEAD')
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(argv)
    sources = {k: getattr(args, k) for k in [*SOURCE_KEYS, 'protocol', 'input_contract', 'source_receipt']}
    # Validate pre-existing trusted contracts before parsing any count values.
    for name in ['input_contract', 'source_receipt']:
        if file_record(sources[name])['sha256'] != getattr(args, name+'_sha256'):
            raise ValueError(f'{name} trusted hash mismatch')
    contract = json.loads(args.input_contract.read_text())
    old_receipt = json.loads(args.source_receipt.read_text())
    if contract.get('schema_version') != 1:
        raise ValueError('unsupported frozen input contract')
    records = {k: file_record(p) for k, p in sources.items()}
    for key, relative in SOURCE_KEYS.items():
        if old_receipt.get('input_hashes', {}).get(relative) != records[key]['sha256']:
            raise ValueError(f'v1 source hash mismatch: {key}')
    for key in ['protocol', 'split']:
        expected = contract.get('expected_inputs', {}).get(key, {})
        if any(records[key][field] != expected.get(field) for field in ['bytes', 'sha256']):
            raise ValueError(f'frozen contract hash mismatch: {key}')
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    if args.source_commit is not None and args.source_commit != revision:
        raise ValueError('source-commit differs from actual Git HEAD')
    code_names = ['src/drugperturb/biology_v2.py', 'scripts/train_biology_v2.py',
                  'src/drugperturb/chemistry.py', 'src/drugperturb/response.py']
    for i, name in enumerate(code_names):
        sources[f'code_{i}'] = ROOT/name
        records[f'code_{i}'] = file_record(ROOT/name)
    seal_path = args.output_root/SEAL
    if seal_path.exists():
        seal = json.loads(seal_path.read_text())
        model_paths, public_paths = {}, {}
        for key in seal['artifact_hashes']:
            kind, relative = key.split(':', 1)
            if Path(relative).is_absolute() or '..' in Path(relative).parts:
                raise ValueError('unsafe sealed artifact reference')
            if kind == 'model' and Path(relative).name == relative:
                model_paths[key] = args.model_dir/relative
            elif kind == 'public' and relative in PUBLIC.values():
                public_paths[key] = args.output_root/relative
            else:
                raise ValueError('unsupported sealed artifact reference')
        verify_seal(seal, sources, model_paths, public_paths)
        print(json.dumps({'status': seal['status'], 'fit_count': seal['fit_count'],
                          'existing_seal_verified': True, 'fits_this_invocation': 0}))
        return seal
    if any((args.output_root/p).exists() for p in PUBLIC.values()) or (args.model_dir.exists() and any(args.model_dir.iterdir())):
        raise ValueError('refusing to overwrite partial or unsealed biology outputs')
    protocol = yaml.safe_load(args.protocol.read_text())
    biology = json.loads(args.biology.read_text())
    cfg = protocol['biology_head']
    if (protocol.get('schema_version') != 2
            or protocol.get('status') != 'PRESPECIFIED_BEFORE_NEW_CONTEXT_EFFECT_EVALUATION'
            or cfg['alpha_candidates'] != list(ALPHAS)
            or cfg['minimum_unique_genes_per_set'] != 10
            or cfg['minimum_fraction_original_source_members'] != .5
            or cfg['feature_scaling'] != 'none_binary_features'
            or cfg['label_scaling'] != 'none' or cfg['fit_intercept'] is not True
            or cfg['refit_train_plus_validation'] is not False
            or cfg['set_ids'] != [s['set_id'] for s in biology['sets']]):
        raise ValueError('protocol differs from the implemented frozen biological-head policy')
    if biology['gene_metadata']['sha256'] != records['gene_metadata']['sha256']:
        raise ValueError('biology gene metadata hash mismatch')
    split = pd.read_csv(args.split, sep='\t', keep_default_na=False)
    actual_counts = split.split.value_counts().to_dict()
    if actual_counts != protocol['split']['original_counts']:
        raise ValueError('frozen split compound counts differ')
    mapping = pd.read_csv(args.mapping, sep='\t', keep_default_na=False)
    source = protocol['scope']['source_context']
    relevant = mapping.loc[mapping.treatment.isin(split.loc[split.split.ne('test'), 'compound_id'])]
    if (not relevant.cell_type.eq(source['cell_type']).all()
            or not relevant.time_point.eq(source['time_hours']).all()
            or not relevant.dose.eq(source['dose_nM']).all()):
        raise ValueError('training/validation mapping has incorrect experimental context')
    genes = pd.read_csv(args.gene_metadata, sep='\t', usecols=['gene_id'], keep_default_na=False)
    if genes.gene_id.duplicated().any() or genes.gene_id.eq('').any():
        raise ValueError('gene metadata has duplicate or missing gene IDs')
    result = develop_head(args.counts, mapping, split, biology,
                          expected_gene_ids=genes.gene_id.tolist())
    # Hashing raw bytes does not parse target response columns. Detect mid-run drift.
    if records != {k: file_record(p) for k, p in sources.items()}:
        raise ValueError('input changed during biological-head development')
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.model_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.biology-public-', dir=args.output_root) as public_tmp, \
            tempfile.TemporaryDirectory(prefix='.biology-model-', dir=args.model_dir.parent) as model_tmp:
        stage, model_stage = Path(public_tmp), Path(model_tmp)
        for relative in PUBLIC.values():
            (stage/relative).parent.mkdir(parents=True, exist_ok=True)
        result['members'].to_csv(stage/PUBLIC['members'], sep='\t', index=False, lineterminator='\n')
        result['coverage'].to_csv(stage/PUBLIC['coverage'], sep='\t', index=False, lineterminator='\n')
        common = {'schema_version': 2, 'status': result['status'], 'fit_count': result['fit_count'],
                  'test_outcomes_loaded': False, 'new_context_outcomes_loaded': False,
                  'validation_counts_read': result['validation_counts_read'],
                  'transfer_A_continues': True, 'source_inputs': records,
                  'code_revision': revision, 'code_revision_is_base': True,
                  'code_files_attested_by_hash': True, 'prevalence': result['prevalence'],
                  'train_well_ids': result['train_well_ids'], 'train_control_ids': result['train_control_ids'],
                  'interpretation': 'Mean fixed-gene-set transcriptional change; not pathway activation or drug efficacy'}
        save_json(stage/PUBLIC['coverage_receipt'], {**common,
                  'coverage': result['coverage'].to_dict(orient='records')})
        selection = {'status': result['status'], 'fit_count': result['fit_count'],
                     'refit_train_plus_validation': False, 'test_outcomes_loaded': False}
        if result['fit_count']:
            selection.update(selected_candidate=result['selected_candidate'], selected_alpha=result['selected_alpha'],
                alpha_candidates=list(ALPHAS), validation_MAE=result['validation_MAE'],
                selection_metric='mean_compound_five_set_MAE', tie_policy='first_candidate_in_frozen_order',
                n_train=result['n_train'], n_validation=result['n_validation'], n_test_queries=result['n_test_queries'],
                train_ids=result['train_ids'], validation_ids=result['validation_ids'],
                test_ids=result['test_ids'], set_ids=result['set_ids'])
            for i, model in enumerate(result['models']):
                joblib.dump(model, model_stage/f'Ridge_c{i}.joblib')
            joblib.dump(result['models'][result['selected_candidate']], model_stage/'selected_model.joblib')
            for name, ids in [('validation', result['validation_ids']), ('test', result['test_ids'])]:
                np.savez_compressed(model_stage/f'{name}_predictions.npz', ids=np.asarray(ids),
                    set_ids=np.asarray(result['set_ids']), **result[f'{name}_predictions'])
        save_json(stage/PUBLIC['selection'], selection)
        public_paths = {'public:'+p: stage/p for p in PUBLIC.values()}
        model_paths = {'model:'+p.name: p for p in sorted(model_stage.iterdir())}
        seal = {**common, 'status': ('SEALED_BEFORE_TARGET_SCORING' if result['fit_count']
                                    else 'FAILED_COVERAGE_ZERO_FITS'),
                'sealed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__,
                'artifact_hashes': {k: file_record(p)['sha256'] for k, p in {**public_paths, **model_paths}.items()}}
        verify_seal(seal, sources, model_paths, public_paths)
        args.model_dir.mkdir(parents=True, exist_ok=True)
        for p in model_stage.iterdir():
            os.replace(p, args.model_dir/p.name)
        for relative in PUBLIC.values():
            target = args.output_root/relative; target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage/relative, target)
        # Write the seal last; partial outputs can never look like a completed run.
        seal_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(stage/'prediction_seal.json', seal)
        os.replace(stage/'prediction_seal.json', seal_path)
    print(json.dumps({'status': seal['status'], 'fit_count': seal['fit_count'],
                      'test_outcomes_loaded': False}))
    return seal


if __name__ == '__main__':
    main()
