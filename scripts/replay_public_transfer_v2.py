"""Replay published transfer scores from public sealed predictions, without fitting."""
from pathlib import Path
import argparse
import datetime
import platform
import importlib.util
import json
import os
import tempfile

import numpy as np
import pandas as pd
import yaml

from drugperturb.biology_v2 import file_record
from drugperturb.transfer_v2 import A_METHODS, B_METHODS, booleans, frozen_cohort

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('drugperturb_public_transfer_evaluator',
                                              ROOT/'scripts/evaluate_transfer_v2.py')
EVALUATOR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EVALUATOR)
digest = EVALUATOR.digest
# This is the published M3 result, not a user-replaceable coherent new analysis.
PUBLISHED_REFERENCE_SHA256 = '38d07b11f386ddced9b97541fb8fdabd366aacb0a049b31999be4b17150713f3'
SCIENTIFIC_INPUTS = {'protocol': 'configs/protocol_v2.yaml', 'split': 'manifests/split_manifest.tsv',
    'genes': 'manifests/selected_genes.tsv', 'members': 'manifests/v2/biology_set_members.tsv'}
SOURCE_SEALS = {'v1': 'receipts/prediction_seal.json',
                'biology': 'receipts/v2_biology_prediction_seal.json'}
SEEDS = (42, 123, 2026)
A_KEYS = {'ids', *A_METHODS, *(f'{m}_s{s}' for m in ['M2', 'M3'] for s in SEEDS)}


def check_record(path, expected, label):
    actual = file_record(path)
    if any(actual[k] != expected.get(k) for k in ['bytes', 'sha256']):
        raise ValueError(f'{label} hash/size mismatch: {path.name}')
    return path


def repo_anchor(root, entry, relative):
    if entry.get('relative_path') != relative:
        raise ValueError('prediction manifest repository anchor differs')
    return check_record(EVALUATOR.safe_child(root, relative), entry, 'repository anchor')


def load_public_predictions(root, manifest_path, expected_hash, test_ids, genes, set_ids):
    """Bind original NPZ bytes directly to public seals; validate numerical schema."""
    EVALUATOR.require_hash(manifest_path, expected_hash, 'prediction manifest')
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('schema_version') != 1 or manifest.get('status') != 'PUBLISHED_SEALED_PREDICTIONS'
            or set(manifest.get('files', {})) != {'gene_predictions.npz', 'biology_predictions.npz'}
            or set(manifest.get('source_seals', {})) != set(SOURCE_SEALS)
            or set(manifest.get('scientific_inputs', {})) != set(SCIENTIFIC_INPUTS)):
        raise ValueError('unsupported public prediction manifest')
    paths = {'prediction_manifest': manifest_path}
    for key, relative in SCIENTIFIC_INPUTS.items():
        paths[key] = repo_anchor(root, manifest['scientific_inputs'][key], relative)
    for key, relative in SOURCE_SEALS.items():
        paths[key+'_seal'] = repo_anchor(root, manifest['source_seals'][key], relative)
    seals = {key: json.loads(paths[key+'_seal'].read_text()) for key in SOURCE_SEALS}
    if (seals['v1'].get('test_outcomes_loaded') is not False
            or seals['biology'].get('status') != 'SEALED_BEFORE_TARGET_SCORING'
            or seals['biology'].get('fit_count') != 4
            or seals['biology'].get('test_outcomes_loaded') is not False
            or seals['biology'].get('new_context_outcomes_loaded') is not False):
        raise ValueError('prediction-only replay requires both pre-target prediction seals')
    expected_sources = {'v1': seals['v1'].get('artifact_hashes', {}).get('models/run_v1/test_predictions.npz'),
        'biology': seals['biology'].get('artifact_hashes', {}).get('model:test_predictions.npz')}
    if manifest.get('source_prediction_hashes') != expected_sources or not all(expected_sources.values()):
        raise ValueError('original prediction hashes differ from public seals')
    if digest(paths['genes']) != seals['v1']['artifact_hashes'].get('manifests/selected_genes.tsv'):
        raise ValueError('original selected-gene order differs from v1 seal')
    for key, expected in seals['biology']['artifact_hashes'].items():
        if key.startswith('public:'):
            relative = key.split(':', 1)[1]
            path = EVALUATOR.safe_child(root, relative)
            EVALUATOR.require_hash(path, expected, 'public biological prediction seal artifact')
            paths[key] = path
    if (seals['biology']['artifact_hashes'].get('public:'+SCIENTIFIC_INPUTS['members'])
            != digest(paths['members'])):
        raise ValueError('biological membership differs from prediction seal')
    arrays = []
    for name, kind, key in [('gene_predictions.npz', 'v1', 'gene_predictions'),
                             ('biology_predictions.npz', 'biology', 'biology_predictions')]:
        path = EVALUATOR.safe_child(manifest_path.parent, name)
        check_record(path, manifest['files'][name], 'public prediction')
        if digest(path) != expected_sources[kind]:
            raise ValueError('public prediction bytes differ from original sealed archive')
        paths[key] = path
        with np.load(path, allow_pickle=False) as source:
            array = {k: source[k].copy() for k in source.files}
        arrays.append(array)
    a, b = arrays
    if set(a) != A_KEYS or set(b) != {'ids', 'set_ids', *B_METHODS}:
        raise ValueError('public prediction archive keys differ from original schema')
    if len(test_ids) != 26 or len(set(test_ids)) != 26 or test_ids != sorted(test_ids):
        raise ValueError('public replay requires the ordered frozen 26 test compounds')
    if len(genes) != 2000 or len(set(genes)) != 2000 or len(set_ids) != 5 or len(set(set_ids)) != 5:
        raise ValueError('public replay requires 2000 ordered genes and five ordered sets')
    for array, width in [(a, 2000), (b, 5)]:
        if array['ids'].ndim != 1 or array['ids'].dtype.kind not in 'US' or array['ids'].tolist() != test_ids:
            raise ValueError('public prediction IDs/order differ from frozen test cohort')
        for key, value in array.items():
            if key in {'ids', 'set_ids'}:
                continue
            if (value.shape != (26, width) or value.dtype.kind not in 'fiu'
                    or not np.isfinite(value).all()):
                raise ValueError('public prediction shape/nonfinite values differ: '+key)
    if b['set_ids'].ndim != 1 or b['set_ids'].dtype.kind not in 'US' or b['set_ids'].tolist() != set_ids:
        raise ValueError('public prediction set order differs')
    for method in ['M2', 'M3']:
        if not np.array_equal(a[method], np.mean([a[f'{method}_s{s}'] for s in SEEDS], axis=0)):
            raise ValueError('original three-seed neural arithmetic-mean ensemble differs')
    return a, b, paths


def check_published_reference(root):
    path = root/'receipts/v2_transfer_evaluation.json'
    EVALUATOR.require_hash(path, PUBLISHED_REFERENCE_SHA256, 'published M3 receipt')
    receipt = json.loads(path.read_text())
    expected_outputs = {'reports/v2/transfer/'+name for name in EVALUATOR.OUTPUTS}
    if (receipt.get('schema_version') != 2 or receipt.get('status') != 'SEALED_TRANSFER_EVALUATION'
            or receipt.get('new_models_fitted') != 0
            or receipt.get('test_outcomes_used_for_selection') is not False
            or set(receipt.get('output_hashes', {})) != expected_outputs):
        raise ValueError('invalid published M3 reference')
    paths = {'published_receipt': path}
    for relative, expected in receipt['output_hashes'].items():
        source = EVALUATOR.safe_child(root, relative)
        EVALUATOR.require_hash(source, expected, 'published M3 output')
        paths['published:'+relative] = source
    return receipt, paths


def check_reference_inputs(reference, sources, target_records):
    """Allow relocated private manifests, requiring identical scientific input bytes."""
    if reference.get('target_sources') != target_records:
        raise ValueError('target input records differ from published M3')
    for key, path in sources.items():
        expected = reference.get('source_inputs', {}).get(key)
        if expected is None:
            raise ValueError('published M3 lacks input reference: '+key)
        check_record(path, expected, 'published M3 input '+key)


def compare_outputs(stage, expected):
    comparisons = {}
    for name in EVALUATOR.OUTPUTS:
        relative = 'reports/v2/transfer/'+name
        actual = digest(stage/name)
        if actual != expected[relative]:
            raise ValueError('replayed output differs from published M3: '+name)
        comparisons[relative] = {'sha256': actual, 'published_sha256': expected[relative], 'matches': True}
    return comparisons


def score_to_stage(stage, protocol, source_data, data, split, a, b, genes, members, set_ids):
    """Use unchanged scientific evaluators and the original six-file serialization."""
    targets = [(protocol['scope']['primary_target_context'], 'primary_test')]
    targets += [(c, 'secondary_test') for c in protocol['scope']['secondary_target_contexts']]
    expected = ['MCF7_24h_1000nM', 'K562_24h_1000nM', 'MCF7_24h_10nM',
                'MCF7_24h_100nM', 'MCF7_24h_10000nM']
    contexts = [f'{t["cell_type"]}_{t["time_hours"]}h_{t["dose_nM"]}nM' for t, _ in targets]
    if contexts != expected:
        raise ValueError('published five-context order differs')
    status, tables, summaries = [], [], {}
    for (target, role), context in zip(targets, contexts):
        cell = target['cell_type']
        cohort = frozen_cohort(data['compounds'], data['qualified'], data['summary'], split, context, role)
        record = {k: v for k, v in cohort.items() if k not in ['ids', 'groups']}
        record.update(context_id=context, context_role=role, sensitivity_compounds=0, sensitivity_groups=0)
        if cell not in source_data:
            record['status'] = 'SOURCE_UNAVAILABLE'
        if record['status'] in ['SOURCE_UNAVAILABLE', 'COUNT_QC_UNAVAILABLE'] or not cohort['ids']:
            status.append(record); summaries[context] = dict(record); continue
        result = EVALUATOR.score_context_source(context, role, source_data[cell], cohort, data, a, b, genes, members, set_ids)
        record.update({k: result[k] for k in ['sensitivity_compounds', 'sensitivity_groups']})
        status.append(record); summaries[context] = {**record, **result['summary']}; tables.append(result)
    pd.DataFrame(status).to_csv(stage/'context_status.tsv', sep='\t', index=False, lineterminator='\n')
    for key, name in [('per_compound', 'per_compound.tsv'), ('per_set', 'per_set.tsv'),
                      ('contrasts', 'contrasts.tsv'), ('pairs', 'sensitivity_pairs.tsv')]:
        available = [r[key] for r in tables if not r[key].empty]
        table = pd.concat(available, ignore_index=True) if available else pd.DataFrame()
        if key == 'pairs' and not table.empty:
            table['alternate_control_Combos'] = table.alternate_control_Combos.map(json.dumps)
        table.to_csv(stage/name, sep='\t', index=False, na_rep='NA', lineterminator='\n')
    summary = {'contexts': summaries, 'primary_context': 'MCF7_24h_1000nM',
        'reporting_rule': 'Failed count gates retain descriptive points only; no CI. Conservative reporting choice; protocol thresholds unchanged.',
        'scope': 'Within-study transport of fixed chemical-structure predictions; no condition features, efficacy or external validation.',
        'uncertainty': 'Pointwise intervals conditional on frozen models; shared controls, plates and only two biological replicates limit inference.'}
    (stage/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-manifest', type=Path, default=ROOT/'reports/v2/predictions/manifest.json')
    parser.add_argument('--prediction-manifest-sha256', required=True)
    parser.add_argument('--target-manifest', type=Path, required=True)
    parser.add_argument('--target-manifest-sha256', required=True)
    parser.add_argument('--input-contract', type=Path, default=ROOT/'configs/v2_metadata_inputs.json')
    parser.add_argument('--input-contract-sha256', required=True)
    parser.add_argument('--qc-receipt', type=Path, default=ROOT/'receipts/v2_context_qc.json')
    parser.add_argument('--qc-receipt-sha256', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(argv)
    for path, expected, label in [(args.target_manifest, args.target_manifest_sha256, 'target manifest'),
        (args.prediction_manifest, args.prediction_manifest_sha256, 'prediction manifest'),
        (args.input_contract, args.input_contract_sha256, 'input contract'),
        (args.qc_receipt, args.qc_receipt_sha256, 'count QC receipt')]:
        EVALUATOR.require_hash(path, expected, label)
    out = args.output_root/'reports/v2/transfer'
    receipt_path = args.output_root/'receipts/v2_public_replay.json'
    if receipt_path.exists() or any((out/name).exists() for name in EVALUATOR.OUTPUTS):
        raise ValueError('refusing to overwrite a prior transfer replay')
    reference, reference_paths = check_published_reference(ROOT)
    protocol_path = ROOT/SCIENTIFIC_INPUTS['protocol']; split_path = ROOT/SCIENTIFIC_INPUTS['split']
    contract = json.loads(args.input_contract.read_text())
    for name, path in [('protocol', protocol_path), ('split', split_path)]:
        check_record(path, contract['expected_inputs'][name], 'frozen scientific '+name)
    protocol = yaml.safe_load(protocol_path.read_text())
    if (protocol['evaluation']['bootstrap_repetitions'] != 2000 or protocol['evaluation']['bootstrap_seed'] != 2026
            or protocol['qc']['minimum_primary_test_compounds'] != 20
            or protocol['qc']['minimum_primary_test_scaffold_groups'] != 10):
        raise ValueError('frozen scoring/resampling policy differs')
    qc, qc_paths, audit_path = EVALUATOR.check_qc_chain(ROOT, args.qc_receipt, protocol_path, args.input_contract)
    split = EVALUATOR.read(split_path); gene_manifest = EVALUATOR.read(ROOT/SCIENTIFIC_INPUTS['genes'])
    if gene_manifest.output_index.tolist() != list(range(2000)) or gene_manifest.gene_id.duplicated().any():
        raise ValueError('frozen transfer requires exactly 2000 uniquely ordered genes')
    genes = gene_manifest.gene_id.tolist(); set_ids = protocol['biology_head']['set_ids']
    ids = sorted(split.loc[split.split.eq('test'), 'compound_id'])
    a, b, prediction_paths = load_public_predictions(ROOT, args.prediction_manifest,
        args.prediction_manifest_sha256, ids, genes, set_ids)
    members = EVALUATOR.read(ROOT/SCIENTIFIC_INPUTS['members'])
    coverage_path = ROOT/'reports/v2/biology_coverage.tsv'; coverage = EVALUATOR.read(coverage_path)
    if (coverage.set_id.tolist() != set_ids or not booleans(coverage.passed, 'coverage passed').all()
            or set(members.set_id) != set(set_ids) or members.duplicated(['set_id', 'gene_id']).any()):
        raise ValueError('sealed five-set coverage/membership is incomplete')
    manifest = json.loads(args.target_manifest.read_text())
    source_data, source_records = EVALUATOR.verify_sources(manifest, args.target_manifest.parent, qc)
    if set(source_data) != {'MCF7', 'K562'}:
        raise ValueError('published replay requires both original target cell sources')
    data = {k: EVALUATOR.read(p) for k, p in qc_paths.items()}
    data['wells'] = EVALUATOR.read(ROOT/'manifests/v2/well_mapping.tsv')
    bound_sources = {'protocol': protocol_path, 'split': split_path,
        'input_contract': args.input_contract, 'qc_receipt': args.qc_receipt, 'metadata_audit': audit_path,
        'genes': prediction_paths['genes'], 'public_v1_seal': prediction_paths['v1_seal'],
        'biology_seal': prediction_paths['biology_seal'], 'v1_predictions': prediction_paths['gene_predictions'],
        'model:test_predictions.npz': prediction_paths['biology_predictions'],
        'context_roles': ROOT/'manifests/v2/context_roles.tsv', 'wells': ROOT/'manifests/v2/well_mapping.tsv', **qc_paths}
    for key, path in prediction_paths.items():
        if key.startswith('public:'):
            bound_sources[key] = path
    for cell, source in source_data.items():
        bound_sources.update({cell+'_'+key: path for key, path in source['files'].items()})
    code_names = ['scripts/replay_public_transfer_v2.py', 'scripts/evaluate_transfer_v2.py',
                  'src/drugperturb/transfer_v2.py', 'src/drugperturb/evaluation.py', 'src/drugperturb/biology_v2.py']
    for name in code_names[1:4]:
        bound_sources['code:'+name] = ROOT/name
    check_reference_inputs(reference, bound_sources, source_records)
    sources = {**bound_sources, **prediction_paths, **reference_paths, 'target_manifest': args.target_manifest,
               'coverage': coverage_path, **{'replay_code:'+n: ROOT/n for n in code_names}}
    # file_record exposes only basename, byte count and SHA; private paths stay private.
    before = {key: file_record(path) for key, path in sources.items()}
    args.output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.public-transfer-replay-', dir=args.output_root) as temporary:
        stage = Path(temporary)
        score_to_stage(stage, protocol, source_data, data, split, a, b, genes, members, set_ids)
        comparisons = compare_outputs(stage, reference['output_hashes'])
        if before != {key: file_record(path) for key, path in sources.items()}:
            raise ValueError('public replay dependency changed during scoring')
        receipt = {'schema_version': 1, 'status': 'PUBLISHED_TRANSFER_REPLAY_MATCHED',
            'replayed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
            'source_inputs': before, 'target_sources': source_records,
            'new_models_fitted': 0, 'test_outcomes_used_for_selection': False,
            'prediction_only_scope': 'Original sealed prediction bytes and target scoring verified; private training data, fitted weights and training execution are not independently reverified.',
            'code_files_attested_by_hash': True,
            'code_hashes': {name: digest(ROOT/name) for name in code_names},
            'published_reference_sha256': PUBLISHED_REFERENCE_SHA256,
            'output_hashes': {key: value['sha256'] for key, value in comparisons.items()},
            'output_comparisons': comparisons}
        (stage/'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)+'\n')
        out.mkdir(parents=True, exist_ok=True); receipt_path.parent.mkdir(parents=True, exist_ok=True)
        # Hard-link publication fails rather than replacing any concurrently created file.
        for name in EVALUATOR.OUTPUTS:
            os.link(stage/name, out/name)
        os.link(stage/'receipt.json', receipt_path)
    print(json.dumps({'status': receipt['status'], 'matched_outputs': len(comparisons), 'new_models_fitted': 0}))
    return receipt


if __name__ == '__main__':
    main()
