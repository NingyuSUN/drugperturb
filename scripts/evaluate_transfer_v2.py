"""Evaluate sealed predictions on predeclared count-qualified target contexts."""
from pathlib import Path
import argparse
import datetime
import json
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
import yaml

from drugperturb.biology_v2 import file_record, verify_seal
from drugperturb.provenance import verify_training_inputs
from drugperturb.transfer_v2 import (A_METHODS, B_METHODS, align_predictions, alternate_controls,
    booleans, checked_file, evaluate_context, frozen_cohort, required_wells)

ROOT = Path(__file__).resolve().parents[1]
QC_FILES = {'compounds': 'manifests/v2/compound_count_qc.tsv',
            'qualified': 'manifests/v2/well_count_qc.tsv', 'summary': 'reports/v2/context_count_qc.tsv'}
OUTPUTS = ['context_status.tsv', 'per_compound.tsv', 'per_set.tsv', 'contrasts.tsv',
           'sensitivity_pairs.tsv', 'summary.json']


def read(path):
    return pd.read_csv(path, sep='\t', keep_default_na=False, dtype={'dose_pattern': str})


def digest(path):
    return file_record(path)['sha256']


def safe_child(root, relative):
    path = Path(root)/relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('unsafe receipt file path')
    return path


def require_hash(path, expected, label):
    if digest(path) != expected:
        raise ValueError(f'{label} hash mismatch')


def check_qc_chain(root, qc_path, protocol_path, contract_path):
    receipt = json.loads(qc_path.read_text())
    if (receipt.get('status') != 'COUNT_QC_ONLY_COMPLETE'
            or receipt.get('expression_effects_read') is not False or receipt.get('models_trained') is not False):
        raise ValueError('a response-blind completed context QC receipt is required')
    paths = {name: root/relative for name, relative in QC_FILES.items()}
    for name, path in paths.items():
        require_hash(path, receipt.get('output_hashes', {}).get(QC_FILES[name]), 'count QC output')
    for name, path in [('well_mapping', root/'manifests/v2/well_mapping.tsv'),
                       ('context_roles', root/'manifests/v2/context_roles.tsv'),
                       ('protocol', protocol_path), ('input_contract', contract_path)]:
        require_hash(path, receipt.get('frozen_input_hashes', {}).get(name), 'count QC dependency')
    audit_path = root/'receipts/v2_metadata_audit.json'
    require_hash(audit_path, receipt.get('metadata_audit_sha256'), 'M0 metadata audit')
    audit = json.loads(audit_path.read_text())
    if audit.get('status') != 'METADATA_ONLY_COMPLETE' or audit.get('new_test_expression_read') is not False:
        raise ValueError('a response-blind completed M0 audit is required')
    require_hash(contract_path, audit.get('input_contract', {}).get('sha256'), 'M0 input contract')
    for relative in ['manifests/v2/well_mapping.tsv', 'manifests/v2/context_roles.tsv']:
        require_hash(root/relative, audit.get('output_hashes', {}).get(relative), 'M0 scientific roles')
    for relative, expected in receipt.get('code_hashes', {}).items():
        require_hash(safe_child(root, relative), expected, 'count QC code')
    return receipt, paths, audit_path


def check_biology_seal(root, v1_root, model_dir, path):
    seal = json.loads(path.read_text())
    allowed = {'SEALED_BEFORE_TARGET_SCORING': 4, 'FAILED_COVERAGE_ZERO_FITS': 0}
    if (seal.get('status') not in allowed or seal.get('fit_count') != allowed[seal['status']]
            or seal.get('test_outcomes_loaded') is not False
            or seal.get('new_context_outcomes_loaded') is not False):
        raise ValueError('invalid biological-head prediction seal')
    sources = {'biology': root/'configs/biology_v1.json', 'code_0': root/'src/drugperturb/biology_v2.py',
        'code_1': root/'scripts/train_biology_v2.py', 'code_2': root/'src/drugperturb/chemistry.py',
        'code_3': root/'src/drugperturb/response.py', 'counts': v1_root/'data/processed/pseudobulk_counts.tsv.gz',
        'gene_metadata': v1_root/'data/processed/gene_metadata.tsv',
        'input_contract': root/'configs/v2_metadata_inputs.json', 'mapping': root/'manifests/qualified_well_mapping.tsv',
        'protocol': root/'configs/protocol_v2.yaml', 'source_receipt': v1_root/'receipts/report_artifacts.json',
        'split': root/'manifests/split_manifest.tsv'}
    model_paths, public_paths = {}, {}
    allowed_public = {'manifests/v2/biology_set_members.tsv', 'reports/v2/biology_coverage.tsv',
                      'receipts/v2_biology_coverage.json', 'receipts/v2_biology_selection.json'}
    allowed_models = {'selected_model.joblib', 'test_predictions.npz', 'validation_predictions.npz',
                      *(f'Ridge_c{i}.joblib' for i in range(4))}
    for key in seal['artifact_hashes']:
        kind, relative = key.split(':', 1)
        if kind == 'model' and relative in allowed_models:
            model_paths[key] = model_dir/relative
        elif kind == 'public' and relative in allowed_public:
            public_paths[key] = root/relative
        else:
            raise ValueError('unexpected biological-head sealed artifact')
    expected = {'public:'+x for x in allowed_public}
    if seal['fit_count']:
        expected |= {'model:'+x for x in allowed_models}
    if set(seal['artifact_hashes']) != expected:
        raise ValueError('biological-head artifact seal is incomplete')
    verify_seal(seal, sources, model_paths, public_paths)
    return seal, {**sources, **model_paths, **public_paths}


def load_sealed_predictions(root, v1_root, b_seal, model_dir, split, genes, set_ids):
    """Read predictions only, after both original and new model chains verify."""
    v1_seal = json.loads((v1_root/'receipts/prediction_seal.json').read_text())
    public_v1 = json.loads((root/'receipts/prediction_seal.json').read_text())
    for relative in ['models/run_v1/test_predictions.npz', 'manifests/selected_genes.tsv']:
        expected = v1_seal['artifact_hashes'].get(relative)
        if expected is None or public_v1['artifact_hashes'].get(relative) != expected:
            raise ValueError('private/public v1 frozen artifact reference differs')
        require_hash(root/relative if relative.startswith('manifests/') else v1_root/relative,
                     expected, 'v1 frozen prediction/gene order')
    with np.load(v1_root/'models/run_v1/test_predictions.npz', allow_pickle=False) as source:
        a = {k: source[k].copy() for k in source.files}
    test_ids = sorted(split.loc[split.split.eq('test'), 'compound_id'])
    if set(a['ids'].tolist()) != set(test_ids):
        raise ValueError('v1 sealed queries differ from the frozen test cohort')
    align_predictions(a, test_ids, A_METHODS, len(genes))
    v1_protocol = yaml.safe_load((v1_root/'configs/protocol_v1.yaml').read_text())
    for method in ['M2', 'M3']:
        arrays = [a[f'{method}_s{seed}'] for seed in v1_protocol['models']['training_seeds']]
        if not np.array_equal(a[method], np.mean(arrays, axis=0)):
            raise ValueError('frozen neural arithmetic-mean ensemble differs')
    b = None
    if b_seal['fit_count']:
        with np.load(model_dir/'test_predictions.npz', allow_pickle=False) as source:
            b = {k: source[k].copy() for k in source.files}
        if b['set_ids'].tolist() != set_ids or set(b['ids'].tolist()) != set(test_ids):
            raise ValueError('new-head sealed query or gene-set order differs')
        align_predictions(b, test_ids, B_METHODS, 5)
    return a, b


def verify_sources(manifest, base, qc_receipt):
    if manifest.get('schema_version') != 1 or not isinstance(manifest.get('sources'), dict):
        raise ValueError('unsupported target source manifest')
    if set(manifest['sources'])-{'MCF7', 'K562'}:
        raise ValueError('target manifest includes an undeclared cell line')
    checked, records = {}, {}
    for cell, entry in manifest['sources'].items():
        files = {name: checked_file(entry[name], base) for name in ['counts', 'gene_metadata',
            'well_qc', 'extraction_summary', 'extractor_code', 'acquisition_receipt', 'extraction_receipt']}
        raw = entry['raw_source']
        if set(raw) != {'basename', 'bytes', 'sha256'}:
            raise ValueError('raw source descriptor must contain basename, bytes and sha256')
        acquisition = json.loads(files['acquisition_receipt'].read_text())
        matches = [r for r in acquisition.get('sources', []) if r.get('filename') == raw['basename']]
        if (len(matches) != 1 or any(matches[0].get(k) != raw[k] for k in ['bytes', 'sha256'])
                or matches[0].get('gzip_integrity_checked') is not True):
            raise ValueError('raw source does not match the acquisition trust anchor')
        if raw['basename'] != f'GSM4150378_sciPlex3_{cell}_24hrs.RDS.gz':
            raise ValueError('source accession/cell filename differs')
        extraction = json.loads(files['extraction_receipt'].read_text())
        executed = extraction.get('sources', {}).get(cell, {})
        if (extraction.get('schema_version') != 1 or executed.get('status') != 'COMPLETE'
                or any(executed.get('source', {}).get(k) != raw[k] for k in ['basename', 'bytes', 'sha256'])):
            raise ValueError('extraction receipt raw source binding failed')
        names = {'counts': 'pseudobulk_counts.tsv.gz', 'gene_metadata': 'gene_metadata.tsv',
                 'well_qc': 'well_qc.tsv', 'extraction_summary': 'extraction_summary.tsv'}
        for key, name in names.items():
            record = file_record(files[key]); expected = executed.get('outputs', {}).get(name, {})
            if any(record[k] != expected.get(k) for k in ['bytes', 'sha256']):
                raise ValueError('extraction receipt output hash/size binding failed')
        for name in ['scripts/export_context_rds.R', 'scripts/csc_utils.R',
                     'manifests/v2/well_mapping.tsv', 'configs/protocol_v2.yaml']:
            expected = executed.get('code_input_hashes', {}).get(name)
            require_hash(ROOT/name, expected, 'executed extraction code/input')
        if digest(files['extractor_code']) != executed['code_input_hashes']['scripts/export_context_rds.R']:
            raise ValueError('extraction exporter snapshot hash differs')
        qc_expected = [(r['sha256'], r['bytes']) for r in qc_receipt['well_qc_inputs']]
        qrecord = file_record(files['well_qc'])
        if (qrecord['sha256'], qrecord['bytes']) not in qc_expected:
            raise ValueError('source well QC is not bound to cohort qualification')
        summary_table = read(files['extraction_summary'])
        if set(summary_table.columns) != {'key', 'value'} or summary_table.key.duplicated().any():
            raise ValueError('source extraction summary keys must be unique')
        summary = dict(zip(summary_table.key, summary_table.value))
        for flag in ['all_human_gene_rows_exported', 'UMI_conserved', 'source_hash_QC_all_pass',
                     'counts_nonnegative_integer']:
            if str(summary.get(flag)).lower() != 'true':
                raise ValueError('source extraction correctness checks are incomplete')
        if (summary.get('context_cell_type') != cell or float(summary.get('context_time_hours', -1)) != 24
                or str(summary.get('treatment_effects_computed')).lower() != 'false'):
            raise ValueError('source extraction context/effect-blind scope differs')
        checked[cell] = {'files': files, 'summary': summary}
        records[cell] = {'files': {k: file_record(p) for k, p in files.items()}, 'raw_source': raw}
    return checked, records


def score_context_source(context, role, source, cohort, data, a, b, genes, members, set_ids):
    cell, _, _ = context.split('_')
    wells = data['wells'][data['wells'].context_id.eq(context)]
    qualified = data['qualified'][data['qualified'].context_id.eq(context)]
    qc = read(source['files']['well_qc'])
    pairs = alternate_controls(wells, qualified, qc, cohort['ids'])
    keys = required_wells(qualified, pairs, cohort['ids'])
    # Requested well projection precedes numeric parsing and all normalization.
    counts = pd.read_csv(source['files']['counts'], sep='\t', usecols=['gene_id', *keys],
                         index_col='gene_id')
    gene_metadata = pd.read_csv(source['files']['gene_metadata'], sep='\t', usecols=['gene_id'])
    if (counts.index.tolist() != gene_metadata.gene_id.tolist()
            or len(counts) != int(source['summary']['source_human_genes'])
            or counts.index.has_duplicates or gene_metadata.gene_id.isna().any()):
        raise ValueError('target count gene identities/order differ from full human gene metadata')
    if 'human_UMI' in qc:
        libraries = counts.sum(axis=0)
        expected = qc.set_index('Combo').human_UMI.reindex(counts.columns)
        if expected.isna().any() or not np.array_equal(libraries.to_numpy(), expected.to_numpy()):
            raise ValueError('target count library sums disagree with source QC UMI totals')
    result = evaluate_context(counts, wells, qualified, qc, cohort,
        align_predictions(a, cohort['ids'], A_METHODS, len(genes)), genes,
        b_predictions=align_predictions(b, cohort['ids'], B_METHODS, 5) if b is not None else None,
        members=members, set_ids=set_ids)
    for table in ['per_compound', 'per_set', 'contrasts', 'pairs']:
        result[table]['context_id'] = context; result[table]['context_role'] = role
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v1-root', type=Path, required=True)
    parser.add_argument('--biology-model-dir', type=Path, required=True)
    parser.add_argument('--target-manifest', type=Path, required=True)
    parser.add_argument('--target-manifest-sha256', required=True)
    parser.add_argument('--input-contract', type=Path, default=ROOT/'configs/v2_metadata_inputs.json')
    parser.add_argument('--input-contract-sha256', required=True)
    parser.add_argument('--biology-seal', type=Path, default=ROOT/'receipts/v2_biology_prediction_seal.json')
    parser.add_argument('--biology-seal-sha256', required=True)
    parser.add_argument('--qc-receipt', type=Path, default=ROOT/'receipts/v2_context_qc.json')
    parser.add_argument('--qc-receipt-sha256', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(argv)
    # First operation: reject a changed externally sealed manifest, before data reads.
    require_hash(args.target_manifest, args.target_manifest_sha256, 'target manifest')
    for path, expected, label in [(args.input_contract, args.input_contract_sha256, 'input contract'),
        (args.biology_seal, args.biology_seal_sha256, 'biological prediction seal'),
        (args.qc_receipt, args.qc_receipt_sha256, 'count QC receipt')]:
        require_hash(path, expected, label)
    protocol_path = ROOT/'configs/protocol_v2.yaml'; split_path = ROOT/'manifests/split_manifest.tsv'
    contract = json.loads(args.input_contract.read_text())
    for name, path in [('protocol', protocol_path), ('split', split_path)]:
        expected = contract['expected_inputs'][name]
        require_hash(path, expected['sha256'], 'frozen scientific '+name)
        if path.stat().st_size != expected['bytes']:
            raise ValueError('frozen scientific input size differs')
    protocol = yaml.safe_load(protocol_path.read_text())
    if (protocol['evaluation']['bootstrap_repetitions'] != 2000 or protocol['evaluation']['bootstrap_seed'] != 2026
            or protocol['qc']['minimum_primary_test_compounds'] != 20
            or protocol['qc']['minimum_primary_test_scaffold_groups'] != 10):
        raise ValueError('frozen scoring/resampling policy differs')
    qc_receipt, qc_paths, audit_path = check_qc_chain(ROOT, args.qc_receipt, protocol_path, args.input_contract)
    verify_training_inputs(args.v1_root, completed=True)  # Deliberately does not read old test truth.
    b_seal, b_dependencies = check_biology_seal(ROOT, args.v1_root, args.biology_model_dir, args.biology_seal)
    manifest = json.loads(args.target_manifest.read_text())
    source_data, source_records = verify_sources(manifest, args.target_manifest.parent, qc_receipt)
    split = read(split_path)
    gene_manifest = read(ROOT/'manifests/selected_genes.tsv')
    if gene_manifest.output_index.tolist() != list(range(2000)) or gene_manifest.gene_id.duplicated().any():
        raise ValueError('frozen transfer requires exactly 2000 uniquely ordered genes')
    genes = gene_manifest.gene_id.tolist(); set_ids = protocol['biology_head']['set_ids']
    members = read(ROOT/'manifests/v2/biology_set_members.tsv')
    if b_seal['fit_count']:
        coverage = read(ROOT/'reports/v2/biology_coverage.tsv')
        if (coverage.set_id.tolist() != set_ids or not booleans(coverage.passed, 'coverage passed').all()
                or set(members.set_id) != set(set_ids) or members.duplicated(['set_id', 'gene_id']).any()):
            raise ValueError('sealed five-set coverage/membership is incomplete')
    a, b = load_sealed_predictions(ROOT, args.v1_root, b_seal, args.biology_model_dir,
                                   split, genes, set_ids)
    data = {k: read(p) for k, p in qc_paths.items()}
    data['wells'] = read(ROOT/'manifests/v2/well_mapping.tsv')
    sources = {'target_manifest': args.target_manifest, 'protocol': protocol_path,
        'input_contract': args.input_contract, 'biology_seal': args.biology_seal,
        'qc_receipt': args.qc_receipt, 'metadata_audit': audit_path, 'split': split_path,
        'genes': ROOT/'manifests/selected_genes.tsv', 'v1_seal': args.v1_root/'receipts/prediction_seal.json',
        'v1_dataset_freeze': args.v1_root/'receipts/dataset_freeze.json',
        'public_v1_seal': ROOT/'receipts/prediction_seal.json',
        'v1_predictions': args.v1_root/'models/run_v1/test_predictions.npz',
        'context_roles': ROOT/'manifests/v2/context_roles.tsv',
        'wells': ROOT/'manifests/v2/well_mapping.tsv', **qc_paths, **b_dependencies}
    # Existing keys above and M2 dependency keys use the same authoritative files.
    for key, cell_source in source_data.items():
        sources.update({key+'_'+name: path for name, path in cell_source['files'].items()})
    for name in ['src/drugperturb/transfer_v2.py', 'scripts/evaluate_transfer_v2.py',
                 'src/drugperturb/evaluation.py']:
        sources['code:'+name] = ROOT/name
    before = {key: file_record(path) for key, path in sources.items()}
    out = args.output_root/'reports/v2/transfer'; receipt_path = args.output_root/'receipts/v2_transfer_evaluation.json'
    if receipt_path.exists() or any((out/name).exists() for name in OUTPUTS):
        raise ValueError('refusing to overwrite a prior transfer evaluation')
    targets = [(protocol['scope']['primary_target_context'], 'primary_test')]
    targets += [(c, 'secondary_test') for c in protocol['scope']['secondary_target_contexts']]
    status, tables, summaries = [], [], {}
    for target, role in targets:
        cell = target['cell_type']; context = f'{cell}_{target["time_hours"]}h_{target["dose_nM"]}nM'
        cohort = frozen_cohort(data['compounds'], data['qualified'], data['summary'], split, context, role)
        record = {k: v for k, v in cohort.items() if k not in ['ids', 'groups']}
        record.update(context_id=context, context_role=role, sensitivity_compounds=0, sensitivity_groups=0)
        if cell not in source_data:
            record['status'] = 'SOURCE_UNAVAILABLE'
        if record['status'] in ['SOURCE_UNAVAILABLE', 'COUNT_QC_UNAVAILABLE'] or not cohort['ids']:
            status.append(record); summaries[context] = dict(record); continue
        result = score_context_source(context, role, source_data[cell], cohort, data, a, b, genes, members, set_ids)
        record.update({k: result[k] for k in ['sensitivity_compounds', 'sensitivity_groups']})
        status.append(record); summaries[context] = {**record, **result['summary']}; tables.append(result)
    if before != {key: file_record(path) for key, path in sources.items()}:
        raise ValueError('sealed scoring dependency changed during target evaluation')
    args.output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.transfer-evaluation-', dir=args.output_root) as temporary:
        stage = Path(temporary)
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
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                                  capture_output=True, text=True).stdout.strip()
        receipt = {'schema_version': 2, 'status': 'SEALED_TRANSFER_EVALUATION',
            'evaluated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source_inputs': before, 'target_sources': source_records,
            'v1_reference_scope': 'Private original chain verified; public/private selected-gene and test-prediction hash references agree. Receipt bytes need not agree.',
            'test_outcomes_used_for_selection': False, 'new_models_fitted': 0,
            'code_revision': revision, 'code_revision_is_base': True, 'code_files_attested_by_hash': True,
            'output_hashes': {'reports/v2/transfer/'+name: digest(stage/name) for name in OUTPUTS}}
        out.mkdir(parents=True, exist_ok=True)
        for name in OUTPUTS:
            os.replace(stage/name, out/name)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        (stage/'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)+'\n')
        os.replace(stage/'receipt.json', receipt_path)
    print(json.dumps({'status': receipt['status'], 'contexts': len(status), 'new_models_fitted': 0}))
    return receipt


if __name__ == '__main__':
    main()
