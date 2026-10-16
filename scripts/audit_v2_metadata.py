"""Audit fixed v2 roles using only source well metadata and frozen identities."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import tempfile

import pandas as pd

from drugperturb.context_design import audit_contexts


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--chemistry', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--input-contract', type=Path, default=root/'configs/v2_metadata_inputs.json',
                        help='Pre-existing frozen hashes; the audit never creates or updates this contract')
    args = parser.parse_args()
    sources = {name: getattr(args, name) for name in ['metadata', 'split', 'chemistry']}
    sources['protocol'] = root/'configs/protocol_v2.yaml'
    contract_hash = digest(args.input_contract)
    contract = json.loads(args.input_contract.read_text())
    expected = contract.get('expected_inputs', {})
    if contract.get('schema_version') != 1 or set(expected) != set(sources):
        raise ValueError('Invalid input contract schema or required entries')
    before = {}
    for name, path in sources.items():
        entry = expected[name]
        if (not isinstance(entry, dict) or not isinstance(entry.get('bytes'), int)
                or isinstance(entry.get('bytes'), bool) or not isinstance(entry.get('sha256'), str)
                or len(entry['sha256']) != 64 or any(c not in '0123456789abcdef' for c in entry['sha256'])):
            raise ValueError(f'Invalid input contract entry: {name}')
        # Check size before reading: unexpected large/non-metadata inputs are rejected early.
        if path.stat().st_size != entry['bytes']:
            raise ValueError(f'Frozen input hash/size mismatch: {name}')
        before[name] = digest(path)
        if before[name] != entry['sha256']:
            raise ValueError(f'Frozen input hash/size mismatch: {name}')
    metadata = pd.read_csv(args.metadata, sep='\t', keep_default_na=False, dtype={'dose_pattern': str})
    chemistry = pd.read_csv(args.chemistry, sep='\t', keep_default_na=False)
    split = pd.read_csv(args.split, sep='\t', keep_default_na=False)
    tables = audit_contexts(metadata, chemistry, split)
    if (before != {name: digest(path) for name, path in sources.items()}
            or contract_hash != digest(args.input_contract)):
        raise ValueError('Metadata inputs changed during audit')
    paths = {'context_roles': 'manifests/v2/context_roles.tsv',
             'well_mapping': 'manifests/v2/well_mapping.tsv',
             'design_inventory': 'reports/v2/design_inventory.tsv'}
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, check=True,
                              capture_output=True, text=True).stdout.strip()
    code = ['scripts/audit_v2_metadata.py', 'src/drugperturb/context_design.py',
            'src/drugperturb/design.py', 'src/drugperturb/chemistry.py', 'configs/protocol_v2.yaml']
    args.output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.metadata-audit-', dir=args.output_root) as temporary:
        stage = Path(temporary)
        hashes = {}
        for name, relative in paths.items():
            target = stage/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            tables[name].to_csv(target, sep='\t', index=False, lineterminator='\n')
            hashes[relative] = digest(target)
        receipt = dict(schema_version=1, status='METADATA_ONLY_COMPLETE',
                       source_inputs={name: {'basename': path.name, 'bytes': path.stat().st_size,
                                             'sha256': before[name]} for name, path in sources.items()},
                       input_contract={'basename': args.input_contract.name, 'sha256': contract_hash,
                                       'bytes': args.input_contract.stat().st_size},
                       code_revision=revision, code_revision_is_base=True,
                       code_files_attested_by_hash=True,
                       code_hashes={relative: digest(root/relative) for relative in code},
                       source_metadata_wells=len(metadata), contexts=len(tables['design_inventory']),
                       context_role_rows=len(tables['context_roles']), well_mapping_rows=len(tables['well_mapping']),
                       unique_selected_wells=tables['well_mapping'].Combo.nunique(),
                       count_qc_done=False, new_test_expression_read=False, models_trained=False,
                       role_policy='Frozen v1 identity/split only; no expression-dependent assignment',
                       extraction_policy='Deduplicate Combo across contexts; preserve all same-plate vehicle rows',
                       limitation='Metadata capacity only; no count-QC eligibility or scientific performance established',
                       output_hashes=hashes)
        target = stage/'receipts/v2_metadata_audit.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)+'\n')
        for relative in [*paths.values(), 'receipts/v2_metadata_audit.json']:
            target = args.output_root/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage/relative, target)
    print(json.dumps({'status': receipt['status'], 'contexts': receipt['contexts'],
                      'context_role_rows': receipt['context_role_rows'], 'count_qc_done': False}))


if __name__ == '__main__':
    main()
