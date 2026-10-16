"""Reproduce the fixed LINCS metadata feasibility gate without reading expression."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import subprocess

import pandas as pd

from drugperturb.external_v2 import audit_lincs_frames


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(path, expected):
    if path.stat().st_size != int(expected['bytes']) or sha(path) != expected['sha256']:
        raise ValueError(f'Frozen metadata input changed: {path.name}')


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    source_manifest = root/'manifests/v2/lincs_sources.tsv'
    sources = list(csv.DictReader(source_manifest.open(), delimiter='\t'))
    files = {}
    for row in sources:
        name = row['filename']
        if Path(name).name != name or not name.endswith('.txt.gz'):
            raise ValueError('Source manifest must contain flat metadata-only filenames')
        path = args.metadata_dir/name
        verify(path, row)
        files[row['role']] = path
    if set(files) != {'signatures', 'perturbations', 'genes', 'metrics'}:
        raise ValueError('Incomplete LINCS metadata manifest')
    contract = json.loads((root/'configs/v2_metadata_inputs.json').read_text())
    project = {'split': root/'manifests/split_manifest.tsv',
               'chemistry': root/'manifests/chemistry_manifest.tsv',
               'protocol': root/'configs/protocol_v2.yaml'}
    for name, path in project.items():
        verify(path, contract['expected_inputs'][name])
    selected = root/'manifests/selected_genes.tsv'
    freeze = json.loads((root/'receipts/dataset_freeze.json').read_text())
    if sha(selected) != freeze['artifact_hashes']['manifests/selected_genes.tsv']:
        raise ValueError('Frozen selected genes changed')
    before = {str(p.relative_to(root)): sha(p) for p in [*project.values(), selected,
              root/'configs/v2_metadata_inputs.json', root/'receipts/dataset_freeze.json', source_manifest]}
    read = lambda p: pd.read_csv(p, sep='\t', dtype=str, keep_default_na=False)
    report = audit_lincs_frames(read(files['signatures']), read(files['perturbations']),
                               read(files['genes']), read(project['split']), read(selected))
    # Inspect only the metrics schema: zero raw overlap cannot be rescued by QC.
    report['signature_metrics_columns'] = list(pd.read_csv(files['metrics'], sep='\t', nrows=0).columns)
    report['source_files'] = sources
    report['project_input_hashes'] = before
    report['code_revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    report['code_revision_is_base'] = True
    report['code_hashes'] = {rel: sha(root/rel) for rel in ['src/drugperturb/external_v2.py',
                            'scripts/audit_lincs_v2.py', 'src/drugperturb/chemistry.py']}
    report['source_manifest_sha256'] = sha(source_manifest)
    for row in sources:
        verify(args.metadata_dir/row['filename'], row)
    if any(sha(root/rel) != value for rel, value in before.items()):
        raise ValueError('Project inputs changed during metadata audit')
    output = args.output_root/'receipts/v2_external_feasibility.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix('.json.part')
    with temp.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(temp, output)
    print(json.dumps({'gate_decision': report['gate_decision'], 'counts': report['counts']}))


if __name__ == '__main__':
    main()
