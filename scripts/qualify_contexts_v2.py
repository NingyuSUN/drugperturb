"""Qualify fixed v2 cohorts using well-level cell counts, never expression effects."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import tempfile

import pandas as pd
import yaml

from drugperturb.context_qc import qualify_contexts


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--well-mapping',type=Path,default=root/'manifests/v2/well_mapping.tsv')
    parser.add_argument('--context-roles',type=Path,default=root/'manifests/v2/context_roles.tsv')
    parser.add_argument('--metadata-audit',type=Path,default=root/'receipts/v2_metadata_audit.json')
    parser.add_argument('--well-qc',type=Path,action='append',required=True,
                        help='Repeat for available extraction well_qc tables; no counts matrix is read')
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    sources={'metadata_audit':args.metadata_audit,'well_mapping':args.well_mapping,
             'context_roles':args.context_roles,'protocol':root/'configs/protocol_v2.yaml',
             'input_contract':root/'configs/v2_metadata_inputs.json'}
    before={name:sha(path) for name,path in sources.items()}
    metadata_audit=json.loads(args.metadata_audit.read_text())
    if (metadata_audit.get('status')!='METADATA_ONLY_COMPLETE' or
            metadata_audit.get('new_test_expression_read') is not False or
            metadata_audit.get('models_trained') is not False or
            metadata_audit.get('count_qc_done') is not False):
        raise ValueError('A completed response-blind M0 audit is required')
    for name in ['well_mapping','context_roles']:
        key=f'manifests/v2/{name}.tsv'
        if metadata_audit.get('output_hashes',{}).get(key)!=before[name]:
            raise ValueError(f'M0 output hash mismatch: {name}')
    if metadata_audit.get('input_contract',{}).get('sha256')!=before['input_contract']:
        raise ValueError('M0 frozen contract hash mismatch')
    contract=json.loads(sources['input_contract'].read_text())
    expected_protocol=contract['expected_inputs']['protocol']['sha256']
    if (before['protocol']!=expected_protocol or
            metadata_audit.get('code_hashes',{}).get('configs/protocol_v2.yaml')!=before['protocol']):
        raise ValueError('M0 scientific protocol hash mismatch')
    protocol=yaml.safe_load(sources['protocol'].read_text())
    read=lambda path:pd.read_csv(path,sep='\t',keep_default_na=False,dtype={'dose_pattern':str})
    wells=read(args.well_mapping)
    roles=read(args.context_roles)
    qc_tables=[]
    qc_inputs=[]
    for path in args.well_qc:
        digest=sha(path)
        frame=read(path)
        if not {'Combo','n_cells'}.issubset(frame):
            raise ValueError('well_qc requires Combo and n_cells')
        qc_inputs.append({'basename':path.name,'sha256':digest,'bytes':path.stat().st_size,
                          'well_records':len(frame)})
        qc_tables.append(frame)
    settings=protocol['qc']
    tables=qualify_contexts(wells,roles,pd.concat(qc_tables,ignore_index=True),
        minimum_cells=settings['minimum_cells_per_treatment_replicate'],
        minimum_test_compounds=settings['minimum_primary_test_compounds'],
        minimum_test_groups=settings['minimum_primary_test_scaffold_groups'])
    if settings['minimum_cells_per_matched_vehicle']!=settings['minimum_cells_per_treatment_replicate']:
        raise ValueError('The frozen equal treatment/control count threshold is required')
    if before!={name:sha(path) for name,path in sources.items()}:
        raise ValueError('Frozen count-QC dependencies changed during qualification')
    for path,entry in zip(args.well_qc,qc_inputs):
        if sha(path)!=entry['sha256']:
            raise ValueError('Well QC input changed during qualification')
    paths={'compound_qualification':'manifests/v2/compound_count_qc.tsv',
           'well_qualification':'manifests/v2/well_count_qc.tsv',
           'context_summary':'reports/v2/context_count_qc.tsv'}
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=root,check=True,capture_output=True,text=True).stdout.strip()
    code=['src/drugperturb/context_qc.py','scripts/qualify_contexts_v2.py','src/drugperturb/design.py']
    args.output_root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.context-qc-',dir=args.output_root) as temporary:
        stage=Path(temporary)
        hashes={}
        for name,relative in paths.items():
            target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True)
            tables[name].to_csv(target,sep='\t',index=False,lineterminator='\n',na_rep='NA')
            hashes[relative]=sha(target)
        summary=tables['context_summary']
        receipt=dict(schema_version=1,status='COUNT_QC_ONLY_COMPLETE',
            metadata_audit_sha256=before['metadata_audit'],
            frozen_input_hashes={name:before[name] for name in ['well_mapping','context_roles','protocol','input_contract']},
            well_qc_inputs=qc_inputs,context_rows=len(summary),
            contexts_with_complete_count_qc=int(summary.count_qc_done.sum()),
            code_revision=revision,code_revision_is_base=True,
            code_hashes={relative:sha(root/relative) for relative in code},
            thresholds={name:settings[name] for name in ['minimum_cells_per_treatment_replicate',
                'minimum_cells_per_matched_vehicle','minimum_primary_test_compounds','minimum_primary_test_scaffold_groups']},
            expression_effects_read=False,models_trained=False,
            limitation='Sample count eligibility only; no predictive or biological validation',output_hashes=hashes)
        target=stage/'receipts/v2_context_qc.json';target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
        for relative in [*paths.values(),'receipts/v2_context_qc.json']:
            target=args.output_root/relative;target.parent.mkdir(parents=True,exist_ok=True)
            os.replace(stage/relative,target)
    print(json.dumps({'status':receipt['status'],'contexts':len(summary),
                      'contexts_with_complete_count_qc':receipt['contexts_with_complete_count_qc']}))


if __name__=='__main__':
    main()
