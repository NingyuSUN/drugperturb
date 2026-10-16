"""Verify replayed context artifacts and write a PRIVATE absolute-path M3 manifest.

Only byte hashes and the small extraction summaries are inspected. This command
does not run extraction, download data, parse expression values or modify seals.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import re

ROOT=Path(__file__).resolve().parents[1]
CELLS=('MCF7','K562')
OUTPUT_NAMES=('pseudobulk_counts.tsv.gz','gene_metadata.tsv','well_qc.tsv',
              'selected_cell_qc.tsv.gz','extraction_summary.tsv')
CODE_INPUTS=('scripts/export_context_rds.R','scripts/csc_utils.R',
             'manifests/v2/well_mapping.tsv','configs/protocol_v2.yaml')
MANIFEST_FILES={'counts':'pseudobulk_counts.tsv.gz','gene_metadata':'gene_metadata.tsv',
                'well_qc':'well_qc.tsv','extraction_summary':'extraction_summary.tsv'}


def present(path):
    return path.exists() or path.is_symlink()


def file_record(path):
    if not path.is_file():raise FileNotFoundError(f'Required input file missing: {path.name}')
    with path.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'bytes':path.stat().st_size,'sha256':digest}


def validate_record(value,label):
    if (not isinstance(value,dict) or set(value)!={'bytes','sha256'} or
            isinstance(value.get('bytes'),bool) or not isinstance(value.get('bytes'),int) or value['bytes']<1 or
            not isinstance(value.get('sha256'),str) or not re.fullmatch(r'[0-9a-f]{64}',value['sha256'])):
        raise ValueError(f'Invalid reference hash/size record: {label}')


def read_summary(path,cell,expected):
    with path.open(newline='',encoding='utf-8') as stream:
        reader=csv.DictReader(stream,delimiter='\t')
        if reader.fieldnames!=['key','value']:raise ValueError('Invalid extraction summary columns')
        rows=list(reader)
    if (any(set(row)!={'key','value'} or row['key'] is None or row['value'] is None for row in rows) or
            len({row['key'] for row in rows})!=len(rows)):
        raise ValueError('Extraction summary keys must be unique and complete')
    summary={row['key']:row['value'] for row in rows}
    if not isinstance(expected,dict) or summary!=expected:
        raise ValueError('Reference summary differs from pinned extraction summary file')
    for flag in ['all_human_gene_rows_exported','UMI_conserved','source_hash_QC_all_pass','counts_nonnegative_integer']:
        if summary.get(flag,'').lower()!='true':raise ValueError(f'Extraction summary check failed: {flag}')
    for flag in ['author_embeddings_reused','treatment_effects_computed']:
        if summary.get(flag,'').lower()!='false':raise ValueError(f'Extraction summary effect-blind check failed: {flag}')
    try:
        correct_time=float(summary.get('context_time_hours',''))==24
    except ValueError:
        correct_time=False
    if summary.get('context_cell_type')!=cell or not correct_time:
        raise ValueError('Extraction summary cell/time context mismatch')
    for flag in ['counts_gene_dimnames_present','counts_cell_dimnames_present']:
        if summary.get(flag,'').lower() not in ['true','false']:
            raise ValueError('Extraction summary lacks explicit assay identity scope')
    if not all(summary.get(field) for field in ['row_positional_binding_scope','col_positional_binding_scope']):
        raise ValueError('Extraction summary lacks positional binding scope')


def prepare(context_root,extraction_receipt,*,root=ROOT):
    """Build the exact existing M3 schema after reference/source/output verification."""
    root=Path(root).resolve();context_root=Path(context_root).resolve()
    extraction_receipt=Path(extraction_receipt).resolve()
    acquisition_path=root/'receipts/v2_source_acquisition.json'
    anchors={path:file_record(path) for path in [acquisition_path,extraction_receipt,
                                                 *(root/name for name in CODE_INPUTS)]}
    extraction=json.loads(extraction_receipt.read_text(encoding='utf-8'))
    acquisition=json.loads(acquisition_path.read_text(encoding='utf-8'))
    if (extraction.get('schema_version')!=1 or not isinstance(extraction.get('sources'),dict) or
            set(extraction['sources'])!=set(CELLS)):
        raise ValueError('Reference extraction must include the two complete fixed cell sources')
    if acquisition.get('schema_version')!=1 or not isinstance(acquisition.get('sources'),list):
        raise ValueError('Invalid public source acquisition record')
    # Check for interrupted artifacts before hashing any of the derived data files.
    for cell in CELLS:
        for name in OUTPUT_NAMES:
            partial=context_root/(cell+'_24h')/(name+'.part')
            if present(partial):raise FileExistsError(f'Existing derived partial preserved: {cell}/{partial.name}')
    manifest={'schema_version':1,'sources':{}}
    for cell in CELLS:
        source=extraction['sources'][cell]
        if (not isinstance(source,dict) or source.get('status')!='COMPLETE' or
                source.get('treatment_effects_computed') is not False):
            raise ValueError('Reference extraction is incomplete or not effect-blind')
        raw=source.get('source',{})
        if (not isinstance(raw,dict) or set(raw)!={'basename','bytes','sha256'} or
                raw['basename']!=f'GSM4150378_sciPlex3_{cell}_24hrs.RDS.gz'):
            raise ValueError('Reference extraction source cell/accession identity differs')
        validate_record({key:raw[key] for key in ['bytes','sha256']},'raw source')
        matches=[entry for entry in acquisition['sources'] if entry.get('filename')==raw['basename']]
        if (len(matches)!=1 or matches[0].get('gzip_integrity_checked') is not True or
                any(matches[0].get(key)!=raw[key] for key in ['bytes','sha256'])):
            raise ValueError('Reference raw source does not match public acquisition')
        code=source.get('code_input_hashes',{})
        if not isinstance(code,dict) or set(code)!=set(CODE_INPUTS):
            raise ValueError('Reference extraction code/input hash list is incomplete')
        for relative in CODE_INPUTS:
            if anchors[root/relative]['sha256']!=code[relative]:
                raise ValueError(f'Current extraction code/input hash mismatch: {relative}')
        outputs=source.get('outputs',{})
        if not isinstance(outputs,dict) or set(outputs)!=set(OUTPUT_NAMES):
            raise ValueError('Reference extraction must record all five derived outputs')
        verified={}
        for name in OUTPUT_NAMES:
            expected=outputs[name];validate_record(expected,name)
            path=context_root/(cell+'_24h')/name
            if not path.is_file() or path.stat().st_size!=expected['bytes']:
                raise ValueError(f'Derived output size/hash mismatch: {cell}/{name}')
            actual=file_record(path)
            if actual!=expected:raise ValueError(f'Derived output hash mismatch: {cell}/{name}')
            verified[name]={'path':str(path.resolve()),**actual}
        read_summary(context_root/(cell+'_24h')/'extraction_summary.tsv',cell,source.get('summary'))
        entry={key:verified[name] for key,name in MANIFEST_FILES.items()}
        for key,path in [('extractor_code',root/'scripts/export_context_rds.R'),
                         ('acquisition_receipt',acquisition_path),('extraction_receipt',extraction_receipt)]:
            entry[key]={'path':str(path.resolve()),**anchors[path]}
        entry['raw_source']=dict(raw)
        manifest['sources'][cell]=entry
    if any(file_record(path)!=expected for path,expected in anchors.items()):
        raise ValueError('Extraction provenance/code inputs changed during verification')
    return manifest


def main(argv=None,*,root=ROOT):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context-root',type=Path,required=True,
                        help='Directory containing MCF7_24h and K562_24h extraction outputs')
    parser.add_argument('--manifest-out',type=Path,required=True,
                        help='PRIVATE manifest destination: contains absolute local paths; do not publish')
    parser.add_argument('--extraction-receipt',type=Path,default=Path(root)/'receipts/v2_context_extraction.json')
    args=parser.parse_args(argv)
    target=args.manifest_out
    partial=target.with_name(target.name+'.part')
    if present(target):raise FileExistsError('Existing private manifest preserved; choose a new destination')
    if present(partial):raise FileExistsError('Existing private manifest partial preserved')
    manifest=prepare(args.context_root,args.extraction_receipt,root=root)
    target.parent.mkdir(parents=True,exist_ok=True)
    with partial.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    # Atomic no-clobber publication; any failure leaves the private partial intact.
    os.link(partial,target)
    partial.unlink()
    print(json.dumps({'private_manifest':True,'sources':2,'manifest_sha256':file_record(target)['sha256'],
                      'note':'Contains absolute local paths; keep this private manifest out of public commits'}))


if __name__=='__main__':
    main()
