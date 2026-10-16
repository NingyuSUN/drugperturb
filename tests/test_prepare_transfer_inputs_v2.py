"""Prepare manifests from tiny pinned byte fixtures without parsing expression."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/prepare_transfer_inputs_v2.py'
    spec=importlib.util.spec_from_file_location('prepare_transfer_inputs_v2',path)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded)
    return loaded


def record(path):
    return {'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def inputs(tmp_path):
    root=tmp_path/'public';contexts=tmp_path/'derived'
    code=['scripts/export_context_rds.R','scripts/csc_utils.R',
          'configs/protocol_v2.yaml','manifests/v2/well_mapping.tsv']
    for relative in code:
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(relative.encode())
    (root/'receipts').mkdir()
    acquisition={'schema_version':1,'sources':[]}
    extraction={'schema_version':1,'sources':{}}
    names=['pseudobulk_counts.tsv.gz','gene_metadata.tsv','well_qc.tsv',
           'selected_cell_qc.tsv.gz','extraction_summary.tsv']
    for cell in ['MCF7','K562']:
        directory=contexts/(cell+'_24h');directory.mkdir(parents=True)
        for name in names[:-1]:
            # Deliberately invalid gzip/numeric payload: this step must not interpret it.
            (directory/name).write_bytes((cell+' opaque bytes '+name).encode())
        summary={'context_cell_type':cell,'context_time_hours':'24',
            'all_human_gene_rows_exported':'TRUE','UMI_conserved':'TRUE',
            'source_hash_QC_all_pass':'TRUE','counts_nonnegative_integer':'TRUE',
            'author_embeddings_reused':'FALSE','treatment_effects_computed':'FALSE',
            'counts_gene_dimnames_present':'TRUE','counts_cell_dimnames_present':'TRUE',
            'row_positional_binding_scope':'Exact assay Dimnames and rowRanges gene_id order',
            'col_positional_binding_scope':'Exact assay Dimnames and colData cell_id order'}
        with (directory/names[-1]).open('w',newline='') as stream:
            writer=csv.writer(stream,delimiter='\t');writer.writerow(['key','value']);writer.writerows(summary.items())
        raw={'basename':f'GSM4150378_sciPlex3_{cell}_24hrs.RDS.gz',
             'bytes':111,'sha256':hashlib.sha256(cell.encode()).hexdigest()}
        acquisition['sources'].append({'filename':raw['basename'],'bytes':raw['bytes'],
            'sha256':raw['sha256'],'gzip_integrity_checked':True,
            'url':'https://ftp.ncbi.nlm.nih.gov/geo/test/'+raw['basename']})
        extraction['sources'][cell]={'status':'COMPLETE','source':raw,
            'treatment_effects_computed':False,'summary':summary,
            'code_input_hashes':{relative:record(root/relative)['sha256'] for relative in code},
            'outputs':{name:record(directory/name) for name in names}}
    (root/'receipts/v2_source_acquisition.json').write_text(json.dumps(acquisition))
    receipt=root/'receipts/v2_context_extraction.json';receipt.write_text(json.dumps(extraction))
    return root,contexts,receipt,extraction,acquisition


def run(inputs,tmp_path,**kwargs):
    root,contexts,receipt,_,_=inputs
    target=tmp_path/'private'/'target_manifest.json'
    module().main(['--context-root',str(contexts),'--manifest-out',str(target),
                   '--extraction-receipt',str(receipt)],root=root,**kwargs)
    return target


def test_cli_emits_the_exact_m3_schema_with_explicit_absolute_paths(inputs,tmp_path,capsys):
    root,contexts,receipt,extraction,_=inputs
    target=run(inputs,tmp_path)
    manifest=json.loads(target.read_text())
    assert set(manifest)=={'schema_version','sources'}
    assert manifest['schema_version']==1 and set(manifest['sources'])=={'MCF7','K562'}
    names={'counts':'pseudobulk_counts.tsv.gz','gene_metadata':'gene_metadata.tsv',
           'well_qc':'well_qc.tsv','extraction_summary':'extraction_summary.tsv'}
    for cell,entry in manifest['sources'].items():
        assert set(entry)=={*names,'raw_source','extractor_code','acquisition_receipt','extraction_receipt'}
        for key,name in names.items():
            path=contexts/(cell+'_24h')/name
            assert entry[key]=={'path':str(path.resolve()),**record(path)}
        assert entry['raw_source']==extraction['sources'][cell]['source']
        assert entry['extractor_code']=={'path':str((root/'scripts/export_context_rds.R').resolve()),**record(root/'scripts/export_context_rds.R')}
        assert entry['acquisition_receipt']=={'path':str((root/'receipts/v2_source_acquisition.json').resolve()),**record(root/'receipts/v2_source_acquisition.json')}
        assert entry['extraction_receipt']=={'path':str(receipt.resolve()),**record(receipt)}
    assert not target.with_name(target.name+'.part').exists()
    assert 'private' in capsys.readouterr().out.lower()


@pytest.mark.parametrize('name',['pseudobulk_counts.tsv.gz','gene_metadata.tsv','well_qc.tsv',
                               'selected_cell_qc.tsv.gz','extraction_summary.tsv'])
@pytest.mark.parametrize('change',['size','hash'])
def test_every_derived_output_is_checked_including_non_manifest_cell_qc(inputs,tmp_path,name,change):
    _,contexts,_,_,_=inputs
    path=contexts/'MCF7_24h'/name;original=path.read_bytes()
    path.write_bytes(original+b'changed' if change=='size' else b'X'+original[1:])
    with pytest.raises(ValueError,match='output.*mismatch'):
        run(inputs,tmp_path)
    assert not (tmp_path/'private/target_manifest.json').exists()


@pytest.mark.parametrize('change',['raw_hash','raw_cell','incomplete','effects','missing_cell',
                                  'missing_code','changed_code'])
def test_invalid_reference_source_or_execution_scope_is_rejected(inputs,tmp_path,change):
    root,_,receipt,extraction,_=inputs
    cell=extraction['sources']['MCF7']
    if change=='raw_hash':cell['source']['sha256']='0'*64
    elif change=='raw_cell':cell['source']['basename']='GSM4150378_sciPlex3_K562_24hrs.RDS.gz'
    elif change=='incomplete':cell['status']='RUNNING'
    elif change=='effects':cell['treatment_effects_computed']=True
    elif change=='missing_cell':del extraction['sources']['K562']
    elif change=='missing_code':del cell['code_input_hashes']['scripts/csc_utils.R']
    elif change=='changed_code':(root/'scripts/csc_utils.R').write_bytes(b'changed')
    receipt.write_text(json.dumps(extraction))
    with pytest.raises(ValueError):run(inputs,tmp_path)


def test_unchecked_acquisition_source_is_rejected(inputs,tmp_path):
    root,_,_,_,acquisition=inputs
    acquisition['sources'][0]['gzip_integrity_checked']=False
    (root/'receipts/v2_source_acquisition.json').write_text(json.dumps(acquisition))
    with pytest.raises(ValueError):run(inputs,tmp_path)


@pytest.mark.parametrize('change',['wrong_cell','wrong_time','failed_flag','effects','duplicate_key'])
def test_rehashed_bad_summary_does_not_bypass_scope_checks(inputs,tmp_path,change):
    _,contexts,receipt,extraction,_=inputs
    source=extraction['sources']['MCF7'];summary=dict(source['summary'])
    if change=='wrong_cell':summary['context_cell_type']='K562'
    elif change=='wrong_time':summary['context_time_hours']='72'
    elif change=='failed_flag':summary['UMI_conserved']='FALSE'
    elif change=='effects':summary['treatment_effects_computed']='TRUE'
    path=contexts/'MCF7_24h/extraction_summary.tsv'
    with path.open('w',newline='') as stream:
        writer=csv.writer(stream,delimiter='\t');writer.writerow(['key','value']);writer.writerows(summary.items())
        if change=='duplicate_key':writer.writerow(['context_cell_type','MCF7'])
    source['summary']=summary;source['outputs']['extraction_summary.tsv']=record(path)
    receipt.write_text(json.dumps(extraction))
    with pytest.raises(ValueError,match='summary|context|effect|check'):run(inputs,tmp_path)


def test_existing_output_or_manifest_partial_is_preserved_before_input_reads(tmp_path):
    target=tmp_path/'manifest.json';target.write_bytes(b'original manifest')
    with pytest.raises(FileExistsError):
        module().main(['--context-root',str(tmp_path/'missing'),'--manifest-out',str(target)],root=tmp_path/'missing')
    assert target.read_bytes()==b'original manifest'
    partial=tmp_path/'new_manifest.json.part';partial.write_bytes(b'partial evidence')
    with pytest.raises(FileExistsError):
        module().main(['--context-root',str(tmp_path/'missing'),'--manifest-out',str(tmp_path/'new_manifest.json')],root=tmp_path/'missing')
    assert partial.read_bytes()==b'partial evidence'


def test_derived_partial_is_not_silently_ignored(inputs,tmp_path):
    _,contexts,_,_,_=inputs
    partial=contexts/'K562_24h/pseudobulk_counts.tsv.gz.part';partial.write_bytes(b'evidence')
    with pytest.raises(FileExistsError,match='partial'):run(inputs,tmp_path)
    assert partial.read_bytes()==b'evidence'


def test_reference_summary_must_agree_with_the_pinned_summary_file(inputs,tmp_path):
    _,_,receipt,extraction,_=inputs
    extraction['sources']['MCF7']['summary']['extra_claim']='not in file'
    receipt.write_text(json.dumps(extraction))
    with pytest.raises(ValueError,match='summary'):run(inputs,tmp_path)
