"""Pinned acquisition tests use tiny files and mocked HTTP only."""
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/fetch_v2_sources.py'
    spec=importlib.util.spec_from_file_location('fetch_v2_sources',path)
    loaded=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture
def catalog(tmp_path):
    root=tmp_path/'catalog'
    (root/'receipts').mkdir(parents=True)
    (root/'manifests/v2').mkdir(parents=True)
    bodies={f'c{i}.bin':f'context{i}'.encode() for i in [1,2]}
    bodies.update({f'l{i}.bin':f'lincs{i}'.encode() for i in [1,2,3,4]})
    def row(name):
        value=bodies[name]
        return {'filename':name,'url':'https://ftp.ncbi.nlm.nih.gov/geo/test/'+name,
                'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()}
    context=[row('c1.bin'),row('c2.bin')]
    # Extra acquisition fields must not leak into a new public verification receipt.
    (root/'receipts/v2_source_acquisition.json').write_text(json.dumps({
        'schema_version':1,'sources':[dict(x,unused_operational_note='/private/location') for x in context]}))
    with (root/'manifests/v2/lincs_sources.tsv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['filename','url','bytes','sha256'],delimiter='\t')
        writer.writeheader();writer.writerows([row(f'l{i}.bin') for i in [1,2,3,4]])
    return root,bodies


def no_http(*args,**kwargs):
    raise AssertionError('HTTP must not be used in this case')


def test_loads_each_existing_record_without_a_second_fact_manifest(catalog):
    root,_=catalog;fetch=module()
    context,path=fetch.load_sources('context',root)
    lincs,lincs_path=fetch.load_sources('lincs',root)
    assert [x['filename'] for x in context]==['c1.bin','c2.bin']
    assert [x['filename'] for x in lincs]==['l1.bin','l2.bin','l3.bin','l4.bin']
    assert path.name=='v2_source_acquisition.json' and lincs_path.name=='lincs_sources.tsv'
    assert set(context[0])=={'filename','url','bytes','sha256'}


@pytest.mark.parametrize('group',['context','lincs'])
def test_successful_stream_is_hash_checked_before_publication(tmp_path,catalog,monkeypatch,group):
    root,bodies=catalog;fetch=module();records,_=fetch.load_sources(group,root)
    calls=[]
    def response(request,timeout):
        calls.append(request.full_url)
        return io.BytesIO(bodies[request.full_url.rsplit('/',1)[1]])
    monkeypatch.setattr(fetch.urllib.request,'urlopen',response)
    output=tmp_path/'downloads'
    result=fetch.fetch_sources(records,output)
    assert len(result)==len(records) and len(calls)==len(records)
    for row in records:
        assert (output/row['filename']).read_bytes()==bodies[row['filename']]
        assert not (output/(row['filename']+'.part')).exists()


def test_valid_cached_sources_are_reused_without_http(tmp_path,catalog,monkeypatch):
    root,bodies=catalog;fetch=module();records,_=fetch.load_sources('context',root)
    output=tmp_path/'cached';output.mkdir()
    for row in records:(output/row['filename']).write_bytes(bodies[row['filename']])
    monkeypatch.setattr(fetch.urllib.request,'urlopen',no_http)
    assert len(fetch.fetch_sources(records,output))==2
    assert len(fetch.fetch_sources(records,output,offline_verify=True))==2


@pytest.mark.parametrize('kind',['size','hash'])
def test_existing_mismatch_is_preserved_and_no_other_source_is_requested(tmp_path,catalog,monkeypatch,kind):
    root,bodies=catalog;fetch=module();records,_=fetch.load_sources('context',root)
    output=tmp_path/'cached';output.mkdir()
    # First source is absent; failure in the second must be found before any HTTP.
    original=b'bad' if kind=='size' else b'XXXXXXXX'
    target=output/records[1]['filename'];target.write_bytes(original)
    monkeypatch.setattr(fetch.urllib.request,'urlopen',no_http)
    with pytest.raises(ValueError,match='mismatch'):
        fetch.fetch_sources(records,output)
    assert target.read_bytes()==original
    assert not (output/(records[0]['filename']+'.part')).exists()


@pytest.mark.parametrize('valid_target',[False,True])
def test_existing_partial_is_preserved_even_beside_valid_target(tmp_path,catalog,monkeypatch,valid_target):
    root,bodies=catalog;fetch=module();records,_=fetch.load_sources('context',root)
    output=tmp_path/'cached';output.mkdir()
    name=records[0]['filename'];partial=output/(name+'.part');partial.write_bytes(b'partial evidence')
    if valid_target:(output/name).write_bytes(bodies[name])
    monkeypatch.setattr(fetch.urllib.request,'urlopen',no_http)
    with pytest.raises(FileExistsError,match='partial'):
        fetch.fetch_sources(records,output)
    assert partial.read_bytes()==b'partial evidence'


def test_offline_missing_source_never_requests_http_or_creates_partial(tmp_path,catalog,monkeypatch):
    root,_=catalog;fetch=module();records,_=fetch.load_sources('context',root)
    output=tmp_path/'missing'
    monkeypatch.setattr(fetch.urllib.request,'urlopen',no_http)
    with pytest.raises(FileNotFoundError):fetch.fetch_sources(records,output,offline_verify=True)
    assert not output.exists()


@pytest.mark.parametrize('body',[b'XXXXXXXX',b'short',b'much too long to match'])
def test_bad_download_leaves_partial_and_never_publishes_target(tmp_path,catalog,monkeypatch,body):
    root,_=catalog;fetch=module();records,_=fetch.load_sources('context',root)
    monkeypatch.setattr(fetch.urllib.request,'urlopen',lambda *a,**k:io.BytesIO(body))
    output=tmp_path/'downloads'
    with pytest.raises(ValueError,match='mismatch'):fetch.fetch_sources(records[:1],output)
    assert not (output/records[0]['filename']).exists()
    assert (output/(records[0]['filename']+'.part')).read_bytes()==body


def test_interrupted_transfer_preserves_received_bytes(tmp_path,catalog,monkeypatch):
    root,_=catalog;fetch=module();records,_=fetch.load_sources('context',root)
    class Interrupted(io.BytesIO):
        def read(self,size=-1):
            if self.tell():raise OSError('simulated connection failure')
            return super().read(3)
    monkeypatch.setattr(fetch.urllib.request,'urlopen',lambda *a,**k:Interrupted(b'context1'))
    output=tmp_path/'downloads'
    with pytest.raises(OSError):fetch.fetch_sources(records[:1],output)
    assert (output/'c1.bin.part').read_bytes()==b'con'
    assert not (output/'c1.bin').exists()


def test_new_target_appearing_during_download_is_not_overwritten(tmp_path,catalog,monkeypatch):
    root,bodies=catalog;fetch=module();records,_=fetch.load_sources('context',root)
    output=tmp_path/'downloads'
    def response(*args,**kwargs):
        (output/'c1.bin').write_bytes(b'other process data')
        return io.BytesIO(bodies['c1.bin'])
    monkeypatch.setattr(fetch.urllib.request,'urlopen',response)
    with pytest.raises(FileExistsError):fetch.fetch_sources(records[:1],output)
    assert (output/'c1.bin').read_bytes()==b'other process data'
    assert (output/'c1.bin.part').read_bytes()==bodies['c1.bin']


@pytest.mark.parametrize('field,value',[('filename','../escape'),('filename','x\\escape'),
    ('url','file:///private/data'),('sha256','not-a-hash'),('bytes',0)])
def test_invalid_catalog_entry_fails_before_any_download(catalog,field,value):
    root,_=catalog;fetch=module()
    path=root/'receipts/v2_source_acquisition.json'
    content=json.loads(path.read_text());content['sources'][0][field]=value
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError):fetch.load_sources('context',root)


def test_explicit_cli_writes_minimal_path_free_receipt_and_reuses_it(tmp_path,catalog,monkeypatch):
    root,bodies=catalog;fetch=module();output=tmp_path/'downloads';output.mkdir()
    for name in ['c1.bin','c2.bin']:(output/name).write_bytes(bodies[name])
    monkeypatch.setattr(fetch.urllib.request,'urlopen',no_http)
    receipt=tmp_path/'public'/'verification.json'
    args=['--group','context','--output-dir',str(output),'--receipt',str(receipt),'--offline-verify']
    fetch.main(args,root=root)
    original=receipt.read_bytes();record=json.loads(original)
    assert record['group']=='context' and len(record['files'])==2
    assert set(record['files'][0])=={'basename','url','bytes','sha256'}
    assert record['source_record']['basename']=='v2_source_acquisition.json'
    assert '/private/location' not in original.decode() and str(tmp_path) not in original.decode()
    fetch.main(args,root=root)
    assert receipt.read_bytes()==original


def test_cli_preserves_mismatched_receipt_before_any_http(tmp_path,catalog,monkeypatch):
    root,_=catalog;fetch=module();receipt=tmp_path/'verification.json'
    receipt.write_bytes(b'prior unrelated result')
    monkeypatch.setattr(fetch.urllib.request,'urlopen',no_http)
    with pytest.raises(ValueError,match='receipt'):
        fetch.main(['--group','context','--output-dir',str(tmp_path/'downloads'),
                    '--receipt',str(receipt)],root=root)
    assert receipt.read_bytes()==b'prior unrelated result'
