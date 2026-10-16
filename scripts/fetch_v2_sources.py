"""Acquire hash-pinned v2 sources from existing public provenance records.

Context files use receipts/v2_source_acquisition.json; LINCS metadata uses
manifests/v2/lincs_sources.tsv. No duplicate source-fact manifest is maintained.
The v1 A549/metadata acquisition command remains separate.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import re
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
CHUNK_BYTES=8*1024*1024


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def _catalog_path(group,root):
    if group=='context':return Path(root)/'receipts/v2_source_acquisition.json'
    if group=='lincs':return Path(root)/'manifests/v2/lincs_sources.tsv'
    raise ValueError('Unknown source group')


def _normalize(rows):
    records=[]
    for row in rows:
        if not isinstance(row,dict):raise ValueError('Invalid source entry')
        name=row.get('filename')
        if (not isinstance(name,str) or not name or name in ('.','..') or
                any(c in name for c in '/\\\0\r\n\t') or Path(name).name!=name):
            raise ValueError('Source filename must be a plain basename')
        url=row.get('url')
        if not isinstance(url,str):raise ValueError('Missing source URL')
        parsed=urllib.parse.urlsplit(url)
        if (parsed.scheme!='https' or parsed.hostname!='ftp.ncbi.nlm.nih.gov' or
                parsed.username or parsed.password or not parsed.path.startswith('/geo/') or
                parsed.query or parsed.fragment or urllib.parse.unquote(parsed.path.rsplit('/',1)[-1])!=name):
            raise ValueError('Expected the exact public GEO source URL')
        size=row.get('bytes')
        if isinstance(size,str) and re.fullmatch(r'[0-9]+',size):size=int(size)
        if isinstance(size,bool) or not isinstance(size,int) or size<1:
            raise ValueError('Source byte count must be a positive integer')
        digest=row.get('sha256')
        if not isinstance(digest,str) or not re.fullmatch(r'[0-9a-f]{64}',digest):
            raise ValueError('Invalid pinned source SHA256')
        records.append(dict(filename=name,url=url,bytes=size,sha256=digest))
    if not records or len({r['filename'] for r in records})!=len(records):
        raise ValueError('Source records must be nonempty with unique basenames')
    return records


def load_sources(group,root=ROOT):
    """Read the existing authoritative record, copying scientific file facts only."""
    path=_catalog_path(group,root)
    if group=='context':
        payload=json.loads(path.read_text(encoding='utf-8'))
        if payload.get('schema_version')!=1 or not isinstance(payload.get('sources'),list):
            raise ValueError('Invalid context acquisition record')
        rows=payload['sources']
    else:
        with path.open(newline='',encoding='utf-8') as stream:
            rows=list(csv.DictReader(stream,delimiter='\t'))
    return _normalize(rows),path


def _present(path):
    return path.exists() or path.is_symlink()


def _publish_new(partial,target):
    # Same-directory hard-link publication is atomic and cannot replace a file
    # created by another process. Unsupported filesystems fail with .part intact.
    try:
        os.link(partial,target)
    except FileExistsError as exc:
        raise FileExistsError(f'New target appeared; files preserved: {target.name}') from exc
    except OSError as exc:
        raise OSError(f'Atomic publication failed; partial preserved: {partial.name}') from exc
    partial.unlink()


def fetch_sources(records,output_dir,*,offline_verify=False):
    """Reuse matching bytes; preserve every pre-existing mismatch or partial."""
    records=_normalize(records)
    output=Path(output_dir)
    missing=[]
    # Preflight the whole group before starting a potentially large transfer.
    for row in records:
        target=output/row['filename'];partial=output/(row['filename']+'.part')
        if _present(partial):
            raise FileExistsError(f'Existing partial preserved: {partial.name}')
        if _present(target):
            if not target.is_file() or target.stat().st_size!=row['bytes'] or sha(target)!=row['sha256']:
                raise ValueError(f'Existing source mismatch preserved: {target.name}')
        elif offline_verify:
            raise FileNotFoundError(f'Offline source missing: {target.name}')
        else:
            missing.append(row)
    if missing:output.mkdir(parents=True,exist_ok=True)
    for row in missing:
        target=output/row['filename'];partial=output/(row['filename']+'.part')
        request=urllib.request.Request(row['url'],headers={'User-Agent':'DrugPerturb/0.2 public-source-reproduction'})
        digest=hashlib.sha256();received=0
        # Exclusive creation prevents accidental resumption or clobbering partial evidence.
        with partial.open('xb') as stream:
            with urllib.request.urlopen(request,timeout=90) as response:
                while block:=response.read(CHUNK_BYTES):
                    stream.write(block);digest.update(block);received+=len(block)
                    if received>row['bytes']:
                        raise ValueError(f'Download size mismatch preserved: {partial.name}')
            stream.flush();os.fsync(stream.fileno())
        if received!=row['bytes'] or digest.hexdigest()!=row['sha256']:
            raise ValueError(f'Download mismatch preserved: {partial.name}')
        _publish_new(partial,target)
    return records


def _check_receipt(path,record):
    if _present(path.with_name(path.name+'.part')):
        raise FileExistsError(f'Existing receipt partial preserved: {path.name}.part')
    if not _present(path):return False
    try:
        same=path.is_file() and json.loads(path.read_text(encoding='utf-8'))==record
    except (ValueError,UnicodeError):
        same=False
    if not same:raise ValueError(f'Existing receipt mismatch preserved: {path.name}')
    return True


def main(argv=None,*,root=ROOT):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--group',choices=['context','lincs'],required=True)
    parser.add_argument('--offline-verify',action='store_true')
    args=parser.parse_args(argv)
    catalog=_catalog_path(args.group,root)
    original_hash=sha(catalog)
    records,_=load_sources(args.group,root)
    protected={catalog.resolve()}
    for row in records:
        target=args.output_dir/row['filename']
        protected.update([target.resolve(),target.with_name(target.name+'.part').resolve()])
    if args.receipt.resolve() in protected:
        raise ValueError('Verification receipt must be separate from source inputs')
    record=dict(schema_version=1,group=args.group,
        source_record={'basename':catalog.name,'sha256':original_hash,'bytes':catalog.stat().st_size},
        files=[{'basename':r['filename'],'url':r['url'],'bytes':r['bytes'],'sha256':r['sha256']} for r in records])
    _check_receipt(args.receipt,record)
    fetch_sources(records,args.output_dir,offline_verify=args.offline_verify)
    if sha(catalog)!=original_hash:raise ValueError('Source provenance record changed during acquisition')
    if not _check_receipt(args.receipt,record):
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        partial=args.receipt.with_name(args.receipt.name+'.part')
        with partial.open('x',encoding='utf-8') as stream:
            stream.write(json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
        _publish_new(partial,args.receipt)
    print(json.dumps({'group':args.group,'files_verified':len(records),'offline_verify':args.offline_verify}))


if __name__=='__main__':
    main()
