"""Fetch only hash-pinned public source files; preserve existing mismatches."""
from pathlib import Path
import argparse, csv, hashlib, json, datetime, urllib.request
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--metadata-only',action='store_true');p.add_argument('--offline-verify',action='store_true');a=p.parse_args()
    rows=list(csv.DictReader((ROOT/'manifests/source_manifest.tsv').open(),delimiter='\t'));result=[]
    (ROOT/'data/raw').mkdir(parents=True,exist_ok=True)
    for r in rows:
        if a.metadata_only and '.RDS.' in r['filename']:continue
        target=ROOT/'data/raw'/r['filename'];cached=target.exists()
        if cached:
            if target.stat().st_size!=int(r['bytes']) or sha(target)!=r['sha256']:raise ValueError(f'Existing source mismatch preserved: {target}')
        else:
            if a.offline_verify:raise FileNotFoundError(target)
            part=target.with_suffix(target.suffix+'.part')
            req=urllib.request.Request(r['url'],headers={'User-Agent':'DrugPerturb/0.1 public-data-reproducibility'})
            with urllib.request.urlopen(req,timeout=90) as response,part.open('wb') as f:
                while chunk:=response.read(8*1024*1024):f.write(chunk)
            if part.stat().st_size!=int(r['bytes']) or sha(part)!=r['sha256']:raise ValueError(f'Download mismatch preserved: {part}')
            part.replace(target)
        result.append(dict(filename=r['filename'],sha256=r['sha256'],bytes=int(r['bytes']),reused_cached=cached))
    record={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':result,'public_sources_only':True}
    (ROOT/'receipts').mkdir(exist_ok=True);(ROOT/'receipts/download_verification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
