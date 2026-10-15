#!/usr/bin/env python3
"""Isolated Nextflow snapshot runner; test mode is wholly synthetic software input.

This is one computational process, not a modular scientific DAG. All real scripts
execute from copied snapshot paths, preserving their __file__-derived ROOT.
"""
from pathlib import Path
import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time

ASSETS = ('src','scripts','configs','locks','conf','containers','docs','main.nf',
          'nextflow.config','pyproject.toml','AGENTS.md','README.md')
PACKAGES = ('numpy','pandas','scipy','scikit-learn','rdkit','torch','PyYAML','joblib','matplotlib')
METADATA = 'GSM4150378_sciPlex3_A549_MCF7_K562_hashTable_metadata.txt.gz'
RDS = 'GSM4150378_sciPlex3_A549_24hrs.RDS.gz'
CORE = ['data/raw/'+METADATA] + ['data/processed/'+n for n in (
    'pseudobulk_counts.tsv.gz','well_qc.tsv','gene_metadata.tsv','extraction_summary.tsv')]
OPTIONAL = ['data/processed/'+n for n in ('selected_cell_qc.tsv.gz','cell_state_well_summary.tsv')]
def validate_manifest(manifest, mode):
    if manifest.get('schema_version') != 1 or manifest.get('mode') != mode:
        raise ValueError('Input manifest schema/mode gate failed')
    hashes=manifest.get('artifact_hashes')
    if not isinstance(hashes,dict) or not hashes:
        raise ValueError('Input manifest requires artifact hashes')
    required=set(CORE if mode=='core' else ['data/raw/'+METADATA,'data/raw/'+RDS])
    allowed=required | (set(OPTIONAL) if mode=='core' else set()) | {'manifests/source_manifest.tsv'}
    for relative, expected in hashes.items():
        if relative not in allowed or relative != Path(relative).as_posix():
            raise ValueError('Input is outside exact mode-specific allowlist: '+str(relative))
        if not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected):
            raise ValueError('Invalid SHA256: '+relative)
    if not required.issubset(hashes):
        raise ValueError('Required inputs missing: '+str(required-set(hashes)))
    return hashes



def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def safe_path(root, relative):
    if not isinstance(relative,str) or Path(relative).is_absolute():
        raise ValueError('Manifest paths must be project-relative strings')
    if relative != Path(relative).as_posix() or relative.startswith('./'):
        raise ValueError('Manifest paths must use canonical relative spelling: '+relative)
    target=root/relative
    if '..' in Path(relative).parts or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Manifest path escapes project: '+relative)
    return target


def runtime(rscript=None):
    versions={}
    for name in PACKAGES:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]='MISSING'
    result=dict(python=platform.python_version(),executable=str(Path(sys.executable).resolve()),
                executable_sha256=sha(Path(sys.executable).resolve()),packages=versions)
    if rscript:
        executable=shutil.which(rscript)
        if not executable: raise FileNotFoundError('Raw mode requires selected Rscript runtime')
        completed=subprocess.run([executable,'--version'],capture_output=True,text=True,check=True)
        result['Rscript']=dict(executable=str(Path(executable).resolve()),executable_sha256=sha(Path(executable).resolve()),
                               version=(completed.stdout+completed.stderr).strip())
    return result


def freeze_inputs(root,mode,out):
    root=root.resolve()
    paths=CORE+OPTIONAL if mode=='core' else ['data/raw/'+METADATA,'data/raw/'+RDS]
    required=set(CORE if mode=='core' else paths)
    hashes={}
    for relative in paths+['manifests/source_manifest.tsv']:
        path=safe_path(root,relative)
        if path.is_file(): hashes[relative]=sha(path)
        elif relative in required: raise FileNotFoundError(path)
    sources=[]
    source_manifest=root/'manifests/source_manifest.tsv'
    if source_manifest.is_file():
        with source_manifest.open() as stream: sources=list(csv.DictReader(stream,delimiter='\t'))
    value=dict(schema_version=1,mode=mode,artifact_hashes=hashes,upstream_sources=sources,
               frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
               scope='Input byte freeze only; not scientific validation')
    if out.exists(): raise FileExistsError('Preserve existing manifest; choose a new path: '+str(out))
    save(out,value)
    print(json.dumps(value,indent=2))


def verify_output(root):
    receipt=json.loads((root/'receipts/workflow_run.json').read_text())
    if receipt['status']!='complete' or not receipt['output_hashes']:
        raise ValueError('Workflow is incomplete')
    for relative,expected in {**receipt['output_hashes'],**receipt.get('input_hashes',{})}.items():
        path=safe_path(root,relative)
        if not path.is_file() or sha(path)!=expected:
            raise ValueError('Workflow output mismatch: '+relative)
    return receipt


def clone_sources(source,destination):
    if destination.exists(): raise FileExistsError('Preserve partial snapshot; use a fresh work directory')
    destination.mkdir(parents=True)
    for name in ASSETS:
        path=source/name
        if path.is_dir():
            shutil.copytree(path,destination/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        else: shutil.copy2(path,destination/name)
    for name in ('data/raw','data/processed','models','manifests','receipts','reports','logs','tmp'):
        (destination/name).mkdir(parents=True,exist_ok=True)


def source_hashes(root):
    result={}
    for name in ASSETS:
        path=root/name
        files=path.rglob('*') if path.is_dir() else [path]
        for item in files:
            if item.is_file() and '__pycache__' not in item.parts and item.suffix!='.pyc':
                result[str(item.relative_to(root))]=sha(item)
    return result


def check_runtime_lock(root,current):
    if not current['python'].startswith('3.12.'):
        raise ValueError('Workflow requires Python3.12')
    lock={}
    for line in (root/'locks/python-cpu.txt').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            name,version=line.split('==',1);lock[name.lower().replace('_','-')]=version
    for name,version in current['packages'].items():
        if version=='MISSING' or lock.get(name.lower().replace('_','-'))!=version:
            raise ValueError(f'Used package does not match pinned lock: {name}={version}')


def synthetic(root):
    # Imports occur only after source is copied and PYTHONPATH is isolated.
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from drugperturb.design import select_design,match_controls,qualify_wells
    from drugperturb.chemistry import standardize_compounds,split_by_structure,morgan_fingerprint,molecular_graph
    from drugperturb.response import build_responses,TrainingTargetTransform,baseline_predictions
    from drugperturb.models import MLP,MolecularGNN,fit_network,predict_network
    from drugperturb.evaluation import compound_metrics,summarize_compound_metrics,paired_scaffold_bootstrap
    rng=np.random.default_rng(2026);rows=[];counts={};gene_ids=[f'SYNTHETIC_GENE_{i}' for i in range(12)]
    base=np.arange(1,13)*20+100
    for rep in (1,2):
        plate=f'SYNTHETIC_PLATE_{rep}'
        for i in range(25):
            well=f'SYNTHETIC_WELL_{i:02d}_';combo=well+plate
            rows.append(dict(cell_type='SYNTHETIC_CELL',time_point=24,replicate=f'rep{rep}',
                             plate_oligo=plate,dose_pattern='2',well_oligo=well,Combo=combo,
                             treatment='vehicle' if i==0 else f'SYNTHETIC_{i:02d}',
                             vehicle=i==0,dose=0 if i==0 else 1000))
            means=base if i==0 else base*np.exp(0.02*i*np.sin(np.arange(12)))
            counts[combo]=rng.poisson(means)+1
    metadata=pd.DataFrame(rows);drugs,controls=select_design(metadata,cell_type='SYNTHETIC_CELL')
    mapping=match_controls(drugs,controls)
    qualified=qualify_wells(mapping,{name:40 for name in counts},minimum_cells=30)
    responses=build_responses(pd.DataFrame(counts,index=gene_ids),qualified)
    records=standardize_compounds([dict(compound_id=f'SYNTHETIC_{i:02d}',smiles='C'*i) for i in range(1,25)])
    divided,split_report=split_by_structure(records,min_test_compounds=2,min_test_groups=2)
    divided=pd.DataFrame(divided).sort_values('compound_id').reset_index(drop=True)
    if responses['compound_ids']!=divided.compound_id.tolist(): raise ValueError('Synthetic ordering mismatch')
    divided.to_csv(root/'manifests/split_manifest.tsv',sep='\t',index=False)
    qualified.to_csv(root/'manifests/control_mapping.tsv',sep='\t',index=False)
    metadata.to_csv(root/'data/processed/synthetic_metadata.tsv',sep='\t',index=False)
    y=responses['delta'];fp=np.stack([morgan_fingerprint(s) for s in divided.canonical_smiles])
    graphs=[molecular_graph(s) for s in divided.canonical_smiles]
    masks={name:divided.split.eq(name).to_numpy() for name in ('train','validation','test')}
    transform=TrainingTargetTransform(8,3).fit(y[masks['train']])
    train_y=transform.select_truth(y[masks['train']]);val_y=transform.select_truth(y[masks['validation']])
    train_z=transform.transform(y[masks['train']]);val_z=transform.transform(y[masks['validation']])
    predictions=baseline_predictions(train_y,fp[masks['train']],fp[masks['test']])
    ridge=Ridge(alpha=1).fit(fp[masks['train']],train_z)
    predictions['M1']=transform.inverse_transform(ridge.predict(fp[masks['test']]))
    training=[]
    for method in ('M2','M3'):
        if method=='M2':
            factory=lambda:MLP(2048,train_z.shape[1],hidden_dim=8,dropout=0)
            inputs={s:fp[mask] for s,mask in masks.items()}
        else:
            factory=lambda:MolecularGNN(output_dim=train_z.shape[1],hidden_dim=8,dropout=0)
            inputs={s:[graphs[i] for i in np.flatnonzero(mask)] for s,mask in masks.items()}
        scorer=lambda p:float(np.sqrt(np.mean((transform.inverse_transform(p)-val_y)**2,axis=1)).mean())
        fitted=fit_network(factory,inputs['train'],train_z,inputs['validation'],val_z,
                           seed=42,lr=.01,max_epochs=12,patience=4,batch_size=8,validation_scorer=scorer)
        predictions[method]=transform.inverse_transform(predict_network(fitted['model'],inputs['test']))
        pd.DataFrame(fitted['history']).to_csv(root/f'receipts/{method}_synthetic_history.tsv',sep='\t',index=False)
        training.append(dict(method=method,best_epoch=fitted['best_epoch'],test_labels_used_for_training=False))
    ids=divided.loc[masks['test'],'compound_id'].tolist();groups=divided.loc[masks['test'],'scaffold_group'].tolist()
    np.savez_compressed(root/'models/test_predictions.npz',ids=np.array(ids),**predictions)
    save(root/'receipts/prediction_seal.json',dict(synthetic=True,test_labels_used_for_training=False,
         prediction_sha256=sha(root/'models/test_predictions.npz'),models=training))
    # Only after sealing every model's predictions do we form the test truth.
    truth=transform.select_truth(y[masks['test']]);tables=[];summary={}
    for method,prediction in predictions.items():
        table=compound_metrics(truth,prediction,ids,groups);summary[method]=summarize_compound_metrics(table)
        table.insert(0,'method',method);tables.append(table)
    metrics=pd.concat(tables,ignore_index=True)
    metrics.to_csv(root/'reports/synthetic_metrics.tsv',sep='\t',index=False)
    comparison=paired_scaffold_bootstrap(truth,predictions['M3'],predictions['B1'],ids,groups)
    comparison.pop('bootstrap_differences')
    save(root/'reports/synthetic_summary.json',dict(synthetic=True,metrics=summary,comparison=comparison))
    (root/'reports/synthetic_report.html').write_text('<!doctype html><meta charset="utf-8"><h1>SYNTHETIC SOFTWARE TEST — NOT BIOLOGICAL EVIDENCE</h1>'+metrics.to_html(index=False))
    return dict(synthetic=True,independent_biological_validation=False,n_compounds=24,n_genes=12,
                selected_genes=8,split=split_report,methods=list(predictions),
                matched_control_wells=int(qualified.control_Combo.nunique()),test_labels_used_for_training=False)


def real_chain(root,mode,manifest_path,staged,settings):
    manifest=json.loads(manifest_path.read_text())
    hashes=validate_manifest(manifest,mode)
    for relative,expected in hashes.items():
        destination=safe_path(root,relative);source=staged/Path(relative).name
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
        if sha(destination)!=expected: raise ValueError('Input hash mismatch: '+relative)
    required=CORE if mode=='core' else ['data/raw/'+METADATA,'data/raw/'+RDS]
    if not set(required).issubset(hashes): raise ValueError('Required inputs missing: '+str(set(required)-set(hashes)))
    # Preserve full upstream source provenance even when core mode stages only metadata.
    save(root/'receipts/workflow_input_manifest.json',manifest)
    if (root/'manifests/source_manifest.tsv').exists():
        shutil.copy2(root/'manifests/source_manifest.tsv',root/'receipts/upstream_source_manifest.tsv')
    scripts=['prepare_design.py','prepare_dataset.py','train.py','development_diagnostics.py','score.py','sensitivity.py','report.py']
    for script in scripts:
        if not (root/'scripts'/script).is_file(): raise FileNotFoundError('Required chain script absent: '+script)
    commands=[[sys.executable,'scripts/prepare_design.py']]
    if mode=='raw':
        rscript=settings.get('rscript','Rscript')
        if not shutil.which(rscript): raise FileNotFoundError('Raw mode requires the explicitly selected Rscript runtime')
        commands.append([rscript,'scripts/export_rds.R','data/raw/'+RDS,'manifests/selected_wells.tsv','data/processed'])
    for script in scripts[1:]:
        command=[sys.executable,'scripts/'+script]
        if script in ('train.py','development_diagnostics.py'):
            command+=['--source-commit',settings.get('source_commit','UNCOMMITTED_SOURCE_CONTENT_HASHED')]
        commands.append(command)
    records=[]
    for index,command in enumerate(commands):
        started=time.perf_counter();log=root/f'logs/{index:02d}_{Path(command[1]).stem}.log'
        with log.open('w') as stream:
            completed=subprocess.run(command,cwd=root,env=os.environ.copy(),stdout=stream,stderr=subprocess.STDOUT)
        records.append(dict(command=command,exit_code=completed.returncode,elapsed_seconds=time.perf_counter()-started,log=str(log.relative_to(root))))
        save(root/'receipts/workflow_steps.json',records)
        if completed.returncode: raise RuntimeError('Workflow step failed; preserved log: '+str(log))
    return dict(synthetic=False,mode=mode,commands=records,upstream_sources=manifest.get('upstream_sources',[]),
                optional_report_context=[p for p in OPTIONAL if p in hashes],
                diagnostics_and_sensitivity_included=True)


def run(args):
    root=args.snapshot.resolve();settings=json.loads(args.settings)
    clone_sources(args.source_root.resolve(),root)
    sys.path.insert(0,str(root/'src'))
    os.environ.update(PYTHONPATH=str(root/'src'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
                      OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TMPDIR=str(root/'tmp'))
    current=runtime(settings.get('rscript','Rscript') if args.mode=='raw' else None);check_runtime_lock(root,current)
    initial=source_hashes(root);started=time.perf_counter()
    try:
        result=synthetic(root) if args.mode=='test' else real_chain(root,args.mode,args.input_manifest.resolve(),args.staged_inputs.resolve(),settings)
        if source_hashes(root)!=initial: raise ValueError('Source/config files changed during workflow')
        # Inputs and logs are retained in task work. Hash outputs, sources and receipts;
        # exclude raw source bytes already bound by the immutable input manifest.
        outputs={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
                 if p.is_file() and p.relative_to(root).parts[0]!='tmp'
                 and 'data/raw' not in str(p.relative_to(root))
                 and '__pycache__' not in p.parts and p.suffix!='.pyc'}
        input_hashes={}
        if args.mode!='test':
            declared=json.loads((root/'receipts/workflow_input_manifest.json').read_text())['artifact_hashes']
            input_hashes={p:h for p,h in declared.items() if p.startswith('data/raw/')}
        receipt=dict(input_hashes=input_hashes,status='complete',mode=args.mode,synthetic=args.mode=='test',
                     settings=settings,runtime=current,source_hashes=initial,output_hashes=outputs,
                     result=result,elapsed_seconds=time.perf_counter()-started,
                     scope='Synthetic software validation only' if args.mode=='test' else 'Isolated same-public-data computational replay; no external validation')
        save(root/'receipts/workflow_run.json',receipt);verify_output(root)
        print(json.dumps({k:v for k,v in receipt.items() if k not in ('source_hashes','output_hashes')},indent=2))
    except Exception as error:
        save(root/'receipts/workflow_failed.json',dict(status='failed',error=repr(error),settings=settings,
             source_hashes=initial,runtime=current,elapsed_seconds=time.perf_counter()-started))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-json',action='store_true');parser.add_argument('--rscript');parser.add_argument('--verify-output',type=Path)
    parser.add_argument('--freeze-inputs',choices=['core','raw']);parser.add_argument('--manifest-out',type=Path)
    parser.add_argument('--mode',choices=['test','core','raw']);parser.add_argument('--source-root',type=Path,default=Path('.'))
    parser.add_argument('--snapshot',type=Path);parser.add_argument('--input-manifest',type=Path)
    parser.add_argument('--staged-inputs',type=Path,default=Path('inputs'));parser.add_argument('--settings',default='{}')
    args=parser.parse_args()
    if args.runtime_json: print(json.dumps(runtime(args.rscript),sort_keys=True));return
    if args.verify_output: verify_output(args.verify_output.resolve());print('All output hashes verified');return
    if args.freeze_inputs:
        if not args.manifest_out: parser.error('--freeze-inputs requires --manifest-out')
        freeze_inputs(args.source_root,args.freeze_inputs,args.manifest_out);return
    if not args.mode or not args.snapshot: parser.error('execution requires --mode and --snapshot')
    run(args)

if __name__=='__main__':main()
