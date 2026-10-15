"""Freeze eligibility/splits, then fit target representation on training only."""
from pathlib import Path
import hashlib, json, datetime
import numpy as np
import pandas as pd
import yaml, joblib
from drugperturb.design import qualify_wells
from drugperturb.chemistry import split_by_structure, morgan_fingerprint, molecular_graph
from drugperturb.response import build_responses, TrainingTargetTransform

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    out=ROOT/'data/processed'; receipt=ROOT/'receipts/dataset_freeze.json'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        for f,h in old['artifact_hashes'].items():
            if sha(ROOT/f)!=h:raise ValueError(f'Frozen artifact changed: {f}')
        print('Existing frozen dataset verified; no rebuild.');return
    protocol=yaml.safe_load((ROOT/'configs/protocol_v1.yaml').read_text())
    if protocol['status']!='FROZEN_BEFORE_TRAINING':raise ValueError('Freeze protocol first')
    mapping=pd.read_csv(ROOT/'manifests/control_mapping.tsv',sep='\t',dtype={'dose_pattern':str})
    chemistry=pd.read_csv(ROOT/'manifests/chemistry_manifest.tsv',sep='\t',keep_default_na=False)
    qc=pd.read_csv(out/'well_qc.tsv',sep='\t').set_index('Combo')
    eligible=qualify_wells(mapping,qc.n_cells,protocol['qc']['minimum_cells_per_treatment_replicate'])
    cell_pass=set(eligible.loc[eligible.eligible,'treatment'])
    chem_status=chemistry.set_index('compound_id')
    eligibility=[]
    for compound,g in eligible.groupby('treatment',sort=True):
        c=chem_status.loc[compound]; reasons=[]
        if compound not in cell_pass:reasons.append(g.exclusion_reason.iloc[0])
        if c.status!='accepted':reasons.append(c.exclusion_reason)
        eligibility.append(dict(compound_id=compound,name=c['name'],eligible=not reasons,
            exclusion_reason=';'.join(reasons),rep1_cells=int(g.loc[g.replicate.eq('rep1'),'treatment_cells'].iloc[0]),
            rep2_cells=int(g.loc[g.replicate.eq('rep2'),'treatment_cells'].iloc[0]),
            minimum_control_cells=int(g.control_cells.min())))
    eligibility=pd.DataFrame(eligibility)
    eligibility.to_csv(ROOT/'manifests/eligibility.tsv',sep='\t',index=False)
    ids=set(eligibility.loc[eligibility.eligible,'compound_id'])
    if len(ids)<protocol['split']['minimum_eligible_compounds']:raise ValueError('Insufficient eligible compounds')
    eligible['eligible'] &= eligible.treatment.isin(ids)
    eligible.to_csv(ROOT/'manifests/qualified_well_mapping.tsv',sep='\t',index=False)
    records=chemistry[chemistry.compound_id.isin(ids)].to_dict('records')
    split,split_report=split_by_structure(records,seed=protocol['split']['seed'],
        min_test_compounds=protocol['split']['minimum_test_compounds'],min_test_groups=protocol['split']['minimum_test_scaffolds'])
    if not split_report['test_minimum_met']:raise ValueError('Test size/group minimum not met')
    split=pd.DataFrame(split).sort_values('compound_id').reset_index(drop=True)
    for identity in ['canonical_smiles','scaffold_group']:
        if split.groupby(identity).split.nunique().max()!=1:raise ValueError('Structure crosses split')
    split=split.merge(eligibility[['compound_id','rep1_cells','rep2_cells','minimum_control_cells']],on='compound_id',validate='one_to_one')
    split.to_csv(ROOT/'manifests/split_manifest.tsv',sep='\t',index=False)
    # All outcome-independent identity and split decisions are now on disk.
    counts=pd.read_csv(out/'pseudobulk_counts.tsv.gz',sep='\t',index_col=0)
    responses=build_responses(counts,eligible)
    if responses['compound_ids']!=split.compound_id.tolist():raise ValueError('Response/structure order mismatch')
    y=responses['delta']; fp=np.stack([morgan_fingerprint(s) for s in split.canonical_smiles])
    graph=[molecular_graph(s) for s in split.canonical_smiles]
    masks={s:split.split.eq(s).to_numpy() for s in ['train','validation','test']}
    transform=TrainingTargetTransform(protocol['response']['max_training_selected_genes'],protocol['response']['max_training_target_pcs']).fit(y[masks['train']])
    joblib.dump(transform,out/'target_transform.joblib')
    selected_genes=pd.read_csv(out/'gene_metadata.tsv',sep='\t').iloc[transform.gene_indices].copy()
    selected_genes.insert(0,'output_index',np.arange(len(selected_genes)))
    selected_genes.to_csv(ROOT/'manifests/selected_genes.tsv',sep='\t',index=False)
    development={}
    for s,key in [('train','train'),('validation','val')]:
        development[key+'_y']=transform.select_truth(y[masks[s]])
        development[key+'_z']=transform.transform(y[masks[s]])
        development[key+'_fp']=fp[masks[s]]
        development[key+'_ids']=split.loc[masks[s],'compound_id'].to_numpy(dtype=str)
    np.savez_compressed(out/'development.npz',**development)
    np.savez_compressed(out/'training_allgenes.npz',y=y[masks['train']],fp=fp[masks['train']],ids=split.loc[masks['train'],'compound_id'].to_numpy(dtype=str))
    np.savez_compressed(out/'validation_allgenes.npz',y=y[masks['validation']],fp=fp[masks['validation']],ids=split.loc[masks['validation'],'compound_id'].to_numpy(dtype=str))
    np.savez_compressed(out/'test_features.npz',fp=fp[masks['test']],ids=split.loc[masks['test'],'compound_id'].to_numpy(dtype=str))
    np.savez_compressed(out/'sealed_test_truth.npz',y=transform.select_truth(y[masks['test']]),ids=split.loc[masks['test'],'compound_id'].to_numpy(dtype=str))
    np.savez_compressed(out/'replicate_responses.npz',delta=responses['replicate_delta'][:,transform.gene_indices],ids=np.repeat(split.compound_id.to_numpy(dtype=str),2))
    for s,mask in masks.items():
        (out/f'graphs_{s}.json').write_text(json.dumps([{k:v.tolist() for k,v in graph[i].items()} for i in np.flatnonzero(mask)],separators=(',',':'))+'\n')
    sensitivity={}
    for threshold in [20,30,50]:
        q=qualify_wells(mapping,qc.n_cells,threshold)
        sensitivity[str(threshold)]={'cell_qualified_labels':int(q.loc[q.eligible,'treatment'].nunique()),
            'unique_parent_representatives':len(set(q.loc[q.eligible,'treatment']) & set(chemistry.loc[chemistry.status.eq('accepted'),'compound_id']))}
    files=['configs/protocol_v1.yaml','manifests/eligibility.tsv','manifests/split_manifest.tsv','manifests/selected_genes.tsv',
        'manifests/qualified_well_mapping.tsv','data/processed/development.npz','data/processed/training_allgenes.npz',
        'data/processed/validation_allgenes.npz','data/processed/test_features.npz','data/processed/sealed_test_truth.npz','data/processed/replicate_responses.npz',
        'data/processed/target_transform.joblib',*[f'data/processed/graphs_{s}.json' for s in masks]]
    result={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_labels':len(chemistry),
        'cell_qualified_labels':len(cell_pass),'eligible_unique_compounds':len(ids),'primary_treatment_cells':int(eligible.loc[eligible.eligible,'treatment_cells'].sum()),
        'primary_control_cells_unique':int(qc.loc[eligible.loc[eligible.eligible,'control_Combo'].unique(),'n_cells'].sum()),
        'source_genes':len(counts),'selected_genes':len(selected_genes),'target_pcs':transform.n_components,
        'training_PCA_explained_variance_fraction':float(transform.explained_variance.sum()),'split':split_report,
        'cell_count_sensitivity':sensitivity,'models_trained':False,'test_outcomes_used_for_selection':False,
        'artifact_hashes':{f:sha(ROOT/f) for f in files}}
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_hashes'},indent=2))
if __name__=='__main__':main()
