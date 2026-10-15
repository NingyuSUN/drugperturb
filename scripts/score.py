"""One frozen final-test evaluation; never refit or change model selections."""
from pathlib import Path
import datetime, hashlib, json
import numpy as np
import pandas as pd
from drugperturb.evaluation import compound_metrics, summarize_compound_metrics, paired_scaffold_bootstrap
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def main():
    final=ROOT/'receipts/final_evaluation.json'
    if final.exists():
        old=json.loads(final.read_text())
        for f,h in old['output_hashes'].items():
            if sha(ROOT/f)!=h:raise ValueError(f'Evaluation output changed: {f}')
        print('Existing final evaluation verified; no scoring repeated.');return
    seal=json.loads((ROOT/'receipts/prediction_seal.json').read_text())
    freeze=json.loads((ROOT/'receipts/dataset_freeze.json').read_text())
    if sha(ROOT/'receipts/dataset_freeze.json')!=seal['dataset_freeze_sha256']:raise ValueError('Dataset freeze is not bound to prediction seal')
    for f,h in freeze['artifact_hashes'].items():
        if sha(ROOT/f)!=h:raise ValueError(f'Frozen evaluation dependency changed: {f}')
    for f,h in seal['artifact_hashes'].items():
        if sha(ROOT/f)!=h:raise ValueError(f'Prediction seal changed: {f}')
    truth_path='data/processed/sealed_test_truth.npz'
    if sha(ROOT/truth_path)!=freeze['artifact_hashes'][truth_path]:raise ValueError('Frozen truth changed')
    # Labels are loaded only after the model, configuration and prediction checks.
    t=np.load(ROOT/truth_path,allow_pickle=False);p=np.load(ROOT/'models/run_v1/test_predictions.npz',allow_pickle=False)
    split=pd.read_csv(ROOT/'manifests/split_manifest.tsv',sep='\t');s=split[split.split.eq('test')].reset_index(drop=True)
    ids=s.compound_id.tolist();groups=s.scaffold_group.tolist()
    if list(t['ids'])!=ids or list(p['ids'])!=ids:raise ValueError('Test identity/order mismatch')
    y=t['y'];out=ROOT/'reports/final_test';out.mkdir(parents=True,exist_ok=True)
    main_methods=['B0','B1','B2','M1','M2','M3'];all_methods=[k for k in p.files if k!='ids']
    tables=[];summary={}
    for method in all_methods:
        table=compound_metrics(y,p[method],ids,groups);summary[method]=summarize_compound_metrics(table)
        table.insert(0,'method',method);tables.append(table)
    metrics=pd.concat(tables,ignore_index=True)
    metrics.to_csv(out/'metrics_by_compound.tsv',sep='\t',index=False)
    pd.DataFrame([dict(method=k,**v) for k,v in summary.items()]).to_csv(out/'metrics_summary.tsv',sep='\t',index=False)
    selection=json.loads((ROOT/'receipts/model_selection.json').read_text());reference=selection['simple_reference']
    paired=[]
    for method in ['M1','M2','M3']:
        for baseline in dict.fromkeys([reference,'B0','B1','B2']):
            if method==baseline:continue
            b=paired_scaffold_bootstrap(y,p[method],p[baseline],ids,groups)
            draws=b.pop('bootstrap_differences');np.save(ROOT/'models/run_v1'/f'bootstrap_{method}_minus_{baseline}.npy',draws)
            b.update(method=method,baseline=baseline,reference_selected_on_validation=baseline==reference)
            paired.append(b)
    save_json(out/'paired_comparisons.json',paired)
    # Record chemical support and experimental reliability for every test compound.
    d=np.load(ROOT/'data/processed/development.npz',allow_pickle=False);tf=np.load(ROOT/'data/processed/test_features.npz',allow_pickle=False)
    inter=tf['fp']@d['train_fp'].T;union=tf['fp'].sum(1)[:,None]+d['train_fp'].sum(1)[None,:]-inter
    sim=np.divide(inter,union,out=np.ones_like(inter),where=union!=0);nearest=sim.argmax(1)
    reps=np.load(ROOT/'data/processed/replicate_responses.npz',allow_pickle=False)
    context=[];replicate_errors=[]
    mapping=pd.read_csv(ROOT/'manifests/qualified_well_mapping.tsv',sep='\t')
    for i,cid in enumerate(ids):
        rr=reps['delta'][reps['ids']==cid]
        if len(rr)!=2:raise ValueError('Replicate count differs from two')
        agreement=compound_metrics(rr[:1],rr[1:],[cid],[groups[i]]).iloc[0]
        m=mapping[mapping.treatment.eq(cid)].sort_values('replicate')
        context.append(dict(compound_id=cid,name=s.iloc[i]['name'],scaffold_group=groups[i],max_train_tanimoto=float(sim[i,nearest[i]]),nearest_training_compound=str(d['train_ids'][nearest[i]]),response_rms=float(np.sqrt(np.mean(y[i]**2))),replicate_rmse=float(agreement.rmse),replicate_spearman=None if pd.isna(agreement.spearman) else float(agreement.spearman),rep1_cells=int(m.iloc[0].treatment_cells),rep2_cells=int(m.iloc[1].treatment_cells),minimum_control_cells=int(m.control_cells.min()),rep1_plate=m.iloc[0].plate_oligo,rep2_plate=m.iloc[1].plate_oligo))
        for j in range(2):
            for method in main_methods:
                replicate_errors.append(dict(compound_id=cid,method=method,replicate=f'rep{j+1}',plate=m.iloc[j].plate_oligo,rmse=float(np.sqrt(np.mean((rr[j]-p[method][i])**2)))))
    context=pd.DataFrame(context);context.to_csv(out/'compound_context.tsv',sep='\t',index=False)
    replicate_errors=pd.DataFrame(replicate_errors);replicate_errors.to_csv(out/'replicate_plate_errors.tsv',sep='\t',index=False)
    metrics[metrics.method.isin(main_methods)].groupby(['method','scaffold_group'],sort=True).agg(n_compounds=('compound_id','count'),mean_rmse=('rmse','mean')).reset_index().to_csv(out/'errors_by_structure_group.tsv',sep='\t',index=False)
    # Fixed coverage rule: do not add genes to rescue a pathway after seeing results.
    biology=json.loads((ROOT/'configs/biology_v1.json').read_text());genes=pd.read_csv(ROOT/'manifests/selected_genes.tsv',sep='\t')
    positions={g:i for i,g in enumerate(genes.gene_id)};coverage=[];path_rows=[]
    for gene_set in biology['sets']:
        columns=[positions[g] for g in gene_set['mapped_unique_gene_ids'] if g in positions]
        fraction=len(columns)/gene_set['source_member_count'];qualified=len(columns)>=10 and fraction>=0.25
        coverage.append(dict(set_id=gene_set['set_id'],name=gene_set['name'],original_members=gene_set['source_member_count'],mapped_unique_genes=gene_set['mapped_unique_gene_count'],selected_genes=len(columns),fraction_original=fraction,quantitatively_evaluable=qualified))
        if not qualified:continue
        for method in main_methods:
            actual=y[:,columns];pred=p[method][:,columns]
            for i,cid in enumerate(ids):
                path_rows.append(dict(set_id=gene_set['set_id'],method=method,compound_id=cid,n_genes=len(columns),measured_mean_delta=float(actual[i].mean()),predicted_mean_delta=float(pred[i].mean()),gene_delta_rmse=float(np.sqrt(np.mean((actual[i]-pred[i])**2))),mean_direction_agrees=bool(np.sign(actual[i].mean())==np.sign(pred[i].mean()))))
    pd.DataFrame(coverage).to_csv(out/'pathway_coverage.tsv',sep='\t',index=False)
    pd.DataFrame(path_rows,columns=['set_id','method','compound_id','n_genes','measured_mean_delta','predicted_mean_delta','gene_delta_rmse','mean_direction_agrees']).to_csv(out/'pathway_response.tsv',sep='\t',index=False)
    # Case rules fixed before scoring: high GNN error, then weak response, then low similarity.
    gnn=metrics[metrics.method.eq('M3')].set_index('compound_id').rmse
    c=context.assign(gnn_rmse=context.compound_id.map(gnn));cases=[];used=set()
    for label,column,ascending in [('highest_GNN_error','gnn_rmse',False),('lowest_response','response_rms',True),('lowest_training_similarity','max_train_tanimoto',True)]:
        row=c[~c.compound_id.isin(used)].sort_values([column,'compound_id'],ascending=[ascending,True]).iloc[0]
        used.add(row.compound_id);cases.append(dict(case_rule=label,**row.to_dict()))
    pd.DataFrame(cases).to_csv(out/'fixed_rule_cases.tsv',sep='\t',index=False)
    save_json(out/'summary.json',{'n_test_compounds':len(ids),'n_structure_groups':len(set(groups)),'n_ring_scaffolds':len({g for g in groups if g.startswith('murcko:')}),'n_genes':y.shape[1],'simple_reference':reference,'primary_model':'M3','metrics':summary,'uncertainty_scope':'Fixed split and fitted models; scaffold-group resampling does not fully cover shared vehicles, plates, retraining or external studies.','biological_scope':'Measured transcript response to A549 24h 1uM perturbations; no efficacy, target engagement or antibody design validation.'})
    output_paths=[str(f.relative_to(ROOT)) for f in sorted(out.glob('*'))]
    record={'evaluated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'prediction_seal_sha256':sha(ROOT/'receipts/prediction_seal.json'),'test_truth_sha256':sha(ROOT/truth_path),'biology_sha256':sha(ROOT/'configs/biology_v1.json'),'evaluation_source_sha256':sha(Path(__file__)),'n_test_compounds':len(ids),'test_used_for_selection':False,'output_hashes':{f:sha(ROOT/f) for f in output_paths}}
    save_json(final,record)
    print(pd.DataFrame([dict(method=m,RMSE=summary[m]['mean_compound_rmse']) for m in main_methods]).to_string(index=False))
    print('Final test evaluation sealed; no model changes made.')
if __name__=='__main__':main()
