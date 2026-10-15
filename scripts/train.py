"""Train frozen candidates; seal test predictions without loading test outcomes."""
from pathlib import Path
import argparse, datetime, hashlib, json, platform, socket, time
import numpy as np
import pandas as pd
import torch, yaml, joblib
from sklearn.linear_model import Ridge
from drugperturb.models import MLP, MolecularGNN, fit_network, predict_network
from drugperturb.response import baseline_predictions
from drugperturb.provenance import verify_training_inputs

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def rmse(y,p):return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2,axis=1)).mean())
def graphs(path):
    return [{k:np.asarray(v,dtype=np.int64 if k=='edge_index' else np.float32) for k,v in g.items()} for g in json.loads(path.read_text())]
def save_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit',required=True);args=parser.parse_args()
    start=time.perf_counter();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
    out=ROOT/'models/run_v1';out.mkdir(parents=True,exist_ok=True)
    seal=ROOT/'receipts/prediction_seal.json'
    if seal.exists():
        verify_training_inputs(ROOT, completed=True)
        print('Existing prediction seal verified; no training repeated.');return
    verify_training_inputs(ROOT)
    protocol=yaml.safe_load((ROOT/'configs/protocol_v1.yaml').read_text()); cfg=protocol['models']
    if protocol['status']!='FROZEN_BEFORE_TRAINING':raise ValueError('Protocol is not frozen')
    split=pd.read_csv(ROOT/'manifests/split_manifest.tsv',sep='\t')
    test_split=split[split.split.eq('test')]
    if len(split)<100 or len(test_split)<20 or test_split.loc[test_split.scaffold_group.str.startswith('murcko:'),'scaffold_group'].nunique()<10:
        raise ValueError('Independent compound / ring scaffold gate failed')
    d=np.load(ROOT/'data/processed/development.npz',allow_pickle=False)
    tf=np.load(ROOT/'data/processed/test_features.npz',allow_pickle=False)
    transform=joblib.load(ROOT/'data/processed/target_transform.joblib')
    target_dim=d['train_z'].shape[1]
    train_graph=graphs(ROOT/'data/processed/graphs_train.json');val_graph=graphs(ROOT/'data/processed/graphs_validation.json');test_graph=graphs(ROOT/'data/processed/graphs_test.json')
    val_pred=baseline_predictions(d['train_y'],d['train_fp'],d['val_fp'])
    test_pred=baseline_predictions(d['train_y'],d['train_fp'],tf['fp'])
    runs=[];ridge_models=[]
    for i,alpha in enumerate(cfg['ridge_alphas']):
        t=time.perf_counter();model=Ridge(alpha=alpha,solver='cholesky').fit(d['train_fp'],d['train_z'])
        pred=transform.inverse_transform(model.predict(d['val_fp']))
        score=rmse(d['val_y'],pred);ridge_models.append((score,model))
        joblib.dump(model,out/f'M1_c{i}.joblib')
        runs.append(dict(method='M1',candidate=i,seed=0,validation_RMSE=score,seconds=time.perf_counter()-t,alpha=alpha))
    ridge_index=min(range(len(ridge_models)),key=lambda i:ridge_models[i][0]);ridge=ridge_models[ridge_index][1]
    val_pred['M1']=transform.inverse_transform(ridge.predict(d['val_fp']));test_pred['M1']=transform.inverse_transform(ridge.predict(tf['fp']))
    selected={'M1':{'candidate':ridge_index,'alpha':cfg['ridge_alphas'][ridge_index]}}
    for method,candidates,train_x,val_x,test_x in [('M2',cfg['mlp_candidates'],d['train_fp'],d['val_fp'],tf['fp']),('M3',cfg['gnn_candidates'],train_graph,val_graph,test_graph)]:
        candidate_predictions={};scores=[]
        for i,hp in enumerate(candidates):
            per_seed=[];predictions=[]
            for seed in cfg['training_seeds']:
                name=f'{method}_c{i}_s{seed}';t=time.perf_counter()
                factory=(lambda h=hp:MLP(d['train_fp'].shape[1],target_dim,h['hidden_dim'],h['dropout'])) if method=='M2' else (lambda h=hp:MolecularGNN(target_dim,hidden_dim=h['hidden_dim'],layers=cfg['gnn_layers'],dropout=h['dropout']))
                result=fit_network(factory,train_x,d['train_z'],val_x,d['val_z'],seed=seed,lr=hp['lr'],max_epochs=cfg['max_epochs'],patience=cfg['patience'],batch_size=cfg['batch_size'],validation_scorer=lambda p:rmse(d['val_y'],transform.inverse_transform(p)))
                if result['selection_metric']!='custom_validation_score':raise AssertionError('Wrong checkpoint score')
                pv=transform.inverse_transform(predict_network(result['model'],val_x))
                if not np.isclose(rmse(d['val_y'],pv),result['best_val_score'],atol=1e-7,rtol=1e-7):raise AssertionError('Restored checkpoint differs from selection score')
                pt=transform.inverse_transform(predict_network(result['model'],test_x))
                predictions.append((pv,pt));per_seed.append(result['best_val_score'])
                torch.save(result['model'].state_dict(),out/f'{name}.pt')
                pd.DataFrame(result['history']).to_csv(out/f'{name}_history.tsv',sep='\t',index=False)
                save_json(out/f'{name}_run.json',{k:v for k,v in result.items() if k not in ['model','history']})
                record=dict(method=method,candidate=i,seed=seed,validation_RMSE=result['best_val_score'],best_epoch=result['best_epoch'],epochs_executed=len(result['history']),parameter_count=result['parameter_count'],seconds=time.perf_counter()-t,**hp)
                runs.append(record);pd.DataFrame(runs).to_csv(ROOT/'receipts/training_runs.tsv',sep='\t',index=False)
                print(json.dumps(record),flush=True)
            candidate_predictions[i]=predictions;scores.append(float(np.mean(per_seed)))
        chosen=int(np.argmin(scores));selected[method]={'candidate':chosen,**candidates[chosen],'mean_seed_validation_RMSE':scores[chosen]}
        chosen_predictions=candidate_predictions[chosen]
        val_pred[method]=np.mean([v for v,t in chosen_predictions],axis=0)
        test_pred[method]=np.mean([t for v,t in chosen_predictions],axis=0)
        for seed,(v,t) in zip(cfg['training_seeds'],chosen_predictions):val_pred[f'{method}_s{seed}']=v;test_pred[f'{method}_s{seed}']=t
    validation_scores={k:rmse(d['val_y'],v) for k,v in val_pred.items()}
    reference=min(['B0','B1','B2','M1'],key=lambda k:validation_scores[k])
    np.savez_compressed(out/'validation_predictions.npz',ids=d['val_ids'],**val_pred)
    np.savez_compressed(out/'test_predictions.npz',ids=tf['ids'],**test_pred)
    pd.DataFrame(runs).to_csv(ROOT/'receipts/training_runs.tsv',sep='\t',index=False)
    selection={'selected':selected,'selection_metric':'mean_seed_validation_mean_compound_gene_delta_RMSE','validation_scores':validation_scores,'simple_reference':reference,'primary_model':'M3','prediction_policy':'three_seed_arithmetic_mean_for_M2_M3','test_outcomes_loaded':False}
    save_json(ROOT/'receipts/model_selection.json',selection)
    files=[str(p.relative_to(ROOT)) for p in sorted(out.glob('*'))]+['receipts/model_selection.json','receipts/training_runs.tsv','configs/protocol_v1.yaml','configs/biology_v1.json','manifests/selected_genes.tsv','src/drugperturb/models.py','src/drugperturb/response.py','scripts/train.py']
    result={'started_utc':utc,'sealed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':args.source_commit,'host':socket.gethostname(),'device':'cpu','python':platform.python_version(),'torch':torch.__version__,'elapsed_seconds':time.perf_counter()-start,'dataset_freeze_sha256':sha(ROOT/'receipts/dataset_freeze.json'),'test_outcomes_loaded':False,'selected':selected,'simple_reference':reference,'artifact_hashes':{f:sha(ROOT/f) for f in files}}
    save_json(seal,result);print(json.dumps(selection,indent=2),flush=True)
if __name__=='__main__':main()
