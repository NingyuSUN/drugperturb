"""Inference checks use only synthetic checkpoints and temporary artifacts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import joblib
import torch
from drugperturb.models import MLP, MolecularGNN
from drugperturb.response import TrainingTargetTransform
from drugperturb.chemistry import morgan_fingerprint


def module():
    spec=importlib.util.spec_from_file_location('predict_cli',Path(__file__).resolve().parents[1]/'scripts/predict.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_rejects_unsupported_context_before_loading_artifacts():
    m=module()
    for kw in [{'cell_type':'MCF7'},{'time_hours':48},{'dose_nM':100},{'time_hours':float('nan')}]:
        with pytest.raises(ValueError,match='A549.*24.*1000'):m.predict_smiles('CCO',**kw)


def test_rejects_invalid_structure_before_loading_artifacts():
    with pytest.raises(ValueError,match='SMILES|undefined|mixture'):
        module().predict_smiles('not_a_molecule')


def toy_artifacts(root):
    paths=['receipts','configs','manifests','data/processed','models/run_v1']
    for p in paths:(root/p).mkdir(parents=True)
    train=np.array([[0,2],[1,0],[3,4]],float)
    tf=TrainingTargetTransform(max_components=2).fit(train)
    joblib.dump(tf,root/'data/processed/target_transform.joblib')
    pd.DataFrame({'output_index':[0,1],'gene_id':['g1','g2']}).to_csv(root/'manifests/selected_genes.tsv',sep='\t',index=False)
    pd.DataFrame({'compound_id':['trainA','trainB'],'split':['train','train']}).to_csv(root/'manifests/split_manifest.tsv',sep='\t',index=False)
    np.savez(root/'data/processed/development.npz',train_fp=np.stack([morgan_fingerprint('CCO'),morgan_fingerprint('CCN')]),train_ids=np.array(['trainA','trainB']))
    cfg={'status':'FROZEN_BEFORE_TRAINING','models':{'training_seeds':[42,123,2026],'neural_final_prediction':'arithmetic_mean_of_three_seed_predictions','gnn_layers':2}}
    import yaml
    (root/'configs/protocol_v1.yaml').write_text(yaml.safe_dump(cfg))
    selection={'selected':{'M2':{'candidate':0,'hidden_dim':2,'dropout':0.0},'M3':{'candidate':0,'hidden_dim':2,'dropout':0.0}},'prediction_policy':'three_seed_arithmetic_mean_for_M2_M3'}
    (root/'receipts/model_selection.json').write_text(json.dumps(selection))
    for method in ['M2','M3']:
        for seed,bias in zip([42,123,2026],[1.,2.,3.]):
            model=MLP(2048,2,2,0.0) if method=='M2' else MolecularGNN(2,hidden_dim=2,dropout=0.0)
            with torch.no_grad():
                for p in model.parameters():p.zero_()
                (model.network[-1] if method=='M2' else model.readout).bias.fill_(bias)
            torch.save(model.state_dict(),root/f'models/run_v1/{method}_c0_s{seed}.pt')
    def sha(rel):return hashlib.sha256((root/rel).read_bytes()).hexdigest()
    frozen=['configs/protocol_v1.yaml','manifests/selected_genes.tsv','manifests/split_manifest.tsv','data/processed/development.npz','data/processed/target_transform.joblib']
    (root/'receipts/dataset_freeze.json').write_text(json.dumps({'artifact_hashes':{f:sha(f) for f in frozen}}))
    sealed=['receipts/model_selection.json',*[f'models/run_v1/{m}_c0_s{s}.pt' for m in ['M2','M3'] for s in [42,123,2026]]]
    (root/'receipts/prediction_seal.json').write_text(json.dumps({'source_commit':'synthetic-only','dataset_freeze_sha256':sha('receipts/dataset_freeze.json'),'artifact_hashes':{f:sha(f) for f in sealed}}))
    return tf


@pytest.mark.parametrize('method',['M2','M3'])
def test_uses_all_three_frozen_checkpoints_and_training_only_similarity(tmp_path,monkeypatch,method):
    m=module();tf=toy_artifacts(tmp_path);monkeypatch.setattr(m,'ROOT',tmp_path)
    table,meta=m.predict_smiles('CCO',model=method)
    assert table.gene_id.tolist()==['g1','g2']
    np.testing.assert_allclose(table.predicted_delta,tf.inverse_transform(np.array([[2.,2.]]))[0])
    assert meta['nearest_training_compound']=='trainA' and meta['max_training_tanimoto']==1.0
    assert meta['training_seeds']==[42,123,2026] and meta['test_outcomes_loaded'] is False
    assert meta['model']==method and len(meta['checkpoint_sha256'])==3


def test_tampered_frozen_artifact_is_rejected_before_deserialization(tmp_path,monkeypatch):
    m=module();toy_artifacts(tmp_path);monkeypatch.setattr(m,'ROOT',tmp_path)
    (tmp_path/'data/processed/target_transform.joblib').write_bytes(b'untrusted replacement')
    with pytest.raises(ValueError,match='hash|changed|mismatch'):m.predict_smiles('CCO',model='M2')
