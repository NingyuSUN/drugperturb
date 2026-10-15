import hashlib,json
from pathlib import Path
import pytest
from drugperturb.provenance import verify_report_inputs

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def put(root,name,value):
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value) if isinstance(value,dict) else value);return digest(p)
@pytest.fixture
def chain(tmp_path):
    r=tmp_path
    truth=put(r,'data/processed/sealed_test_truth.npz','sealed truth')
    prediction=put(r,'models/run_v1/test_predictions.npz','sealed predictions')
    model=put(r,'receipts/model_selection.json',{'selected':'fixed'})
    freeze=put(r,'receipts/dataset_freeze.json',{'artifact_hashes':{'data/processed/sealed_test_truth.npz':truth}})
    seal=put(r,'receipts/prediction_seal.json',{'dataset_freeze_sha256':freeze,'artifact_hashes':{'models/run_v1/test_predictions.npz':prediction}})
    summary=put(r,'reports/final_test/summary.json',{'result':'observed'})
    final=put(r,'receipts/final_evaluation.json',{'prediction_seal_sha256':seal,'test_truth_sha256':truth,'output_hashes':{'reports/final_test/summary.json':summary}})
    diag=put(r,'receipts/development_diagnostics.tsv','complete diagnostic rows')
    put(r,'receipts/development_diagnostics.json',{'status':'COMPLETE_EXPLORATORY_DEVELOPMENT_ONLY','input_hashes':{'receipts/model_selection.json':model},'artifact_hashes':{'receipts/development_diagnostics.tsv':diag}})
    sens=put(r,'reports/sensitivity/metrics_summary.tsv','complete sensitivity rows')
    put(r,'receipts/sensitivity.json',{'status':'COMPLETE_FROZEN_PREDICTION_SENSITIVITY','prediction_seal_sha256':seal,'final_evaluation_sha256':final,'input_hashes':{'receipts/final_evaluation.json':final},'output_hashes':{'reports/sensitivity/metrics_summary.tsv':sens}})
    return r

def test_complete_chain_is_verified(chain):
    hashes=verify_report_inputs(chain)
    assert hashes['receipts/development_diagnostics.tsv']==digest(chain/'receipts/development_diagnostics.tsv')
@pytest.mark.parametrize('file',['receipts/development_diagnostics.tsv','reports/sensitivity/metrics_summary.tsv'])
def test_changed_secondary_analysis_is_rejected(chain,file):
    (chain/file).write_text('misleading replacement')
    with pytest.raises(ValueError,match='changed'):verify_report_inputs(chain)
def test_table_without_completion_receipt_is_rejected(chain):
    (chain/'receipts/development_diagnostics.json').unlink()
    with pytest.raises(FileNotFoundError):verify_report_inputs(chain)
def test_incomplete_status_is_rejected(chain):
    p=chain/'receipts/development_diagnostics.json';d=json.loads(p.read_text());d['status']='RUNNING';p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='incomplete'):verify_report_inputs(chain)
def test_sensitivity_must_bind_current_evaluation(chain):
    p=chain/'receipts/sensitivity.json';d=json.loads(p.read_text());d['final_evaluation_sha256']='0'*64;p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='binding'):verify_report_inputs(chain)
