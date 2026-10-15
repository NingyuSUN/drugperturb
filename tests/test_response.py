import numpy as np
import pandas as pd
import pytest


def api():
    from drugperturb.response import build_responses, TrainingTargetTransform, baseline_predictions
    return build_responses, TrainingTargetTransform, baseline_predictions


def fixture():
    # Unequal libraries and cell counts must never weight replicate responses.
    counts = pd.DataFrame({'d2':[10,90], 'c1':[20,20], 'd1':[9,1], 'c2':[5,5], 'unused':[-1,0]}, index=['gB','gA'], dtype=float)
    mapping = pd.DataFrame({'treatment':['drug','drug'], 'replicate':['rep2','rep1'], 'Combo':['d2','d1'], 'control_Combo':['c2','c1'], 'eligible':[True,True], 'treatment_cells':[100,31]})
    return counts,mapping


def test_response_is_mean_of_exact_control_matched_log_cpm_deltas():
    build,_,_=api(); counts,mapping=fixture()
    result=build(counts,mapping)
    expected=np.array([[np.log1p(900000)-np.log1p(500000), np.log1p(100000)-np.log1p(500000)], [np.log1p(100000)-np.log1p(500000),np.log1p(900000)-np.log1p(500000)]])
    np.testing.assert_allclose(result['replicate_delta'],expected)
    np.testing.assert_allclose(result['delta'],expected.mean(axis=0)[None,:])
    assert result['compound_ids']==['drug'] and result['gene_ids']==['gB','gA']
    assert result['replicate_manifest'].replicate.tolist()==['rep1','rep2']
    # Pooling counts gives an asymmetric response; equal well deltas are symmetric.
    pooled=np.log1p(np.array([19,91])/110*1e6)-np.log1p(500000)
    assert not np.allclose(result['delta'][0],pooled)
    pd.testing.assert_frame_equal(counts,fixture()[0])


@pytest.mark.parametrize('bad',[-1,0.5,np.nan,np.inf])
def test_selected_invalid_counts_rejected(bad):
    build,_,_=api(); counts,mapping=fixture(); counts.loc['gA','d1']=bad
    with pytest.raises(ValueError,match='count'):
        build(counts,mapping)


@pytest.mark.parametrize('change, message', [('missing_control','control'),('missing_rep','rep'),('duplicate_combo','Combo'),('zero_library','librar'),('unknown_rep','rep'),('duplicate_genes','gene')])
def test_invalid_mapping_or_libraries_rejected(change,message):
    build,_,_=api(); counts,mapping=fixture()
    if change=='missing_control': mapping.loc[0,'control_Combo']='absent'
    if change=='missing_rep': mapping=mapping.iloc[:1]
    if change=='duplicate_combo': mapping.loc[0,'Combo']='d1'
    if change=='zero_library': counts.loc[:,'c1']=0
    if change=='unknown_rep': mapping.loc[0,'replicate']='rep3'
    if change=='duplicate_genes': counts.index=['g','g']
    with pytest.raises(ValueError,match=message): build(counts,mapping)


def test_eligible_selection_is_respected_without_qc_reselection():
    build,_,_=api(); counts,mapping=fixture()
    excluded=mapping.copy(); excluded['treatment']='excluded';excluded['Combo']=['absent1','absent2'];excluded['eligible']=False
    result=build(counts,pd.concat([excluded,mapping],ignore_index=True))
    assert result['compound_ids']==['drug']
    assert len(result['replicate_manifest'])==2


def test_sorted_compounds_share_controls_without_duplicate_count_columns():
    build,_,_=api(); counts,mapping=fixture()
    second=mapping.copy();second['treatment']='alpha';second['Combo']=['a2','a1']
    counts['a1']=[30,10];counts['a2']=[15,5]
    result=build(counts,pd.concat([mapping,second],ignore_index=True))
    assert result['compound_ids']==['alpha','drug']
    assert result['delta'].shape==(2,2)


def test_transform_uses_train_only_and_preserves_real_selected_truth():
    _,Transform,_=api()
    train=np.array([[0,3,7,1],[2,0,7,0],[4,5,7,2],[6,9,7,0]],float)
    tr=Transform(max_genes=3,max_components=1).fit(train)
    assert tr.gene_indices.tolist()==[1,0,3]
    assert tr.n_components==1 and tr.explained_variance.shape==(1,)
    original=[tr.gene_indices.copy(),tr.scaler.mean_.copy(),tr.scaler.scale_.copy(),tr.pca.components_.copy()]
    heldout=np.array([[99,-150,7,80],[2000,4000,7,-600]],float)
    z=tr.transform(heldout)
    assert z.shape==(2,1)
    for before,after in zip(original,[tr.gene_indices,tr.scaler.mean_,tr.scaler.scale_,tr.pca.components_]): np.testing.assert_array_equal(before,after)
    np.testing.assert_array_equal(tr.select_truth(heldout),heldout[:,[1,0,3]])
    assert not np.allclose(tr.inverse_transform(z),tr.select_truth(heldout))
    np.testing.assert_allclose(tr.inverse_transform(np.zeros((1,1))),train[:,[1,0,3]].mean(axis=0)[None,:])


def test_transform_retained_subspace_roundtrip_and_stable_variance_ties():
    _,Transform,_=api(); train=np.array([[0,0,4],[1,1,4],[2,2,4]],float)
    tr=Transform(max_components=20).fit(train)
    assert tr.gene_indices.tolist()==[0,1] and tr.n_components==2
    z=tr.transform(train)
    reconstructed=train.copy(); reconstructed[:,tr.gene_indices]=tr.inverse_transform(z)
    np.testing.assert_allclose(tr.transform(reconstructed),z,atol=1e-12)
    np.testing.assert_allclose(tr.inverse_transform(z),tr.select_truth(train),atol=1e-12)


@pytest.mark.parametrize('train',[np.ones((3,2)),np.ones((1,2)),np.array([[1,np.nan],[2,3]]),np.array([1,2])])
def test_invalid_training_targets_rejected(train):
    _,Transform,_=api()
    with pytest.raises(ValueError): Transform().fit(train)


def test_baselines_are_zero_train_mean_and_nearest_tanimoto_not_euclidean():
    _,_,baseline=api()
    train=np.array([[3,4],[8,9]],float)
    # Query 111100; A 100000 has Hamming distance 3, Tanimoto .25.
    # B 111111 has Hamming distance 2, Tanimoto 4/6 (nearest).
    fp=np.array([[1,0,0,0,0,0],[1,1,1,1,1,1]])
    query=np.array([[1,1,1,1,0,0],[0,0,0,0,0,0]])
    out=baseline(train,fp,query)
    assert set(out)=={'B0','B1','B2'}
    np.testing.assert_array_equal(out['B0'],np.zeros((2,2)))
    np.testing.assert_array_equal(out['B1'],[[5.5,6.5],[5.5,6.5]])
    np.testing.assert_array_equal(out['B2'],[[8,9],[3,4]])


def test_tanimoto_and_euclidean_can_choose_different_neighbors():
    _,_,baseline=api()
    # q has 3 bits. A overlaps 1 bit (distance2,Tanimoto1/3),
    # B overlaps all3 with 3 extras (distance3,Tanimoto1/2).
    query=np.array([[1,1,1,0,0,0]])
    fp=np.array([[1,0,0,0,0,0],[1,1,1,1,1,1]])
    assert np.argmin(((fp-query)**2).sum(axis=1))==0
    np.testing.assert_array_equal(baseline(np.array([[10],[20]]),fp,query)['B2'],[[20]])


@pytest.mark.parametrize("method, value", [("transform",np.ones((2,3))),("inverse_transform",np.ones((2,1))),("select_truth",np.ones((2,3)))])
def test_unfitted_transform_methods_reject_clearly(method,value):
    _,Transform,_=api()
    with pytest.raises(ValueError,match="fitted"):
        getattr(Transform(),method)(value)


def test_transform_dimensions_and_nonfinite_inputs_are_rejected():
    _,Transform,_=api(); tr=Transform(max_components=1).fit([[0,1],[1,2],[3,0]])
    for y in [np.ones((2,3)),np.array([[np.inf,2]])]:
        with pytest.raises(ValueError): tr.transform(y)
    for z in [np.ones((2,2)),np.array([[np.nan]])]:
        with pytest.raises(ValueError): tr.inverse_transform(z)


@pytest.mark.parametrize("targets, train, query", [(np.ones((2,1)),np.ones((1,3)),np.ones((2,3))), (np.ones((2,1)),np.ones((2,3)),np.ones((2,4))), (np.ones((2,1)),np.full((2,3),0.5),np.ones((2,3))), (np.array([[np.nan],[1]]),np.ones((2,3)),np.ones((2,3)))])
def test_baseline_invalid_inputs_rejected(targets,train,query):
    _,_,baseline=api()
    with pytest.raises(ValueError): baseline(targets,train,query)
