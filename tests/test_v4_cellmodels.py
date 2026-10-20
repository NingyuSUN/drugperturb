"""Synthetic engineering checks for the v4 chemCPA arms (cell-model env only).

Tiny networks and data exercise the objectives and their plumbing; they say
nothing about real sci-Plex3 or OP3 performance.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

pytest.importorskip('descriptastorus')
torch = pytest.importorskip('torch')

from drugperturb.v3 import chemcpa, runner
from drugperturb.v3.descriptors import VEHICLE_ID, compound_table
from drugperturb.v3.pseudobulk import normalize_counts, pseudobulk_log_cpm
from drugperturb.v4 import chemcpa_v4, counterfactual

SMILES = {'S1': 'CCO', 'S2': 'c1ccccc1O', 'S3': 'CC(=O)Nc1ccc(O)cc1', 'S4': 'Cn1cnc2c1c(=O)n(C)c(=O)n2C'}
GENES = 12
HPARAMS = dict(adversary_depth=1, adversary_lr=1e-3, adversary_steps=3, adversary_wd=1e-6, adversary_width=16,
               autoencoder_depth=2, autoencoder_lr=3e-3, autoencoder_wd=1e-7, autoencoder_width=32, batch_size=32,
               dim=8, dosers_depth=1, dosers_lr=1e-3, dosers_wd=1e-7, dosers_width=8, embedding_encoder_depth=1,
               embedding_encoder_width=16, penalty_adversary=0.5, reg_adversary=1.0, reg_adversary_cov=1.0,
               step_size_lr=6)


def synthetic():
    rng = np.random.default_rng(1)
    rows, blocks = [], []
    effect = {'DMSO': 0.0, 'S1': 1.0, 'S2': 2.0, 'S3': 1.5, 'S4': 0.5}
    wells = [('v1', 'DMSO', 'vehicle', 'rep1'), ('v2', 'DMSO', 'vehicle', 'rep2'), ('a1', 'S1', 'train', 'rep1'),
             ('a2', 'S1', 'train', 'rep2'), ('b1', 'S2', 'train', 'rep1'), ('b2', 'S2', 'train', 'rep2'),
             ('c1', 'S3', 'validation', 'rep1'), ('c2', 'S3', 'validation', 'rep2')]
    for combo, compound, role, rep in wells:
        rate = np.full(GENES, 4.0)
        rate[:4] *= 1+effect[compound]
        blocks.append(rng.poisson(rate, size=(40, GENES)))
        rows += [dict(cell_id=f'{combo}_{i}', Combo=combo, cell_type='A549', replicate=rep, plate_oligo='p',
                      compound_id=compound, role=role, dose_nM=0.0 if compound == 'DMSO' else 1000.0) for i in range(40)]
    counts = sparse.csr_matrix(np.vstack(blocks))
    library = np.asarray(counts.sum(axis=1)).ravel()+100.0
    obs = pd.DataFrame(rows)
    query = pd.DataFrame([dict(compound_id=c, context_id='A549_24h_1000nM', split=s, replicate=r, cell_type='A549',
                               dose_nM=1000.0, vehicle_Combo=v)
                          for c, s in [('S1', 'train'), ('S2', 'train'), ('S3', 'validation'), ('S4', 'test')]
                          for r, v in [('rep1', 'v1'), ('rep2', 'v2')]])
    return obs, counts, library, query


def log_cpm(counts, library):
    return np.log1p(np.asarray(counts.sum(axis=0)).ravel()/library.sum()*1e6)


def setup(likelihood='gaussian', weight=0.0, seed=42, activation='ReLU'):
    obs, counts, library, query = synthetic()
    X = normalize_counts(counts, library)
    descriptors, _ = compound_table(list(SMILES), list(SMILES.values()), ['S1', 'S2'])
    vocabulary = [VEHICLE_ID, *SMILES]
    vehicles = runner.vehicle_wells(obs, counts, library, ['v1', 'v2'])
    queries = {s: counterfactual.build_queries(query, ['A549_24h_1000nM'], [s], 2) for s in ['train', 'validation', 'test']}
    truth = {}
    for (cid, ctx) in queries['train']:
        reps = [log_cpm(counts[obs.Combo.eq(f'{w}').to_numpy()], library[obs.Combo.eq(f'{w}').to_numpy()])
                for w in ({'S1': ['a1', 'a2'], 'S2': ['b1', 'b2']}[cid])]
        vs = [log_cpm(counts[obs.Combo.eq(v).to_numpy()], library[obs.Combo.eq(v).to_numpy()]) for v in ['v1', 'v2']]
        truth[(cid, ctx)] = np.mean([r - v for r, v in zip(reps, vs)], axis=0)
    target = None
    if weight:
        keys = sorted(queries['train'])
        pairs = [dict(compound=k[0], dose_nM=1000.0, cell_type='A549', wells=['v1', 'v2']) for k in keys]
        target = chemcpa_v4.PseudobulkTarget(pairs=pairs, truth=np.vstack([truth[k] for k in keys]),
                                              vehicle_cells=vehicles, n_replicates=2, pairs_per_update=2,
                                              cells_per_well=16, seed=seed+1, device='cpu')
    train = obs.role.isin(['train', 'vehicle']).to_numpy()
    offset = chemcpa_v4.gene_offsets(counts[train], library[train]) if likelihood == 'nb' else None
    model = chemcpa_v4.ChemCPAv4(likelihood=likelihood, gene_offset=offset, pseudobulk_weight=weight,
                                 pseudobulk_target=target, hparams=HPARAMS,
                                 additional=dict(doser_type='amortized', decoder_activation=activation),
                                 n_genes=GENES, drug_vocabulary=vocabulary,
                                 drug_matrix=np.vstack([descriptors[c] for c in vocabulary]), cell_types=['A549'],
                                 use_cell_type=False, device='cpu', seed=seed)
    return obs, counts, library, X, model, vehicles, queries, truth, train


def run_fit(model, obs, counts, library, X, vehicles, queries, train, fit=chemcpa_v4.fit, epochs=4):
    predict = lambda well, condition: model.predict_mean(well['values'], condition)
    truth = {k: np.full(GENES, 0.1) for k in queries['validation']}
    validate = lambda m: runner.mean_query_rmse(
        counterfactual.counterfactual_deltas(predict, queries['validation'], vehicles, gene_count=GENES,
                                             n_replicates=2)[0], truth)
    t = obs[train]
    kwargs = dict(batch_size=32, max_epochs=epochs, evaluate_every=2, patience=5, max_minutes=5, seed=42,
                  validate=validate)
    if fit is chemcpa_v4.fit:
        kwargs.update(counts=counts[train], library=library[train])
    return fit(model, X[train], t.compound_id.tolist(), t.dose_nM.tolist(), t.cell_type.tolist(), **kwargs)


def test_torch_pseudobulk_equals_numpy_mapping():
    rng = np.random.default_rng(0)
    values = rng.uniform(-0.5, 3.0, size=(2, 7, 5))
    library = rng.uniform(100, 900, size=(2, 7))
    got = chemcpa_v4.torch_pseudobulk(torch.as_tensor(values), torch.as_tensor(library)).numpy()
    expected = np.vstack([pseudobulk_log_cpm(values[i], library[i]) for i in range(2)])
    assert np.allclose(got, expected, rtol=1e-12, atol=1e-12)
    values[0, 0, 0] = 900.0  # divergent output: capped identically in torch and numpy (amendment A2)
    got = chemcpa_v4.torch_pseudobulk(torch.as_tensor(values), torch.as_tensor(library)).numpy()
    expected = np.vstack([counterfactual.capped_pseudobulk(values[i], library[i]) for i in range(2)])
    assert np.isfinite(got).all() and np.allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_a0_repeats_the_v3_training_step_exactly():
    obs, counts, library, X, model, vehicles, queries, _, train = setup()
    descriptors, _ = compound_table(list(SMILES), list(SMILES.values()), ['S1', 'S2'])
    vocabulary = [VEHICLE_ID, *SMILES]
    reference = chemcpa.ChemCPA(hparams=HPARAMS, additional=dict(doser_type='amortized', decoder_activation='ReLU'),
                                n_genes=GENES, drug_vocabulary=vocabulary,
                                drug_matrix=np.vstack([descriptors[c] for c in vocabulary]), cell_types=['A549'],
                                use_cell_type=False, device='cpu', seed=42)
    a = run_fit(model, obs, counts, library, X, vehicles, queries, train)
    b = run_fit(reference, obs, counts, library, X, vehicles, queries, train, fit=chemcpa.fit)
    assert a['best_epoch'] == b['best_epoch'] and a['best_validation_rmse'] == b['best_validation_rmse']
    for (k, x), (_, y) in zip(model.model.state_dict().items(), reference.model.state_dict().items()):
        assert torch.equal(x, y), k
    assert all(r['pseudobulk'] == 0.0 for r in a['history'])


def test_nb_arm_trains_with_finite_loss_and_maps_to_count_scale():
    obs, counts, library, X, model, vehicles, queries, _, train = setup(likelihood='nb', activation='linear')
    result = run_fit(model, obs, counts, library, X, vehicles, queries, train)
    assert all(np.isfinite(r['reconstruction']) for r in result['history'])
    values = model.predict_mean(vehicles['v1']['values'], dict(compound='DMSO', dose_nM=0.0, cell_type='A549'))
    assert values.shape == (40, GENES) and (values >= 0).all() and np.isfinite(values).all()
    # the offset alone maps to the pooled training fraction of each gene
    fraction = np.exp(chemcpa_v4.gene_offsets(counts[train], library[train]))
    assert np.allclose(np.expm1(np.log1p(1e4*fraction))/1e4, fraction)
    with pytest.raises(ValueError):
        chemcpa_v4.ChemCPAv4(likelihood='nb', gene_offset=None, hparams=HPARAMS,
                             additional=dict(doser_type='amortized', decoder_activation='linear'), n_genes=GENES,
                             drug_vocabulary=['DMSO', 'S1'], drug_matrix=np.zeros((2, 3)), cell_types=['A549'],
                             use_cell_type=False, device='cpu', seed=1)


@pytest.mark.parametrize('likelihood,activation', [('gaussian', 'ReLU'), ('nb', 'linear')])
def test_pseudobulk_loss_reaches_the_drug_pathway_and_is_sampled_within_wells(likelihood, activation):
    obs, counts, library, X, model, vehicles, queries, truth, train = setup(likelihood, 1.0, activation=activation)
    chosen, rows = model.pseudobulk.sample()
    assert rows.shape == (2, 2, 16)
    for p, i in enumerate(chosen.tolist()):
        for r in range(2):
            start, size = int(model.pseudobulk.start[i, r]), int(model.pseudobulk.size[i, r])
            assert ((rows[p, r] >= start) & (rows[p, r] < start+size)).all()
    model.model.train()
    loss = model.pseudobulk_loss()
    assert model.model.training  # restored after the evaluation-mode pass
    loss.backward()
    grads = [p.grad for p in [*model.model.drug_embedding_encoder.parameters(), *model.model.dosers.parameters()]]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    result = run_fit(model, obs, counts, library, X, vehicles, queries, train, epochs=6)
    assert all(np.isfinite(r['pseudobulk']) and r['pseudobulk'] > 0 for r in result['history'])


def test_pseudobulk_arm_fits_training_deltas_better_than_a0():
    """Engineering check: the counterfactual loss lowers training-pair contrast error (final epoch, no selection)."""
    def training_error(weight):
        obs, counts, library, X, model, vehicles, queries, truth, train = setup(weight=weight)
        t = obs[train]
        chemcpa_v4.fit(model, X[train], t.compound_id.tolist(), t.dose_nM.tolist(), t.cell_type.tolist(),
                       batch_size=32, max_epochs=30, evaluate_every=10**6, patience=5, max_minutes=5, seed=42,
                       validate=lambda m: 0.0)
        predict = lambda well, condition: model.predict_mean(well['values'], condition)
        contrast, _ = counterfactual.counterfactual_deltas(predict, queries['train'], vehicles, gene_count=GENES,
                                                           n_replicates=2)
        return np.mean([np.sqrt(np.mean((contrast[k]-truth[k])**2)) for k in truth])
    assert training_error(1.0) < training_error(0.0)
