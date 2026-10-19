"""Synthetic engineering checks for the chemCPA/biolord wrappers (cell-model env only).

Tiny networks and data exercise the training/prediction plumbing; they say
nothing about the published configurations or real sci-Plex3 performance.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

pytest.importorskip('descriptastorus')
torch = pytest.importorskip('torch')

from drugperturb.v3 import chemcpa, runner
from drugperturb.v3.descriptors import TrainingDescriptorScaler, VEHICLE_ID, compound_table
from drugperturb.v3.pseudobulk import normalize_counts

SMILES = {'S1': 'CCO', 'S2': 'c1ccccc1O', 'S3': 'CC(=O)Nc1ccc(O)cc1', 'S4': 'Cn1cnc2c1c(=O)n(C)c(=O)n2C'}
GENES = 12


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
                          for c, s in [('S3', 'validation'), ('S4', 'test')] for r, v in [('rep1', 'v1'), ('rep2', 'v2')]])
    return obs, counts, library, query


def test_descriptor_scaler_fits_training_rows_only():
    train = np.array([[1.0, 5.0, np.nan], [3.0, 5.0, 2.0]])
    scaler = TrainingDescriptorScaler().fit(train)
    assert scaler.keep.tolist() == [True, False, False]
    assert np.allclose(scaler.transform(np.array([[2.0, 0.0, 9.0]])), [[0.0]])
    table, kept = compound_table(list(SMILES), list(SMILES.values()), ['S1', 'S2'])
    assert set(table) == {*SMILES, VEHICLE_ID} and all(np.isfinite(v).all() for v in table.values())


def test_chemcpa_trains_and_predicts_counterfactuals():
    obs, counts, library, query = synthetic()
    X = normalize_counts(counts, library)
    descriptors, _ = compound_table(list(SMILES), list(SMILES.values()), ['S1', 'S2'])
    vocabulary = [VEHICLE_ID, *SMILES]
    hparams = dict(adversary_depth=1, adversary_lr=1e-3, adversary_steps=3, adversary_wd=1e-6, adversary_width=16,
                   autoencoder_depth=2, autoencoder_lr=3e-3, autoencoder_wd=1e-7, autoencoder_width=32, batch_size=32,
                   dim=8, dosers_depth=1, dosers_lr=1e-3, dosers_wd=1e-7, dosers_width=8, embedding_encoder_depth=1,
                   embedding_encoder_width=16, penalty_adversary=0.5, reg_adversary=1.0, reg_adversary_cov=1.0,
                   step_size_lr=6)
    model = chemcpa.ChemCPA(hparams=hparams, additional=dict(doser_type='amortized', decoder_activation='ReLU'),
                            n_genes=GENES, drug_vocabulary=vocabulary,
                            drug_matrix=np.vstack([descriptors[c] for c in vocabulary]), cell_types=['A549'],
                            use_cell_type=False, device='cpu', seed=42)
    vehicles = runner.vehicle_wells(obs, counts, library, ['v1', 'v2'])
    queries = runner.build_queries(query, ['A549_24h_1000nM'], ['validation'])
    truth = {('S3', 'A549_24h_1000nM'): np.full(GENES, 0.1)}
    predict = lambda well, condition: model.predict_mean(well['values'], condition)
    validate = lambda m: runner.mean_query_rmse(runner.predict_queries(predict, queries, vehicles, GENES)[0], truth)
    train = obs.role.isin(['train', 'vehicle']).to_numpy()
    t = obs[train]
    result = chemcpa.fit(model, X[train], t.compound_id.tolist(), t.dose_nM.tolist(), t.cell_type.tolist(),
                         batch_size=32, max_epochs=6, evaluate_every=3, patience=5, max_minutes=5, seed=42,
                         validate=validate)
    assert result['best_epoch'] in {3, 6} and np.isfinite(result['best_validation_rmse'])
    assert np.isclose(validate(model), result['best_validation_rmse'])  # restored checkpoint
    tests = runner.build_queries(query, ['A549_24h_1000nM'], ['test'])
    contrast, anchored = runner.predict_queries(predict, tests, vehicles, GENES)
    assert list(contrast) == [('S4', 'A549_24h_1000nM')] and contrast[('S4', 'A549_24h_1000nM')].shape == (GENES,)


def test_biolord_trains_and_predicts_from_vehicle_latents():
    pytest.importorskip('biolord')
    from drugperturb.v3 import biolord_model
    obs, counts, library, query = synthetic()
    sub = obs.reset_index(drop=True)
    X = normalize_counts(counts, library)
    descriptors, _ = compound_table(list(SMILES), list(SMILES.values()), ['S1', 'S2'])
    adata = biolord_model.build_adata(X, sub, descriptors, ['A549'], False)
    assert (adata.obs.split == 'test').sum() == 80
    candidate = dict(n_latent=4, module_params=dict(decoder_width=16, decoder_depth=1, attribute_nn_width=8,
                     attribute_nn_depth=1, n_latent_attribute_ordered=4, n_latent_attribute_categorical=2,
                     gene_likelihood='normal', reconstruction_penalty=1.0, unknown_attribute_penalty=0.1,
                     unknown_attribute_noise_param=0.1, attribute_dropout_rate=0.1, use_batch_norm=False,
                     use_layer_norm=False),
                     trainer_params=dict(n_epochs_warmup=0, latent_lr=1e-3, latent_wd=1e-4, decoder_lr=1e-3,
                     decoder_wd=1e-4, attribute_nn_lr=1e-3, attribute_nn_wd=4e-8, step_size_lr=45,
                     cosine_scheduler=True, scheduler_final_lr=1e-5),
                     train=dict(max_epochs=3, batch_size=64, early_stopping=True, early_stopping_patience=20,
                                check_val_every_n_epoch=1, max_minutes_per_fit=5))
    model = biolord_model.Biolord(adata, candidate=candidate, seed=42, use_cell_type=False, cell_types=['A549'],
                                  model_name='test')
    history = model.fit(candidate, 'cpu')
    assert len(history)
    vehicles = runner.vehicle_wells(sub, counts, library, ['v1', 'v2'])
    predict = lambda well, condition: model.predict_mean(well, condition, descriptors)
    tests = runner.build_queries(query, ['A549_24h_1000nM'], ['test'])
    contrast, _ = runner.predict_queries(predict, tests, vehicles, GENES)
    assert np.isfinite(contrast[('S4', 'A549_24h_1000nM')]).all()
