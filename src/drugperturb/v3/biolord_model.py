"""biolord with the published sci-Plex3 settings, wrapped for counterfactual queries.

biolord learns a latent code per training cell, so counterfactuals start from
vehicle cells that were training samples, as in the published sci-Plex3 code.
Validation-compound cells form biolord's internal validation split; their codes
are not trained and they are never used as counterfactual sources.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

DOSE_SCALE = 1e4
ATTRIBUTE = 'rdkit2d_dose'


def attribute_matrix(compounds, doses_nM, descriptors):
    """[scaled descriptors, dose / 10 uM]; the DMSO vehicle receives dose zero."""
    rows = [np.append(descriptors[c], 0.0 if c == 'DMSO' else float(d)/DOSE_SCALE)
            for c, d in zip(compounds, doses_nM)]
    return np.asarray(rows, dtype=np.float32)


def build_adata(X, obs, descriptors, cell_types, use_cell_type):
    import anndata
    frame = pd.DataFrame({'split': np.where(obs.role.eq('validation'), 'test', 'train'),
                          'compound_id': obs.compound_id.to_numpy()}, index=obs.cell_id.to_numpy())
    if use_cell_type:
        frame['cell_type'] = pd.Categorical(obs.cell_type.to_numpy(), categories=cell_types)
    adata = anndata.AnnData(X=np.asarray(X, dtype=np.float32), obs=frame)
    adata.obsm[ATTRIBUTE] = attribute_matrix(obs.compound_id, obs.dose_nM.fillna(0), descriptors)
    return adata


class Biolord:
    def __init__(self, adata, *, candidate, seed, use_cell_type, cell_types, model_name):
        import biolord
        import scvi
        scvi.settings.seed = seed
        biolord.Biolord.setup_anndata(adata, ordered_attributes_keys=[ATTRIBUTE],
                                      categorical_attributes_keys=['cell_type'] if use_cell_type else None)
        params = dict(candidate['module_params'], seed=seed)
        self.adata, self.descriptors = adata, None
        self.use_cell_type, self.cell_types = use_cell_type, list(cell_types)
        self.model = biolord.Biolord(adata=adata, n_latent=candidate['n_latent'], model_name=model_name,
                                     module_params=params, train_classifiers=False, split_key='split',
                                     train_split='train', valid_split='test', test_split='ood')

    def fit(self, candidate, accelerator):
        import datetime
        train = candidate['train']
        extra = {}
        if train.get('max_minutes_per_fit'):
            extra['max_time'] = datetime.timedelta(minutes=train['max_minutes_per_fit'])
        self.model.train(max_epochs=train['max_epochs'], batch_size=train['batch_size'], **extra,
                         plan_kwargs=dict(candidate['trainer_params']),
                         early_stopping=train['early_stopping'],
                         early_stopping_patience=train['early_stopping_patience'],
                         check_val_every_n_epoch=train['check_val_every_n_epoch'],
                         num_workers=0, enable_checkpointing=False, accelerator=accelerator,
                         enable_progress_bar=False)
        history = pd.DataFrame(self.model.training_plan.epoch_history)
        return history

    @torch.no_grad()
    def predict_mean(self, well, condition, descriptors):
        module = self.model.module
        module.eval()
        index = np.asarray(well['index'])
        tensors = self.model.get_dataset(self.adata, indices=index)
        device = next(module.parameters()).device
        tensors = {k: v.to(device) for k, v in tensors.items()}
        n = len(index)
        target = attribute_matrix([condition['compound']], [condition['dose_nM']], descriptors)
        tensors[ATTRIBUTE] = torch.as_tensor(np.repeat(target, n, axis=0), device=device)
        if self.use_cell_type:
            code = float(self.cell_types.index(condition['cell_type']))
            tensors['cell_type'] = torch.full_like(tensors['cell_type'], code)
        mean, _ = module.get_expression(tensors)
        return mean.double().cpu().numpy()
