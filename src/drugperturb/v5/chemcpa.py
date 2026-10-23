"""v4 chemCPA with a replicate-weighted counterfactual pseudobulk loss for v5.

The v4 loss averages a fixed number R of replicate contrasts per training pair.
Tahoe-100M pairs have one to three replicates, so each pair's vehicle wells are
padded to the largest count by repeating its first well, and the contrast is the
weighted sum with weight 1/r on its r real replicates and 0 on padding: the
equal-weight replicate mean, as in the measured truth. Sampling, modules and
everything else are v4's (drugperturb.v4.chemcpa_v4).
"""
from __future__ import annotations

import numpy as np
import torch

from ..v4.chemcpa_v4 import ChemCPAv4, PseudobulkTarget, torch_pseudobulk


class WeightedPseudobulkTarget(PseudobulkTarget):
    def __init__(self, *, pairs, truth, vehicle_cells, pairs_per_update, cells_per_well, seed, device):
        counts = [len(p['wells']) for p in pairs]
        if not pairs or min(counts) < 1:
            raise ValueError('every pseudobulk pair needs at least one vehicle well')
        width = max(counts)
        padded = [dict(p, wells=list(p['wells'])+[p['wells'][0]]*(width-len(p['wells']))) for p in pairs]
        super().__init__(pairs=padded, truth=truth, vehicle_cells=vehicle_cells, n_replicates=width,
                         pairs_per_update=pairs_per_update, cells_per_well=cells_per_well, seed=seed, device=device)
        weights = np.zeros((len(pairs), width), np.float32)
        for i, n in enumerate(counts):
            weights[i, :n] = 1.0/n
        self.weights = torch.as_tensor(weights, device=device)
        self.replicates = counts


class ChemCPAv5(ChemCPAv4):
    def pseudobulk_loss(self):
        """Counterfactual pseudobulk MSE with replicate weights (evaluation-mode modules)."""
        target = self.pseudobulk
        chosen, rows = target.sample()
        P, R, M = rows.shape
        flat = rows.reshape(-1)
        values, library = target.values[flat], target.library[flat].reshape(P*R, M)
        pairs = [target.pairs[i] for i in chosen.tolist()]
        per_cell = lambda key: [p[key] for p in pairs for _ in range(R*M)]
        treated = self.encode_conditions(per_cell('compound'), per_cell('dose_nM'), per_cell('cell_type'))
        vehicle = self.encode_conditions(['DMSO']*len(flat), [0.0]*len(flat), per_cell('cell_type'))
        was_training = self.model.training
        self.model.eval()
        try:
            outputs = []
            for drugs, dose, covariates in (treated, vehicle):
                recon, _ = self.model.predict(genes=values, drugs_idx=drugs, dosages=dose, covariates=covariates)
                outputs.append(torch_pseudobulk(self.model_values(recon).reshape(P*R, M, -1), library))
        finally:
            self.model.train(was_training)
        weights = target.weights[chosen.to(self.device)].unsqueeze(-1)
        contrast = ((outputs[0]-outputs[1]).reshape(P, R, -1)*weights).sum(dim=1)
        return ((contrast-target.truth[chosen.to(self.device)])**2).mean()
