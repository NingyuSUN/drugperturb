"""chemCPA v4 arms: negative binomial likelihood and a counterfactual pseudobulk loss.

The upstream ComPert module stays unchanged; only the objective changes.

* A0 (v3 reference): exactly the v3 training step (Gaussian NLL on log1p values).
* NB likelihood: the decoder's first gene block is a log mean fraction eta;
  mu = library * exp(min(eta + b, 0)) with b the training-cell log fraction of
  each gene; the second block (softplus, from ComPert) is the inverse dispersion.
  The upstream NBLoss is applied to raw counts.
* Pseudobulk loss: each autoencoder/doser update adds the mean squared error
  between measured training deltas and counterfactual contrasts computed as at
  inference (vehicle cells under the compound minus the same cells under the
  vehicle, library-weighted pseudobulk, replicate mean) on cells sampled from the
  matched vehicle wells, with modules in evaluation mode.

Either way the encoder still sees log1p(count / library * 1e4).
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from ..third_party._chemcpa_model_upstream import NBLoss
from ..v3 import chemcpa as v3_chemcpa
from ..v3.pseudobulk import MODEL_SCALE, TO_CPM

NB_THETA_EPS = 1e-4
OFFSET_FLOOR = 1e-12


def gene_offsets(counts, library):
    """log pooled fraction of each gene over the given (training) cells."""
    totals = np.asarray(counts.sum(axis=0), dtype=np.float64).ravel()
    return np.log(np.maximum(totals/float(np.sum(library)), OFFSET_FLOOR))


def torch_pseudobulk(values, library):
    """log1p library-weighted mean CPM; values (groups, cells, genes), library (groups, cells)."""
    expression = torch.clamp(torch.expm1(values), min=0.0)*TO_CPM
    weights = library/library.sum(dim=1, keepdim=True)
    return torch.log1p(torch.einsum('gc,gcn->gn', weights, expression))


class PseudobulkTarget:
    """Training (compound, context) pairs, their measured deltas and matched vehicle cells on device.

    ``pairs`` is a list of dicts with compound, dose_nM, cell_type and wells
    (one vehicle well per replicate); ``truth`` is (pairs, genes). Vehicle cells
    are stored contiguously per well so sampling is an offset plus a draw.
    """

    def __init__(self, *, pairs, truth, vehicle_cells, n_replicates, pairs_per_update, cells_per_well, seed, device):
        if not pairs or len(pairs) != len(truth):
            raise ValueError('pseudobulk target needs aligned nonempty pairs and truth')
        wells = sorted({w for p in pairs for w in p['wells']})
        values, library, starts, sizes, offset = [], [], {}, {}, 0
        for well in wells:
            cells = vehicle_cells[well]
            values.append(np.asarray(cells['values'], np.float32))
            library.append(np.asarray(cells['library'], np.float32))
            starts[well], sizes[well] = offset, cells['n_cells']
            offset += cells['n_cells']
        self.values = torch.as_tensor(np.vstack(values), device=device)
        self.library = torch.as_tensor(np.concatenate(library), device=device)
        if any(len(p['wells']) != n_replicates for p in pairs):
            raise ValueError(f'every pair needs {n_replicates} vehicle wells')
        self.start = torch.as_tensor([[starts[w] for w in p['wells']] for p in pairs], dtype=torch.long)
        self.size = torch.as_tensor([[sizes[w] for w in p['wells']] for p in pairs], dtype=torch.long)
        self.truth = torch.as_tensor(np.asarray(truth, np.float32), device=device)
        self.pairs, self.n_replicates = pairs, n_replicates
        self.pairs_per_update, self.cells_per_well = pairs_per_update, cells_per_well
        self.generator = torch.Generator().manual_seed(seed)
        self.device = device

    def sample(self):
        """Indices of P pairs (without replacement) and (P, R, m) vehicle-cell rows."""
        chosen = torch.randperm(len(self.pairs), generator=self.generator)[:self.pairs_per_update]
        draws = torch.rand((len(chosen), self.n_replicates, self.cells_per_well), generator=self.generator)
        rows = self.start[chosen].unsqueeze(-1)+(draws*self.size[chosen].unsqueeze(-1)).long()
        return chosen, rows.to(self.device)


class ChemCPAv4(v3_chemcpa.ChemCPA):
    def __init__(self, *, likelihood='gaussian', gene_offset=None, pseudobulk_weight=0.0, pseudobulk_target=None,
                 **kwargs):
        super().__init__(**kwargs)
        if likelihood not in ('gaussian', 'nb'):
            raise ValueError('likelihood must be gaussian or nb')
        if likelihood == 'nb':
            if gene_offset is None:
                raise ValueError('the NB likelihood needs training-cell gene offsets')
            self.offset = torch.as_tensor(np.asarray(gene_offset, np.float32), device=self.device)
            self.nb_loss = NBLoss()
        if pseudobulk_weight and pseudobulk_target is None:
            raise ValueError('a pseudobulk weight needs a pseudobulk target')
        self.likelihood, self.pseudobulk_weight, self.pseudobulk = likelihood, float(pseudobulk_weight), pseudobulk_target

    def log_fraction(self, recon):
        dim = recon.shape[1]//2
        return torch.clamp(recon[:, :dim]+self.offset, max=0.0)

    def model_values(self, recon):
        """Per-cell values on the log1p(fraction * 1e4) scale that the pseudobulk mapping expects."""
        if self.likelihood == 'gaussian':
            return recon[:, :recon.shape[1]//2]
        return torch.log1p(MODEL_SCALE*torch.exp(self.log_fraction(recon)))

    def _reconstruction(self, recon, genes, counts, library):
        dim = recon.shape[1]//2
        if self.likelihood == 'gaussian':
            return self.model.loss_autoencoder(recon[:, :dim], genes, recon[:, dim:])
        mu = library.unsqueeze(1)*torch.exp(self.log_fraction(recon))
        return self.nb_loss(torch.cat([mu, recon[:, dim:]+NB_THETA_EPS], dim=1), counts)

    def _losses(self, genes, drugs, dose, covariates, counts=None, library=None):
        m = self.model
        recon, _, (basal, _, _) = m.predict(genes=genes, drugs_idx=drugs, dosages=dose,
                                            covariates=covariates, return_latent_basal=True)
        reconstruction = self._reconstruction(recon, genes, counts, library)
        drug_logits = m.adversary_drugs(basal)
        drug_loss = m.loss_adversary_drugs(drug_logits, torch.nn.functional.one_hot(drugs, m.num_drugs).float())
        cov_logits, cov_loss = [], torch.zeros((), device=self.device)
        if self.use_cell_type:
            for i, adversary in enumerate(m.adversary_covariates):
                logits = adversary(basal)
                cov_logits.append(logits)
                cov_loss = cov_loss+m.loss_adversary_covariates[i](logits, covariates[i].argmax(1))
        return reconstruction, drug_loss, cov_loss, basal, drug_logits, cov_logits

    def pseudobulk_loss(self):
        """Counterfactual pseudobulk MSE on one sampled set of training pairs (evaluation-mode modules)."""
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
        contrast = (outputs[0]-outputs[1]).reshape(P, R, -1).mean(dim=1)
        return ((contrast-target.truth[chosen.to(self.device)])**2).mean()

    def train_batch(self, genes, drugs, dose, covariates, counts=None, library=None):
        h = self.model.hparams
        self.model.train()
        reconstruction, drug_loss, cov_loss, basal, drug_logits, cov_logits = self._losses(
            genes, drugs, dose, covariates, counts, library)
        pseudobulk = torch.zeros((), device=self.device)
        if self.batches % h['adversary_steps'] == 0:
            self.opt_adv.zero_grad()
            penalty = torch.autograd.grad(drug_logits.sum(), basal, create_graph=True)[0].pow(2).mean()
            for logits in cov_logits:
                penalty = penalty+torch.autograd.grad(logits.sum(), basal, create_graph=True)[0].pow(2).mean()
            (drug_loss+cov_loss+h['penalty_adversary']*penalty).backward()
            self.opt_adv.step()
        else:
            self.opt_ae.zero_grad()
            self.opt_dos.zero_grad()
            loss = reconstruction-h['reg_adversary']*drug_loss-h.get('reg_adversary_cov', 1.0)*cov_loss
            if self.pseudobulk_weight:
                pseudobulk = self.pseudobulk_loss()
                loss = loss+self.pseudobulk_weight*pseudobulk
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ae_params, 1.0)
            torch.nn.utils.clip_grad_norm_(self.doser_params, 1.0)
            self.opt_ae.step()
            self.opt_dos.step()
        self.batches += 1
        return (float(reconstruction.detach()), float(drug_loss.detach()), float(cov_loss.detach()),
                float(pseudobulk.detach()))

    @torch.no_grad()
    def predict_mean(self, values, condition):
        self.model.eval()
        genes = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        n = genes.shape[0]
        drugs, dose, cov = self.encode_conditions([condition['compound']]*n, [condition['dose_nM']]*n,
                                                  [condition['cell_type']]*n)
        recon, _ = self.model.predict(genes=genes, drugs_idx=drugs, dosages=dose, covariates=cov)
        return self.model_values(recon).double().cpu().numpy()

    @torch.no_grad()
    def latent_summary(self, vehicle_values, conditions, cell_type):
        """Mean L2 norms of the basal latent, drug terms and drug-minus-vehicle terms.

        ``conditions`` are (compound, dose_nM) pairs; vehicle cells come from one
        cell type (the covariate embedding cancels in the contrast).
        """
        self.model.eval()
        basal = self.model.encoder(torch.as_tensor(vehicle_values, dtype=torch.float32, device=self.device))
        compounds = [c for c, _ in conditions]
        drugs, dose, _ = self.encode_conditions(compounds, [d for _, d in conditions], [cell_type]*len(compounds))
        term = self.model.compute_drug_embeddings_(drugs_idx=drugs, dosages=dose)
        v_drugs, v_dose, _ = self.encode_conditions(['DMSO'], [0.0], [cell_type])
        vehicle = self.model.compute_drug_embeddings_(drugs_idx=v_drugs, dosages=v_dose)
        basal_norm = float(basal.norm(dim=1).mean())
        contrast = float((term-vehicle).norm(dim=1).mean())
        return dict(basal_norm=basal_norm, drug_term_norm=float(term.norm(dim=1).mean()),
                    vehicle_term_norm=float(vehicle.norm(dim=1).mean()), drug_contrast_norm=contrast,
                    latent_ratio=contrast/basal_norm if basal_norm > 0 else float('nan'))


def fit(model, X, compounds, doses, cell_types, *, batch_size, max_epochs, evaluate_every, patience, max_minutes,
        seed, validate, counts=None, library=None):
    """v3 ``fit`` plus raw-count batches for the NB likelihood and a pseudobulk-loss history column.

    Cell-batch order uses the same generator calls as v3, so A0 repeats the v3
    fit for the same seed and device.
    """
    genes = torch.as_tensor(X, dtype=torch.float32, device=model.device)
    drugs, dose, covariates = model.encode_conditions(compounds, doses, cell_types)
    nb = model.likelihood == 'nb'
    if nb:
        if counts is None or library is None:
            raise ValueError('the NB likelihood needs raw counts and libraries')
        counts = counts.tocsr()
        library_t = torch.as_tensor(np.asarray(library, np.float32), device=model.device)
    generator = torch.Generator().manual_seed(seed)
    history, best, start, stale = [], dict(score=np.inf, epoch=0, state=None), time.perf_counter(), 0
    n = genes.shape[0]
    for epoch in range(1, max_epochs+1):
        order_cpu = torch.randperm(n, generator=generator)
        order = order_cpu.to(model.device)
        sums, batches = np.zeros(4), 0
        for begin in range(0, n, batch_size):
            index = order[begin:begin+batch_size]
            if index.numel() < 2:  # BatchNorm cannot train on a single cell.
                continue
            cov = None if covariates is None else [c[index] for c in covariates]
            extra = {}
            if nb:
                rows = order_cpu[begin:begin+batch_size].numpy()
                extra = dict(counts=torch.as_tensor(counts[rows].toarray(), dtype=torch.float32, device=model.device),
                             library=library_t[index])
            sums += model.train_batch(genes[index], drugs[index], dose[index], cov, **extra)
            batches += 1
        for scheduler in model.schedulers:
            scheduler.step()
        record = dict(epoch=epoch, reconstruction=sums[0]/batches, adversary_drug=sums[1]/batches,
                      adversary_cell_type=sums[2]/batches, pseudobulk=sums[3]/batches,
                      minutes=(time.perf_counter()-start)/60)
        if epoch % evaluate_every == 0 or epoch == max_epochs:
            record['validation_rmse'] = score = validate(model)
            if score < best['score']:
                best = dict(score=score, epoch=epoch, state=copy.deepcopy(model.model.state_dict()))
                stale = 0
            else:
                stale += 1
        history.append(record)
        if not np.isfinite(record['reconstruction']):
            break
        if stale >= patience or record['minutes'] >= max_minutes:
            break
    if best['state'] is None:
        raise RuntimeError('no validation evaluation completed')
    model.model.load_state_dict(best['state'])
    stopped = ('nonfinite_loss' if not np.isfinite(history[-1]['reconstruction']) else 'patience' if stale >= patience
               else 'time' if history[-1]['minutes'] >= max_minutes else 'max_epochs')
    return dict(best_validation_rmse=best['score'], best_epoch=best['epoch'], epochs_executed=len(history),
                history=history, stopped=stopped)
