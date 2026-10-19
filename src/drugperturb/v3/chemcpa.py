"""chemCPA fitting without Lightning, following chemCPA/lightning_module.py.

The upstream ComPert module is used unchanged. The training step reproduces the
published losses and updates: Gaussian NLL reconstruction; drug (BCE) and
cell-type (CE) adversaries on the basal latent with a gradient penalty; the
autoencoder and doser maximize adversary loss; gradient-norm clipping at 1; Adam
optimizers with StepLR(gamma=0.9) stepped each epoch. The adversary replaces the
autoencoder update on every ``adversary_steps``-th mini-batch, by batch count;
Lightning's manual-optimization global-step counter is not replicated.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..third_party._chemcpa_model_upstream import ComPert

VEHICLE_DOSE = 1.0
DOSE_SCALE = 1e4


class ChemCPA:
    def __init__(self, *, hparams, additional, n_genes, drug_vocabulary, drug_matrix, cell_types,
                 use_cell_type, device, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = device
        self.vocabulary = {cid: i for i, cid in enumerate(drug_vocabulary)}
        self.cell_types = {c: i for i, c in enumerate(cell_types)}
        self.use_cell_type = use_cell_type
        embedding = torch.nn.Embedding.from_pretrained(
            torch.as_tensor(np.asarray(drug_matrix), dtype=torch.float32, device=device), freeze=True)
        self.model = ComPert(
            num_genes=n_genes, num_drugs=len(drug_vocabulary),
            num_covariates=[len(cell_types)] if use_cell_type else [0], device=device, seed=seed,
            doser_type=additional['doser_type'], decoder_activation=additional['decoder_activation'],
            hparams=dict(hparams), drug_embeddings=embedding, use_drugs_idx=True).to(device)
        h = self.model.hparams
        ae = [*self.model.encoder.parameters(), *self.model.decoder.parameters(),
              *self.model.drug_embedding_encoder.parameters()]
        adversaries = [*self.model.adversary_drugs.parameters()]
        if use_cell_type:
            ae += [p for e in self.model.covariates_embeddings for p in e.parameters()]
            adversaries += [p for a in self.model.adversary_covariates for p in a.parameters()]
        self.ae_params, self.doser_params = ae, list(self.model.dosers.parameters())
        self.opt_ae = torch.optim.Adam(ae, lr=h['autoencoder_lr'], weight_decay=h['autoencoder_wd'])
        self.opt_adv = torch.optim.Adam(adversaries, lr=h['adversary_lr'], weight_decay=h['adversary_wd'])
        self.opt_dos = torch.optim.Adam(self.doser_params, lr=h['dosers_lr'], weight_decay=h['dosers_wd'])
        self.schedulers = [torch.optim.lr_scheduler.StepLR(o, step_size=h['step_size_lr'], gamma=0.9)
                           for o in (self.opt_ae, self.opt_adv, self.opt_dos)]
        self.batches = 0

    def encode_conditions(self, compounds, doses_nM, cell_types):
        drugs = torch.as_tensor([self.vocabulary[c] for c in compounds], dtype=torch.long, device=self.device)
        dose = np.where(np.asarray(compounds) == 'DMSO', VEHICLE_DOSE, np.asarray(doses_nM, float)/DOSE_SCALE)
        dose = torch.as_tensor(dose, dtype=torch.float32, device=self.device)
        covariates = None
        if self.use_cell_type:
            index = torch.as_tensor([self.cell_types[c] for c in cell_types], device=self.device)
            covariates = [F.one_hot(index, len(self.cell_types)).float()]
        return drugs, dose, covariates

    def _losses(self, genes, drugs, dose, covariates):
        m = self.model
        recon, _, (basal, _, _) = m.predict(genes=genes, drugs_idx=drugs, dosages=dose,
                                            covariates=covariates, return_latent_basal=True)
        dim = recon.shape[1]//2
        reconstruction = m.loss_autoencoder(recon[:, :dim], genes, recon[:, dim:])
        drug_logits = m.adversary_drugs(basal)
        drug_loss = m.loss_adversary_drugs(drug_logits, F.one_hot(drugs, m.num_drugs).float())
        cov_logits, cov_loss = [], torch.zeros((), device=self.device)
        if self.use_cell_type:
            for i, adversary in enumerate(m.adversary_covariates):
                logits = adversary(basal)
                cov_logits.append(logits)
                cov_loss = cov_loss+m.loss_adversary_covariates[i](logits, covariates[i].argmax(1))
        return reconstruction, drug_loss, cov_loss, basal, drug_logits, cov_logits

    def train_batch(self, genes, drugs, dose, covariates):
        h = self.model.hparams
        self.model.train()
        reconstruction, drug_loss, cov_loss, basal, drug_logits, cov_logits = self._losses(genes, drugs, dose, covariates)
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
            (reconstruction-h['reg_adversary']*drug_loss-h.get('reg_adversary_cov', 1.0)*cov_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.ae_params, 1.0)
            torch.nn.utils.clip_grad_norm_(self.doser_params, 1.0)
            self.opt_ae.step()
            self.opt_dos.step()
        self.batches += 1
        return float(reconstruction.detach()), float(drug_loss.detach()), float(cov_loss.detach())

    @torch.no_grad()
    def predict_mean(self, values, condition):
        self.model.eval()
        genes = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        n = genes.shape[0]
        drugs, dose, cov = self.encode_conditions([condition['compound']]*n, [condition['dose_nM']]*n,
                                                  [condition['cell_type']]*n)
        recon, _ = self.model.predict(genes=genes, drugs_idx=drugs, dosages=dose, covariates=cov)
        return recon[:, :recon.shape[1]//2].double().cpu().numpy()


def fit(model, X, compounds, doses, cell_types, *, batch_size, max_epochs, evaluate_every,
        patience, max_minutes, seed, validate):
    """Train; every ``evaluate_every`` epochs call ``validate(model) -> RMSE``.

    Returns the lowest-validation state, its score/epoch and the full history.
    """
    genes = torch.as_tensor(X, dtype=torch.float32, device=model.device)
    drugs, dose, covariates = model.encode_conditions(compounds, doses, cell_types)
    generator = torch.Generator().manual_seed(seed)
    history, best, start, stale = [], dict(score=np.inf, epoch=0, state=None), time.perf_counter(), 0
    n = genes.shape[0]
    for epoch in range(1, max_epochs+1):
        order = torch.randperm(n, generator=generator).to(model.device)
        sums = np.zeros(3)
        batches = 0
        for begin in range(0, n, batch_size):
            index = order[begin:begin+batch_size]
            if index.numel() < 2:  # BatchNorm cannot train on a single cell.
                continue
            cov = None if covariates is None else [c[index] for c in covariates]
            sums += model.train_batch(genes[index], drugs[index], dose[index], cov)
            batches += 1
        for scheduler in model.schedulers:
            scheduler.step()
        record = dict(epoch=epoch, reconstruction=sums[0]/batches, adversary_drug=sums[1]/batches,
                      adversary_cell_type=sums[2]/batches, minutes=(time.perf_counter()-start)/60)
        if epoch % evaluate_every == 0 or epoch == max_epochs:
            record['validation_rmse'] = score = validate(model)
            if score < best['score']:
                best = dict(score=score, epoch=epoch, state=copy.deepcopy(model.model.state_dict()))
                stale = 0
            else:
                stale += 1
        history.append(record)
        if stale >= patience or record['minutes'] >= max_minutes:
            break
    if best['state'] is None:
        raise RuntimeError('no validation evaluation completed')
    model.model.load_state_dict(best['state'])
    return dict(best_validation_rmse=best['score'], best_epoch=best['epoch'],
                epochs_executed=len(history), history=history,
                stopped=('patience' if stale >= patience else 'time' if history[-1]['minutes'] >= max_minutes
                         else 'max_epochs'))
