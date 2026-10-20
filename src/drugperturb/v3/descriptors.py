"""RDKit 2D normalized descriptors scaled on training compounds only.

chemCPA and biolord both used descriptastorus RDKit2DNormalized features for
sci-Plex3. Descriptors are label-free, but scaling, column removal and missing
value fills are still fitted on training compounds alone.
"""
from __future__ import annotations

import numpy as np

VEHICLE_ID = 'DMSO'
VEHICLE_SMILES = 'CS(C)=O'

# RDKit2DNormalized values lie in [0, 1]. A column that is constant on the training
# compounds still gets a floating-point standard deviation of up to ~3e-15, while the
# least variable informative column in the sci-Plex3 and OP3 training sets has 2e-3.
# The v3/v4 rule (std > 0) keeps such columns; see docs/POST_V4_ENGINEERING.md.
NOISE_STD_TOLERANCE = 1e-8


def rdkit2d_normalized(smiles):
    """Raw descriptor matrix (NaN where a descriptor is undefined) and names."""
    from descriptastorus.descriptors import rdNormalizedDescriptors
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    names = [name for name, _ in generator.columns]
    rows = []
    for value in smiles:
        result = generator.process(value)
        if result is None or not result[0]:
            raise ValueError(f'descriptor generation failed for {value!r}')
        rows.append(np.asarray(result[1:], dtype=np.float64))
    matrix = np.vstack(rows)
    matrix[~np.isfinite(matrix)] = np.nan
    if matrix.shape[1] != len(names):
        raise ValueError('descriptor width differs from generator names')
    return matrix, names


class TrainingDescriptorScaler:
    """Drop training columns with std <= min_std, fill NaN with training means, z-score.

    The default min_std=0 is the v3/v4 rule, which keeps columns that are constant up
    to floating-point noise; min_std=NOISE_STD_TOLERANCE drops them.
    """

    def __init__(self, min_std=0.0):
        if not min_std >= 0:
            raise ValueError('min_std must be nonnegative')
        self.min_std = float(min_std)

    def fit(self, train):
        values = np.asarray(train, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2:
            raise ValueError('need at least two training compounds')
        mean = np.nanmean(values, axis=0)
        filled = np.where(np.isnan(values), mean, values)
        std = filled.std(axis=0)
        keep = np.isfinite(mean) & (std > self.min_std)
        if not keep.any():
            raise ValueError('no informative training descriptor columns')
        self.keep, self.mean, self.std = keep, mean[keep], std[keep]
        self.n_input = values.shape[1]
        return self

    def transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_input:
            raise ValueError('descriptor width differs from the fitted scaler')
        kept = values[:, self.keep]
        kept = np.where(np.isnan(kept), self.mean, kept)
        return ((kept - self.mean) / self.std).astype(np.float32)


def compound_table(compound_ids, smiles, train_ids, min_std=0.0):
    """Scaled descriptors for every compound plus the DMSO vehicle, keyed by ID.

    min_std is passed to TrainingDescriptorScaler; the default reproduces v3/v4.
    """
    ids = list(compound_ids)
    if VEHICLE_ID in ids or len(set(ids)) != len(ids) or len(ids) != len(smiles):
        raise ValueError('compound IDs must be unique, aligned with SMILES, and exclude DMSO')
    train = set(train_ids)
    if not train or not train.issubset(ids):
        raise ValueError('training IDs must be a nonempty subset of the compound IDs')
    raw, names = rdkit2d_normalized([*smiles, VEHICLE_SMILES])
    order = [*ids, VEHICLE_ID]
    scaler = TrainingDescriptorScaler(min_std).fit(raw[[i for i, cid in enumerate(ids) if cid in train]])
    scaled = scaler.transform(raw)
    kept = [name for name, flag in zip(names, scaler.keep) if flag]
    return {cid: scaled[i] for i, cid in enumerate(order)}, kept
