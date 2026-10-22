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
# Sealing refuses fits whose scaled descriptors exceed this |z| (scripts/v4_seal.py). Under the
# NOISE_STD_TOLERANCE rule the largest |z| of any compound is 9.2 in OP3 and 10.9 in sci-Plex3; a kept
# training-constant column gives ~1e16 to a compound that differs from training in it.
DESCRIPTOR_Z_LIMIT = 1e3


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


def json_float(value):
    """A |z| beyond the float32 range becomes None (JSON null); the gate treats it as above any limit."""
    return float(value) if np.isfinite(value) else None


def scaling_diagnostics(table, splits, limit=DESCRIPTOR_Z_LIMIT):
    """Largest |z| per split and the compounds above limit, for a compound_table result.

    splits maps compound ID to split; the vehicle is its own split. Label-free: descriptors only.
    """
    largest, above = {}, {}
    for cid, row in table.items():
        split = 'vehicle' if cid == VEHICLE_ID else splits[cid]
        z = float(np.abs(row).max())
        largest[split] = max(largest.get(split, 0.0), z)
        if not z <= limit:
            above.setdefault(split, {})[cid] = z
    return dict(z_limit=float(limit), max_abs_z={s: json_float(z) for s, z in largest.items()},
                above_limit={s: {c: json_float(z) for c, z in found.items()} for s, found in above.items()})


def descriptor_gate(records, override=False, limit=DESCRIPTOR_Z_LIMIT):
    """Check the fit records' descriptor_scaling before sealing; the summary is stored in the seal.

    records maps fit name to fit record. Raises, naming the compounds, if any |z| exceeds limit or a
    record has no diagnostics (written before this gate), unless override is set.
    """
    largest, above, missing = {}, {}, []
    for name, record in records.items():
        found = record.get('descriptor_scaling')
        if found is None:
            missing.append(name)
            continue
        for split, z in found['max_abs_z'].items():
            largest[split] = max(largest.get(split, 0.0), np.inf if z is None else z)
        for split, compounds in found['above_limit'].items():
            for cid, z in compounds.items():
                if z is None or z > limit:
                    above.setdefault(split, {})[cid] = np.inf if z is None else z
    passed = not missing and all(z <= limit for z in largest.values())
    if not passed and not override:
        problems = ['largest |z| per split ' + ', '.join(f'{s} {z:.3g}' for s, z in largest.items())] if largest else []
        if above:
            problems.append(f'compounds above {limit:g}: '
                            + ', '.join(f'{c} ({s}, |z| = {z:.3g})' for s, found in above.items() for c, z in found.items()))
        if missing:
            problems.append(f'fits without descriptor diagnostics: {", ".join(missing)}')
        raise ValueError(f'descriptor gate: {"; ".join(problems)}. Refusing to seal; an explicit override '
                         '(v4_seal.py --override-descriptor-gate) is recorded in the seal')
    return dict(z_limit=float(limit), max_abs_z={s: json_float(z) for s, z in largest.items()},
                above_limit={s: {c: json_float(z) for c, z in found.items()} for s, found in above.items()},
                fits_without_diagnostics=missing, passed=passed, override=bool(override))
