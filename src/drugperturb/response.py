"""Well-level responses and transformations fitted only to training compounds."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def build_responses(counts: pd.DataFrame, mapping: pd.DataFrame) -> dict:
    """CPM/log1p each well, subtract its matched control, average two deltas.

    Rows marked ineligible are ignored, without changing upstream selection.
    Output replicate rows are sorted by compound, then rep1/rep2; shared control
    columns are allowed but treatment Combo keys must be unique.
    """
    required = ['treatment', 'replicate', 'Combo', 'control_Combo', 'eligible']
    if not isinstance(counts, pd.DataFrame) or not isinstance(mapping, pd.DataFrame):
        raise ValueError('counts and mapping must be DataFrames')
    if not set(required).issubset(mapping.columns):
        raise ValueError(f'mapping requires columns {required}')
    if counts.index.has_duplicates or counts.index.isna().any() or len(counts.index) == 0:
        raise ValueError('gene index must be nonempty, unique, and nonmissing')
    if counts.columns.has_duplicates:
        raise ValueError('count Combo columns must be unique')
    if mapping['eligible'].isna().any() or not mapping['eligible'].isin([True, False]).all():
        raise ValueError('eligible must contain boolean values')
    selected = mapping.loc[mapping['eligible'].astype(bool)].copy()
    if selected.empty:
        raise ValueError('no eligible treatment replicates')
    if selected[required[:-1]].isna().any().any():
        raise ValueError('selected mapping keys and control keys cannot be missing')
    if selected['Combo'].duplicated().any():
        raise ValueError('treatment Combo must map one-to-one to count columns')
    for compound, group in selected.groupby('treatment', sort=False):
        if len(group) != 2 or set(group['replicate']) != {'rep1', 'rep2'}:
            raise ValueError(f'{compound}: expected exactly rep1 and rep2')
    selected = selected.sort_values(['treatment', 'replicate'], kind='stable').reset_index(drop=True)
    missing_treatment = set(selected['Combo']) - set(counts.columns)
    missing_control = set(selected['control_Combo']) - set(counts.columns)
    if missing_treatment:
        raise ValueError(f'missing treatment Combo count columns: {sorted(missing_treatment)}')
    if missing_control:
        raise ValueError(f'missing control count columns: {sorted(missing_control)}')
    keys = list(dict.fromkeys([*selected['Combo'], *selected['control_Combo']]))
    try:
        values = counts.loc[:, keys].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('selected counts must be numeric') from exc
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError('selected counts must be finite nonnegative integers')
    libraries = values.sum(axis=0)
    if not np.isfinite(libraries).all() or (libraries <= 0).any():
        raise ValueError('all selected libraries must have finite positive count sums')
    normalized = np.log1p(values / libraries[None, :] * 1e6)
    positions = {key: i for i, key in enumerate(keys)}
    treatment_columns = [positions[key] for key in selected['Combo']]
    control_columns = [positions[key] for key in selected['control_Combo']]
    replicate_delta = (normalized[:, treatment_columns] - normalized[:, control_columns]).T
    compound_ids = selected['treatment'].drop_duplicates().tolist()
    # Exactly two validated, adjacent replicate rows per compound.
    delta = replicate_delta.reshape(len(compound_ids), 2, len(counts.index)).mean(axis=1)
    return dict(compound_ids=compound_ids, gene_ids=counts.index.tolist(),
                delta=delta, replicate_delta=replicate_delta, replicate_manifest=selected)


def _finite_matrix(value, name):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f'{name} must be a finite 2D matrix')
    return array


class TrainingTargetTransform:
    """Select variable genes, standardize, and fit PCA using training data only.

    ``explained_variance`` contains the per-component explained variance ratios.
    Truth selection never projects/reconstructs data. Inversion returns selected
    gene responses on their original delta scale, generally the training mean
    for an all-zero PC vector; it must not implement the zero-response baseline.
    """
    def __init__(self, max_genes=2000, max_components=20):
        for name, value in [('max_genes', max_genes), ('max_components', max_components)]:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        self.max_genes = int(max_genes)
        self.max_components = int(max_components)

    def fit(self, y_train):
        train = _finite_matrix(y_train, 'y_train')
        if train.shape[0] < 2 or train.shape[1] == 0:
            raise ValueError('fit requires at least two training rows and one gene')
        variances = np.var(train, axis=0)
        if not np.isfinite(variances).all():
            raise ValueError('training gene variance overflow')
        positive = np.flatnonzero(variances > 0)
        if positive.size == 0:
            raise ValueError('no positive-variance training genes')
        indices = positive[np.argsort(-variances[positive], kind='stable')][:self.max_genes]
        scaler = StandardScaler().fit(train[:, indices])
        n_components = min(self.max_components, train.shape[0] - 1, len(indices))
        pca = PCA(n_components=n_components, svd_solver='full').fit(scaler.transform(train[:, indices]))
        self.n_input_genes = train.shape[1]
        self.gene_indices = indices
        self.scaler = scaler
        self.pca = pca
        self.n_components = n_components
        self.explained_variance = pca.explained_variance_ratio_.copy()
        return self

    def _check_fitted(self):
        if not hasattr(self, 'gene_indices'):
            raise ValueError('transform must be fitted on training targets first')

    def select_truth(self, y):
        self._check_fitted()
        array = _finite_matrix(y, 'y')
        if array.shape[1] != self.n_input_genes:
            raise ValueError('y gene count differs from training input')
        return array[:, self.gene_indices].copy()

    def transform(self, y):
        self._check_fitted()
        return self.pca.transform(self.scaler.transform(self.select_truth(y)))

    def inverse_transform(self, z):
        self._check_fitted()
        z = _finite_matrix(z, 'z')
        if z.shape[1] != self.n_components:
            raise ValueError('z component count differs from fitted PCA')
        return self.scaler.inverse_transform(self.pca.inverse_transform(z))


def baseline_predictions(y_train_selected, fp_train, fp_query) -> dict:
    """B0 zero delta, B1 training mean, B2 nearest binary Tanimoto neighbor.

    Ties use the first training row. Two empty fingerprints have similarity one;
    an empty/nonempty pair has similarity zero. No target transform is fitted.
    """
    targets = _finite_matrix(y_train_selected, 'y_train_selected')
    train = _finite_matrix(fp_train, 'fp_train')
    query = _finite_matrix(fp_query, 'fp_query')
    if len(targets) == 0 or targets.shape[1] == 0 or len(train) != len(targets):
        raise ValueError('training responses and fingerprints need equal nonzero row counts')
    if train.shape[1] == 0 or query.shape[1] != train.shape[1]:
        raise ValueError('fingerprint widths must be equal and nonzero')
    if not np.isin(train, [0, 1]).all() or not np.isin(query, [0, 1]).all():
        raise ValueError('Tanimoto fingerprints must be binary')
    intersection = query @ train.T
    union = query.sum(axis=1)[:, None] + train.sum(axis=1)[None, :] - intersection
    similarity = np.divide(intersection, union, out=np.ones_like(intersection), where=union != 0)
    nearest = similarity.argmax(axis=1)
    shape = (len(query), targets.shape[1])
    return {'B0': np.zeros(shape), 'B1': np.broadcast_to(targets.mean(axis=0), shape).copy(),
            'B2': targets[nearest].copy()}
