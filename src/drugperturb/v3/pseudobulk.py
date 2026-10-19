"""Map cell-level model outputs onto the v1/v2 well-level response space.

Models see log1p(count / all-gene library * 1e4) for the frozen genes. The v1/v2
truth is log1p(CPM) of summed well counts. A library-weighted mean of
expm1(cell value) * 100 is exactly that CPM when the cell values are the measured
ones, so an identity model reproduces zero change exactly.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

MODEL_SCALE = 1e4
TO_CPM = 1e6 / MODEL_SCALE


def _library(library, n):
    values = np.asarray(library, dtype=np.float64)
    if values.shape != (n,) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('libraries must be finite positive all-gene UMI totals, one per cell')
    return values


def normalize_counts(counts, library) -> np.ndarray:
    """Dense float32 model input from raw frozen-gene counts and all-gene libraries."""
    matrix = sparse.csr_matrix(counts, dtype=np.float64)
    if matrix.nnz and (matrix.data < 0).any():
        raise ValueError('counts must be nonnegative')
    lib = _library(library, matrix.shape[0])
    # log1p(0) = 0: transform stored entries in float64, then densify once as float32
    # (a float64 dense intermediate would need ~7 GB for the Q2 cell archive).
    scaled = (sparse.diags(MODEL_SCALE / lib) @ matrix).tocsr()
    scaled.data = np.log1p(scaled.data)
    return scaled.astype(np.float32).toarray()


def pseudobulk_log_cpm(cell_values, library) -> np.ndarray:
    """log1p of the library-weighted mean CPM implied by per-cell model values."""
    values = np.asarray(cell_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError('cell values must be a finite nonempty (cells, genes) matrix')
    lib = _library(library, values.shape[0])
    expression = np.clip(np.expm1(values), 0.0, None) * TO_CPM
    return np.log1p(lib @ expression / lib.sum())


def measured_log_cpm(counts, library) -> np.ndarray:
    """Measured well log1p(CPM) for frozen genes from cell counts and libraries."""
    matrix = sparse.csr_matrix(counts, dtype=np.float64)
    lib = _library(library, matrix.shape[0])
    return np.log1p(np.asarray(matrix.sum(axis=0)).ravel() / lib.sum() * 1e6)


def compound_delta(replicate_deltas) -> np.ndarray:
    """Equal-weight mean of exactly two replicate deltas, as in v1/v2 truth."""
    values = np.asarray(replicate_deltas, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 2 or not np.isfinite(values).all():
        raise ValueError('expected two finite replicate delta vectors')
    return values.mean(axis=0)


def counterfactual_deltas(predict, queries, vehicle_cells, *, gene_count):
    """Primary contrast and anchored sensitivity for each compound/context query.

    ``predict(well, condition)`` returns per-cell model values for one vehicle
    well's cells under a condition dict. ``queries`` maps a key to two replicate
    records, each holding the treated condition, the vehicle condition and the
    strict matched vehicle well ID. ``vehicle_cells[well]`` holds the model input
    values, raw-count CSR, libraries and any model-specific cell indices.
    """
    contrast, anchored = {}, {}
    cache = {}
    for key, replicates in queries.items():
        if len(replicates) != 2:
            raise ValueError(f'{key}: expected exactly two replicate queries')
        c_rows, a_rows = [], []
        for record in replicates:
            well = vehicle_cells[record['vehicle_well']]
            treated = pseudobulk_log_cpm(predict(well, record['treated']), well['library'])
            cache_key = (record['vehicle_well'], tuple(sorted(record['vehicle'].items())))
            if cache_key not in cache:
                cache[cache_key] = pseudobulk_log_cpm(predict(well, record['vehicle']), well['library'])
            if treated.shape != (gene_count,):
                raise ValueError('prediction width differs from the frozen gene space')
            c_rows.append(treated - cache[cache_key])
            a_rows.append(treated - measured_log_cpm(well['counts'], well['library']))
        contrast[key] = compound_delta(c_rows)
        anchored[key] = compound_delta(a_rows)
    return contrast, anchored
