"""v6 extraction: v5's streaming extraction with every unit split into a selection and an evaluation half.

The v6 endpoint needs two independent measurements of each unit, so each unit's
cells are partitioned once, deterministically, into half S and half E
(protocol_v6 measurement_split). The partition is recorded per cell, so it is
auditable and identical for every method and arm.

The split is implemented by doubling the unit space: the accumulator sees one
row per (unit, half), named ``<unit_id>:S`` and ``<unit_id>:E``, which lets
v5's shard loop, decoding and part files work unchanged. Model-cell selection
runs first, on whole units, so the 32-cell cap is a cap per unit and not per
half.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from ..v5.extract import (MODEL_ROLES, OBS_COLUMNS, PASS_VALUE, SHARD_COLUMNS, Accumulator,  # noqa: F401
                          decode_cells, design_cells, gene_lookup, load_npz, save_part, select_model_cells,
                          treatment_unit, unit_table, vehicle_unit)

HALVES = ('S', 'E')
SPLIT_SEED = 606  # protocol_v6 measurement_split


def assign_halves(cells, *, seed=SPLIT_SEED):
    """Half label per cell: within each unit, barcodes are permuted once and cut in two.

    A unit with an odd number of cells gives the extra cell to the evaluation
    half, as the protocol states. The permutation is drawn from a single
    ``default_rng(seed)``, visiting units in unit order and barcodes in sorted
    order within a unit, so the labels depend on nothing but the cell table.
    """
    barcode, unit = cells.barcode.to_numpy(), cells.unit.to_numpy()
    order = np.lexsort((barcode, unit))
    unit_sorted = unit[order]
    starts = np.flatnonzero(np.r_[True, unit_sorted[1:] != unit_sorted[:-1]])
    rng = np.random.default_rng(seed)
    half = np.empty(len(cells), dtype='<U1')
    for start, stop in zip(starts, np.r_[starts[1:], len(order)]):
        members = order[start:stop]
        shuffled = members[rng.permutation(len(members))]
        half[shuffled[:len(members)//2]] = 'S'
        half[shuffled[len(members)//2:]] = 'E'
    return half


def half_units(units):
    """Two rows per unit, S then E, with the parent unit and half recorded."""
    frames = []
    for half in HALVES:
        frame = units.copy()
        frame['parent_unit'] = units.unit_id.to_numpy()
        frame['half'] = half
        frame['unit_id'] = units.unit_id.astype(str)+':'+half
        frames.append(frame)
    doubled = pd.concat(frames, ignore_index=True)
    order = np.lexsort((doubled.half.map({h: i for i, h in enumerate(HALVES)}).to_numpy(),
                        doubled.parent_unit.to_numpy()))
    return doubled.iloc[order].reset_index(drop=True)


def half_unit_positions(cells, halves):
    """Row of ``halves`` for each cell, from its unit and half label."""
    position = pd.Series(np.arange(len(halves)), index=halves.unit_id)
    keys = pd.Index(cells.parent_unit.astype(str)+':'+cells.half.astype(str))
    located = position.reindex(keys)
    if located.isna().any():
        raise ValueError('a cell names a (unit, half) absent from the doubled unit table')
    return located.to_numpy(np.int64)


def prepared_cells(obs, units, *, seed=SPLIT_SEED, pass_value=PASS_VALUE):
    """v5 design cells plus the half label, the parent unit id and the doubled-unit row."""
    cells = design_cells(obs, units, pass_value=pass_value)
    cells = cells.assign(parent_unit=units.unit_id.to_numpy(str)[cells.unit.to_numpy()],
                         half=assign_halves(cells, seed=seed))
    return cells


def accumulator_cells(cells, halves):
    """The cell table the accumulator sees: ``unit`` points into the doubled unit table."""
    return cells.assign(unit=half_unit_positions(cells, halves))


def expected_half_cells(cells, halves):
    return np.bincount(half_unit_positions(cells, halves), minlength=len(halves)).astype(np.int64)


def split_halves(ids, counts, library, n_cells):
    """Split a doubled-unit result back into {'S': ..., 'E': ...} keyed by the parent unit id.

    Returns, per half, the parent unit ids in their doubled-table order and the
    rows of ``counts``, ``library`` and ``n_cells`` belonging to that half.
    """
    ids = np.asarray(ids, str)
    suffix = np.asarray([i.rsplit(':', 1)[-1] for i in ids])
    parent = np.asarray([i.rsplit(':', 1)[0] for i in ids])
    unknown = sorted(set(suffix)-set(HALVES))
    if unknown:
        raise ValueError(f'unit ids carry unknown half labels: {unknown}')
    output = {}
    for half in HALVES:
        rows = np.flatnonzero(suffix == half)
        output[half] = dict(unit_id=parent[rows], counts=counts[rows], library=library[rows],
                            n_cells=n_cells[rows], row_of={u: i for i, u in enumerate(parent[rows])})
    if list(output['S']['unit_id']) != list(output['E']['unit_id']):
        raise ValueError('the two halves must cover the same units in the same order')
    return output


def merge_parts(parts, halves, *, expected_cells, model_barcodes, cap, shard_manifest):
    """Combine v6 extraction parts and run the extraction anchors; raises on any failure.

    Anchors: every manifest shard processed exactly once; per-(unit, half) cell
    numbers equal the obs_metadata pass-filter counts; the two halves of a unit
    are disjoint and together hold all of its cells; half libraries equal half
    sums; every selected model cell found once; a unit's model cells never
    exceed its two halves' sums, and equal them where the unit holds no more
    than ``cap`` cells.
    """
    shards = pd.concat([pd.read_csv(p/'shards.tsv', sep='\t') for p in map(_path, parts)], ignore_index=True)
    if shards.shard.duplicated().any() or set(shards.shard) != set(shard_manifest.path):
        raise ValueError('extraction parts must cover every manifest shard exactly once')
    total, library, n_cells, gene_ids = None, 0, 0, None
    detected, vehicle_cells, model = 0, 0, []
    for part in map(_path, parts):
        u = load_npz(part/'unit_counts.npz')
        if not np.array_equal(u['unit_ids'], halves.unit_id.to_numpy(str)):
            raise ValueError(f'{part}: unit order differs')
        matrix = sparse.csr_matrix((u['data'], u['indices'], u['indptr']), shape=tuple(u['shape'])).astype(np.int64)
        total = matrix if total is None else total+matrix
        library, n_cells, gene_ids = library+u['library'], n_cells+u['n_cells'], u['gene_ids']
        v = load_npz(part/'vehicle_detection.npz')
        detected, vehicle_cells = detected+v['detected'], vehicle_cells+int(v['vehicle_cells'])
        m = load_npz(part/'model_cells.npz')
        model.append((m['barcode'], m['unit_id'], m['library'],
                      sparse.csr_matrix((m['data'], m['indices'], m['indptr']), shape=tuple(m['shape']))))
    checks = dict(shards=len(shards))
    checks['half_cells_equal_obs_pass_filter'] = bool(np.array_equal(n_cells, expected_cells))
    checks['half_library_equals_half_sums'] = bool(np.array_equal(np.asarray(total.sum(axis=1)).ravel(), library))
    parent = halves.parent_unit.to_numpy(str)
    fold = sparse.csr_matrix((np.ones(len(halves), np.int64),
                              (pd.factorize(parent, sort=True)[0], np.arange(len(halves)))))
    whole_cells = np.asarray(fold@n_cells).ravel()
    checks['halves_partition_each_unit'] = bool((np.asarray(fold.sum(axis=1)).ravel() == len(HALVES)).all())
    barcode, unit_id, cell_library, counts = _stack_model(model)
    checks['model_cells_found_once'] = bool(len(barcode) == len(set(barcode)) == len(model_barcodes)
                                            and set(barcode) == set(model_barcodes))
    whole_counts = (fold@total).tocsr()
    whole_ids = np.asarray(sorted(set(parent)), str)
    position = pd.Series(np.arange(len(whole_ids)), index=whole_ids)
    model_parent = np.asarray([u.rsplit(':', 1)[0] for u in unit_id], str)
    rows = position.reindex(model_parent)
    if rows.isna().any():
        raise ValueError('a model cell names a unit absent from the extraction')
    indicator = sparse.csr_matrix((np.ones(len(rows), np.int64), (rows.to_numpy(np.int64), np.arange(len(rows)))),
                                  shape=(len(whole_ids), len(rows)))
    model_sums = (indicator@counts.astype(np.int64)).tocsr()
    present = np.unique(rows.to_numpy(np.int64))
    difference = (whole_counts[present]-model_sums[present]).tocsr()
    checks['model_cells_within_unit_sums'] = bool(difference.nnz == 0 or difference.data.min() >= 0)
    complete = present[whole_cells[present] <= cap]
    checks['model_cells_equal_unit_sums_where_all_kept'] = bool((whole_counts[complete] != model_sums[complete]).nnz == 0)
    checks['model_library_equals_counts'] = bool(np.array_equal(np.asarray(counts.sum(axis=1)).ravel(), cell_library))
    failed = [k for k, v in checks.items() if v is False]
    if failed:
        raise ValueError(f'extraction anchors failed: {failed}')
    stats = {k: int(shards[k].sum()) for k in ['rows', 'matched_rows', 'markers', 'unknown_token_entries',
                                               'unknown_token_umi', 'duplicate_token_entries'] if k in shards}
    return dict(half_counts=total.tocsr(), half_library=library, half_cells=n_cells, half_ids=halves.unit_id.to_numpy(str),
                whole_counts=whole_counts, whole_ids=whole_ids, whole_cells=whole_cells,
                whole_library=np.asarray(fold@library).ravel(), gene_ids=gene_ids,
                detected=detected, vehicle_cells=vehicle_cells,
                model=dict(barcode=barcode, unit_id=model_parent, library=cell_library, counts=counts),
                checks=checks, shard_stats=stats)


def _path(part):
    from pathlib import Path
    return Path(part)


def _stack_model(model):
    barcode = np.concatenate([m[0] for m in model]) if model else np.zeros(0, str)
    unit_id = np.concatenate([m[1] for m in model]) if model else np.zeros(0, str)
    library = np.concatenate([m[2] for m in model]) if model else np.zeros(0, np.int64)
    counts = sparse.vstack([m[3] for m in model]).tocsr() if model else sparse.csr_matrix((0, 0), dtype=np.int32)
    order = np.lexsort((barcode, unit_id))
    return barcode[order], unit_id[order], library[order], counts[order]
