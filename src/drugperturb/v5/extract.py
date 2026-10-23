"""Tahoe-100M extraction for v5: one streaming pass over hash-verified expression shards.

Cells are matched to the design through ``obs_metadata`` by BARCODE_SUB_LIB_ID.
Only ``pass_filter == 'full'`` cells of the selected cell lines, in samples of
vehicles and of accepted compounds, are kept. For every (sample, line) unit the
pass sums all-gene counts and counts cells; it counts gene detection in vehicle
cells; and it keeps the per-cell all-gene counts of the model cells. Model cells
are at most ``cap`` per unit, drawn from ``obs_metadata`` with numpy
``default_rng(seed)`` before any count is read, and only from units of training
and validation compounds and of vehicles. No response (delta) is computed here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import sparse

SHARD_COLUMNS = ('genes', 'expressions', 'sample', 'cell_line_id', 'BARCODE_SUB_LIB_ID')
OBS_COLUMNS = ('BARCODE_SUB_LIB_ID', 'sample', 'cell_line', 'pass_filter')
PASS_VALUE = 'full'
MODEL_ROLES = ('train', 'validation', 'vehicle')


def treatment_unit(sample, line):
    return f'{sample}:{line}'


def vehicle_unit(plate, line):
    """The strict matched vehicle unit: all vehicle cells of one line on one plate."""
    return f'{plate}:vehicle:{line}'


def gene_lookup(gene_metadata):
    """Token -> gene column (columns sorted by token, as the Tahoe tutorial), Ensembl IDs and symbols."""
    order = gene_metadata.sort_values('token_id')
    tokens = order.token_id.to_numpy(np.int64)
    if (tokens < 0).any() or len(np.unique(tokens)) != len(tokens) or order.ensembl_id.duplicated().any():
        raise ValueError('gene tokens and Ensembl IDs must be unique and nonnegative')
    lookup = np.full(int(tokens.max())+1, -1, np.int64)
    lookup[tokens] = np.arange(len(tokens))
    return lookup, order.ensembl_id.astype(str).to_numpy(), order.gene_symbol.astype(str).to_numpy()


def unit_table(samples, compounds, lines):
    """The (sample, line) units to extract: vehicle samples and samples of accepted compounds, all splits."""
    split = compounds.loc[compounds.status.eq('accepted')].set_index('compound_id').split
    chosen = samples[samples.role.eq('vehicle') | (samples.role.eq('compound') & samples.drug.isin(split.index))]
    rows = []
    for s in chosen.itertuples(index=False):
        vehicle = s.role == 'vehicle'
        for line in sorted(lines):
            rows.append(dict(unit_id=treatment_unit(s.sample, line), sample=s.sample, plate=s.plate,
                             cell_line_id=line, role='vehicle' if vehicle else split[s.drug],
                             compound_id='DMSO' if vehicle else s.drug, dose_nM=0 if vehicle else int(s.dose_nM),
                             vehicle_unit=vehicle_unit(s.plate, line)))
    table = pd.DataFrame(rows).sort_values('unit_id').reset_index(drop=True)
    if not table.role.isin(['train', 'validation', 'test', 'vehicle']).all():
        raise ValueError('every extracted unit needs a split or the vehicle role')
    return table


def design_cells(obs, units, *, pass_value=PASS_VALUE):
    """Pass-filter obs cells of extracted units: barcode, unit code and design labels, sorted by barcode."""
    frame = obs[list(OBS_COLUMNS)].astype(str)
    frame = frame[frame.pass_filter.eq(pass_value)]
    frame = frame.assign(unit_id=frame['sample']+':'+frame.cell_line)
    code = pd.Series(np.arange(len(units)), index=units.unit_id)
    frame = frame[frame.unit_id.isin(code.index)]
    if frame.BARCODE_SUB_LIB_ID.duplicated().any():
        raise ValueError('BARCODE_SUB_LIB_ID must identify one cell')
    cells = pd.DataFrame(dict(barcode=frame.BARCODE_SUB_LIB_ID.to_numpy(), sample=frame['sample'].to_numpy(),
                              cell_line_id=frame.cell_line.to_numpy(), unit=code[frame.unit_id].to_numpy()))
    return cells.sort_values('barcode').reset_index(drop=True)


def select_model_cells(cells, units, *, cap, seed):
    """Boolean mask over ``cells``: at most ``cap`` cells per training, validation or vehicle unit.

    Units are visited in sorted unit-ID order and barcodes in sorted order within
    a unit; one ``default_rng(seed)`` draws without replacement only where a unit
    holds more than ``cap`` cells. Test-compound units contribute no model cell.
    """
    eligible = units.role.isin(MODEL_ROLES).to_numpy()
    rng = np.random.default_rng(seed)
    order = np.lexsort((cells.barcode.to_numpy(), cells.unit.to_numpy()))
    unit_sorted = cells.unit.to_numpy()[order]
    starts = np.flatnonzero(np.r_[True, unit_sorted[1:] != unit_sorted[:-1]])
    keep = np.zeros(len(cells), bool)
    for start, stop in zip(starts, np.r_[starts[1:], len(order)]):
        if not eligible[unit_sorted[start]]:
            continue
        members = order[start:stop]
        if len(members) > cap:
            members = members[np.sort(rng.choice(len(members), size=cap, replace=False))]
        keep[members] = True
    return keep


def expected_unit_cells(cells, units):
    return np.bincount(cells.unit.to_numpy(), minlength=len(units)).astype(np.int64)


def _list_parts(column):
    array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if array.null_count:
        raise ValueError('gene and expression lists must not be null')
    offsets = np.asarray(array.offsets, dtype=np.int64)
    values = array.values.to_numpy(zero_copy_only=False)[offsets[0]:offsets[-1]]
    return offsets-offsets[0], values


def decode_cells(genes, expressions, lookup):
    """Per-cell entries of a shard's list columns -> (cell, gene column, count) and decoding statistics.

    A cell's leading entry with a negative expression is the CLS marker and is
    dropped (Tahoe tutorial). Any other negative or non-integer value raises.
    Tokens outside the gene vocabulary are dropped and counted.
    """
    offsets, tokens = _list_parts(genes)
    expression_offsets, values = _list_parts(expressions)
    if not np.array_equal(offsets, expression_offsets):
        raise ValueError('gene and expression lists differ in length')
    lengths = np.diff(offsets)
    cell = np.repeat(np.arange(len(lengths)), lengths)
    first = np.zeros(len(tokens), bool)
    first[offsets[:-1][lengths > 0]] = True
    values = values.astype(np.float64)
    marker = first & (values < 0)
    if (values[~marker] < 0).any() or not np.array_equal(values[~marker], np.rint(values[~marker])):
        raise ValueError('expressions must be nonnegative integer counts after the marker')
    inside = (tokens >= 0) & (tokens < len(lookup))
    column = np.full(len(tokens), -1, np.int64)
    column[inside] = lookup[tokens[inside]]
    unknown = ~marker & (column < 0)
    keep = ~marker & ~unknown
    stats = dict(cells=int(len(lengths)), entries=int(len(tokens)), markers=int(marker.sum()),
                 unknown_token_entries=int(unknown.sum()), unknown_token_umi=int(values[unknown].sum()))
    return cell[keep], column[keep], values[keep].astype(np.int64), stats


class Accumulator:
    """Running unit sums, cell numbers, vehicle detection and model-cell rows over shards."""

    def __init__(self, units, cells, model_mask, n_genes):
        self.units, self.n_genes = units, n_genes
        self.barcodes = pd.Index(cells.barcode.to_numpy())  # its hash table is built once, not per shard
        self.cell_unit = cells.unit.to_numpy()
        self.cell_sample = cells['sample'].to_numpy()
        self.cell_line = cells.cell_line_id.to_numpy()
        self.model_mask = np.asarray(model_mask, bool)
        self.vehicle_unit = units.role.eq('vehicle').to_numpy()
        self.sums = np.zeros((len(units), n_genes), np.int32)
        self.library = np.zeros(len(units), np.int64)
        self.n_cells = np.zeros(len(units), np.int64)
        self.detected = np.zeros(n_genes, np.int64)
        self.vehicle_cells = 0
        self.seen = np.zeros(len(cells), bool)
        self.model_parts, self.shards = [], []

    def add(self, table, lookup, name):
        """Add one shard (pyarrow Table with SHARD_COLUMNS); returns its decoding statistics."""
        position = self.barcodes.get_indexer(table.column('BARCODE_SUB_LIB_ID').to_pandas().astype(str))
        rows = np.flatnonzero(position >= 0)
        cell = position[rows].astype(np.int64)
        stats = dict(shard=name, rows=table.num_rows, matched_rows=int(rows.size))
        if self.seen[cell].any() or len(np.unique(cell)) != len(cell):
            raise ValueError(f'{name}: a cell appears more than once in the expression shards')
        sub = table.take(pa.array(rows))
        for column, expected in [('sample', self.cell_sample), ('cell_line_id', self.cell_line)]:
            if not np.array_equal(sub.column(column).to_pandas().astype(str).to_numpy(), expected[cell]):
                raise ValueError(f'{name}: shard {column} disagrees with obs_metadata')
        entry_cell, gene, count, decoded = decode_cells(sub.column('genes'), sub.column('expressions'), lookup)
        stats.update(decoded)
        matrix = sparse.csr_matrix((count, (entry_cell, gene)), shape=(len(rows), self.n_genes), dtype=np.int64)
        matrix.sum_duplicates()
        stats['duplicate_token_entries'] = int(len(count)-matrix.nnz)
        library = np.asarray(matrix.sum(axis=1)).ravel()
        unit = self.cell_unit[cell]
        indicator = sparse.csr_matrix((np.ones(len(rows), np.int64), (unit, np.arange(len(rows)))),
                                      shape=(len(self.units), len(rows)))
        summed = (indicator@matrix).tocoo()
        self.sums[summed.row, summed.col] += summed.data.astype(np.int32)
        np.add.at(self.library, unit, library)
        self.n_cells += np.bincount(unit, minlength=len(self.units))
        vehicle = self.vehicle_unit[unit]
        if vehicle.any():
            self.detected += np.bincount(matrix[vehicle].indices, minlength=self.n_genes)
            self.vehicle_cells += int(vehicle.sum())
        model = self.model_mask[cell]
        if model.any():
            if (library[model] <= 0).any():
                raise ValueError(f'{name}: a model cell has an empty library')
            part = matrix[model].astype(np.int32)
            self.model_parts.append((cell[model], part, library[model]))
        self.seen[cell] = True
        self.shards.append(stats)
        return stats

    def model_cells(self):
        """Model cells sorted by (unit, barcode): cell positions, all-gene counts and libraries."""
        if not self.model_parts:
            return np.zeros(0, np.int64), sparse.csr_matrix((0, self.n_genes), dtype=np.int32), np.zeros(0, np.int64)
        cells = np.concatenate([p[0] for p in self.model_parts])
        counts = sparse.vstack([p[1] for p in self.model_parts]).tocsr()
        library = np.concatenate([p[2] for p in self.model_parts])
        order = np.argsort(cells, kind='stable')
        return cells[order], counts[order], library[order]


def save_part(out, acc, cells, gene_ids, extra):
    """Write one extraction part: unit sums, vehicle detection, model cells and the shard table."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (acc.sums.astype(np.int64).sum(axis=1) != acc.library).any():
        raise ValueError('unit sums overflowed int32 or lost counts')
    unit = sparse.csr_matrix(acc.sums)
    np.savez_compressed(out/'unit_counts.npz', data=unit.data, indices=unit.indices.astype(np.int32),
                        indptr=unit.indptr.astype(np.int64), shape=np.asarray(unit.shape, np.int64),
                        unit_ids=acc.units.unit_id.to_numpy(str), gene_ids=np.asarray(gene_ids, str),
                        library=acc.library, n_cells=acc.n_cells)
    np.savez_compressed(out/'vehicle_detection.npz', detected=acc.detected, vehicle_cells=acc.vehicle_cells)
    position, counts, library = acc.model_cells()
    np.savez_compressed(out/'model_cells.npz', data=counts.data.astype(np.int32),
                        indices=counts.indices.astype(np.int32), indptr=counts.indptr.astype(np.int64),
                        shape=np.asarray(counts.shape, np.int64), barcode=cells.barcode.to_numpy(str)[position],
                        unit_id=acc.units.unit_id.to_numpy(str)[cells.unit.to_numpy()[position]], library=library)
    pd.DataFrame(acc.shards).to_csv(out/'shards.tsv', sep='\t', index=False)
    for name, frame in extra.items():
        frame.to_csv(out/name, sep='\t', index=False)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def merge_parts(parts, units, *, expected_cells, model_barcodes, cap, shard_manifest):
    """Combine extraction parts and run the extraction anchors; raises on any failure.

    Anchors: every manifest shard was processed exactly once; per-unit cell numbers
    equal the obs_metadata pass-filter counts; every selected model cell was found
    once; unit libraries equal unit sums; model cells of a unit never exceed the
    unit sums, and equal them where the unit has no more than ``cap`` cells.
    """
    shards = pd.concat([pd.read_csv(Path(p)/'shards.tsv', sep='\t') for p in parts], ignore_index=True)
    if shards.shard.duplicated().any() or set(shards.shard) != set(shard_manifest.path):
        raise ValueError('extraction parts must cover every manifest shard exactly once')
    total, library, n_cells, detected, vehicle_cells = None, 0, 0, 0, 0
    model = []
    for part in parts:
        u = load_npz(Path(part)/'unit_counts.npz')
        if not np.array_equal(u['unit_ids'], units.unit_id.to_numpy(str)):
            raise ValueError(f'{part}: unit order differs')
        matrix = sparse.csr_matrix((u['data'], u['indices'], u['indptr']), shape=tuple(u['shape'])).astype(np.int64)
        total = matrix if total is None else total+matrix
        library, n_cells, gene_ids = library+u['library'], n_cells+u['n_cells'], u['gene_ids']
        v = load_npz(Path(part)/'vehicle_detection.npz')
        detected, vehicle_cells = detected+v['detected'], vehicle_cells+int(v['vehicle_cells'])
        m = load_npz(Path(part)/'model_cells.npz')
        model.append((m['barcode'], m['unit_id'], m['library'],
                      sparse.csr_matrix((m['data'], m['indices'], m['indptr']), shape=tuple(m['shape']))))
    checks = dict(shards=len(shards))
    checks['unit_cells_equal_obs_pass_filter'] = bool(np.array_equal(n_cells, expected_cells))
    checks['unit_library_equals_unit_sums'] = bool(np.array_equal(np.asarray(total.sum(axis=1)).ravel(), library))
    barcode = np.concatenate([m[0] for m in model])
    unit_id = np.concatenate([m[1] for m in model])
    cell_library = np.concatenate([m[2] for m in model])
    counts = sparse.vstack([m[3] for m in model]).tocsr()
    order = np.lexsort((barcode, unit_id))
    barcode, unit_id, cell_library, counts = barcode[order], unit_id[order], cell_library[order], counts[order]
    checks['model_cells_found_once'] = bool(len(barcode) == len(set(barcode)) == len(model_barcodes)
                                            and set(barcode) == set(model_barcodes))
    position = pd.Series(np.arange(len(units)), index=units.unit_id)[unit_id].to_numpy()
    indicator = sparse.csr_matrix((np.ones(len(position)), (position, np.arange(len(position)))),
                                  shape=(len(units), len(position)))
    model_sums = (indicator@counts.astype(np.int64)).tocsr()
    present = np.unique(position)
    difference = (total[present]-model_sums[present]).tocsr()
    checks['model_cells_within_unit_sums'] = bool(difference.nnz == 0 or difference.data.min() >= 0)
    complete = present[n_cells[present] <= cap]
    checks['model_cells_equal_unit_sums_where_all_kept'] = bool((total[complete] != model_sums[complete]).nnz == 0)
    checks['model_library_equals_counts'] = bool(np.array_equal(np.asarray(counts.sum(axis=1)).ravel(), cell_library))
    failed = [k for k, v in checks.items() if v is False]
    if failed:
        raise ValueError(f'extraction anchors failed: {failed}')
    stats = {k: int(shards[k].sum()) for k in ['rows', 'matched_rows', 'markers', 'unknown_token_entries',
                                               'unknown_token_umi', 'duplicate_token_entries'] if k in shards}
    return dict(unit_counts=total.tocsr(), library=library, n_cells=n_cells, gene_ids=gene_ids, detected=detected,
                vehicle_cells=vehicle_cells, model=dict(barcode=barcode, unit_id=unit_id, library=cell_library,
                                                         counts=counts), checks=checks, shard_stats=stats)
