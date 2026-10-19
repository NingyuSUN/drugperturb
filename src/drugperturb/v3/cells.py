"""Cell-level inputs for v3: R exports in, response-free model cell archives out.

The model archive holds only training-compound, validation-compound and vehicle
cells. Test-compound cells are dropped here, before any model code can see them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

OBS_FIELDS = ('cell_id', 'Combo', 'cell_type', 'replicate', 'plate_oligo', 'compound_id',
              'role', 'dose_nM')
ROLES = ('train', 'validation', 'vehicle')


def read_cell_export(directory, frozen_gene_ids):
    """Read one R cell export and verify gene order and UMI conservation."""
    root = Path(directory)
    table = pd.read_csv(root/'cells.tsv.gz', sep='\t', keep_default_na=False,
                        dtype={'dose_pattern': str, 'replicate': str})
    genes = pd.read_csv(root/'genes.tsv', sep='\t', keep_default_na=False)
    if genes.gene_id.tolist() != list(frozen_gene_ids):
        raise ValueError(f'{root}: exported gene order differs from the frozen manifest')
    arrays = {name: np.fromfile(root/f'{name}.i32', dtype='<i4') for name in ['indptr', 'indices', 'data']}
    n = len(table)
    if arrays['indptr'].shape != (n+1,) or arrays['indptr'][0] != 0 or arrays['indptr'][-1] != len(arrays['data']):
        raise ValueError(f'{root}: CSR pointer array is inconsistent with the cell table')
    if len(arrays['indices']) != len(arrays['data']) or (arrays['data'] < 0).any():
        raise ValueError(f'{root}: CSR entries are inconsistent or negative')
    counts = sparse.csr_matrix((arrays['data'], arrays['indices'], arrays['indptr']),
                               shape=(n, len(frozen_gene_ids)))
    counts.sort_indices()
    if not np.array_equal(np.asarray(counts.sum(axis=1)).ravel(), table.frozen_gene_UMI.to_numpy(np.int64)):
        raise ValueError(f'{root}: per-cell frozen-gene UMI totals differ from the table')
    library = table.human_UMI.to_numpy(np.float64)
    if (library <= 0).any() or (library < table.frozen_gene_UMI.to_numpy()).any():
        raise ValueError(f'{root}: invalid all-gene libraries')
    return table, counts, library


def assign_roles(table, wells, split):
    """Label each exported cell train/validation/vehicle/test/excluded by its well.

    ``wells`` is the v2 well mapping; drug wells are unique per context, while a
    vehicle well can serve two dose contexts. Labels never use expression.
    """
    drugs = wells[wells.row_kind.eq('drug')].drop_duplicates('Combo')
    vehicles = set(wells.loc[wells.row_kind.eq('vehicle'), 'Combo'])
    if set(drugs.Combo) & vehicles:
        raise ValueError('a well is both drug and vehicle')
    roles = split.set_index('compound_id').split
    drug_compound = drugs.set_index('Combo').treatment
    drug_dose = drugs.set_index('Combo').context_dose_nM.astype(float)
    out = table.copy()
    is_vehicle = out.Combo.isin(vehicles)
    compound = out.Combo.map(drug_compound)
    if (~is_vehicle & compound.isna()).any():
        raise ValueError('exported cell belongs to no mapped well')
    out['compound_id'] = np.where(is_vehicle, 'DMSO', compound.fillna(''))
    role = compound.map(roles).fillna('excluded')
    out['role'] = np.where(is_vehicle, 'vehicle', role)
    out['dose_nM'] = np.where(is_vehicle, 0.0, out.Combo.map(drug_dose).fillna(np.nan))
    return out


def build_model_archive(parts):
    """Concatenate (table, counts, library) parts keeping only allowed roles."""
    tables, matrices, libraries, dropped = [], [], [], {}
    for table, counts, library in parts:
        keep = table.role.isin(ROLES).to_numpy()
        for role, n in table.loc[~keep, 'role'].value_counts().items():
            dropped[role] = dropped.get(role, 0)+int(n)
        tables.append(table.loc[keep, list(OBS_FIELDS)].reset_index(drop=True))
        matrices.append(counts[keep]); libraries.append(library[keep])
    obs = pd.concat(tables, ignore_index=True)
    if obs.cell_id.duplicated().any():
        raise ValueError('cell IDs must be unique across exports')
    if (obs.role.ne('vehicle') & obs.dose_nM.isna()).any():
        raise ValueError('drug cells require a context dose')
    return obs, sparse.vstack(matrices).tocsr(), np.concatenate(libraries), dropped


def save_model_archive(path, obs, counts, library):
    np.savez_compressed(path, indptr=counts.indptr.astype(np.int64), indices=counts.indices.astype(np.int32),
                        data=counts.data.astype(np.int32), shape=np.asarray(counts.shape, np.int64),
                        library=library, **{f'obs_{k}': obs[k].to_numpy(dtype=str if k != 'dose_nM' else float)
                                             for k in OBS_FIELDS})


def load_model_archive(path):
    with np.load(path, allow_pickle=False) as z:
        counts = sparse.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))
        obs = pd.DataFrame({k: z[f'obs_{k}'] for k in OBS_FIELDS})
        library = z['library']
    if not obs.role.isin(ROLES).all():
        raise ValueError('model archive contains a disallowed cell role')
    return obs, counts, library
