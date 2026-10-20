"""Build (or check) the OP3 design manifests from obs metadata only; never reads expression.

Reads the plate/well/donor/compound/dose/time/control/SMILES/library columns of
GSE279945_sc_counts_processed.h5ad, standardizes structures with the v1 policy,
splits whole scaffold groups (seed 42; 60/15/25), and records the well layout
with each treated well's same-row DMSO vehicle. ``--check`` regenerates the
tables from a local, hash-verified file and requires them to equal the
committed manifests. ``--vehicle-counts`` also records cell-type counts of the
DMSO wells (vehicle cells only), which informed the count-gate feasibility note.
"""
from pathlib import Path
import argparse
import json

import h5py
import numpy as np
import pandas as pd
import yaml

from drugperturb.chemistry import standardize_compounds
from drugperturb.v4.design import (CELL_CODES, DESIGN_COLUMNS, VEHICLE_NAME, build_wells, compound_records,
                                   split_by_structure_fractions)
from drugperturb.v4.h5io import HTTPRangeFile, read_obs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'manifests/v4'


def tables(design, split_spec):
    records = standardize_compounds(compound_records(design))
    records, report = split_by_structure_fractions(
        records, seed=split_spec['seed'], test_fraction=split_spec['test_fraction'],
        validation_fraction=split_spec['validation_fraction'],
        min_test_compounds=split_spec['min_test_compounds'], min_test_groups=split_spec['min_test_groups'])
    compounds = pd.DataFrame(records)[['compound_id', 'sm_name', 'original_smiles', 'canonical_smiles', 'scaffold_smiles',
                                       'scaffold_group', 'split', 'n_original_fragments']]
    compounds['removed_fragments'] = [';'.join(r['removed_fragments']) for r in records]
    return compounds.sort_values('compound_id').reset_index(drop=True), build_wells(design), report


def read(path):
    return pd.read_csv(path, sep='\t', keep_default_na=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--url')
    source.add_argument('--h5ad', type=Path)
    parser.add_argument('--check', action='store_true', help='compare with committed manifests instead of writing')
    parser.add_argument('--vehicle-counts', action='store_true')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    protocol = yaml.safe_load((ROOT/'configs/protocol_v4.yaml').read_text())
    split_spec = protocol['datasets']['op3']['split']
    handle = HTTPRangeFile(args.url) if args.url else open(args.h5ad, 'rb')
    with h5py.File(handle, 'r') as h5:
        columns = [*DESIGN_COLUMNS, 'cell_type'] if args.vehicle_counts else list(DESIGN_COLUMNS)
        design = read_obs(h5, columns)
    compounds, wells, report = tables(design, split_spec)
    result = dict(cells=len(design), wells=len(wells), compounds=len(compounds), split=report,
                  obs_columns_read=columns, expression_read=False)
    if args.vehicle_counts:
        vehicle = design[design.sm_name.astype(str).eq(VEHICLE_NAME)]
        units = vehicle.groupby(['plate_name', 'well', 'cell_type'], observed=True).size().unstack(fill_value=0)
        result['vehicle_unit_cells'] = {CELL_CODES[c]: dict(units[c].describe().round(1)) for c in units.columns}
    if args.check:
        mismatches = {}
        for name, frame in [('op3_compounds.tsv', compounds), ('op3_wells.tsv', wells)]:
            committed = read(OUT/name)
            regenerated = read_roundtrip(frame)
            mismatches[name] = not committed.equals(regenerated)
        result['committed_manifests_equal'] = not any(mismatches.values())
        if any(mismatches.values()):
            raise SystemExit(f'regenerated design differs from committed manifests: {mismatches}')
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        compounds.to_csv(OUT/'op3_compounds.tsv', sep='\t', index=False)
        wells.to_csv(OUT/'op3_wells.tsv', sep='\t', index=False)
        (OUT/'op3_split_report.json').write_text(json.dumps(report, indent=2)+'\n')
    if args.url:
        result['http_range_requests'] = handle.requests
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x))+'\n')
    print(json.dumps({k: result[k] for k in ['cells', 'wells', 'compounds', 'split']}, indent=2, default=str))


def read_roundtrip(frame):
    """Serialize and re-read so dtype differences cannot hide or invent mismatches."""
    import io
    buffer = io.StringIO()
    frame.to_csv(buffer, sep='\t', index=False)
    buffer.seek(0)
    return pd.read_csv(buffer, sep='\t', keep_default_na=False)


if __name__ == '__main__':
    main()
