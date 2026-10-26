"""Tahoe-100M v6 extraction of one part; splits every unit into two halves, never computes a response.

The same streaming extraction as v5, with one addition: each unit's cells are
partitioned into a selection and an evaluation half (protocol_v6
measurement_split), and sums are accumulated per (unit, half). The partition
comes from the committed seed and the obs_metadata cell table, so it is
reproducible without storing a label per cell; the receipt records its hash.

Inputs are the hash-verified obs_metadata and gene_metadata and the committed
manifests/v6 design. Each expression shard is read from --shard-dir or
downloaded from the pinned revision (--download), checked against
manifests/v6/expression_source.tsv, processed and, when downloaded, deleted.
Model cells are chosen before any shard is read, on whole units, so the cap is
a cap per unit and not per half.
"""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import platform
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import yaml

from drugperturb.v5 import extract as v5_extract
from drugperturb.v6 import extract

ROOT = Path(__file__).resolve().parents[1]
MODEL_CELL_SEED = 42  # protocol_v6 model_cells, unchanged from v5
PROTOCOL_STATUS = 'PRESPECIFIED_BEFORE_V6_FITTING'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verified(path, row):
    path = Path(path)
    if path.stat().st_size != int(row['bytes']) or sha256(path) != row['sha256']:
        raise ValueError(f'{path} does not match the source manifest ({row["path"]})')
    return path


def download(row, directory, attempts=5):
    target = Path(directory)/Path(row['path']).name
    for attempt in range(attempts):
        try:
            urllib.request.urlretrieve(row['url'], target)
            return verified(target, row)
        except (OSError, ValueError) as error:
            if attempt == attempts-1:
                raise
            print(f'retry {row["path"]}: {error}', flush=True)
            time.sleep(10*(attempt+1))


def read_tsv(path):
    return pd.read_csv(path, sep='\t', keep_default_na=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--obs', type=Path, required=True)
    parser.add_argument('--gene-metadata', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--shard-dir', type=Path)
    source.add_argument('--download', type=Path, metavar='DIR', help='download shards into DIR, delete after use')
    parser.add_argument('--shard-start', type=int, default=0)
    parser.add_argument('--shard-stop', type=int)
    parser.add_argument('--prefetch', type=int, default=2)
    parser.add_argument('--manifests', type=Path, default=ROOT/'manifests/v6')
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/protocol_v6.yaml')
    args = parser.parse_args()
    started = time.perf_counter()
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol['status'] != PROTOCOL_STATUS:
        raise ValueError('v6 protocol is not prespecified')
    cap = int(protocol['model_cells']['cap_per_well_and_line'])
    sources = pd.concat([read_tsv(args.manifests/'source.tsv'), read_tsv(args.manifests/'expression_source.tsv')])
    role = {r['role']: r for r in sources.to_dict('records') if r['role'] != 'expression_shard'}
    verified(args.obs, role['obs_metadata'])
    verified(args.gene_metadata, role['gene_metadata'])
    shards = sources[sources.role.eq('expression_shard')].sort_values('path').to_dict('records')
    stop = len(shards) if args.shard_stop is None else args.shard_stop
    if not 0 <= args.shard_start < stop <= len(shards):
        raise ValueError('shard range outside the manifest')
    part = shards[args.shard_start:stop]
    out = args.output_root/f'part_{args.shard_start:05d}_{stop:05d}'
    if (out/'part_receipt.json').exists():
        raise ValueError(f'{out} already exists')

    samples, compounds = read_tsv(args.manifests/'samples.tsv'), read_tsv(args.manifests/'compounds.tsv')
    contexts = read_tsv(args.manifests/'contexts.tsv')
    lines = sorted(contexts.cell_line_id.unique())
    units = v5_extract.unit_table(samples, compounds, lines)
    obs = pq.read_table(args.obs, columns=list(extract.OBS_COLUMNS), filters=[('cell_line', 'in', lines)]).to_pandas()
    cells = extract.prepared_cells(obs, units, seed=extract.SPLIT_SEED)
    del obs
    model = v5_extract.select_model_cells(cells, units, cap=cap, seed=MODEL_CELL_SEED)
    halves = extract.half_units(units)
    accumulated = extract.accumulator_cells(cells, halves)
    lookup, gene_ids, symbols = extract.gene_lookup(pd.read_parquet(args.gene_metadata))
    acc = extract.Accumulator(halves, accumulated, model, len(gene_ids))

    def fetch(row):
        return download(row, args.download) if args.download else verified(args.shard_dir/Path(row['path']).name, row)
    if args.download:
        args.download.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.prefetch)) as pool:
        pending = [pool.submit(fetch, row) for row in part[:args.prefetch+1]]
        for i, row in enumerate(part):
            path = pending[i].result()
            if i+args.prefetch+1 < len(part):
                pending.append(pool.submit(fetch, part[i+args.prefetch+1]))
            stats = acc.add(pq.read_table(path, columns=list(extract.SHARD_COLUMNS)), lookup, row['path'])
            if args.download:
                path.unlink()
            if i % 50 == 0 or i == len(part)-1:
                print(json.dumps(dict(done=i+1, of=len(part), minutes=round((time.perf_counter()-started)/60, 1),
                                      **{k: stats[k] for k in ['rows', 'matched_rows']})), flush=True)

    expected = pd.DataFrame(dict(unit_id=halves.unit_id, parent_unit=halves.parent_unit, half=halves.half,
                                 expected_cells=extract.expected_half_cells(cells, halves)))
    selection = pd.DataFrame(dict(barcode=cells.barcode[model],
                                  unit_id=halves.unit_id.to_numpy()[accumulated.unit.to_numpy()[model]]))
    genes = pd.DataFrame(dict(gene_id=gene_ids, gene_symbol=symbols))
    extract.save_part(out, acc, accumulated, gene_ids,
                      {'units.tsv': units, 'half_units.tsv': halves, 'expected_half_cells.tsv': expected,
                       'model_selection.tsv': selection, 'genes.tsv': genes})
    half_digest = hashlib.sha256('\n'.join(f'{b}:{h}' for b, h in
                                           zip(cells.barcode, cells.half)).encode()).hexdigest()
    shard_table = pd.DataFrame(acc.shards)
    sizes = expected.groupby('half').expected_cells.describe().to_dict('index')
    receipt = dict(status='EXTRACTED_PART_TEST_RESPONSES_NOT_COMPUTED', test_outcomes_computed=False,
                   shard_start=args.shard_start, shard_stop=stop, shards=len(part), units=len(units),
                   half_units=len(halves), design_cells=len(cells), split_seed=extract.SPLIT_SEED,
                   half_label_sha256=half_digest,
                   half_cells={k: {m: float(v[m]) for m in ['count', 'min', '50%', 'max']} for k, v in sizes.items()},
                   model_cells_selected=int(model.sum()), model_cell_cap=cap, model_cell_seed=MODEL_CELL_SEED,
                   cells_found_in_part=int(acc.seen.sum()), vehicle_cells_in_part=acc.vehicle_cells,
                   decoding={k: int(shard_table[k].sum()) for k in ['rows', 'matched_rows', 'markers',
                             'unknown_token_entries', 'unknown_token_umi', 'duplicate_token_entries']},
                   inputs=dict(obs_metadata=role['obs_metadata']['sha256'], gene_metadata=role['gene_metadata']['sha256'],
                               protocol=sha256(args.protocol)),
                   versions=dict(python=sys.version.split()[0], platform=platform.platform(), numpy=np.__version__,
                                 pandas=pd.__version__, pyarrow=pyarrow.__version__),
                   seconds=time.perf_counter()-started, command=' '.join(['python', *sys.argv]),
                   outputs={p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out/'part_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: receipt[k] for k in ['shards', 'units', 'half_units', 'design_cells', 'half_cells',
                                              'model_cells_selected', 'cells_found_in_part', 'decoding']}, indent=2))


if __name__ == '__main__':
    main()
