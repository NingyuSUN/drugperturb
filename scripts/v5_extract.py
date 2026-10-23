"""Tahoe-100M v5 extraction of one part (a range of expression shards); never computes a response.

Inputs are the hash-verified obs_metadata and gene_metadata and the committed
manifests/v5 design. Each expression shard is read from --shard-dir or
downloaded from the pinned revision (--download), checked against
manifests/v5/expression_source.tsv (size and SHA256), processed and, when
downloaded, deleted. Model cells are chosen from obs_metadata before any shard
is read. The part is written to <output-root>/part_<start>_<stop>; preparation
merges the parts and runs the extraction anchors.
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

from drugperturb.v5 import extract

ROOT = Path(__file__).resolve().parents[1]
MODEL_CELL_SEED = 42  # protocol_v5 model_cells: "sampled with numpy default_rng(42)"


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
    parser.add_argument('--manifests', type=Path, default=ROOT/'manifests/v5')
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/protocol_v5.yaml')
    args = parser.parse_args()
    started = time.perf_counter()
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol['status'] != 'PRESPECIFIED_BEFORE_V5_FITTING':
        raise ValueError('v5 protocol is not prespecified')
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
    units = extract.unit_table(samples, compounds, lines)
    obs = pq.read_table(args.obs, columns=list(extract.OBS_COLUMNS), filters=[('cell_line', 'in', lines)]).to_pandas()
    cells = extract.design_cells(obs, units)
    del obs
    model = extract.select_model_cells(cells, units, cap=cap, seed=MODEL_CELL_SEED)
    lookup, gene_ids, symbols = extract.gene_lookup(pd.read_parquet(args.gene_metadata))
    acc = extract.Accumulator(units, cells, model, len(gene_ids))

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

    expected = pd.DataFrame(dict(unit_id=units.unit_id, expected_cells=extract.expected_unit_cells(cells, units)))
    selection = pd.DataFrame(dict(barcode=cells.barcode[model], unit_id=units.unit_id.to_numpy()[cells.unit[model]]))
    genes = pd.DataFrame(dict(gene_id=gene_ids, gene_symbol=symbols))
    extract.save_part(out, acc, cells, gene_ids, {'units.tsv': units, 'expected_unit_cells.tsv': expected,
                                                  'model_selection.tsv': selection, 'genes.tsv': genes})
    shard_table = pd.DataFrame(acc.shards)
    receipt = dict(status='EXTRACTED_PART_TEST_RESPONSES_NOT_COMPUTED', test_outcomes_computed=False,
                   shard_start=args.shard_start, shard_stop=stop, shards=len(part), units=len(units),
                   design_cells=len(cells), model_cells_selected=int(model.sum()), model_cell_cap=cap,
                   model_cell_seed=MODEL_CELL_SEED, cells_found_in_part=int(acc.seen.sum()),
                   vehicle_cells_in_part=acc.vehicle_cells,
                   decoding={k: int(shard_table[k].sum()) for k in ['rows', 'matched_rows', 'markers',
                             'unknown_token_entries', 'unknown_token_umi', 'duplicate_token_entries']},
                   inputs=dict(obs_metadata=role['obs_metadata']['sha256'], gene_metadata=role['gene_metadata']['sha256'],
                               protocol=sha256(args.protocol)),
                   versions=dict(python=sys.version.split()[0], platform=platform.platform(), numpy=np.__version__,
                                 pandas=pd.__version__, pyarrow=pyarrow.__version__),
                   seconds=time.perf_counter()-started, command=' '.join(['python', *sys.argv]),
                   outputs={p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out/'part_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: receipt[k] for k in ['shards', 'units', 'design_cells', 'model_cells_selected',
                                              'cells_found_in_part', 'decoding']}, indent=2))


if __name__ == '__main__':
    main()
