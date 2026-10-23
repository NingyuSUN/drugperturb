"""Build response-free v5 model inputs and training/validation truth from extraction parts; never test truth.

Merges the parts (scripts/v5_extract.py) and runs the extraction anchors, then
writes: count QC and gates G3/G4; the frozen genes, chosen from vehicle-cell
detection and training-compound deltas only; training and validation truth and
per-replicate training deltas; the counterfactual query table for all splits;
all-cell vehicle levels for the anchored comparators; and the model-cell archive
(capped training, validation and vehicle cells; frozen genes) in the v3 layout.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import yaml

from drugperturb.v3.cells import save_model_archive
from drugperturb.v4.design import log_cpm, select_genes
from drugperturb.v4.diagnostics import finite
from drugperturb.v5 import extract, prepare

ROOT = Path(__file__).resolve().parents[1]
SHARED = ('units.tsv', 'expected_unit_cells.tsv', 'model_selection.tsv', 'genes.tsv')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path):
    return pd.read_csv(path, sep='\t', keep_default_na=False)


def key_strings(keys):
    return np.asarray([f'{c}|{x}' for c, x in keys])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extraction-root', type=Path, nargs='+', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, default=ROOT/'manifests/v5')
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/protocol_v5.yaml')
    args = parser.parse_args()
    out = args.output_root
    if (out/'prepare_receipt.json').exists():
        raise ValueError('prepare outputs already exist')
    (out/'count_qc').mkdir(parents=True, exist_ok=True)
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol['status'] != 'PRESPECIFIED_BEFORE_V5_FITTING':
        raise ValueError('v5 protocol is not prespecified')
    cap = int(protocol['model_cells']['cap_per_well_and_line'])
    parts = sorted({p for root in args.extraction_root for p in root.rglob('part_*') if (p/'part_receipt.json').exists()})
    if not parts:
        raise ValueError('no extraction part found')
    for name in SHARED:
        if len({sha256(p/name) for p in parts}) != 1:
            raise ValueError(f'extraction parts disagree on {name}')
    units = read_tsv(parts[0]/'units.tsv')
    expected = read_tsv(parts[0]/'expected_unit_cells.tsv')
    if not expected.unit_id.equals(units.unit_id):
        raise ValueError('expected cell counts are not in unit order')
    selection = pd.read_csv(parts[0]/'model_selection.tsv', sep='\t', keep_default_na=False)
    genes = read_tsv(parts[0]/'genes.tsv')
    shards = read_tsv(args.manifests/'expression_source.tsv')
    merged = extract.merge_parts(parts, units, expected_cells=expected.expected_cells.to_numpy(np.int64),
                                 model_barcodes=selection.barcode.to_numpy(str), cap=cap,
                                 shard_manifest=shards[shards.role.eq('expression_shard')])
    if not np.array_equal(merged['gene_ids'], genes.gene_id.to_numpy(str)):
        raise ValueError('gene order differs between parts and genes.tsv')

    compounds, contexts = read_tsv(args.manifests/'compounds.tsv'), read_tsv(args.manifests/'contexts.tsv')
    pairs = prepare.response_pairs(units, compounds, contexts)
    ids, counts, library, n_cells = prepare.with_pooled_vehicles(units, merged['unit_counts'], merged['library'],
                                                                 merged['n_cells'])
    row_of = {u: i for i, u in enumerate(ids)}
    replicate_qc, compound_qc, summary = prepare.count_qc(pairs, dict(zip(ids, n_cells)))
    gate = prepare.study_gate(summary, contexts, **prepare.STUDY_GATE)
    for name, frame in [('replicate_qc', replicate_qc), ('compound_qc', compound_qc), ('context_summary', summary)]:
        frame.to_csv(out/'count_qc'/f'{name}.tsv', sep='\t', index=False)

    # Gene space: vehicle-cell detection and eligible training-compound deltas only.
    detection = merged['detected']/merged['vehicle_cells']
    candidates = np.flatnonzero(detection >= prepare.MIN_DETECTION)
    eligible = compound_qc[compound_qc.eligible]
    keys = {s: sorted(map(tuple, eligible.loc[eligible.split.eq(s), ['compound_id', 'context_id']].to_numpy()))
            for s in ['train', 'validation', 'test']}
    table, delta = prepare.replicate_deltas(pairs, keys['train'], row_of, counts, library, candidates)
    training = prepare.key_means(table, delta, keys['train'])
    chosen = select_genes(training, detection[candidates], n_genes=int(protocol['gene_space']['n_genes']),
                          min_detection=prepare.MIN_DETECTION)
    selected = candidates[chosen]
    frozen = pd.DataFrame(dict(output_index=np.arange(len(selected)), gene_id=genes.gene_id.to_numpy()[selected],
                               gene_symbol=genes.gene_symbol.to_numpy()[selected], source_index=selected,
                               training_delta_variance=training.var(axis=0)[chosen],
                               vehicle_detection=detection[selected]))
    frozen.to_csv(out/'frozen_genes.tsv', sep='\t', index=False)
    del table, delta, training

    # Truth: training and validation only. Test keys never enter a delta computation here.
    truth_pairs = {}
    for split in ['train', 'validation']:
        table, delta = prepare.replicate_deltas(pairs, keys[split], row_of, counts, library, selected)
        np.savez_compressed(out/f'{split}_truth.npz', keys=key_strings(keys[split]),
                            delta=prepare.key_means(table, delta, keys[split]))
        truth_pairs[split] = len(keys[split])
        if split == 'train':
            np.savez_compressed(out/'train_replicate_deltas.npz', **{c: table[c].to_numpy(str) for c in table.columns},
                                delta=delta)
    queries = pairs.rename(columns=dict(treatment_unit='treatment_Combo', vehicle_unit='vehicle_Combo'))
    flags = set(map(tuple, eligible[['compound_id', 'context_id']].to_numpy()))
    queries['count_eligible'] = [(c, x) in flags for c, x in zip(queries.compound_id, queries.context_id)]
    queries.to_csv(out/'queries.tsv', sep='\t', index=False)
    (queries[['vehicle_Combo', 'context_id']].drop_duplicates().rename(columns={'vehicle_Combo': 'Combo'})
     .sort_values(['Combo', 'context_id']).to_csv(out/'vehicle_contexts.tsv', sep='\t', index=False))
    vehicles = sorted(units.loc[units.role.eq('vehicle'), 'vehicle_unit'].unique())
    rows = np.asarray([row_of[v] for v in vehicles])
    np.savez_compressed(out/'vehicle_levels.npz', vehicle_unit=np.asarray(vehicles, str),
                        level=log_cpm(counts[rows][:, selected].toarray(), library[rows]),
                        n_cells=n_cells[rows], library=library[rows])

    model = merged['model']
    cells = pd.DataFrame(dict(barcode=model['barcode'], unit_id=model['unit_id']))
    obs = prepare.model_obs(cells, units)
    if not obs.role.isin(extract.MODEL_ROLES).all():
        raise ValueError('model cells include a disallowed role')
    model_counts = model['counts'][:, selected].astype(np.int32).tocsr()
    model_counts.sort_indices()
    save_model_archive(out/'model_cells.npz', obs, model_counts, model['library'].astype(np.float64))
    checks = dict(merged['checks'], truth_keys_exclude_test=not (set(keys['train']) | set(keys['validation'])) & set(keys['test']),
                  model_cells_exclude_test=bool(obs.role.isin(extract.MODEL_ROLES).all()),
                  model_cells_per_unit_at_most_cap=bool(cells.unit_id.value_counts().max() <= cap))
    if not all(checks.values()):
        raise ValueError(f'prepare checks failed: {[k for k, v in checks.items() if not v]}')
    receipt = dict(status='PREPARED_RESPONSE_FREE_FOR_TEST', test_outcomes_computed=False, checks=checks,
                   study_gate_G4=gate, contexts_passing_count_gate=int(summary.test_count_gate_met.sum()),
                   context_summary=summary.to_dict('records'), shard_stats=merged['shard_stats'],
                   units=len(units), vehicle_units=len(vehicles), vehicle_cells=int(merged['vehicle_cells']),
                   candidate_genes=int(candidates.size), frozen_genes=int(len(selected)),
                   frozen_gene_detection_min=float(frozen.vehicle_detection.min()),
                   truth_pairs=truth_pairs, eligible_test_pairs=len(keys['test']), queries=len(queries),
                   model_cells=len(obs), model_cell_roles=obs.role.value_counts().to_dict(),
                   versions=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__),
                   inputs={str(p.relative_to(p.parents[1])): sha256(p) for part in parts for p in sorted(part.iterdir())
                           if p.is_file()},
                   outputs={str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob('*')) if p.is_file()})
    (out/'prepare_receipt.json').write_text(json.dumps(finite(receipt), indent=2)+'\n')
    print(json.dumps(finite({k: receipt[k] for k in ['units', 'model_cells', 'frozen_genes', 'truth_pairs',
                                                     'eligible_test_pairs', 'study_gate_G4', 'checks']}), indent=2))


if __name__ == '__main__':
    main()
