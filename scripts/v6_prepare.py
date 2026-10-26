"""Build response-free v6 model inputs, DE sets, the responder gate and training truth; never test truth.

Merges the extraction parts and runs the v6 anchors, then writes: count QC on
both measurement halves and gates G3/G4; the frozen genes, from vehicle-cell
detection and training-compound deltas only; each key's selection-half delta,
standardized statistic, responder status and DE set; the vehicle-versus-vehicle
null and the responder threshold; training and validation truth, including the
evaluation-half truth the validation endpoint needs; leave-one-out residual
targets for arm M2; the query table; all-cell vehicle levels; and the
model-cell archive.

Which half is used where, as the protocol specifies:

- DE sets, the responder statistic and the null come from the SELECTION half;
- validation scoring, like test scoring, uses the EVALUATION half;
- model training targets, the baselines and gene selection use WHOLE units, so
  the split costs the model nothing.

No evaluation-half test delta is computed here. Test keys get their
selection-half quantities, which the endpoint needs and the protocol allows.
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
from drugperturb.v5 import prepare as v5_prepare
from drugperturb.v6 import endpoint, extract, prepare

ROOT = Path(__file__).resolve().parents[1]
SHARED = ('units.tsv', 'half_units.tsv', 'expected_half_cells.tsv', 'model_selection.tsv', 'genes.tsv')
PROTOCOL_STATUS = 'PRESPECIFIED_BEFORE_V6_FITTING'


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
    parser.add_argument('--manifests', type=Path, default=ROOT/'manifests/v6')
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/protocol_v6.yaml')
    args = parser.parse_args()
    out = args.output_root
    if (out/'prepare_receipt.json').exists():
        raise ValueError('prepare outputs already exist')
    (out/'count_qc').mkdir(parents=True, exist_ok=True)
    (out/'endpoint').mkdir(parents=True, exist_ok=True)
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol['status'] != PROTOCOL_STATUS:
        raise ValueError('v6 protocol is not prespecified')
    cap = int(protocol['model_cells']['cap_per_well_and_line'])
    de_k = int(protocol['response_endpoint']['de_gene_set']['k'])
    parts = sorted({p for root in args.extraction_root for p in root.rglob('part_*') if (p/'part_receipt.json').exists()})
    if not parts:
        raise ValueError('no extraction part found')
    for name in SHARED:
        if len({sha256(p/name) for p in parts}) != 1:
            raise ValueError(f'extraction parts disagree on {name}')
    units = read_tsv(parts[0]/'units.tsv')
    halves = read_tsv(parts[0]/'half_units.tsv')
    expected = read_tsv(parts[0]/'expected_half_cells.tsv')
    if not expected.unit_id.equals(halves.unit_id):
        raise ValueError('expected cell counts are not in half-unit order')
    selection = read_tsv(parts[0]/'model_selection.tsv')
    genes = read_tsv(parts[0]/'genes.tsv')
    shards = read_tsv(args.manifests/'expression_source.tsv')
    merged = extract.merge_parts(parts, halves, expected_cells=expected.expected_cells.to_numpy(np.int64),
                                 model_barcodes=selection.barcode.to_numpy(str), cap=cap,
                                 shard_manifest=shards[shards.role.eq('expression_shard')])
    if not np.array_equal(merged['gene_ids'], genes.gene_id.to_numpy(str)):
        raise ValueError('gene order differs between parts and genes.tsv')

    compounds, contexts = read_tsv(args.manifests/'compounds.tsv'), read_tsv(args.manifests/'contexts.tsv')
    pairs = v5_prepare.response_pairs(units, compounds, contexts)
    whole = {u: i for i, u in enumerate(merged['whole_ids'])}
    order = np.asarray([whole[u] for u in units.unit_id], np.int64)
    ids, counts, library, n_cells = v5_prepare.with_pooled_vehicles(
        units, merged['whole_counts'][order], merged['whole_library'][order], merged['whole_cells'][order])
    row_of = {u: i for i, u in enumerate(ids)}
    half = prepare.half_tables(merged['half_ids'], merged['half_counts'], merged['half_library'],
                               merged['half_cells'], units)

    # Count QC on both halves, then the gates.
    half_cells_of = {u: {h: int(half[h]['n_cells'][half[h]['row_of'][u]]) for h in extract.HALVES} for u in ids}
    replicate_qc = prepare.half_count_qc(pairs, half_cells_of)
    compound_qc = (replicate_qc.groupby(['context_id', 'compound_id', 'split', 'scaffold_group'], as_index=False)
                   .agg(replicates=('replicate', 'nunique'), passing=('replicate_passes', 'sum'),
                        min_half_cells=('min_half_cells', 'min')))
    compound_qc['eligible'] = compound_qc.replicates.ge(1) & compound_qc.passing.eq(compound_qc.replicates)

    # Gene space: vehicle-cell detection and eligible training-compound deltas on whole units only.
    detection = merged['detected']/merged['vehicle_cells']
    candidates = np.flatnonzero(detection >= prepare.MIN_DETECTION)
    eligible = compound_qc[compound_qc.eligible]
    keys = {s: sorted(map(tuple, eligible.loc[eligible.split.eq(s), ['compound_id', 'context_id']].to_numpy()))
            for s in ['train', 'validation', 'test']}
    table, delta = v5_prepare.replicate_deltas(pairs, keys['train'], row_of, counts, library, candidates)
    training = v5_prepare.key_means(table, delta, keys['train'])
    chosen = select_genes(training, detection[candidates], n_genes=int(protocol['gene_space']['n_genes']),
                          min_detection=prepare.MIN_DETECTION)
    selected = candidates[chosen]
    pd.DataFrame(dict(output_index=np.arange(len(selected)), gene_id=genes.gene_id.to_numpy()[selected],
                      gene_symbol=genes.gene_symbol.to_numpy()[selected], source_index=selected,
                      training_delta_variance=training.var(axis=0)[chosen],
                      vehicle_detection=detection[selected])).to_csv(out/'frozen_genes.tsv', sep='\t', index=False)
    del table, delta, training

    # The responder null and threshold, from the selection half and vehicle wells only (amendment A1).
    null = prepare.vehicle_null(units, half['S'], selected)
    threshold = endpoint.responder_threshold(null.statistic.to_numpy())
    null.to_csv(out/'endpoint'/'vehicle_null.tsv', sep='\t', index=False)

    # Selection-half quantities for every eligible key, including test keys.
    all_keys = sorted(set().union(*keys.values()))
    delta_s = prepare.half_delta(pairs, all_keys, half['S'], selected)
    error_s = prepare.half_standard_error(pairs, all_keys, half['S'], selected)
    statistic = endpoint.response_statistic(delta_s, error_s)
    de_sets = endpoint.de_gene_sets(delta_s, k=de_k)
    responder = statistic > threshold
    split_of = {k: s for s, members in keys.items() for k in members}
    endpoint_table = pd.DataFrame(dict(
        compound_id=[k[0] for k in all_keys], context_id=[k[1] for k in all_keys],
        split=[split_of[k] for k in all_keys], statistic=statistic, responder=responder,
        selection_norm=np.linalg.norm(delta_s, axis=1),
        selection_cells=prepare.key_cells(pairs, all_keys, half['S'])))
    endpoint_table.to_csv(out/'endpoint'/'keys.tsv', sep='\t', index=False)
    np.savez_compressed(out/'endpoint'/'de_sets.npz', keys=key_strings(all_keys), genes=de_sets.astype(np.int32),
                        selection_delta=delta_s, selection_error=error_s, statistic=statistic,
                        responder=responder, threshold=threshold, k=de_k)
    compound_qc = compound_qc.merge(endpoint_table[['compound_id', 'context_id', 'responder']],
                                    on=['compound_id', 'context_id'], how='left')
    compound_qc['responder'] = compound_qc.responder.fillna(False)
    summary = prepare.context_gate(compound_qc)
    gate = v5_prepare.study_gate(summary, contexts, **prepare.STUDY_GATE)
    for name, frame in [('replicate_qc', replicate_qc), ('compound_qc', compound_qc), ('context_summary', summary)]:
        frame.to_csv(out/'count_qc'/f'{name}.tsv', sep='\t', index=False)

    # Truth. Training targets use whole units; validation also gets the evaluation half, which its endpoint needs.
    truth_pairs = {}
    for split in ['train', 'validation']:
        table, delta = v5_prepare.replicate_deltas(pairs, keys[split], row_of, counts, library, selected)
        whole_means = v5_prepare.key_means(table, delta, keys[split])
        payload = dict(keys=key_strings(keys[split]), delta=whole_means)
        if split == 'validation':
            payload['delta_evaluation'] = prepare.half_delta(pairs, keys[split], half['E'], selected)
            payload['delta_selection'] = prepare.half_delta(pairs, keys[split], half['S'], selected)
        np.savez_compressed(out/f'{split}_truth.npz', **payload)
        truth_pairs[split] = len(keys[split])
        if split == 'train':
            np.savez_compressed(out/'train_replicate_deltas.npz',
                                **{c: table[c].to_numpy(str) for c in table.columns}, delta=delta)
            means, loo, plate_counts = prepare.leave_one_out_plate_means(table, delta)
            residual, kept, missing = prepare.residual_targets(
                keys[split], whole_means, {k: loo.get((k[1], _plate_of(table, k), k[0])) for k in keys[split]})
            np.savez_compressed(out/'train_residual_targets.npz', keys=key_strings(kept), residual=residual,
                                fallback_keys=key_strings(missing))
            pd.DataFrame([dict(context_id=c, plate=p, compounds=n) for (c, p), n in sorted(plate_counts.items())]
                         ).to_csv(out/'endpoint'/'plate_compounds.tsv', sep='\t', index=False)

    queries = pairs.rename(columns=dict(treatment_unit='treatment_Combo', vehicle_unit='vehicle_Combo'))
    flags = set(map(tuple, eligible[['compound_id', 'context_id']].to_numpy()))
    responders = set(map(tuple, endpoint_table.loc[endpoint_table.responder, ['compound_id', 'context_id']].to_numpy()))
    queries['count_eligible'] = [(c, x) in flags for c, x in zip(queries.compound_id, queries.context_id)]
    queries['responder'] = [(c, x) in responders for c, x in zip(queries.compound_id, queries.context_id)]
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
    obs = v5_prepare.model_obs(cells, units)
    if not obs.role.isin(extract.MODEL_ROLES).all():
        raise ValueError('model cells include a disallowed role')
    model_counts = model['counts'][:, selected].astype(np.int32).tocsr()
    model_counts.sort_indices()
    save_model_archive(out/'model_cells.npz', obs, model_counts, model['library'].astype(np.float64))

    checks = dict(merged['checks'],
                  truth_keys_exclude_test=not (set(keys['train']) | set(keys['validation'])) & set(keys['test']),
                  model_cells_exclude_test=bool(obs.role.isin(extract.MODEL_ROLES).all()),
                  model_cells_per_unit_at_most_cap=bool(cells.unit_id.value_counts().max() <= cap),
                  de_sets_within_frozen_genes=bool(de_sets.max() < len(selected) and de_sets.min() >= 0),
                  every_half_has_min_cells=bool(replicate_qc.loc[replicate_qc.replicate_passes,
                                                                 'min_half_cells'].min() >= prepare.MIN_CELLS))
    if not all(checks.values()):
        raise ValueError(f'prepare checks failed: {[k for k, v in checks.items() if not v]}')
    receipt = dict(status='PREPARED_RESPONSE_FREE_FOR_TEST', test_outcomes_computed=False,
                   evaluation_half_test_delta_computed=False, checks=checks, study_gate_G4=gate,
                   contexts_passing_count_gate=int(summary.test_count_gate_met.sum()),
                   context_summary=summary.to_dict('records'), shard_stats=merged['shard_stats'],
                   units=len(units), half_units=len(halves), vehicle_units=len(vehicles),
                   vehicle_cells=int(merged['vehicle_cells']), candidate_genes=int(candidates.size),
                   frozen_genes=int(len(selected)), de_k=de_k,
                   responder=dict(threshold=threshold, nulls=len(null),
                                  percentile=endpoint.RESPONDER_PERCENTILE,
                                  null_statistic_median=float(null.statistic.median()),
                                  responders=int(endpoint_table.responder.sum()), keys=len(endpoint_table),
                                  by_split=endpoint_table.groupby('split').responder.sum().to_dict()),
                   truth_pairs=truth_pairs, eligible_test_pairs=len(keys['test']), queries=len(queries),
                   model_cells=len(obs), model_cell_roles=obs.role.value_counts().to_dict(),
                   versions=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__),
                   inputs={str(p.relative_to(p.parents[1])): sha256(p) for part in parts for p in sorted(part.iterdir())
                           if p.is_file()},
                   outputs={str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob('*')) if p.is_file()})
    (out/'prepare_receipt.json').write_text(json.dumps(finite(receipt), indent=2)+'\n')
    print(json.dumps(finite({k: receipt[k] for k in ['units', 'frozen_genes', 'responder', 'truth_pairs',
                                                     'eligible_test_pairs', 'study_gate_G4', 'checks']}), indent=2))


def _plate_of(table, key):
    """The plate of a key's first replicate; keys with several plates fall back on the residual reference."""
    rows = table[table.compound_id.eq(key[0]) & table.context_id.eq(key[1])]
    plates = sorted(set(rows.plate))
    return plates[0] if len(plates) == 1 else None


if __name__ == '__main__':
    main()
