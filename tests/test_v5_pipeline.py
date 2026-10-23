"""Synthetic checks of the v5 Tahoe-100M extraction, preparation and comparator sealing.

A small Tahoe-like dataset (pooled lines, CLS markers, unknown tokens, minimal-pass
cells, an unselected line, an excluded compound, a reference control and a
replicate plate) runs through the scripts. Fixtures establish software behaviour
only; they say nothing about Tahoe-100M.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from scipy import sparse

pytest.importorskip('rdkit')
pytest.importorskip('sklearn')
from drugperturb.v3.cells import load_model_archive  # noqa: E402
from drugperturb.v4.counterfactual import load_truth  # noqa: E402
from drugperturb.v5 import baselines, extract, prepare  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GENES = 60
LINES = ['L1', 'L2']
SMILES = {'C01': 'c1ccccc1O', 'C02': 'c1ccncc1', 'C03': 'C1CCCCC1N', 'C04': 'c1ccoc1', 'C05': 'c1ccsc1',
          'C06': 'C1CCNCC1', 'C07': 'c1ccc2ccccc2c1', 'C08': 'C1CCOC1', 'C09': 'c1cnccn1', 'C10': 'C1CC1C(=O)O',
          'C11': 'c1ccc(Cl)cc1', 'C12': 'O=C1CCCN1', 'X1': 'CCO.CCN'}
SPLIT = {**{f'C0{i}': 'train' for i in range(1, 8)}, 'C08': 'validation', 'C09': 'validation',
         'C10': 'test', 'C11': 'test', 'C12': 'test'}
PLATES = {'plate1': 0.05, 'plate2': 0.5, 'plate3': 5.0, 'plate4': 5.0}
REPLICATED = ('C01', 'C03', 'C08', 'C10')  # also on plate4 at 5 uM
CAP = 8


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_tsv(frame, path):
    frame.to_csv(path, sep='\t', index=False)


def make_dataset(root, perturb_test=False):
    """Write synthetic metadata, shards and v5-style manifests; returns paths and the ground-truth cell table."""
    rng = np.random.default_rng(7)
    root.mkdir(parents=True)
    samples, n = [], 0
    for plate, dose in PLATES.items():
        for v in 'ab':
            samples.append(dict(sample=f'{plate}_v{v}', plate=plate, role='vehicle', drug='DMSO_TF', dose_uM=0.0))
        samples.append(dict(sample=f'{plate}_ref', plate=plate, role='reference_control', drug='Adagrasib',
                            dose_uM=0.05))
        for c in SMILES:
            if plate != 'plate4' or c in REPLICATED:
                samples.append(dict(sample=f'{plate}_{c}', plate=plate, role='compound', drug=c, dose_uM=dose))
    samples = pd.DataFrame(samples)
    samples['plate_number'] = samples.plate.str[5:].astype(int)
    samples['dose_nM'] = (samples.dose_uM*1000).round().astype(int)
    samples['drug_source'] = samples.drug
    vehicles = samples[samples.role.eq('vehicle')].groupby('plate')['sample'].agg(lambda s: ';'.join(sorted(s)))
    samples['vehicle_samples'] = samples.plate.map(vehicles)
    effect = {c: rng.normal(0, 0.6, GENES) for c in SMILES}
    base = {line: rng.uniform(0.02, 1.5, GENES) for line in ['L1', 'L2', 'L3']}
    rows, truth = [], []
    for s in samples.itertuples(index=False):
        for line in ['L1', 'L2', 'L3']:
            count = 12 if (s.drug == 'C12' and s.dose_uM == 0.5 and line == 'L2') else int(rng.integers(34, 45))
            rate = base[line]*(np.exp(effect[s.drug]*s.dose_uM) if s.role == 'compound' else 1.0)
            for _ in range(count):
                barcode = f'bc{n:06d}'
                n += 1
                values = rng.poisson(rate*rng.uniform(0.7, 1.3))
                genes = np.flatnonzero(values)
                if perturb_test and SPLIT.get(s.drug) == 'test' and s.role == 'compound':
                    values = values*3+(values > 0)
                passing = 'minimal' if rng.random() < 0.05 else 'full'
                tokens, expressions = list(genes+3), list(values[genes].astype(np.float32))
                unknown = rng.random() < 0.1
                if unknown:
                    tokens.append(9000)
                    expressions.append(2.0)
                if rng.random() < 0.95:
                    tokens, expressions = [1]+tokens, [-2.0]+expressions
                rows.append(dict(genes=tokens, expressions=expressions, drug=s.drug, sample=s.sample,
                                 BARCODE_SUB_LIB_ID=barcode, cell_line_id=line, plate=s.plate))
                truth.append(dict(barcode=barcode, sample=s.sample, line=line, passing=passing, counts=values))
    order = rng.permutation(len(rows))
    shard_dir = root/'shards'
    shard_dir.mkdir()
    schema = pa.schema([('genes', pa.list_(pa.int64())), ('expressions', pa.list_(pa.float32())),
                        ('drug', pa.string()), ('sample', pa.string()), ('BARCODE_SUB_LIB_ID', pa.string()),
                        ('cell_line_id', pa.string()), ('plate', pa.string())])
    shard_rows = []
    for i, chunk in enumerate(np.array_split(order, 3)):
        path = shard_dir/f'train-{i:05d}-of-00003.parquet'
        pq.write_table(pa.Table.from_pylist([rows[j] for j in chunk], schema=schema), path)
        shard_rows.append(dict(role='expression_shard', path=f'data/{path.name}', bytes=path.stat().st_size,
                               sha256=sha256(path)))
    obs = pd.DataFrame(dict(BARCODE_SUB_LIB_ID=[t['barcode'] for t in truth], sample=[t['sample'] for t in truth],
                            cell_line=[t['line'] for t in truth], pass_filter=[t['passing'] for t in truth]))
    obs.to_parquet(root/'obs_metadata.parquet')
    pd.DataFrame(dict(gene_symbol=[f'S{i}' for i in range(GENES)], ensembl_id=[f'ENSG{i:05d}' for i in range(GENES)],
                      token_id=np.arange(3, 3+GENES))).to_parquet(root/'gene_metadata.parquet')
    manifests = root/'manifests'
    manifests.mkdir()
    write_tsv(samples, manifests/'samples.tsv')
    compounds = pd.DataFrame(dict(compound_id=list(SMILES), canonical_smiles=list(SMILES.values()),
                                  status=['excluded' if c == 'X1' else 'accepted' for c in SMILES],
                                  split=[SPLIT.get(c, '') for c in SMILES],
                                  scaffold_group=[f'group:{c}' for c in SMILES]))
    write_tsv(compounds, manifests/'compounds.tsv')
    write_tsv(pd.DataFrame([dict(context_id=f'{line}_{int(d*1000)}nM', cell_line_id=line, dose_uM=d, dose_nM=int(d*1000))
                            for line in LINES for d in (0.05, 0.5, 5.0)]), manifests/'contexts.tsv')
    write_tsv(pd.DataFrame([dict(role='obs_metadata', path='metadata/obs_metadata.parquet',
                                 bytes=(root/'obs_metadata.parquet').stat().st_size, sha256=sha256(root/'obs_metadata.parquet'))]),
              manifests/'source.tsv')
    write_tsv(pd.DataFrame(shard_rows+[dict(role='gene_metadata', path='metadata/gene_metadata.parquet',
                                            bytes=(root/'gene_metadata.parquet').stat().st_size,
                                            sha256=sha256(root/'gene_metadata.parquet'))]).assign(url=''),
              manifests/'expression_source.tsv')
    protocol = yaml.safe_load((REPO/'configs/protocol_v5.yaml').read_text())
    protocol['model_cells']['cap_per_well_and_line'] = CAP
    protocol['gene_space']['n_genes'] = 10
    (root/'protocol.yaml').write_text(yaml.safe_dump(protocol))
    return pd.DataFrame(truth)


def run(script, *args):
    env = {**os.environ, 'PYTHONPATH': str(REPO/'src'), 'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, str(REPO/'scripts'/script), *map(str, args)], cwd=REPO, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-3000:]+result.stderr[-3000:]
    return result


def extract_part(root, out, start, stop):
    return run('v5_extract.py', '--obs', root/'obs_metadata.parquet', '--gene-metadata', root/'gene_metadata.parquet',
               '--shard-dir', root/'shards', '--output-root', out, '--shard-start', start, '--shard-stop', stop,
               '--manifests', root/'manifests', '--protocol', root/'protocol.yaml')


def pipeline(root, parts):
    for start, stop in parts:
        extract_part(root, root/'extraction', start, stop)
    common = ['--manifests', root/'manifests', '--protocol', root/'protocol.yaml']
    run('v5_prepare.py', '--extraction-root', root/'extraction', '--output-root', root/'prepared', *common)
    run('v5_baselines.py', '--prepared', root/'prepared', '--output-root', root/'sealed', *common)
    return root/'prepared', root/'sealed'


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('v5')
    truth = make_dataset(tmp/'two')
    two = pipeline(tmp/'two', [(0, 2), (2, 3)])
    make_dataset(tmp/'one')
    one = pipeline(tmp/'one', [(0, 3)])
    make_dataset(tmp/'perturbed', perturb_test=True)
    perturbed = pipeline(tmp/'perturbed', [(0, 3)])
    return dict(truth=truth, two=two, one=one, perturbed=perturbed, root=tmp)


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def test_decode_cells_drops_markers_and_unknown_tokens():
    genes = pa.array([[1, 3, 5, 9000], [4, 3], []], pa.list_(pa.int64()))
    expressions = pa.array([[-2.0, 2.0, 1.0, 7.0], [3.0, 1.0], []], pa.list_(pa.float32()))
    lookup, _, _ = extract.gene_lookup(pd.DataFrame(dict(token_id=[3, 4, 5], ensembl_id=['a', 'b', 'c'],
                                                         gene_symbol=['A', 'B', 'C'])))
    cell, gene, count, stats = extract.decode_cells(genes, expressions, lookup)
    assert cell.tolist() == [0, 0, 1, 1] and gene.tolist() == [0, 2, 1, 0] and count.tolist() == [2, 1, 3, 1]
    assert stats['markers'] == 1 and stats['unknown_token_entries'] == 1 and stats['unknown_token_umi'] == 7
    with pytest.raises(ValueError, match='integer'):
        extract.decode_cells(pa.array([[3]], pa.list_(pa.int64())), pa.array([[1.5]], pa.list_(pa.float32())), lookup)
    with pytest.raises(ValueError, match='integer'):
        extract.decode_cells(pa.array([[3, 4]], pa.list_(pa.int64())), pa.array([[1.0, -1.0]], pa.list_(pa.float32())),
                             lookup)


def test_model_cells_are_capped_seeded_and_never_from_test_units():
    units = pd.DataFrame(dict(unit_id=['a', 'b', 'c'], role=['train', 'test', 'vehicle']))
    cells = pd.DataFrame(dict(barcode=[f'x{i:02d}' for i in range(28)], unit=[0]*12+[1]*12+[2]*4))
    keep = extract.select_model_cells(cells, units, cap=5, seed=42)
    assert keep[:12].sum() == 5 and not keep[12:24].any() and keep[24:].all()
    shuffled = cells.sample(frac=1, random_state=1).reset_index(drop=True)
    again = extract.select_model_cells(shuffled, units, cap=5, seed=42)
    assert set(shuffled.barcode[again]) == set(cells.barcode[keep])
    # (seeds 42 and 43 happen to draw the same 5 of 12 in numpy's generator; 44 does not)
    assert set(cells.barcode[extract.select_model_cells(cells, units, cap=5, seed=44)]) != set(cells.barcode[keep])


def test_extraction_unit_sums_match_ground_truth(runs):
    truth = runs['truth']
    receipt = json.loads((runs['two'][0]/'prepare_receipt.json').read_text())
    assert receipt['checks'] and all(receipt['checks'].values())
    parts = sorted((runs['root']/'two'/'extraction').glob('part_*'))
    units = pd.read_csv(parts[0]/'units.tsv', sep='\t')
    kept = truth[truth.passing.eq('full') & truth.line.isin(LINES) & truth['sample'].isin(units['sample'])]
    total = 0
    for p in parts:
        u = npz(p/'unit_counts.npz')
        total = total+sparse.csr_matrix((u['data'], u['indices'], u['indptr']), shape=tuple(u['shape'])).toarray()
    row = {unit: i for i, unit in enumerate(u['unit_ids'])}
    groups = kept.groupby(kept['sample']+':'+kept.line)
    assert len(groups) == len(units)
    for unit, group in groups:
        assert np.array_equal(total[row[unit]], np.sum(np.vstack(group['counts']), axis=0))


def test_partitioning_does_not_change_any_prepared_output(runs):
    for name in ['train_truth.npz', 'validation_truth.npz', 'model_cells.npz', 'vehicle_levels.npz',
                 'train_replicate_deltas.npz']:
        a, b = npz(runs['two'][0]/name), npz(runs['one'][0]/name)
        assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a), name
    for name in ['frozen_genes.tsv', 'queries.tsv', 'vehicle_contexts.tsv', 'count_qc/compound_qc.tsv']:
        assert (runs['two'][0]/name).read_bytes() == (runs['one'][0]/name).read_bytes(), name


def test_model_archive_excludes_test_cells_and_respects_the_cap(runs):
    obs, counts, library = load_model_archive(runs['one'][0]/'model_cells.npz')
    assert set(obs.role) == {'train', 'validation', 'vehicle'}
    assert not obs.compound_id.isin(['C10', 'C11', 'C12', 'X1', 'Adagrasib']).any()
    per_sample_line = obs.groupby(['replicate', 'cell_type']).size()
    assert per_sample_line.max() <= CAP
    vehicle = obs[obs.role.eq('vehicle')]
    assert vehicle.Combo.str.contains(':vehicle:').all() and not obs.cell_type.eq('L3').any()
    assert counts.shape == (len(obs), 10) and (library > 0).all()


def test_truth_matches_a_manual_replicate_mean(runs):
    truth, frame = runs['truth'], runs['one'][0]
    frozen = pd.read_csv(frame/'frozen_genes.tsv', sep='\t').source_index.to_numpy()
    kept = truth[truth.passing.eq('full')]

    def level(rows):
        counts = np.sum(np.vstack(rows['counts']), axis=0)
        return np.log1p(counts[frozen]/counts.sum()*1e6)
    expected = np.mean([level(kept[kept['sample'].eq(f'{p}_C01') & kept.line.eq('L1')])
                        - level(kept[kept['sample'].isin([f'{p}_va', f'{p}_vb']) & kept.line.eq('L1')])
                        for p in ['plate3', 'plate4']], axis=0)
    got = load_truth(frame/'train_truth.npz')[('C01', 'L1_5000nM')]
    assert np.allclose(got, expected, atol=1e-12)
    assert not any(k[0] in {'C10', 'C11', 'C12'} for k in load_truth(frame/'train_truth.npz'))


def test_count_qc_flags_the_small_unit_and_records_gates(runs):
    qc = pd.read_csv(runs['one'][0]/'count_qc/compound_qc.tsv', sep='\t')
    row = qc[qc.compound_id.eq('C12') & qc.context_id.eq('L2_500nM')].iloc[0]
    assert not row.eligible and row.min_treatment_cells < 30
    receipt = pd.read_json(runs['one'][0]/'prepare_receipt.json', typ='series')
    assert receipt['study_gate_G4']['met'] is False  # three test compounds cannot pass a 20-compound gate
    assert receipt['test_outcomes_computed'] is False


def test_test_responses_never_reach_prepared_inputs_or_sealed_comparators(runs):
    for name in ['train_truth.npz', 'validation_truth.npz', 'model_cells.npz', 'vehicle_levels.npz',
                 'train_replicate_deltas.npz']:
        a, b = npz(runs['one'][0]/name), npz(runs['perturbed'][0]/name)
        assert all(np.array_equal(a[k], b[k]) for k in a), name
    assert (runs['one'][0]/'frozen_genes.tsv').read_bytes() == (runs['perturbed'][0]/'frozen_genes.tsv').read_bytes()
    a, b = npz(runs['one'][1]/'baselines_v5_predictions.npz'), npz(runs['perturbed'][1]/'baselines_v5_predictions.npz')
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert {'B0', 'B1c', 'B2c', 'M1c', 'B1a', 'M1a', 'B1p'} <= set(a)


def test_anchored_and_plate_comparators_follow_their_definitions(runs):
    prepared, sealed = runs['one']
    pred = npz(sealed/'baselines_v5_predictions.npz')
    keys = [tuple(k.split('|')) for k in pred['keys']]
    queries = pd.read_csv(prepared/'queries.tsv', sep='\t', keep_default_na=False, dtype={'replicate': str})
    levels = npz(prepared/'vehicle_levels.npz')
    level = dict(zip(levels['vehicle_unit'], levels['level']))
    train = load_truth(prepared/'train_truth.npz')
    i = keys.index(('C10', 'L1_5000nM'))
    recs = baselines.records(queries, 'train')
    context = [level_mean for k, level_mean in baselines.query_vehicle_level(
        {k: v for k, v in recs.items() if k in train and k[1] == 'L1_5000nM'}, level).items()]
    vbar = np.mean(context, axis=0)
    own = np.mean([level['plate3:vehicle:L1'], level['plate4:vehicle:L1']], axis=0)
    assert np.allclose(pred['B1a'][i], pred['B1c'][i]+vbar-own)
    rep = npz(prepared/'train_replicate_deltas.npz')
    on3 = (rep['context_id'] == 'L1_5000nM') & (rep['plate'] == 'plate3')
    plate3 = np.mean([rep['delta'][on3 & (rep['compound_id'] == c)].mean(axis=0) for c in set(rep['compound_id'][on3])],
                     axis=0)
    plate4 = pred['B1c'][i]+vbar-level['plate4:vehicle:L1']  # two training compounds on plate4: falls back
    assert np.allclose(pred['B1p'][i], (plate3+plate4)/2)


def test_extraction_refuses_a_tampered_shard(tmp_path):
    make_dataset(tmp_path/'d')
    shard = next((tmp_path/'d'/'shards').glob('*.parquet'))
    shard.write_bytes(shard.read_bytes()+b'x')
    env = {**os.environ, 'PYTHONPATH': str(REPO/'src')}
    result = subprocess.run([sys.executable, str(REPO/'scripts/v5_extract.py'), '--obs', tmp_path/'d'/'obs_metadata.parquet',
                             '--gene-metadata', tmp_path/'d'/'gene_metadata.parquet', '--shard-dir', tmp_path/'d'/'shards',
                             '--output-root', tmp_path/'out', '--manifests', tmp_path/'d'/'manifests',
                             '--protocol', tmp_path/'d'/'protocol.yaml'], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'does not match the source manifest' in result.stderr


def test_merge_refuses_missing_shards(runs):
    part = next((runs['root']/'two'/'extraction').glob('part_00000_00002'))
    units = pd.read_csv(part/'units.tsv', sep='\t', keep_default_na=False)
    expected = pd.read_csv(part/'expected_unit_cells.tsv', sep='\t').expected_cells.to_numpy()
    selection = pd.read_csv(part/'model_selection.tsv', sep='\t')
    shards = pd.read_csv(runs['root']/'two'/'manifests'/'expression_source.tsv', sep='\t')
    with pytest.raises(ValueError, match='every manifest shard'):
        extract.merge_parts([part], units, expected_cells=expected, model_barcodes=selection.barcode, cap=CAP,
                            shard_manifest=shards[shards.role.eq('expression_shard')])


def test_b1p_falls_back_where_a_plate_has_too_few_training_compounds():
    table = pd.DataFrame(dict(compound_id=['a', 'b', 'a'], context_id='x', replicate=['s1', 's2', 's3'],
                              plate=['p1', 'p1', 'p2']))
    delta = np.array([[1.0], [3.0], [10.0]])
    means, counts = baselines.plate_means(table, delta, min_compounds=2)
    assert means == {('x', 'p1'): np.array([2.0])} and counts == {('x', 'p1'): 2, ('x', 'p2'): 1}
    recs = {('q', 'x'): [dict(replicate='r1', plate='p1', vehicle_unit='v1'), dict(replicate='r2', plate='p2',
                                                                                  vehicle_unit='v2')]}
    out, fallback = baselines.plate_anchored(recs, {('q', 'x'): np.array([5.0])}, {'v1': np.array([0.0]),
                                             'v2': np.array([1.0])}, {'x': np.array([0.5])}, means)
    assert fallback == 1 and np.allclose(out[('q', 'x')], [(2.0+(5.0+0.5-1.0))/2])


def test_prepare_constants_match_the_protocol_text():
    text = ' '.join((REPO/'configs/protocol_v5.yaml').read_text().split())
    assert f'>= {prepare.MIN_CELLS} cells' in text
    assert (f'at least {prepare.MIN_TEST_COMPOUNDS} eligible test compounds in at least {prepare.MIN_TEST_GROUPS} '
            'scaffold groups') in text
    gate = prepare.STUDY_GATE
    assert (f"at least {gate['min_contexts']} gate-passing contexts covering at least {gate['min_lines']} cell lines "
            f"and {gate['min_doses']} doses") in text
    assert 'nonzero counts in at least 5% of vehicle cells' in text and prepare.MIN_DETECTION == 0.05
    assert 'default_rng(42)' in text and f'fewer than {baselines.B1P_MIN_COMPOUNDS} such compounds' in text
