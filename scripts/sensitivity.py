#!/usr/bin/env python3
"""Post-primary, predeclared sensitivities with the same six sealed predictions."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import yaml
from drugperturb.evaluation import compound_metrics, summarize_compound_metrics

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('B0', 'B1', 'B2', 'M1', 'M2', 'M3')
GROUP_KEYS = ('cell_type', 'time_point', 'replicate', 'plate_oligo')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def boolean_column(values, name):
    result = values.astype(str).str.lower().map({'true': True, 'false': False})
    if result.isna().any():
        raise ValueError(f'{name} requires explicit True/False values')
    return result.astype(bool)

def qualify_alternate_controls(wells, mapping, qc, compound_ids, minimum=30):
    """Use metadata/count thresholds only; never inspect expression outcomes."""
    wells = wells.copy()
    mapping = mapping.copy()
    if wells.Combo.duplicated().any() or qc.index.duplicated().any():
        raise ValueError('Well metadata and QC must have unique Combo keys')
    wells['vehicle'] = boolean_column(wells.vehicle, 'vehicle')
    wells['dose'] = pd.to_numeric(wells.dose, errors='raise')
    mapping['eligible'] = boolean_column(mapping.eligible, 'eligible')
    for column in GROUP_KEYS + ('dose_pattern',):
        wells[column] = wells[column].astype(str)
        mapping[column] = mapping[column].astype(str)
    q = pd.to_numeric(qc.n_cells, errors='raise')
    if not np.isfinite(q).all() or (q < 0).any() or not np.equal(q, np.floor(q)).all():
        raise ValueError('Cell counts must be nonnegative integers')
    metadata = wells.set_index('Combo')
    records = []
    for cid in compound_ids:
        rows = mapping.loc[mapping.treatment.eq(cid)].sort_values('replicate')
        if len(rows) != 2 or set(rows.replicate) != {'rep1', 'rep2'} or not rows.eligible.all():
            raise ValueError(f'{cid}: frozen primary test mapping is not two eligible replicates')
        for row in rows.itertuples(index=False):
            if row.Combo not in metadata.index or row.control_Combo not in metadata.index:
                raise ValueError(f'{cid}: primary treatment/control absent from well metadata')
            treatment = metadata.loc[row.Combo]
            strict = metadata.loc[row.control_Combo]
            if treatment.vehicle or treatment.dose != 1000 or not strict.vehicle or strict.dose != 0:
                raise ValueError(f'{cid}: primary treatment/control identity changed')
            if treatment.treatment != cid or any(str(treatment[key]) != str(getattr(row, key)) for key in GROUP_KEYS):
                raise ValueError(f'{cid}: primary treatment grouping disagrees with frozen mapping')
            if any(str(strict[key]) != str(getattr(row, key)) for key in GROUP_KEYS + ('dose_pattern',)):
                raise ValueError(f'{cid}: strict control does not match frozen treatment keys')
            if row.Combo not in q.index or row.control_Combo not in q.index:
                raise ValueError(f'{cid}: primary cell counts absent')
            if q[row.Combo] != row.treatment_cells or q[row.control_Combo] != row.control_cells:
                raise ValueError(f'{cid}: primary QC counts disagree with frozen mapping')
            group = wells.copy()
            for key in GROUP_KEYS:
                group = group.loc[group[key].eq(str(getattr(row, key)))]
            # Include malformed zero-dose/vehicle-labelled candidates to flag them,
            # rather than silently hiding an inconsistent control annotation.
            controls = group.loc[group.vehicle | group.dose.eq(0)].sort_values('Combo')
            reasons = []
            if len(controls) != 2:
                reasons.append('expected_exactly_two_same_plate_vehicle_wells')
            if not (controls.vehicle & controls.dose.eq(0)).all():
                reasons.append('control_not_explicit_vehicle_true_and_dose_zero')
            if controls.dose_pattern.nunique() != 2:
                reasons.append('vehicle_dose_patterns_not_distinct')
            if row.control_Combo not in set(controls.Combo):
                reasons.append('primary_strict_control_not_among_alternate_pair')
            counts = controls.Combo.map(q)
            if counts.isna().any():
                reasons.append('alternate_vehicle_QC_missing')
            elif (counts < minimum).any():
                reasons.append('alternate_vehicle_below_minimum_cells')
            records.append({'compound_id': cid, 'replicate': row.replicate, 'treatment_Combo': row.Combo,
                            'primary_control_Combo': row.control_Combo, 'plate_oligo': row.plate_oligo,
                            'cell_type': row.cell_type, 'time_point': row.time_point,
                            'treatment_cells': int(q[row.Combo]), 'primary_control_cells': int(q[row.control_Combo]),
                            'alternate_control_Combos': controls.Combo.tolist(),
                            'alternate_control_cell_counts': [None if pd.isna(v) else int(v) for v in counts],
                            'alternate_dose_patterns': controls.dose_pattern.tolist(),
                            'eligible': not reasons, 'exclusion_reason': ';'.join(reasons)})
    result = pd.DataFrame(records)
    accepted = result.groupby('compound_id').eligible.all().to_dict()
    result['compound_eligible'] = result.compound_id.map(accepted)
    return result

def normalize_allgene_wells(counts):
    """Normalize each complete all-human-gene library before selecting genes."""
    if counts.index.has_duplicates or counts.columns.has_duplicates or counts.empty:
        raise ValueError('Counts require nonempty unique gene and well keys')
    values = counts.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError('Counts must be finite nonnegative integers')
    libraries = values.sum(axis=0)
    if (libraries <= 0).any() or not np.isfinite(libraries).all():
        raise ValueError('Selected treatment/control libraries must be positive')
    return pd.DataFrame(np.log1p(values / libraries[None, :] * 1e6), index=counts.index, columns=counts.columns)

def responses_for_qualified(normalized, qualifications, compound_ids, selected_genes):
    """Equal-weight replicate means; alternate vehicle mean after normalization."""
    selected = normalized.loc[selected_genes]
    primary, alternate = [], []
    for cid in compound_ids:
        rows = qualifications.loc[qualifications.compound_id.eq(cid)].sort_values('replicate')
        if len(rows) != 2:
            raise ValueError('Expected two replicate records')
        p, a = [], []
        for row in rows.itertuples(index=False):
            treated = selected[row.treatment_Combo].to_numpy()
            p.append(treated - selected[row.primary_control_Combo].to_numpy())
            if row.compound_eligible:
                if len(row.alternate_control_Combos) != 2:
                    raise ValueError('Qualified alternate comparison requires exactly two vehicles')
                a.append(treated - selected[row.alternate_control_Combos].mean(axis=1).to_numpy())
        primary.append(np.mean(p, axis=0))
        alternate.append(np.mean(a, axis=0) if len(a) == 2 else np.full(len(selected_genes), np.nan))
    return np.asarray(primary), np.asarray(alternate)

def main():
    started = time.perf_counter()
    seal_path = ROOT / 'receipts/prediction_seal.json'
    final_path = ROOT / 'receipts/final_evaluation.json'
    if not seal_path.exists() or not final_path.exists():
        raise FileNotFoundError('Both prediction_seal and final_evaluation must exist before sensitivity; no expression data loaded')
    seal = json.loads(seal_path.read_text())
    final = json.loads(final_path.read_text())
    freeze_path = ROOT / 'receipts/dataset_freeze.json'
    freeze = json.loads(freeze_path.read_text())
    seal_digest = sha(seal_path)
    if final['prediction_seal_sha256'] != seal_digest or sha(freeze_path) != seal['dataset_freeze_sha256']:
        raise ValueError('Main evaluation, prediction seal and dataset freeze are not bound')
    for name, digest in seal['artifact_hashes'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'Sealed model/prediction/configuration changed: {name}')
    for name, digest in final['output_hashes'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'Primary evaluation output changed: {name}')
    required = ['configs/protocol_v1.yaml', 'manifests/selected_wells.tsv', 'manifests/qualified_well_mapping.tsv',
                'manifests/selected_genes.tsv', 'manifests/split_manifest.tsv', 'data/processed/pseudobulk_counts.tsv.gz',
                'data/processed/well_qc.tsv', 'data/processed/sealed_test_truth.npz',
                'models/run_v1/test_predictions.npz', 'receipts/prediction_seal.json',
                'receipts/final_evaluation.json', 'receipts/dataset_freeze.json', 'scripts/sensitivity.py',
                'src/drugperturb/evaluation.py']
    hashes = {name: sha(ROOT / name) for name in required}
    receipt = ROOT / 'receipts/sensitivity.json'
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old['input_hashes'] != hashes:
            raise ValueError('Completed sensitivity inputs changed; refuse silent rerun')
        for name, digest in old['output_hashes'].items():
            if sha(ROOT / name) != digest:
                raise ValueError(f'Sensitivity output changed: {name}')
        print('Existing sensitivity receipt verified; no scoring repeated.')
        return
    for name in ['configs/protocol_v1.yaml', 'manifests/qualified_well_mapping.tsv', 'manifests/selected_genes.tsv',
                 'manifests/split_manifest.tsv', 'data/processed/sealed_test_truth.npz']:
        if freeze['artifact_hashes'].get(name) != hashes[name]:
            raise ValueError(f'Frozen sensitivity dependency changed: {name}')
    if final['test_truth_sha256'] != hashes['data/processed/sealed_test_truth.npz']:
        raise ValueError('Primary truth differs from final evaluation')
    protocol = yaml.safe_load((ROOT / 'configs/protocol_v1.yaml').read_text())
    if protocol['controls']['sensitivity'] != 'same_plate_both_vehicle_wells':
        raise ValueError('Alternate control was not predeclared')
    minimum = protocol['qc']['minimum_cells_per_matched_vehicle']
    if minimum != 30:
        raise ValueError('This predeclared alternate-control sensitivity requires >=30 cells per vehicle')
    split = pd.read_csv(ROOT / 'manifests/split_manifest.tsv', sep='\t')
    test = split.loc[split.split.eq('test')].reset_index(drop=True)
    ids, groups = test.compound_id.tolist(), test.scaffold_group.tolist()
    with np.load(ROOT / 'models/run_v1/test_predictions.npz', allow_pickle=False) as archive:
        if archive['ids'].tolist() != ids or not set(METHODS).issubset(archive.files):
            raise ValueError('Frozen six-method prediction IDs/methods disagree')
        predictions = {method: archive[method].copy() for method in METHODS}
    with np.load(ROOT / 'data/processed/sealed_test_truth.npz', allow_pickle=False) as archive:
        if archive['ids'].tolist() != ids:
            raise ValueError('Frozen primary truth IDs disagree')
        primary_truth = archive['y'].copy()
    wells = pd.read_csv(ROOT / 'manifests/selected_wells.tsv', sep='\t', dtype={'dose_pattern': str})
    mapping = pd.read_csv(ROOT / 'manifests/qualified_well_mapping.tsv', sep='\t', dtype={'dose_pattern': str})
    qc = pd.read_csv(ROOT / 'data/processed/well_qc.tsv', sep='\t').set_index('Combo')
    qualification = qualify_alternate_controls(wells, mapping, qc, ids, minimum)
    alternate_mask = np.array([bool(qualification.loc[qualification.compound_id.eq(cid), 'compound_eligible'].all()) for cid in ids])
    count50_mask = np.array([bool((qualification.loc[qualification.compound_id.eq(cid), 'treatment_cells'] >= 50).all()) for cid in ids])
    genes_table = pd.read_csv(ROOT / 'manifests/selected_genes.tsv', sep='\t')
    if genes_table.output_index.tolist() != list(range(len(genes_table))):
        raise ValueError('Frozen output gene order is invalid')
    genes = genes_table.gene_id.tolist()
    keys = set(qualification.treatment_Combo) | set(qualification.primary_control_Combo)
    for row in qualification.itertuples(index=False):
        if row.compound_eligible:
            keys.update(row.alternate_control_Combos)
    counts = pd.read_csv(ROOT / 'data/processed/pseudobulk_counts.tsv.gz', sep='\t', index_col=0,
                         usecols=lambda c: c == 'gene_id' or c in keys)
    if set(counts.columns) != keys or len(counts) != freeze['source_genes']:
        raise ValueError('Required all-gene libraries are missing or truncated')
    if not np.array_equal(counts.sum(axis=0).to_numpy(), qc.loc[counts.columns, 'human_UMI'].to_numpy()):
        raise ValueError('Raw well libraries disagree with QC UMI totals')
    normalized = normalize_allgene_wells(counts)
    reconstructed, alternate_truth = responses_for_qualified(normalized, qualification, ids, genes)
    if reconstructed.shape != primary_truth.shape or not np.allclose(reconstructed, primary_truth, rtol=1e-10, atol=1e-10):
        raise ValueError('Reconstructed strict-control response differs from frozen primary truth')
    scenarios = [
        ('primary_all_test', np.ones(len(ids), dtype=bool), primary_truth),
        ('primary_on_alternate_qualified', alternate_mask, primary_truth),
        ('alternate_two_vehicle_mean', alternate_mask, alternate_truth),
        ('primary_treatment_both_replicates_ge50', count50_mask, primary_truth),
    ]
    all_metrics, summaries = [], []
    ids_array, groups_array = np.asarray(ids), np.asarray(groups)
    for scenario, mask, truth in scenarios:
        for method in METHODS:
            if mask.any():
                metrics = compound_metrics(truth[mask], predictions[method][mask], ids_array[mask].tolist(), groups_array[mask].tolist())
                summary = summarize_compound_metrics(metrics)
                metrics.insert(0, 'method', method)
                metrics.insert(0, 'scenario', scenario)
                all_metrics.append(metrics)
                summaries.append({'scenario': scenario, 'method': method, 'status': 'scored', 'primary_test_denominator': len(ids), **summary})
            else:
                summaries.append({'scenario': scenario, 'method': method, 'status': 'NA_no_qualified_compounds',
                                  'primary_test_denominator': len(ids), 'n_compounds': 0, 'n_genes': len(genes), 'mean_compound_rmse': None})
    metrics = pd.concat(all_metrics, ignore_index=True)
    primary_same = metrics.loc[metrics.scenario.eq('primary_on_alternate_qualified'), ['method', 'compound_id', 'rmse']].rename(columns={'rmse': 'primary_rmse'})
    alternative = metrics.loc[metrics.scenario.eq('alternate_two_vehicle_mean'), ['method', 'compound_id', 'rmse']].rename(columns={'rmse': 'alternate_rmse'})
    changes = primary_same.merge(alternative, on=['method', 'compound_id'], how='outer', validate='one_to_one')
    changes['alternate_minus_primary_rmse'] = changes.alternate_rmse - changes.primary_rmse
    truth_shifts = []
    for i, cid in enumerate(ids):
        subset = qualification.loc[qualification.compound_id.eq(cid)]
        qualified = bool(alternate_mask[i])
        difference = alternate_truth[i] - primary_truth[i] if qualified else None
        truth_shifts.append({'compound_id': cid, 'alternate_qualified': qualified, 'both_treatment_replicates_ge50': bool(count50_mask[i]),
                             'alternate_exclusion_reasons': ';'.join(sorted(set(reason for reason in subset.exclusion_reason if reason))),
                             'n_selected_genes': len(genes), 'truth_shift_RMS': float(np.sqrt(np.mean(difference ** 2))) if qualified else None,
                             'truth_shift_mean': float(np.mean(difference)) if qualified else None})
    # Source eligibility is copied from the original freeze; no new predictions
    # or response-based eligibility decisions are made at 20/30/50 cells.
    source_cutoffs = []
    for cutoff in ('20', '30', '50'):
        source_cutoffs.append({'minimum_cells': int(cutoff), 'source_labels_denominator': freeze['source_labels'],
                               **freeze['cell_count_sensitivity'][cutoff],
                               'source': 'existing_dataset_freeze', 'new_compound_predictions': False,
                               'eligibility_definition': 'both treatment replicates and their strict controls satisfy cutoff; then chemistry qualification'})
    config = {'control_sensitivity': 'mean_of_two_separately_CPM_log1p_normalized_same_plate_vehicles',
              'matching_keys': list(GROUP_KEYS), 'number_of_vehicles': 2, 'require_distinct_dose_patterns': True,
              'require_explicit_vehicle_true_and_dose_zero': True, 'minimum_cells_per_alternate_vehicle': 30,
              'replicate_combination': 'equal_weight_mean_of_two_deltas', 'normalization_library': 'all_human_genes_before_selected_gene_subsetting',
              'cutoff50_definition': 'both treatment replicates >=50; primary truth/control requirements unchanged',
              'models_retrained': False, 'predictions_changed': False, 'selected_genes_changed': False,
              'primary_test_compounds': len(ids), 'alternate_qualified_compounds': int(alternate_mask.sum()),
              'alternate_excluded_compounds': int((~alternate_mask).sum()), 'cutoff50_test_compounds': int(count50_mask.sum()),
              'methods': list(METHODS), 'n_frozen_output_genes': len(genes)}
    out = ROOT / 'reports/sensitivity'
    out.mkdir(parents=True, exist_ok=True)
    public_qualification = qualification.copy()
    for column in ('alternate_control_Combos', 'alternate_control_cell_counts', 'alternate_dose_patterns'):
        public_qualification[column] = public_qualification[column].map(json.dumps)
    public_qualification.to_csv(out / 'control_qualification.tsv', sep='\t', index=False)
    metrics.to_csv(out / 'metrics_by_compound.tsv', sep='\t', index=False)
    pd.DataFrame(summaries).to_csv(out / 'metrics_summary.tsv', sep='\t', index=False)
    changes.to_csv(out / 'paired_control_RMSE_changes.tsv', sep='\t', index=False)
    pd.DataFrame(truth_shifts).to_csv(out / 'compound_sensitivity_context.tsv', sep='\t', index=False)
    pd.DataFrame(source_cutoffs).to_csv(out / 'source_cutoff_eligibility.tsv', sep='\t', index=False)
    save_json(out / 'summary.json', {'config': config, 'summaries': summaries, 'source_cutoff_eligibility': source_cutoffs,
                                   'interpretation': 'Changed-truth and fixed-subset sensitivity with primary frozen predictions; not independent validation or a retrained alternate-control benchmark.'})
    if sha(seal_path) != seal_digest or any(sha(ROOT / name) != digest for name, digest in seal['artifact_hashes'].items()):
        raise ValueError('A sealed artifact changed during sensitivity')
    save_json(receipt, {'status': 'COMPLETE_FROZEN_PREDICTION_SENSITIVITY', 'completed_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                        'config': config, 'input_hashes': hashes, 'prediction_seal_sha256': seal_digest,
                        'final_evaluation_sha256': sha(final_path), 'primary_response_reconstruction_verified': True,
                        'output_hashes': {str(path.relative_to(ROOT)): sha(path) for path in sorted(out.glob('*')) if path.is_file()},
                        'elapsed_seconds': time.perf_counter() - started, 'python': platform.python_version(),
                        'limitations': ['Changing the evaluation reference changes truth, not fitted predictions.',
                                        'Two controls are equally weighted after separate normalization, not pooled by library or cell count.',
                                        'Cutoff50 is a technical subset of the primary test set; selective survival can change its composition.',
                                        'No new predictions for newly eligible cutoff20 compounds; source counts are historical frozen eligibility only.',
                                        'Shared vehicles, plate effects and two biological replicates do not provide independent validation.']})
    print(json.dumps(config, indent=2), flush=True)

if __name__ == '__main__':
    main()
