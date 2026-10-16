"""Measured cross-context errors with fixed prediction and control definitions."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .biology_v2 import file_record
from .evaluation import compound_metrics, summarize_compound_metrics

A_METHODS = ('B0', 'B1', 'B2', 'M1', 'M2', 'M3')
B_METHODS = ('B0', 'B1', 'B2', 'Ridge')
GROUP_KEYS = ('cell_type', 'time_point', 'replicate', 'plate_oligo')


def frozen_cohort(compounds, qualified, summary, split, context_id, expected_role):
    """Use existing count eligibility, rejecting altered identities or summaries."""
    rows = compounds[compounds.context_id.eq(context_id)].copy()
    info = summary[summary.context_id.eq(context_id)]
    tests = split[split.split.eq('test')].set_index('compound_id')
    if len(info) != 1 or rows.compound_id.duplicated().any() or tests.index.has_duplicates:
        raise ValueError('context QC identities and summary must be unique')
    rows = rows[rows.split.eq('test')].set_index('compound_id')
    if set(rows.index) != set(tests.index):
        raise ValueError('context QC differs from frozen test compound identities')
    rows = rows.loc[tests.index]
    for key in ['scaffold_group', 'canonical_smiles']:
        if not rows[key].equals(tests[key]):
            raise ValueError(f'context QC differs from frozen {key} identity')
    if not rows.role.eq(expected_role).all():
        raise ValueError('context QC has incorrect frozen scientific role')
    complete = bool(booleans(info.count_qc_done, 'summary count_qc_done').iloc[0])
    selected = rows[booleans(rows.eligible, 'compound eligible')].sort_index()
    n, k = len(selected), selected.scaffold_group.nunique()
    if (n != int(info.eligible_test_compounds.iloc[0])
            or k != int(info.eligible_test_structure_groups.iloc[0])):
        raise ValueError('context count-QC summary denominators disagree')
    gate = complete and n >= 20 and k >= 10
    if gate != bool(booleans(info.evaluation_gate_met, 'evaluation_gate_met').iloc[0]):
        raise ValueError('context count-QC gate disagrees with frozen thresholds')
    if complete:
        current = qualified[qualified.context_id.eq(context_id)]
        selected_rows = current[current.treatment.isin(selected.index)]
        for cid in selected.index:
            well_rows = selected_rows[selected_rows.treatment.eq(cid)]
            if (len(well_rows) != 2 or set(well_rows.replicate) != {'rep1', 'rep2'}
                    or not booleans(well_rows.eligible, 'well eligible').all()
                    or not booleans(well_rows.count_qc_done, 'well count_qc_done').all()):
                raise ValueError('qualified compound lacks two eligible replicate wells')
    return {'ids': selected.index.tolist() if complete else [],
            'groups': selected.scaffold_group.tolist() if complete else [],
            'gate_met': gate, 'count_qc_done': complete, 'original_test_compounds': len(tests),
            'eligible_test_compounds': n, 'eligible_test_structure_groups': k,
            'status': ('COUNT_QC_UNAVAILABLE' if not complete else 'COUNT_GATE_PASS'
                       if gate else 'DESCRIPTIVE_NO_COUNT_GATE')}


def checked_file(entry, base):
    """Check externally pinned manifest bytes before any data interpretation."""
    path = Path(entry['path'])
    path = path if path.is_absolute() else Path(base)/path
    actual = file_record(path)
    if actual['bytes'] != entry['bytes'] or actual['sha256'] != entry['sha256']:
        raise ValueError(f'manifest file hash/size mismatch: {path.name}')
    return path


def booleans(values, label):
    result = values.astype(str).str.lower().map({'true': True, 'false': False})
    if result.isna().any():
        raise ValueError(f'{label} requires explicit boolean values')
    return result.astype(bool)


def alternate_controls(wells, qualified, qc, compound_ids):
    """Qualify the prespecified two vehicles without reselecting strict samples."""
    wells, qualified, qc = wells.copy(), qualified.copy(), qc.copy()
    if wells.Combo.duplicated().any() or qc.Combo.duplicated().any():
        raise ValueError('well metadata and count QC keys must be unique')
    if len(set(compound_ids)) != len(compound_ids):
        raise ValueError('compound IDs must be unique')
    wells['vehicle'] = booleans(wells.vehicle, 'vehicle')
    wells['dose'] = pd.to_numeric(wells.dose, errors='raise')
    qualified['eligible'] = booleans(qualified.eligible, 'eligible')
    qualified['count_qc_done'] = booleans(qualified.count_qc_done, 'count_qc_done')
    q = pd.to_numeric(qc.n_cells, errors='raise')
    if not np.isfinite(q).all() or (q < 0).any() or not np.equal(q, np.floor(q)).all():
        raise ValueError('QC cell counts must be nonnegative integers')
    cell_counts = pd.Series(q.to_numpy(), index=qc.Combo)
    metadata = wells.set_index('Combo')
    records = []
    for cid in compound_ids:
        rows = qualified[qualified.treatment.eq(cid)].sort_values('replicate')
        if (len(rows) != 2 or set(rows.replicate) != {'rep1', 'rep2'}
                or rows.Combo.duplicated().any() or not rows.eligible.all() or not rows.count_qc_done.all()):
            raise ValueError('expected two unique qualified replicate records')
        for row in rows.itertuples(index=False):
            if row.Combo not in metadata.index or row.control_Combo not in metadata.index:
                raise ValueError('qualified treatment/control absent from frozen metadata')
            treated, strict = metadata.loc[row.Combo], metadata.loc[row.control_Combo]
            if (treated.vehicle or treated.treatment != cid or strict.vehicle is False
                    or not strict.vehicle or strict.dose != 0
                    or any(str(treated[k]) != str(strict[k]) for k in [*GROUP_KEYS, 'dose_pattern'])):
                raise ValueError('qualified strict control identity differs from frozen mapping')
            if 'control_Combo' in treated and treated.control_Combo != row.control_Combo:
                raise ValueError('strict control differs from frozen metadata')
            for key, expected in [(row.Combo, row.treatment_cells), (row.control_Combo, row.control_cells)]:
                if key not in cell_counts or cell_counts[key] != expected or expected < 30:
                    raise ValueError('qualified count QC disagrees with source well counts')
            same = wells.copy()
            for key in GROUP_KEYS:
                same = same[same[key].astype(str).eq(str(treated[key]))]
            controls = same[same.vehicle | same.dose.eq(0)].sort_values('Combo')
            reasons = []
            if len(controls) != 2:
                reasons.append('expected_exactly_two_same_plate_vehicle_wells')
            if not (controls.vehicle & controls.dose.eq(0)).all():
                reasons.append('invalid_vehicle_identity')
            if controls.dose_pattern.astype(str).nunique() != 2:
                reasons.append('vehicle_dose_patterns_not_distinct')
            if row.control_Combo not in set(controls.Combo):
                reasons.append('strict_control_not_in_pair')
            control_counts = controls.Combo.map(cell_counts)
            if control_counts.isna().any():
                reasons.append('missing_vehicle_count_QC')
            elif control_counts.lt(30).any():
                reasons.append('vehicle_below_30_cells')
            records.append(dict(compound_id=cid, replicate=row.replicate, treatment_Combo=row.Combo,
                strict_control_Combo=row.control_Combo, alternate_control_Combos=controls.Combo.tolist(),
                replicate_eligible=not reasons, exclusion_reason=';'.join(reasons)))
    result = pd.DataFrame(records)
    if result.empty:
        return pd.DataFrame(columns=['compound_id', 'replicate', 'treatment_Combo',
            'strict_control_Combo', 'alternate_control_Combos', 'replicate_eligible',
            'exclusion_reason', 'compound_eligible'])
    result['compound_eligible'] = result.compound_id.map(result.groupby('compound_id').replicate_eligible.all())
    return result


def required_wells(qualified, pairs, ids):
    selected = qualified[qualified.treatment.isin(ids)]
    keys = [*selected.Combo, *selected.control_Combo]
    for row in pairs.itertuples(index=False):
        if row.compound_eligible:
            keys.extend(row.alternate_control_Combos)
    return sorted(set(keys))


def measured_responses(counts, qualified, pairs, ids, gene_ids):
    """All-human CPM/log1p first; equal replicate and normalized-vehicle means."""
    if (counts.empty or counts.index.has_duplicates or counts.columns.has_duplicates
            or len(set(ids)) != len(ids) or len(set(gene_ids)) != len(gene_ids)):
        raise ValueError('gene, well and compound identities must be nonempty and unique')
    missing = set(gene_ids) - set(counts.index)
    if missing:
        raise ValueError(f'missing fixed gene identities: {sorted(missing)}')
    values = counts.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError('counts must be finite nonnegative integers')
    library = values.sum(axis=0)
    if (library <= 0).any() or not np.isfinite(library).all():
        raise ValueError('selected all-human libraries must be positive')
    normalized = pd.DataFrame(np.log1p(values / library[None, :] * 1e6),
                              index=counts.index, columns=counts.columns).loc[gene_ids]
    strict, alternate = [], []
    for cid in ids:
        rows = pairs[pairs.compound_id.eq(cid)].sort_values('replicate')
        if len(rows) != 2 or set(rows.replicate) != {'rep1', 'rep2'}:
            raise ValueError('expected exactly two response replicate records')
        p, a = [], []
        for row in rows.itertuples(index=False):
            treatment = normalized[row.treatment_Combo].to_numpy()
            p.append(treatment - normalized[row.strict_control_Combo].to_numpy())
            if row.compound_eligible:
                if len(row.alternate_control_Combos) != 2:
                    raise ValueError('sensitivity requires exactly two vehicles')
                a.append(treatment-normalized[row.alternate_control_Combos].mean(axis=1).to_numpy())
        strict.append(np.mean(p, axis=0))
        alternate.append(np.mean(a, axis=0) if len(a) == 2 else np.full(len(gene_ids), np.nan))
    return np.asarray(strict), np.asarray(alternate)


def align_predictions(sealed, ids, methods, width):
    labels = np.asarray(sealed['ids']).tolist()
    if (not all(isinstance(x, str) and x for x in labels) or len(set(labels)) != len(labels)
            or len(set(ids)) != len(ids)):
        raise ValueError('sealed and query compound IDs must be unique strings')
    missing = set(ids)-set(labels)
    if missing:
        raise ValueError(f'missing sealed compound IDs: {sorted(missing)}')
    order = [labels.index(x) for x in ids]
    output = {}
    for method in methods:
        values = np.asarray(sealed[method], dtype=float)
        if values.shape != (len(labels), width) or not np.isfinite(values).all():
            raise ValueError(f'sealed {method} has wrong dimensions or nonfinite predictions')
        output[method] = values[order]
    return output


def project_sets(delta, gene_ids, members, set_ids):
    if len(set(gene_ids)) != len(gene_ids) or members.duplicated(['set_id', 'gene_id']).any():
        raise ValueError('fixed gene/member identities must be unique')
    if set(members.set_id) != set(set_ids):
        raise ValueError('fixed gene-set identities differ')
    missing = set(members.gene_id)-set(gene_ids)
    if missing:
        raise ValueError(f'missing fixed set gene identities: {sorted(missing)}')
    positions = {gene: i for i, gene in enumerate(gene_ids)}
    return np.stack([np.asarray(delta)[:, [positions[g] for g in
        members.loc[members.set_id.eq(s), 'gene_id']]].mean(axis=1) for s in set_ids], axis=1)


def score_predictions(truth, predictions, ids, groups, *, metric, feature_ids=None, analysis_unit='gene'):
    if metric not in ['rmse', 'mae']:
        raise ValueError('unsupported primary error metric')
    tables, summaries, feature_rows = [], {}, []
    for method, predicted in predictions.items():
        table = compound_metrics(truth, predicted, ids, groups)
        summary = summarize_compound_metrics(table)
        summary['n_outputs'] = summary.pop('n_genes')
        summary['analysis_unit'] = analysis_unit
        summary['mean_error'] = summary['mean_compound_'+metric]
        summary['error_metric'] = metric
        table = table.rename(columns={'n_genes': 'n_outputs', 'n_sign_genes': 'n_sign_outputs'})
        table['analysis_unit'] = analysis_unit
        table['error'] = table[metric]; table['method'] = method; table['error_metric'] = metric
        summaries[method] = summary; tables.append(table)
        if feature_ids is not None:
            if len(feature_ids) != np.asarray(truth).shape[1]:
                raise ValueError('feature IDs do not match truth width')
            for i, cid in enumerate(ids):
                for j, feature in enumerate(feature_ids):
                    feature_rows.append(dict(compound_id=cid, scaffold_group=groups[i], method=method,
                        feature_id=feature, measured_delta=float(truth[i, j]),
                        predicted_delta=float(predicted[i, j]), absolute_error=float(abs(predicted[i, j]-truth[i, j]))))
    return pd.concat(tables, ignore_index=True), summaries, pd.DataFrame(feature_rows)


def bootstrap_error_difference(difference, ids, groups):
    """Resample K groups, keep each sampled group's compounds with multiplicity."""
    difference = np.asarray(difference, float)
    if (difference.shape != (len(ids),) or len(ids) != len(groups) or len(set(ids)) != len(ids)
            or not np.isfinite(difference).all() or len(set(groups)) < 2):
        raise ValueError('finite paired compound differences and at least two groups required')
    order = np.argsort(np.asarray(ids)); difference = difference[order]
    groups = np.asarray(groups)[order]; unique = sorted(set(groups))
    sums = np.array([difference[groups == g].sum() for g in unique])
    sizes = np.array([np.count_nonzero(groups == g) for g in unique])
    sampled = np.random.default_rng(2026).integers(0, len(unique), size=(2000, len(unique)))
    n = sizes[sampled].sum(axis=1); draws = sums[sampled].sum(axis=1)/n
    lo, hi = np.quantile(draws, [.025, .975], method='linear')
    return dict(point_estimate=float(difference.mean()), ci_lower=float(lo), ci_upper=float(hi),
        bootstrap_differences=draws, n_compounds=len(ids), n_groups=len(unique), draws=2000, seed=2026,
        sampled_compounds_min=int(n.min()), sampled_compounds_max=int(n.max()),
        method='paired_scaffold_cluster_percentile_bootstrap', aggregation_unit='compound',
        quantile_method='linear', confidence_level=.95)


def contrast_summary(table, candidate, baseline, *, gate_met):
    a = table[table.method.eq(candidate)].set_index('compound_id')
    b = table[table.method.eq(baseline)].set_index('compound_id')
    if a.index.has_duplicates or b.index.has_duplicates or set(a.index) != set(b.index) or a.empty:
        raise ValueError('contrasts require paired unique compound rows')
    a = a.sort_index(); b = b.loc[a.index]
    if not a.scaffold_group.equals(b.scaffold_group):
        raise ValueError('paired scaffold groups disagree')
    difference = a.error.to_numpy()-b.error.to_numpy()
    result = dict(point_estimate=float(difference.mean()), ci_lower=None, ci_upper=None,
        n_compounds=len(a), n_groups=a.scaffold_group.nunique(), candidate=candidate, baseline=baseline,
        comparison='candidate_minus_baseline_'+a.iloc[0].get('error_metric', 'error'),
        negative_favors='candidate', status='DESCRIPTIVE_NO_COUNT_GATE',
        no_CI_reason='Prespecified compound/scaffold feasibility gate not met')
    if gate_met:
        result.update(bootstrap_error_difference(difference, a.index.tolist(), a.scaffold_group.tolist()),
                      status='POINTWISE_CONDITIONAL_INTERVAL', no_CI_reason=None)
        result.pop('bootstrap_differences')
    return result


def score_control_sensitivity(strict, alternative, predictions, ids, groups, *, metric, feature_ids=None,
                              analysis_unit='gene'):
    valid = np.isfinite(alternative).all(axis=1)
    if ((np.isfinite(alternative).any(axis=1)) & ~valid).any():
        raise ValueError('alternate truth cannot be partially missing within a compound')
    selected = np.flatnonzero(valid)
    subset_ids = [ids[i] for i in selected]; subset_groups = [groups[i] for i in selected]
    if not len(selected):
        return {'compound_ids': [], 'groups': [], 'strict_summary': {}, 'alternative_summary': {},
                'strict_table': pd.DataFrame(), 'alternative_table': pd.DataFrame(),
                'strict_feature_table': pd.DataFrame(), 'alternative_feature_table': pd.DataFrame()}
    sliced = {m: p[selected] for m, p in predictions.items()}
    st, ss, sf = score_predictions(strict[selected], sliced, subset_ids, subset_groups,
                                   metric=metric, feature_ids=feature_ids, analysis_unit=analysis_unit)
    at, ast, af = score_predictions(alternative[selected], sliced, subset_ids, subset_groups,
                                    metric=metric, feature_ids=feature_ids, analysis_unit=analysis_unit)
    return {'compound_ids': subset_ids, 'groups': subset_groups, 'strict_summary': ss,
            'alternative_summary': ast, 'strict_table': st, 'alternative_table': at,
            'strict_feature_table': sf, 'alternative_feature_table': af}


def evaluate_analysis(strict, alternative, predictions, ids, groups, *, analysis,
                      gate_met, feature_ids=None, unique_member_genes=None):
    """Three explicit comparison cohorts; never compare full strict to subset alt."""
    metric, unit = ('rmse', 'gene') if analysis == 'A' else ('mae', 'gene_set')
    strict_table, strict_summary, strict_features = score_predictions(strict, predictions, ids, groups,
        metric=metric, feature_ids=feature_ids, analysis_unit=unit)
    sensitivity = score_control_sensitivity(strict, alternative, predictions, ids, groups,
        metric=metric, feature_ids=feature_ids, analysis_unit=unit)
    subset_gate = len(sensitivity['compound_ids']) >= 20 and len(set(sensitivity['groups'])) >= 10
    comparisons = [
        ('strict_full', strict_table, strict_summary, strict_features, gate_met),
        ('strict_sensitivity_subset', sensitivity['strict_table'], sensitivity['strict_summary'],
         sensitivity['strict_feature_table'], subset_gate),
        ('alternate_sensitivity_subset', sensitivity['alternative_table'], sensitivity['alternative_summary'],
         sensitivity['alternative_feature_table'], subset_gate),
    ]
    contrasts = [('M3', 'M1'), ('M3', 'B1')] if analysis == 'A' else [('Ridge', 'B1')]
    tables, features, contrast_rows, summaries = [], [], [], {}
    for name, table, summary, feature_table, gate in comparisons:
        summaries[name] = {'metrics': summary, 'gate_met': gate,
            'n_compounds': len(ids) if name == 'strict_full' else len(sensitivity['compound_ids']),
            'n_groups': len(set(groups)) if name == 'strict_full' else len(set(sensitivity['groups']))}
        if table.empty:
            summaries[name]['status'] = 'NO_ELIGIBLE_SENSITIVITY_COMPOUNDS'
            continue
        table = table.assign(analysis=analysis, comparison=name)
        if analysis == 'B':
            table['n_gene_sets'] = len(feature_ids)
            table['n_unique_member_genes'] = unique_member_genes
            for value in summary.values():
                value['n_gene_sets'] = len(feature_ids)
                value['n_unique_member_genes'] = unique_member_genes
        tables.append(table)
        if not feature_table.empty:
            features.append(feature_table.rename(columns={'feature_id': 'set_id'}).assign(
                analysis=analysis, comparison=name))
        summaries[name]['contrasts'] = []
        for candidate, baseline in contrasts:
            result = contrast_summary(table, candidate, baseline, gate_met=gate)
            result.update(analysis=analysis, comparison=name, error_metric=metric)
            contrast_rows.append(result); summaries[name]['contrasts'].append(result)
    return dict(per_compound=pd.concat(tables, ignore_index=True),
        per_set=pd.concat(features, ignore_index=True) if features else pd.DataFrame(),
        contrasts=pd.DataFrame(contrast_rows), summary=summaries,
        sensitivity_compound_ids=sensitivity['compound_ids'])


def evaluate_context(counts, wells, qualified, qc, cohort, a_predictions, a_gene_ids,
                     *, b_predictions=None, members=None, set_ids=None):
    ids, groups = cohort['ids'], cohort['groups']
    pairs = alternate_controls(wells, qualified, qc, ids)
    genes = list(a_gene_ids)
    if b_predictions is not None:
        genes += sorted(set(members.gene_id)-set(genes))
    strict, alternate = measured_responses(counts, qualified, pairs, ids, genes)
    a = evaluate_analysis(strict[:, :len(a_gene_ids)], alternate[:, :len(a_gene_ids)], a_predictions,
                          ids, groups, analysis='A', gate_met=cohort['gate_met'])
    analyses = {'A': a}; summary = {'A': a['summary']}
    if b_predictions is not None:
        b = evaluate_analysis(project_sets(strict, genes, members, set_ids),
            project_sets(alternate, genes, members, set_ids), b_predictions, ids, groups,
            analysis='B', gate_met=cohort['gate_met'], feature_ids=set_ids,
            unique_member_genes=members.gene_id.nunique())
        analyses['B'] = b; summary['B'] = b['summary']
    else:
        summary['B'] = {'status': 'FAILED_COVERAGE_ZERO_FITS_NO_PRIMARY_HEAD'}
    tables = {}
    for name in ['per_compound', 'per_set', 'contrasts']:
        selected = [a[name] for a in analyses.values() if not a[name].empty]
        tables[name] = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    return {**tables, 'summary': summary, 'pairs': pairs,
            'sensitivity_compounds': len(a['sensitivity_compound_ids']),
            'sensitivity_groups': len({groups[ids.index(i)] for i in a['sensitivity_compound_ids']})}
