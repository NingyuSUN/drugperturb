"""Train-only fixed gene-set labels and a bounded, direct biological head."""
from __future__ import annotations

import csv
import gzip
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from drugperturb.chemistry import morgan_fingerprint
from drugperturb.response import baseline_predictions, build_responses

ALPHAS = (0.1, 1.0, 10.0, 100.0)


def file_record(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'basename': path.name, 'bytes': path.stat().st_size, 'sha256': digest}


def project_mapping(mapping, compound_ids):
    """Select requested compounds before validating any response-bearing wells."""
    required = {'treatment', 'replicate', 'Combo', 'control_Combo', 'eligible'}
    if not required.issubset(mapping):
        raise ValueError('mapping lacks required treatment/control fields')
    ids = list(compound_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('compound IDs must be nonempty and unique')
    selected = mapping.loc[mapping.treatment.isin(ids)].copy()
    if set(selected.treatment) != set(ids):
        raise ValueError('missing source treatment mapping')
    if selected[list(required)].isna().any().any():
        raise ValueError('missing selected mapping key')
    if not selected.eligible.isin([True, False]).all() or not selected.eligible.all():
        raise ValueError('frozen source compounds require eligible treatment/control pairs')
    if selected.Combo.duplicated().any():
        raise ValueError('duplicate treatment Combo')
    for compound, rows in selected.groupby('treatment'):
        if len(rows) != 2 or set(rows.replicate) != {'rep1', 'rep2'}:
            raise ValueError(f'{compound}: expected exactly rep1 and rep2')
    if set(selected.Combo) & set(selected.control_Combo):
        raise ValueError('treatment wells overlap control wells')
    return selected.sort_values(['treatment', 'replicate'], kind='stable').reset_index(drop=True)


def read_projected_counts(path, selected_mapping):
    """Parse only requested treatment and unique matched vehicle columns.

    All human gene rows remain present for the library-size denominator. Other
    count columns are never parsed as numeric arrays, even for input validation.
    """
    path = Path(path)
    keys = list(dict.fromkeys([*selected_mapping.Combo, *selected_mapping.control_Combo]))
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        header = next(csv.reader(stream, delimiter='\t'))
    if not header or header[0] != 'gene_id':
        raise ValueError('counts require a first gene_id column')
    if any(header.count(key) != 1 for key in ['gene_id', *keys]):
        raise ValueError('missing or duplicate requested count column')
    counts = pd.read_csv(path, sep='\t', usecols=['gene_id', *keys], index_col='gene_id')
    counts = counts.loc[:, keys]
    _cpm(counts)
    return counts


def _cpm(counts):
    if (counts.empty or counts.index.has_duplicates or counts.index.isna().any()
            or counts.columns.has_duplicates):
        raise ValueError('count gene/well identities must be nonempty and unique')
    try:
        values = counts.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('projected counts must be numeric') from exc
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.equal(values, np.floor(values)).all()):
        raise ValueError('projected counts must be finite nonnegative integers')
    libraries = values.sum(axis=0)
    if not np.isfinite(libraries).all() or (libraries <= 0).any():
        raise ValueError('all human-gene count libraries must be positive and finite')
    return pd.DataFrame(values / libraries[None, :] * 1e6,
                        index=counts.index, columns=counts.columns)


def training_prevalence(train_counts, train_mapping):
    """CPM>=1 in ceil(.2*N) treatment OR unique matched vehicle wells."""
    treatment = list(dict.fromkeys(train_mapping.Combo))
    controls = list(dict.fromkeys(train_mapping.control_Combo))
    keys = list(dict.fromkeys([*treatment, *controls]))
    if not set(keys).issubset(train_counts.columns):
        raise ValueError('missing train treatment/control count column')
    cpm = _cpm(train_counts.loc[:, keys])
    nt, nc = math.ceil(.2 * len(treatment)), math.ceil(.2 * len(controls))
    mask = (cpm.loc[:, treatment].ge(1).sum(axis=1).ge(nt)
            | cpm.loc[:, controls].ge(1).sum(axis=1).ge(nc))
    return mask, {'treatment_wells': len(treatment), 'unique_control_wells': len(controls),
                  'treatment_required_wells': nt, 'control_required_wells': nc,
                  'all_human_genes': len(cpm), 'expressed_genes': int(mask.sum())}


def freeze_members(prevalence, biology):
    """Preserve source denominators, deduplicate mapped IDs, fail absent IDs."""
    definitions = biology.get('sets', [])
    if len(definitions) != 5 or len({s['set_id'] for s in definitions}) != 5:
        raise ValueError('exactly five unique frozen gene sets are required')
    rows, retained = [], []
    for definition in definitions:
        mapped = sorted(set(definition['mapped_unique_gene_ids']))
        original = definition['source_member_count']
        if isinstance(original, bool) or not isinstance(original, int) or original <= 0:
            raise ValueError('source member count must be a positive integer')
        missing = set(mapped) - set(prevalence.index)
        if missing:
            raise ValueError(f'missing mapped gene identities for {definition["set_id"]}: {sorted(missing)}')
        members = [g for g in mapped if bool(prevalence.loc[g])]
        rows.append({'set_id': definition['set_id'], 'source_member_count': original,
                     'mapped_unique_genes': len(mapped), 'n_retained': len(members),
                     'fraction_original_members': len(members) / original,
                     'passed': len(members) >= 10 and len(members) / original >= .5})
        retained.extend({'set_id': definition['set_id'], 'gene_id': g} for g in members)
    return pd.DataFrame(rows), pd.DataFrame(retained, columns=['set_id', 'gene_id'])


def gene_set_labels(counts, mapping, members, set_ids):
    missing = set(members.gene_id) - set(counts.index)
    if missing:
        raise ValueError(f'missing fixed member gene identities: {sorted(missing)}')
    responses = build_responses(counts, mapping)
    positions = {g: i for i, g in enumerate(responses['gene_ids'])}
    labels = []
    for set_id in set_ids:
        genes = members.loc[members.set_id.eq(set_id), 'gene_id'].tolist()
        if not genes or len(set(genes)) != len(genes):
            raise ValueError('fixed set member IDs must be nonempty and unique')
        labels.append(responses['delta'][:, [positions[g] for g in genes]].mean(axis=1))
    return responses['compound_ids'], np.asarray(labels, dtype=float).T


def _binary_fp(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2048 or not np.isin(values, [0, 1]).all():
        raise ValueError(f'{name} must be unscaled binary Morgan2048 features')
    return values


def fit_candidates(train_fp, train_y, val_fp, val_y, train_ids, query_fp):
    train_fp, val_fp, query_fp = [_binary_fp(x, n) for x, n in
        [(train_fp, 'train'), (val_fp, 'validation'), (query_fp, 'query')]]
    train_y, val_y = np.asarray(train_y, float), np.asarray(val_y, float)
    if (train_y.shape != (len(train_fp), 5) or val_y.shape != (len(val_fp), 5)
            or not len(val_y) or not len(train_y) or not np.isfinite(train_y).all()
            or not np.isfinite(val_y).all() or len(train_ids) != len(train_y)
            or len(set(train_ids)) != len(train_ids)):
        raise ValueError('finite five-set train/validation targets and unique IDs are required')
    # ID-sorted training makes nearest-neighbor ties deterministic and explicit.
    order = np.argsort(np.asarray(train_ids, dtype=str), kind='stable')
    x, y = train_fp[order], train_y[order]
    models, scores = [], []
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, fit_intercept=True, solver='cholesky').fit(x, y)
        scores.append(float(np.abs(val_y - model.predict(val_fp)).mean(axis=1).mean()))
        models.append(model)
    chosen = int(np.argmin(scores))
    predictions = []
    for query in [val_fp, query_fp]:
        output = baseline_predictions(y, x, query)
        output['Ridge'] = models[chosen].predict(query)
        predictions.append(output)
    return {'models': models, 'fit_count': 4, 'selected_candidate': chosen,
            'validation_MAE': scores, 'selected_alpha': ALPHAS[chosen],
            'validation_predictions': predictions[0], 'test_predictions': predictions[1]}


def develop_head(counts_path, mapping, split, biology, *, expected_gene_ids=None):
    """Read train first; failed coverage exits before validation or any fitting."""
    if split.compound_id.duplicated().any() or not split.split.isin(['train', 'validation', 'test']).all():
        raise ValueError('invalid frozen compound split')
    train = split.loc[split.split.eq('train')].sort_values('compound_id')
    train_mapping = project_mapping(mapping, train.compound_id.tolist())
    counts = read_projected_counts(counts_path, train_mapping)
    if expected_gene_ids is not None and counts.index.tolist() != list(expected_gene_ids):
        raise ValueError('count gene identities/order differ from frozen human gene metadata')
    mask, prevalence = training_prevalence(counts, train_mapping)
    coverage, members = freeze_members(mask, biology)
    result = {'status': 'FAILED_COVERAGE_ZERO_FITS', 'fit_count': 0,
              'validation_counts_read': False, 'transfer_A_continues': True,
              'prevalence': prevalence, 'coverage': coverage, 'members': members,
              'train_well_ids': sorted(set(train_mapping.Combo)),
              'train_control_ids': sorted(set(train_mapping.control_Combo))}
    if not coverage.passed.all():
        return result
    set_ids = coverage.set_id.tolist()
    train_ids, train_y = gene_set_labels(counts, train_mapping, members, set_ids)
    del counts
    validation = split.loc[split.split.eq('validation')].sort_values('compound_id')
    val_mapping = project_mapping(mapping, validation.compound_id.tolist())
    val_counts = read_projected_counts(counts_path, val_mapping)
    val_ids, val_y = gene_set_labels(val_counts, val_mapping, members, set_ids)
    test = split.loc[split.split.eq('test')].sort_values('compound_id')
    if test.empty:
        raise ValueError('test query structures are required without test labels')
    by_id = split.set_index('compound_id')
    fingerprints = lambda ids: np.stack([morgan_fingerprint(by_id.loc[i, 'canonical_smiles']) for i in ids])
    fitted = fit_candidates(fingerprints(train_ids), train_y, fingerprints(val_ids), val_y,
                            train_ids, fingerprints(test.compound_id.tolist()))
    result.update(fitted, status='FITTED_AND_PREDICTED_PENDING_SEAL', validation_counts_read=True,
                  train_ids=train_ids, validation_ids=val_ids, test_ids=test.compound_id.tolist(),
                  set_ids=set_ids, n_train=len(train_ids), n_validation=len(val_ids),
                  n_test_queries=len(test), refit_train_plus_validation=False)
    return result


def verify_seal(seal, source_paths, model_paths, public_paths):
    """Check exact source/model/output bytes; paths never come from a receipt."""
    if set(source_paths) != set(seal['source_inputs']):
        raise ValueError('seal source input keys differ')
    for key, path in source_paths.items():
        if file_record(path) != seal['source_inputs'][key]:
            raise ValueError(f'sealed input changed: {key}')
    paths = {**model_paths, **public_paths}
    if set(paths) != set(seal['artifact_hashes']):
        raise ValueError('seal artifact keys differ')
    for key, path in paths.items():
        if file_record(path)['sha256'] != seal['artifact_hashes'][key]:
            raise ValueError(f'sealed artifact changed: {key}')
