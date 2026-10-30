"""Bind reports/v1/report.md to the sealed v1 result files.

The markdown report restates numbers that live in reports/final_test, reports/sensitivity and
receipts/development_diagnostics.tsv. These checks fail if the prose and the receipts drift apart; they
establish nothing about biology, only that the report quotes what was actually scored.
"""
import csv
import json
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports' / 'v1' / 'report.md'
FINAL_TEST = ROOT / 'reports' / 'final_test'
METHODS = ('B0', 'B1', 'B2', 'M1', 'M2', 'M3')
MINUS = '−'


def text():
    return REPORT.read_text(encoding='utf-8')


def prose():
    """The report with its hard line wrapping and emphasis removed, for sentence checks."""
    return ' '.join(text().replace('**', '').split())


def plain(value):
    return (MINUS if value < 0 else '') + f'{abs(value):.4f}'


def signed(value):
    return (MINUS if value < 0 else '+') + f'{abs(value):.4f}'


def rows(name):
    with (FINAL_TEST / name).open(encoding='utf-8') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def contrasts():
    return json.loads((FINAL_TEST / 'paired_comparisons.json').read_text(encoding='utf-8'))


def table_rows(heading):
    """The '| ... |' lines of the section introduced by `heading`."""
    section = text().split(heading, 1)[1].split('\n## ', 1)[0]
    return [line for line in section.splitlines() if line.startswith('|') and '---' not in line]


@pytest.mark.parametrize('method', METHODS)
def test_each_method_row_quotes_the_sealed_metrics(method):
    metrics = json.loads((FINAL_TEST / 'summary.json').read_text(encoding='utf-8'))['metrics'][method]
    row = next(line.replace('**', '') for line in table_rows('## Every method')
               if f'| {method},' in line.replace('**', ''))
    for key in ('mean_compound_rmse', 'mean_compound_mae', 'mean_spearman', 'mean_cosine_zero',
                'mean_sign_agreement'):
        value = metrics[key]
        if value is None:
            assert 'NA' in row, f'{method}.{key} is undefined and the row must say NA'
        else:
            assert f'{value:.4f}' in row, f'{method}.{key} missing from its row'


def test_every_paired_contrast_appears_with_its_interval():
    quoted = {}
    for line in table_rows('## Confirmatory contrasts'):
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        if cells[0] in ('Contrast', ''):
            continue
        pair = cells[0].replace(' (primary)', '').replace('**', '')
        candidate, baseline = (part.strip() for part in pair.split(f' {MINUS} '))
        quoted[(candidate, baseline)] = (cells[1], cells[2])
    expected = {(r['method'], r['baseline']) for r in contrasts()}
    assert set(quoted) == expected, 'the table must hold every prespecified contrast and no others'
    for record in contrasts():
        estimate, verdict = quoted[(record['method'], record['baseline'])]
        assert estimate == (f"{signed(record['point_estimate'])} "
                            f"[{signed(record['ci_lower'])}, {signed(record['ci_upper'])}]")
        reliable = record['ci_lower'] > 0 or record['ci_upper'] < 0
        assert verdict.startswith('yes' if reliable else 'no'), (record['method'], record['baseline'])
        if reliable:
            better = record['method'] if record['point_estimate'] < 0 else record['baseline']
            assert better in verdict, 'a reliable contrast must name the method it favours'


def test_the_primary_contrast_is_not_reported_as_reliable():
    primary = next(r for r in contrasts() if r['method'] == 'M3' and r['baseline'] == 'M1')
    assert primary['ci_lower'] < 0 < primary['ci_upper'], 'the sealed primary interval includes zero'
    assert 'a reliable advantage over the strong simple baselines was not established' in prose()


@pytest.mark.parametrize('scenario', ('primary_all_test', 'alternate_two_vehicle_mean',
                                      'primary_treatment_both_replicates_ge50'))
def test_the_control_sensitivity_table_matches_the_frozen_scenarios(scenario):
    with (ROOT / 'reports' / 'sensitivity' / 'metrics_summary.tsv').open(encoding='utf-8') as handle:
        scored = {row['method']: float(row['mean_compound_rmse'])
                  for row in csv.DictReader(handle, delimiter='\t') if row['scenario'] == scenario}
    report = text()
    for method in METHODS:
        assert f'{scored[method]:.4f}' in report, f'{scenario}/{method} missing'
    lowest = min(scored, key=scored.get)
    assert f'**{scored[lowest]:.4f}**' in report, f'{scenario} must mark {lowest} as its lowest error'


def test_the_replicate_noise_claim_is_recomputed():
    context = rows('compound_context.tsv')
    noisier = [r for r in context if float(r['replicate_rmse']) > float(r['response_rms'])]
    assert len(noisier) == len(context) == 26
    assert f'{len(noisier)} of {len(context)}' in prose()
    for column in ('response_rms', 'replicate_rmse', 'replicate_spearman'):
        values = [float(r[column]) for r in context]
        assert f'{statistics.median(values):.4f}' in prose(), f'median {column}'


def test_no_biological_set_passes_the_coverage_gate():
    report = text()
    for row in rows('pathway_coverage.tsv'):
        assert row['quantitatively_evaluable'] == 'False'
        assert f"| {row['original_members']} | {row['selected_genes']} |" in report
        assert f"{float(row['fraction_original']):.3f}" in report
    assert 'no pathway-recovery claim is made' in prose()


def test_the_development_diagnostics_stay_exploratory():
    with (ROOT / 'receipts' / 'development_diagnostics.tsv').open(encoding='utf-8') as handle:
        folds = [row for row in csv.DictReader(handle, delimiter='\t')
                 if row['diagnostic'] == 'compound_cv' and row['method'] in ('M1', 'M3')]
    report = text()
    for row in folds:
        assert row['exploratory_single_seed'] == 'True'
        assert f"{float(row['mean_compound_RMSE']):.4f}" in report
    ridge = {r['run']: float(r['mean_compound_RMSE']) for r in folds if r['method'] == 'M1'}
    graph = {r['run']: float(r['mean_compound_RMSE']) for r in folds if r['method'] == 'M3'}
    assert all(graph[run] > ridge[run] for run in ridge), 'the GNN is worse in every development fold'
    assert 'all three compound cross-validation folds' in prose()
