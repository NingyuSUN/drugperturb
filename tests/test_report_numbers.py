"""Bind every interval the version reports print to a sealed receipt record.

Each report quotes estimates that were computed elsewhere and committed as JSON or TSV. These checks
re-read those files and require that every `estimate [lower, upper]` triple a report prints matches a
committed record at the precision it is printed, so prose and receipts cannot drift apart. Two figures in
the v4 report have no committed table of their own; their point estimates are recomputed here from the
committed per-compound errors instead, and the exception list is pinned so a new untraceable number fails.

These checks establish bookkeeping, not biology: they say a number was scored, never that it is meaningful.
"""
import csv
import functools
import json
import math
import re
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRIPLE = re.compile(r'([+−-]?\d+\.\d+)\s*\[\s*([+−-]?\d+\.\d+)\s*,\s*([+−-]?\d+\.\d+)\s*\]')
REPORTS = ('reports/v1/report.md', 'reports/v2/report.md', 'reports/v2/report.zh-CN.md',
           'reports/v3/report.md', 'reports/v4/report.md', 'reports/v4_corrected/report.md',
           'reports/v5/report.md', 'reports/v6/report.md', 'reports/synthesis.md',
           'reports/synthesis.zh-CN.md', 'docs/MODEL_CARD_V3_V6.md')
# docs/MODEL_CARD.md is deliberately absent: it separates each estimate from its interval ("**−0.0286**,
# paired 95% structure-group interval **[−0.0680,0.0087]**"), which the triple pattern cannot see.
# reports/v1/report.md carries the same figures in a form this module and tests/test_v1_report.py bind.

# The v4 report states these two OP3 baseline contrasts in prose as descriptive post hoc comparisons, and
# no committed table carries them. test_the_untraced_v4_figures_are_recomputed checks their point estimates
# against reports/v4/op3/per_compound.tsv; the intervals cannot be checked, because reproducing the cluster
# bootstrap needs the gene-level matrices, which are deliberately not in Git.
UNTRACED = {
    ('reports/v4/report.md', '-0.013 [-0.023, -0.003]'),   # B1c - B0, pooled over the three OP3 contexts
    ('reports/v4/report.md', '0.075 [0.064, 0.086]'),      # B1a - B1c, stated as a reduction in error
}


def normalize(token):
    return token.replace('−', '-')


def decimals(token):
    return len(token.split('.')[1])


def _json_records(node, path, out):
    if isinstance(node, dict):
        if {'point_estimate', 'ci_lower', 'ci_upper'} <= set(node):
            try:
                out.append((path, float(node['point_estimate']), float(node['ci_lower']),
                            float(node['ci_upper'])))
            except (TypeError, ValueError):
                pass
        for key, value in node.items():
            _json_records(value, f'{path}.{key}', out)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _json_records(value, f'{path}[{index}]', out)


def _tsv_records(path, out):
    with path.open(encoding='utf-8') as handle:
        for number, row in enumerate(csv.DictReader(handle, delimiter='\t')):
            point = [k for k in row if k and 'point' in k.lower()]
            lower = [k for k in row if k and 'lower' in k.lower()]
            upper = [k for k in row if k and 'upper' in k.lower()]
            if not (point and lower and upper):
                return
            try:
                out.append((f'row {number}', float(row[point[0]]), float(row[lower[0]]),
                            float(row[upper[0]])))
            except (TypeError, ValueError):
                pass


@functools.cache
def receipts():
    """Every (point, lower, upper) record committed under reports/ and receipts/."""
    records = []
    for base in ('reports', 'receipts'):
        for path in sorted((ROOT / base).rglob('*')):
            if path.is_dir() or path.suffix not in ('.json', '.tsv'):
                continue
            found = []
            try:
                if path.suffix == '.json':
                    _json_records(json.loads(path.read_text(encoding='utf-8')), '', found)
                else:
                    _tsv_records(path, found)
            except (UnicodeDecodeError, json.JSONDecodeError, csv.Error):
                continue
            records.extend((str(path.relative_to(ROOT)), label, *values) for label, *values in found)
    assert len(records) > 1500, 'the receipt pool collapsed; check the reports and receipts trees'
    return records


def quoted(relative):
    """The triples a report prints, with line wrapping removed."""
    text = ' '.join((ROOT / relative).read_text(encoding='utf-8').split())
    for match in TRIPLE.finditer(text):
        printed = [normalize(group) for group in match.groups()]
        yield printed, '{} [{}, {}]'.format(*printed)


def agrees(printed, value, *, bound=None):
    """True if `value` prints as `printed`, rounded to nearest or - for a bound - outward."""
    target, places = float(printed), decimals(printed)
    if round(value, places) == target:
        return True
    step = 10 ** -places
    if bound == 'lower':
        return math.isclose(target, math.floor(value / step) * step, abs_tol=step / 2)
    if bound == 'upper':
        return math.isclose(target, math.ceil(value / step) * step, abs_tol=step / 2)
    return False


def traced(printed):
    for record in receipts():
        point, lower, upper = record[2:]
        if (agrees(printed[0], point) and agrees(printed[1], lower, bound='lower')
                and agrees(printed[2], upper, bound='upper')):
            return record
    return None


@pytest.mark.parametrize('relative', REPORTS)
def test_every_interval_a_report_prints_traces_to_a_receipt(relative):
    untraced = [shown for printed, shown in quoted(relative)
                if (relative, shown) not in UNTRACED and traced(printed) is None]
    assert not untraced, f'{relative} prints intervals no committed record carries: {untraced}'


def test_the_reports_print_the_intervals_at_all():
    counts = {relative: sum(1 for _ in quoted(relative)) for relative in REPORTS}
    assert all(counts[r] for r in REPORTS if r != 'reports/v2/report.zh-CN.md'), counts
    assert counts['docs/MODEL_CARD_V3_V6.md'] >= 15, 'the v3-v6 model card must quote its results'
    assert sum(counts.values()) > 150, counts


def test_the_exception_list_stays_pinned():
    """A report may not quietly add another figure that no table carries."""
    found = {(relative, shown) for relative in REPORTS for _, shown in quoted(relative)
             if traced(_normalize_printed(shown)) is None}
    assert found == UNTRACED, f'untraceable intervals changed: {found ^ UNTRACED}'


def _normalize_printed(shown):
    return [normalize(token) for token in TRIPLE.match(shown).groups()]


def test_the_untraced_v4_figures_are_recomputed():
    """The two prose contrasts: recompute their pooled point estimates from the committed per-compound errors."""
    summary = json.loads((ROOT / 'reports/v4/op3/summary.json').read_text(encoding='utf-8'))
    contexts = summary['endpoints']['gate_passing_contexts']
    with (ROOT / 'reports/v4/op3/per_compound.tsv').open(encoding='utf-8') as handle:
        errors = {}
        for row in csv.DictReader(handle, delimiter='\t'):
            errors.setdefault((row['context_id'], row['method']), {})[row['compound_id']] = \
                float(row['rmse'])

    def pooled(candidate, baseline):
        per_context = []
        for context in contexts:
            left, right = errors[(context, candidate)], errors[(context, baseline)]
            assert set(left) == set(right), (context, candidate, baseline)
            per_context.append(statistics.fmean(left[c] - right[c] for c in sorted(left)))
        return statistics.fmean(per_context)

    # A sanity anchor: the same recomputation reproduces two contrasts that do have committed records.
    for candidate, baseline, key in (('I1_nb_likelihood', 'B0', 'I1_nb_likelihood_minus_B0'),
                                     ('I2_counterfactual_pseudobulk', 'B0',
                                      'I2_counterfactual_pseudobulk_minus_B0')):
        assert math.isclose(pooled(candidate, baseline), summary['pooled'][key]['point_estimate'],
                            rel_tol=1e-9, abs_tol=1e-12), key

    report = ' '.join((ROOT / 'reports/v4/report.md').read_text(encoding='utf-8').split())
    assert agrees('-0.013', pooled('B1c', 'B0')), 'B1c - B0 no longer prints as -0.013'
    assert '−0.013 [−0.023, −0.003]' in report
    reduction = pooled('B1a', 'B1c')
    assert reduction < 0, 'the vehicle-anchored level must lower error for the prose to read as it does'
    assert agrees('0.075', abs(reduction)), 'B1a - B1c no longer prints as 0.075'
    assert 'lowers error by a further 0.075 [0.064, 0.086]' in report


def test_the_v3_q1_table_matches_its_contrast_file():
    """The row that was misprinted as +0.049 stays bound to the sealed 0.0484731."""
    with (ROOT / 'reports/v3/contrasts.tsv').open(encoding='utf-8') as handle:
        rows = {(row['candidate'], row['baseline']): row for row in csv.DictReader(handle, delimiter='\t')
                if row['exposure'] == 'opened'}
    report = ' '.join((ROOT / 'reports/v3/report.md').read_text(encoding='utf-8').split())
    for (candidate, baseline), label in ((('biolord_Q1', 'B1'), 'biolord − training mean'),
                                         (('chemcpa_Q1', 'B1'), 'chemCPA − training mean'),
                                         (('biolord_Q1', 'M1'), 'biolord − Ridge'),
                                         (('chemcpa_Q1', 'M1'), 'chemCPA − Ridge')):
        row = rows[(candidate, baseline)]
        point = float(row['point_estimate'])
        cell = f'| {label} | +{point:.3f} ['.replace('+-', '−')
        assert cell in report, f'{label} must print {point:+.3f}'
