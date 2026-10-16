"""Render verified, sealed v2 tables without recomputing scientific endpoints."""
from pathlib import Path
import argparse
import datetime
import json
import os
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from drugperturb.biology_v2 import file_record

ROOT = Path(__file__).resolve().parents[1]
CONTEXTS = ['MCF7_24h_1000nM', 'K562_24h_1000nM', 'MCF7_24h_10nM',
            'MCF7_24h_100nM', 'MCF7_24h_10000nM']
PRIMARY = CONTEXTS[0]
SETS = ['author_s_phase', 'author_g2m_phase', 'R-HSA-3214815', 'R-HSA-3214847', 'R-HSA-2262752']
SET_LABELS = ['S phase', 'G2/M', 'Histone\ndeacetylation',
              'Histone\nacetylation', 'Cellular\nstress']
TABLES = ['context_status.tsv', 'per_compound.tsv', 'per_set.tsv', 'contrasts.tsv',
          'sensitivity_pairs.tsv', 'summary.json']
FIGURES = ['contrast_intervals.png', 'biology_set_errors.png', 'control_sensitivity.png']
METHODS = {'A': ['B0', 'B1', 'B2', 'M1', 'M2', 'M3'], 'B': ['B0', 'B1', 'B2', 'Ridge']}
COLORS = {'B0': '#7a8491', 'B1': '#b07a3d', 'B2': '#238b8d',
          'M1': '#6a73ad', 'M2': '#a45b8c', 'M3': '#2463a3', 'Ridge': '#2463a3'}
METHOD_LABELS = {'B0': 'Zero change', 'B1': 'Source mean', 'B2': 'Chemical neighbor',
                 'M1': 'Fingerprint Ridge', 'M2': 'Fingerprint MLP',
                 'M3': 'Molecular GNN', 'Ridge': 'Five-set Ridge'}


def context_label(context):
    cell, hours, dose = context.split('_')
    label = f'{cell} · {hours} · {int(dose[:-2]):,} nM'
    return label + ('  [PRIMARY]' if context == PRIMARY else '')


def load_verified(root):
    """Hash every sealed output, including files not needed for a particular plot."""
    receipt_path = root/'receipts/v2_transfer_evaluation.json'
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get('status') != 'SEALED_TRANSFER_EVALUATION'
            or receipt.get('new_models_fitted') != 0
            or receipt.get('test_outcomes_used_for_selection') is not False):
        raise ValueError('a sealed, selection-free transfer evaluation is required')
    paths = {'reports/v2/transfer/'+name: root/'reports/v2/transfer'/name for name in TABLES}
    if set(receipt.get('output_hashes', {})) != set(paths):
        raise ValueError('evaluation receipt must bind exactly the six transfer outputs')
    records = {'evaluation_receipt': file_record(receipt_path)}
    for key, path in paths.items():
        record = file_record(path)
        if record['sha256'] != receipt['output_hashes'][key]:
            raise ValueError(f'sealed transfer output hash mismatch: {path.name}')
        records[key] = record
    data = {}
    for name in TABLES:
        if name.endswith('.json'):
            data[name] = json.loads(paths['reports/v2/transfer/'+name].read_text())
        else:
            try:
                data[name] = pd.read_csv(paths['reports/v2/transfer/'+name], sep='\t')
            except pd.errors.EmptyDataError:
                data[name] = pd.DataFrame()
    status = data['context_status.tsv']
    if (status.context_id.duplicated().any() or set(status.context_id) != set(CONTEXTS)
            or data['summary.json'].get('primary_context') != PRIMARY):
        raise ValueError('sealed context identities differ from the fixed five contexts')
    expected_roles = np.where(status.context_id.eq(PRIMARY), 'primary_test', 'secondary_test')
    if not np.array_equal(status.context_role.to_numpy(), expected_roles):
        raise ValueError('sealed primary/secondary context roles differ')
    for name in ['per_compound.tsv', 'per_set.tsv', 'contrasts.tsv']:
        frame = data[name]
        if not frame.empty and not set(frame.context_id).issubset(CONTEXTS):
            raise ValueError('undeclared context in scoring output')
    return data, records, paths, receipt_path


def prettify(ax, *, axis='x'):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis=axis, color='#e5e8ed', linewidth=.7, zorder=0)
    ax.set_axisbelow(True)


def draw_contrasts(data, path):
    table = data['contrasts.tsv']
    specifications = [('A', 'M3', 'M1'), ('A', 'M3', 'B1'), ('B', 'Ridge', 'B1')]
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.7), sharey=True)
    for ax, (analysis, candidate, baseline) in zip(axes, specifications):
        rows = table if table.empty else table[
            table.analysis.eq(analysis) & table.comparison.eq('strict_full')
            & table.candidate.eq(candidate) & table.baseline.eq(baseline)]
        if not rows.empty and rows.context_id.duplicated().any():
            raise ValueError('duplicate contrast/context rows')
        indexed = rows.set_index('context_id') if not rows.empty else rows
        for y, context in enumerate(CONTEXTS):
            if rows.empty or context not in indexed.index:
                ax.text(.98, y, 'Not evaluated', transform=ax.get_yaxis_transform(),
                        ha='right', va='center', color='#697382', fontsize=8)
                continue
            row = indexed.loc[context]; x = float(row.point_estimate)
            if not np.isfinite(x):
                raise ValueError('contrast point estimates must be finite')
            lo, hi = row.ci_lower, row.ci_upper
            if pd.notna(lo) != pd.notna(hi):
                raise ValueError('partially missing confidence interval')
            has_interval = pd.notna(lo) and pd.notna(hi)
            if has_interval:
                if not np.isfinite([lo, hi]).all() or lo > hi:
                    raise ValueError('invalid contrast interval')
                ax.plot([lo, hi], [y, y], color=COLORS[candidate], linewidth=2, zorder=2)
                ax.plot([lo, hi], [y, y], '|', color=COLORS[candidate], markersize=7)
            ax.scatter(x, y, s=45, facecolors=COLORS[candidate] if has_interval else 'white',
                       edgecolors=COLORS[candidate], linewidths=1.5, zorder=3)
            annotation = f'n={int(row.n_compounds)}, g={int(row.n_groups)}'
            if not has_interval:
                annotation += ' · no CI'
            ax.text(.98, y+.27, annotation, transform=ax.get_yaxis_transform(),
                    ha='right', va='center', fontsize=8, color='#586474')
        ax.axvline(0, color='#697382', linewidth=1, linestyle='--')
        ax.axhspan(-.45, .45, color='#edf3fa', zorder=-1)
        metric = 'gene RMSE' if analysis == 'A' else 'five-set MAE'
        ax.set_title(f'{analysis} · {METHOD_LABELS[candidate]}\n− {METHOD_LABELS[baseline]}',
                     loc='left', fontweight='bold')
        ax.set_xlabel(f'Difference in mean per-compound {metric}\n(log1p CPM delta units)')
        ax.set_yticks(range(len(CONTEXTS)), [context_label(c) for c in CONTEXTS])
        ax.set_ylim(len(CONTEXTS)-.55, -.55); prettify(ax)
    fig.suptitle('Fixed prediction contrasts across experimental conditions', x=.02,
                 ha='left', fontsize=15, fontweight='bold')
    fig.text(.02, .015, 'Negative favors candidate. Bars: pointwise 95% paired scaffold-group bootstrap intervals; '
             'open points: count gate failed, interval unavailable.\nn = compounds; g = structure groups. '
             'All panels use strict matched controls; shared controls and two replicates limit inference.',
             fontsize=8.5, color='#586474')
    fig.tight_layout(rect=(0, .115, 1, .94)); fig.savefig(path, dpi=200); plt.close(fig)


def draw_sets(data, path):
    table = data['per_set.tsv']; values = np.full((5, 5), np.nan); counts = np.zeros((5, 5), int)
    if not table.empty:
        table = table[table.analysis.eq('B') & table.comparison.eq('strict_full')
                      & table.method.isin(['Ridge', 'B1'])]
        if not set(table.set_id).issubset(SETS):
            raise ValueError('unexpected biology set label')
        keys = ['context_id', 'compound_id', 'scaffold_group', 'set_id']
        if table.duplicated([*keys, 'method']).any():
            raise ValueError('duplicate per-set prediction rows')
        ridge = table[table.method.eq('Ridge')][keys+['absolute_error']]
        baseline = table[table.method.eq('B1')][keys+['absolute_error']]
        paired = ridge.merge(baseline, on=keys, how='outer', suffixes=('_ridge', '_mean'),
                             validate='one_to_one', indicator=True)
        if not paired._merge.eq('both').all():
            raise ValueError('per-set Ridge/B1 compound identities differ')
        if not np.isfinite(paired[['absolute_error_ridge', 'absolute_error_mean']]).all().all():
            raise ValueError('per-set absolute errors must be finite')
        paired['difference'] = paired.absolute_error_ridge-paired.absolute_error_mean
        for i, context in enumerate(CONTEXTS):
            previous = None
            for j, sid in enumerate(SETS):
                rows = paired[paired.context_id.eq(context) & paired.set_id.eq(sid)]
                ids = set(rows.compound_id)
                if previous is not None and ids != previous:
                    raise ValueError('five fixed sets must use the same compound cohort')
                previous = ids
                if not rows.empty:
                    values[i, j] = rows.difference.mean(); counts[i, j] = len(rows)
    maximum = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.
    maximum = maximum if maximum > 0 else 1e-12
    cmap = plt.get_cmap('RdBu_r').copy(); cmap.set_bad('#edf0f3')
    fig, ax = plt.subplots(figsize=(11.7, 5.7))
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap,
        norm=TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum), aspect='auto')
    for i in range(5):
        for j in range(5):
            finite = np.isfinite(values[i, j])
            label = f'{values[i, j]:+.3f}\nn={counts[i, j]}' if finite else 'NA\nn=0'
            ax.text(j, i, label, ha='center', va='center', fontsize=9,
                    color='white' if finite and abs(values[i, j]) > .65*maximum else '#233346')
    ax.set_xticks(range(5), SET_LABELS); ax.set_yticks(range(5), [context_label(c) for c in CONTEXTS])
    ax.tick_params(length=0); ax.spines[:].set_visible(False)
    fig.colorbar(image, ax=ax, shrink=.82, label='Mean paired absolute-error difference\n(log1p CPM delta units)')
    ax.set_title('B · Ridge − source-training mean (B1), per fixed gene set',
                 loc='left', fontsize=14, fontweight='bold', pad=15)
    fig.text(.02, .02, 'Negative (blue) favors Ridge; positive (red) favors B1. Equal weight per compound; '
             'strict matched controls.\nFive fixed transcriptional readouts; these values do not establish pathway activation. '
             'n = compounds per set; NA is not zero.', fontsize=8.5, color='#586474')
    fig.tight_layout(rect=(0, .12, 1, 1)); fig.savefig(path, dpi=200); plt.close(fig)


def draw_sensitivity(data, path):
    table = data['per_compound.tsv']
    fig, axes = plt.subplots(2, 5, figsize=(18, 8.1), squeeze=False, sharey='row')
    for i, analysis in enumerate(['A', 'B']):
        metric = 'gene RMSE' if analysis == 'A' else 'five-set MAE'
        for j, context in enumerate(CONTEXTS):
            ax = axes[i, j]; methods = METHODS[analysis]
            rows = table if table.empty else table[table.context_id.eq(context) & table.analysis.eq(analysis)
                & table.comparison.isin(['strict_sensitivity_subset', 'alternate_sensitivity_subset'])]
            n, groups, cohort_ids = 0, 0, None
            for k, method in enumerate(methods):
                current = rows if rows.empty else rows[rows.method.eq(method)]
                if current.empty:
                    continue
                strict = current[current.comparison.eq('strict_sensitivity_subset')].set_index('compound_id')
                alternate = current[current.comparison.eq('alternate_sensitivity_subset')].set_index('compound_id')
                if (strict.index.has_duplicates or alternate.index.has_duplicates
                        or set(strict.index) != set(alternate.index) or strict.empty):
                    raise ValueError('sensitivity plots require identical paired compound subsets')
                alternate = alternate.loc[strict.index]
                if not strict.scaffold_group.equals(alternate.scaffold_group):
                    raise ValueError('sensitivity scaffold identities disagree')
                if (current.error_metric.ne('rmse' if analysis == 'A' else 'mae').any()
                        or not np.isfinite(current.error).all()):
                    raise ValueError('sensitivity error units/values disagree')
                if cohort_ids is not None and set(strict.index) != cohort_ids:
                    raise ValueError('sensitivity methods must share the same compound subset')
                cohort_ids = set(strict.index)
                n, groups = len(strict), strict.scaffold_group.nunique()
                ax.plot([0, 1], [strict.error.mean(), alternate.error.mean()],
                        marker='o', linewidth=1.6, markersize=4, color=COLORS[method], label=method)
            if not n:
                ax.text(.5, .5, 'No eligible paired subset', ha='center', va='center',
                        transform=ax.transAxes, fontsize=8, color='#697382')
            ax.set_title(context_label(context).replace(' · ', '\n', 1)+f'\nn={n}, g={groups}', fontsize=9)
            if context == PRIMARY:
                ax.set_facecolor('#f2f6fb')
            ax.set_xticks([0, 1], ['Strict', 'Two vehicles']); ax.set_xlim(-.15, 1.15)
            ax.set_ylim(bottom=0); prettify(ax, axis='y')
            if j == 0:
                ax.set_ylabel(f'{analysis} · Mean per-compound {metric}\n(log1p CPM delta units)')
    legend = [Line2D([0], [0], color=COLORS[m], marker='o', markersize=4, label=METHOD_LABELS[m])
              for m in ['B0', 'B1', 'B2', 'M1', 'M2', 'M3', 'Ridge']]
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, .055), ncol=7, frameon=False)
    fig.suptitle('Control sensitivity on the same eligible compounds', x=.02, ha='left',
                 fontsize=15, fontweight='bold')
    fig.text(.02, .015, 'Every line compares the strict control with exactly two matched vehicle wells '
             'on the identical subset; full strict-cohort errors are excluded.\nn = paired compounds; '
             'g = structure groups. Lines are descriptive changes in mean error, not confidence intervals.',
             fontsize=8.5, color='#586474')
    fig.tight_layout(rect=(0, .13, 1, .94)); fig.savefig(path, dpi=200); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    data, records, paths, evaluation_path = load_verified(args.result_root)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlepad': 10, 'axes.labelcolor': '#233346', 'text.color': '#233346',
        'xtick.color': '#455467', 'ytick.color': '#455467', 'figure.facecolor': 'white',
        'savefig.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False})
    code_record = file_record(Path(__file__).resolve())
    args.output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.v2-transfer-figures-', dir=args.output_root) as temporary:
        stage = Path(temporary)
        draw_contrasts(data, stage/FIGURES[0]); draw_sets(data, stage/FIGURES[1])
        draw_sensitivity(data, stage/FIGURES[2])
        current = {'evaluation_receipt': file_record(evaluation_path),
                   **{key: file_record(path) for key, path in paths.items()}}
        if records != current or code_record != file_record(Path(__file__).resolve()):
            raise ValueError('sealed rendering input/code changed while plotting')
        receipt = {'schema_version': 1, 'status': 'VERIFIED_TRANSFER_FIGURES_RENDERED',
            'rendered_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'input_files': records, 'code': code_record, 'primary_context': PRIMARY,
            'scientific_endpoints_recomputed': False,
            'render_aggregations': 'Equal-compound means of sealed per-set errors and paired-subset errors only.',
            'output_files': {'reports/v2/figures/'+name: file_record(stage/name) for name in FIGURES},
            'runtime': {'matplotlib': matplotlib.__version__, 'numpy': np.__version__, 'pandas': pd.__version__}}
        destination = args.output_root/'reports/v2/figures'; destination.mkdir(parents=True, exist_ok=True)
        for name in FIGURES:
            os.replace(stage/name, destination/name)
        receipt_path = args.output_root/'receipts/v2_transfer_rendering.json'
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        (stage/'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)+'\n')
        os.replace(stage/'receipt.json', receipt_path)
    print(json.dumps({'status': receipt['status'], 'figures': FIGURES}))
    return receipt


if __name__ == '__main__':
    main()
