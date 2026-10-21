"""The post hoc corrected OP3 analysis: plan, labels and kernel wiring agree with the code.

Software checks only; the end-to-end synthetic run of the corrected path is in
tests/test_v4_op3_pipeline.py.
"""
from pathlib import Path
import ast
import importlib.util

import numpy as np
import pandas as pd
import pytest
import yaml

from drugperturb.v3.descriptors import NOISE_STD_TOLERANCE

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    """Import a script by path (kaggle/ and scripts/ both hold a v4_fit.py)."""
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pytest.importorskip('torch')
v4_fit = load('scripts/v4_fit.py', 'v4_fit_script')


def test_plan_matches_the_code_constant_and_keeps_the_v4_protocol():
    plan = yaml.safe_load((ROOT/'configs/v4_corrected_posthoc.yaml').read_text())
    protocol = yaml.safe_load((ROOT/plan['based_on']).read_text())
    assert plan['status'] == 'POST_HOC_PLAN_BEFORE_CORRECTED_FITS'
    assert protocol['status'] == 'PRESPECIFIED_BEFORE_V4_FITTING'
    assert plan['change']['descriptor_min_std'] == NOISE_STD_TOLERANCE == 1e-8
    assert v4_fit.CORRECTED == dict(analysis=plan['label'], descriptor_min_std=NOISE_STD_TOLERANCE)


def test_labels_are_absent_by_default_and_must_match_the_requested_rule():
    labelled = dict(name='fit', **v4_fit.CORRECTED)
    assert v4_fit.descriptor_labels(dict(name='fit')) == {}
    assert v4_fit.descriptor_labels(labelled) == v4_fit.CORRECTED
    v4_fit.require_labels(dict(name='fit'), {}, 'default')
    v4_fit.require_labels(labelled, v4_fit.CORRECTED, 'corrected')
    with pytest.raises(ValueError, match='descriptor rule'):
        v4_fit.require_labels(labelled, {}, 'corrected fit under the default rule')
    with pytest.raises(ValueError, match='descriptor rule'):
        v4_fit.require_labels(dict(name='fit'), v4_fit.CORRECTED, 'default fit under the corrected rule')
    with pytest.raises(ValueError, match='descriptor rule'):
        v4_fit.require_labels(dict(labelled, descriptor_min_std=1e-6), v4_fit.CORRECTED, 'other tolerance')


def calls(path):
    """Script name and arguments of every list literal that names a scripts/*.py file."""
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.List):
            strings = [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            scripts = [s for s in strings if s.startswith('scripts/')]
            if scripts:
                found.append((scripts[0], strings))
    return found


def test_corrected_kernel_passes_the_flag_to_every_fit_and_seal_and_has_its_own_code_dataset():
    corrected = calls(ROOT/'kaggle/v4c_fit_op3.py')
    assert {s for s, _ in corrected} == {'scripts/v4_fit.py', 'scripts/v4_seal.py'}
    assert all('--descriptor-tolerance' in args for _, args in corrected)
    assert not any('--descriptor-tolerance' in args for _, args in calls(ROOT/'kaggle/v4_fit.py'))
    push = load('kaggle/push.py', 'kaggle_push')
    spec = push.KERNELS['v4c-fit-op3']
    assert spec['file'] == 'v4c_fit_op3.py' and spec['sources'] == ['dp-v4-op3-prepare'] and spec['gpu']
    assert spec['code'] == 'drugperturb-v4c-code' in push.CODE_DATASETS
    assert all(s.get('code') != spec['code'] for name, s in push.KERNELS.items() if name != 'v4c-fit-op3')


def synthetic_analysis(shift, selected, label=None):
    """Minimal evaluation outputs of one analysis: every pooled contrast and endpoint equals shift."""
    side = load('scripts/v4c_side_by_side.py', 'v4c_side_by_side')
    estimate = dict(point_estimate=shift, ci_lower=shift-0.1, ci_upper=shift+0.1)
    arms = {a: dict(validation_rmse=0.4+0.01*i, non_collapse=dict(non_collapsed=a != 'I3_nb_plus_pseudobulk'))
            for i, a in enumerate(side.ARMS)}
    selection = dict(selected=selected, arms=arms, **({'analysis': label} if label else {}))
    summary = dict(selection=selection, pooled={f'{a}_minus_{b}': estimate for a in side.ARMS for b in side.COMPARATORS},
                   endpoints=dict(selected_arm=selected, E1_non_collapse=dict(estimate=estimate, supported=False),
                                  E2_improvement=dict(estimates=dict(B1c=estimate, M1c=estimate), supported=False),
                                  E3_beyond_vehicle_anchoring=dict(estimates=dict(B1a=estimate, M1a=estimate), supported=False)),
                   **({'analysis': label} if label else {}))
    methods = [*side.BASELINES, *side.ARMS, 'I1_nb_likelihood_anchored']
    per_compound = pd.DataFrame([dict(compound_id=c, context_id='Tcells_24h_1000nM', method=m,
                                      rmse=(5.0 if c == 'LSM-43293' and m == 'I1_nb_likelihood' and not label else 0.5))
                                 for c in ['LSM-43293', 'LSM-1011', 'LSM-1020', 'LSM-9'] for m in methods])
    per_context = pd.DataFrame([dict(context_id='Tcells_24h_1000nM', gate_met=True, method=m, mean_compound_rmse=0.3+shift)
                                for m in methods])
    return side, summary, selection, per_compound, per_context


def test_side_by_side_pairs_each_analysis_with_its_own_selected_arm():
    side, v4, v4_sel, v4_pc, v4_ctx = synthetic_analysis(0.17, 'I1_nb_likelihood')
    _, corr, corr_sel, corr_pc, corr_ctx = synthetic_analysis(0.01, 'I2_counterfactual_pseudobulk', 'v4_corrected_posthoc')
    selection, endpoints, pooled, context, compounds = side.tables(
        dict(v4=v4, corrected=corr), dict(v4=v4_sel, corrected=corr_sel), dict(v4=v4_pc, corrected=corr_pc),
        dict(v4=v4_ctx, corrected=corr_ctx))
    assert selection.set_index('arm').loc['I1_nb_likelihood', 'v4_selected']
    assert selection.set_index('arm').loc['I2_counterfactual_pseudobulk', 'corrected_selected']
    assert len(endpoints) == 5 and set(endpoints.v4_arm) == {'I1_nb_likelihood'}
    assert set(endpoints.corrected_arm) == {'I2_counterfactual_pseudobulk'}
    assert np.allclose(endpoints.v4_point_estimate, 0.17) and np.allclose(endpoints.corrected_ci_upper, 0.11)
    assert len(pooled) == 4*5 and 'I1_nb_likelihood_anchored' not in set(context.method)
    assert set(compounds.compound_id) == set(side.AFFECTED) and set(compounds.method) == set(side.ARMS)
    ixa = compounds.set_index(['compound_id', 'method']).loc[('LSM-43293', 'I1_nb_likelihood')]
    assert ixa.v4 == 5.0 and ixa.corrected == 0.5 and ixa['name'] == 'Ixabepilone'


def test_replay_comparison_reports_bytes_or_the_largest_numeric_difference(tmp_path):
    local = load('scripts/v4c_evaluate_local.py', 'v4c_evaluate_local')
    a, b = tmp_path/'replay', tmp_path/'committed'
    for root, value in [(a, 0.1), (b, 0.1 + 1e-12)]:
        root.mkdir()
        for name in local.TABLES:
            pd.DataFrame(dict(method=['B0'], rmse=[value if name == 'summary.tsv' else 0.1])).to_csv(
                root/name, sep='\t', index=False)
        (root/'summary.json').write_text('{"x": 1}')
    result = local.compare(a, b)
    assert result['per_compound.tsv'] == dict(bytes_identical=True) and result['summary.json']['equal_as_data']
    differing = result['summary.tsv']
    assert not differing['bytes_identical'] and differing['text_columns_equal']
    assert 0 < differing['max_abs_numeric_difference'] < 1e-11
