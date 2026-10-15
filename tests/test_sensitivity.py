"""Small synthetic checks of reference construction and metadata-only qualification."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('sensitivity', Path(__file__).resolve().parents[1] / 'scripts/sensitivity.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    wells, mapping, qc = [], [], []
    for replicate, plate in [('rep1', 'p1'), ('rep2', 'p2')]:
        for name, vehicle, dose, pattern in [('t', False, 1000, '2'), ('c1', True, 0, '2'), ('c2', True, 0, '1')]:
            combo = replicate + name
            wells.append(dict(Combo=combo, vehicle=vehicle, dose=dose, dose_pattern=pattern, cell_type='A549', time_point=24, replicate=replicate, plate_oligo=plate, treatment='drug' if not vehicle else 'S0000'))
            qc.append(dict(Combo=combo, n_cells=40))
        mapping.append(dict(treatment='drug', Combo=replicate+'t', control_Combo=replicate+'c1', dose_pattern='2', cell_type='A549', time_point=24, replicate=replicate, plate_oligo=plate, eligible=True, treatment_cells=40, control_cells=40))
    return pd.DataFrame(wells), pd.DataFrame(mapping), pd.DataFrame(qc).set_index('Combo')


def test_controls_are_normalized_separately_before_average_and_use_full_library():
    wells, mapping, qc = fixture()
    qualification = module.qualify_alternate_controls(wells, mapping, qc, ['drug'])
    counts = pd.DataFrame({r+s:v for r in ['rep1','rep2'] for s,v in [('t',[2,8]),('c1',[90,10]),('c2',[1,9])]}, index=['g1','g2'])
    normalized = module.normalize_allgene_wells(counts)
    primary, alternate = module.responses_for_qualified(normalized, qualification, ['drug'], ['g1'])
    expected = np.log1p(200000) - (np.log1p(900000) + np.log1p(100000))/2
    assert alternate[0,0] == pytest.approx(expected)
    assert primary[0,0] == pytest.approx(np.log1p(200000)-np.log1p(900000))
    pooled_wrong = np.log1p(200000) - np.log1p(91/110*1e6)
    assert not np.isclose(expected, pooled_wrong)
    # Normalizing selected g1 alone would make every library one million.
    assert alternate[0,0] != pytest.approx(0)


@pytest.mark.parametrize('defect', ['low_cells','false_vehicle','nonzero_dose','same_pattern','extra_vehicle'])
def test_invalid_alternate_control_excludes_entire_compound(defect):
    wells, mapping, qc = fixture()
    affected = wells.Combo.eq('rep2c2')
    if defect == 'low_cells': qc.loc['rep2c2','n_cells'] = 29
    if defect == 'false_vehicle': wells.loc[affected,'vehicle'] = False
    if defect == 'nonzero_dose': wells.loc[affected,'dose'] = 10
    if defect == 'same_pattern': wells.loc[affected,'dose_pattern'] = '2'
    if defect == 'extra_vehicle':
        extra = wells.loc[affected].copy(); extra['Combo']='rep2c3'
        wells = pd.concat([wells, extra], ignore_index=True); qc.loc['rep2c3','n_cells']=40
    result = module.qualify_alternate_controls(wells, mapping, qc, ['drug'])
    assert not result.compound_eligible.any()
    assert result.eligible.tolist() == [True, False]
    assert result.exclusion_reason.iloc[1]


def test_preprimary_gate_prevents_any_expression_read(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(np, 'load', lambda *a, **kw: pytest.fail('Outcome load before both primary receipts'))
    with pytest.raises(FileNotFoundError, match='Both prediction_seal'):
        module.main()
