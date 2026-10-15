import pandas as pd
import pytest
from drugperturb.design import select_design, match_controls, qualify_wells


def metadata():
    rows=[]
    for rep, plate in [("rep1","plate1"),("rep2","plate2")]:
        for well, treatment, vehicle, dose, pattern in [("A1","drug1",False,1000,"2"),("A2","drug2",False,1000,"2"),("C1","S0000",True,0,"1"),("C2","S0000",True,0,"2")]:
            rows.append(dict(well_oligo=well,plate_oligo=plate,Combo=well+plate,cell_type="A549",time_point=24,replicate=rep,treatment=treatment,vehicle=vehicle,dose=dose,dose_pattern=pattern,SMILES="CCO"))
    return pd.DataFrame(rows)


def test_retains_zero_dose_vehicle_when_selecting_one_micromolar_drugs():
    drugs, controls=select_design(metadata())
    assert len(drugs)==4 and len(controls)==4
    assert controls.dose.eq(0).all()


def test_exact_plate_replicate_and_pattern_control_matching():
    drugs,controls=select_design(metadata())
    mapping=match_controls(drugs,controls)
    assert len(mapping)==4
    assert dict(zip(mapping.Combo,mapping.control_Combo))=={"A1plate1":"C2plate1","A2plate1":"C2plate1","A1plate2":"C2plate2","A2plate2":"C2plate2"}


def test_missing_strict_control_is_not_replaced_by_other_pattern():
    drugs,controls=select_design(metadata())
    controls=controls[controls.Combo.ne("C2plate1")]
    with pytest.raises(ValueError,match="control"):
        match_controls(drugs,controls)


def test_ambiguous_control_is_rejected():
    drugs,controls=select_design(metadata())
    with pytest.raises(ValueError,match="control"):
        match_controls(drugs,pd.concat([controls,controls.iloc[[1]]],ignore_index=True))


def test_duplicate_well_key_rejected_before_join():
    d=metadata()
    with pytest.raises(ValueError,match="Combo"):
        select_design(pd.concat([d,d.iloc[[0]]],ignore_index=True))


def test_inconsistent_barcode_concatenation_rejected():
    d=metadata();d.loc[0,"Combo"]="invented"
    with pytest.raises(ValueError,match="Combo"):
        select_design(d)


def test_one_low_cell_replicate_excludes_whole_treatment():
    drugs,controls=select_design(metadata());mapping=match_controls(drugs,controls)
    counts={x:40 for x in metadata().Combo};counts["A1plate2"]=29
    result=qualify_wells(mapping,counts)
    assert not result.loc[result.treatment.eq("drug1"),"eligible"].any()
    assert result.loc[result.treatment.eq("drug2"),"eligible"].all()
    assert result.treatment_cells.sum()==149


def test_low_shared_control_excludes_both_treatments():
    drugs,controls=select_design(metadata());mapping=match_controls(drugs,controls)
    counts={x:40 for x in metadata().Combo};counts["C2plate1"]=0
    result=qualify_wells(mapping,counts)
    assert not result.eligible.any()
    assert "low_control_cells" in result.exclusion_reason.iloc[0]


def test_negative_or_fractional_cell_counts_are_rejected():
    drugs,controls=select_design(metadata());mapping=match_controls(drugs,controls)
    for invalid in [-1, 2.5]:
        counts={x:40 for x in metadata().Combo};counts["A1plate1"]=invalid
        with pytest.raises(ValueError,match="count"):
            qualify_wells(mapping,counts)
