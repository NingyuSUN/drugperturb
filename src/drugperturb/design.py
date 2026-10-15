"""Well-level design qualification; never match controls across plates or patterns."""
from __future__ import annotations
import numpy as np
import pandas as pd

MATCH_KEYS = ["cell_type", "time_point", "replicate", "plate_oligo", "dose_pattern"]
REQUIRED = MATCH_KEYS + ["well_oligo", "Combo", "treatment", "vehicle", "dose"]


def select_design(metadata, cell_type="A549", time_hours=24, dose_nM=1000):
    """Return target-dose drug wells and ALL same-cell/time vehicle wells."""
    d = metadata.copy()
    missing = set(REQUIRED) - set(d.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    if d["Combo"].isna().any() or d["Combo"].duplicated().any():
        raise ValueError("Combo must be unique and nonmissing")
    expected = d.well_oligo.astype(str) + d.plate_oligo.astype(str)
    if not d.Combo.astype(str).eq(expected).all():
        raise ValueError("Combo differs from well_oligo + plate_oligo")
    truth = d.vehicle.astype(str).str.lower()
    if not truth.isin(["true", "false"]).all():
        raise ValueError("vehicle must be explicit TRUE/FALSE")
    d["vehicle"] = truth.eq("true")
    d["dose"] = pd.to_numeric(d.dose, errors="raise")
    d["time_point"] = pd.to_numeric(d.time_point, errors="raise")
    for key in ["cell_type", "replicate", "plate_oligo", "dose_pattern"]:
        if d[key].isna().any():
            raise ValueError(f"Missing matching key {key}")
        d[key] = d[key].astype(str)
    scoped = d[d.cell_type.eq(cell_type) & d.time_point.eq(time_hours)]
    drugs = scoped[~scoped.vehicle & scoped.dose.eq(dose_nM)].copy()
    controls = scoped[scoped.vehicle].copy()
    if drugs.empty or controls.empty:
        raise ValueError("No drugs or vehicle controls in selected scope")
    if drugs.duplicated(["treatment", "replicate"]).any():
        raise ValueError("Multiple wells per treatment/replicate require an explicit policy")
    if not controls.dose.eq(0).all():
        raise ValueError("Expected vehicle control dose zero")
    return drugs.reset_index(drop=True), controls.reset_index(drop=True)


def match_controls(drugs, controls):
    """Require exactly one vehicle for every strict matching key."""
    if controls.duplicated(MATCH_KEYS).any():
        raise ValueError("Ambiguous control: multiple wells for strict matching key")
    right = controls[MATCH_KEYS + ["Combo"]].rename(columns={"Combo": "control_Combo"})
    mapped = drugs.merge(right, how="left", on=MATCH_KEYS, validate="many_to_one")
    if mapped.control_Combo.isna().any():
        bad = mapped.loc[mapped.control_Combo.isna(), "Combo"].tolist()
        raise ValueError(f"Missing strict control for {bad}")
    if len(mapped) != len(drugs):
        raise ValueError("Control join changed treatment-well denominator")
    return mapped


def qualify_wells(mapping, cell_counts, minimum_cells=30):
    """Exclude an entire treatment unless BOTH replicate/control pairs qualify."""
    if minimum_cells < 1:
        raise ValueError("Minimum cell count must be positive")
    counts = pd.Series(cell_counts, dtype=float)
    if counts.index.duplicated().any() or not np.isfinite(counts.to_numpy()).all():
        raise ValueError("Cell counts must be unique and finite")
    if (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Cell counts must be nonnegative integers")
    d = mapping.copy()
    d["treatment_cells"] = d.Combo.map(counts).fillna(0).astype(int)
    d["control_cells"] = d.control_Combo.map(counts).fillna(0).astype(int)
    d["eligible"] = False
    d["exclusion_reason"] = ""
    for _, group in d.groupby("treatment", sort=True):
        reasons = []
        if len(group) != 2 or set(group.replicate) != {"rep1", "rep2"}:
            reasons.append("missing_or_duplicate_replicate")
        if (group.treatment_cells < minimum_cells).any():
            reasons.append("low_treatment_cells")
        if (group.control_cells < minimum_cells).any():
            reasons.append("low_control_cells")
        d.loc[group.index, "eligible"] = not reasons
        d.loc[group.index, "exclusion_reason"] = ";".join(reasons)
    return d
