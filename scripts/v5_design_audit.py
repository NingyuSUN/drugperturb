"""Build (or check) the Tahoe-100M v5 design manifests from metadata only; never reads expression.

Reads, at the pinned revision and after verifying each file's SHA256 against
manifests/v5/source.tsv:

- sample_metadata: the design columns sample, plate, drug, drugname_drugconc
  (not the mean_* columns, which summarize expression);
- drug_metadata: drug, canonical_smiles, pubchem_cid;
- cell_line_metadata: cell_name, Cell_ID_Cellosaur, Organ;
- obs_metadata: sample, cell_line, pass_filter, for rows of vehicle samples only
  (the row filter is applied in the parquet read). No cell count of a compound
  sample is computed.

Standardizes structures with the v1 policy and v1 duplicate-parent rule, splits
whole scaffold groups (protocol_v5 split), counts vehicle cells per plate and
line, applies the v5 cell-line rule and writes manifests/v5. ``--check``
regenerates the tables and requires them to equal the committed manifests.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import platform
import sys

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import yaml
from rdkit import __version__ as RDKIT_VERSION

from drugperturb.chemistry import standardize_smiles
from drugperturb.v5.design import (CELL_LINE_COLUMNS, DRUG_COLUMNS, SAMPLE_COLUMNS, build_samples, compound_table,
                                   context_table, replicate_table, select_cell_lines, split_compounds,
                                   vehicle_units)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'manifests/v5'
OBS_COLUMNS = ('sample', 'cell_line', 'pass_filter')
PASS_VALUE = 'full'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(16*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verified_paths(metadata_dir):
    source = pd.read_csv(OUT/'source.tsv', sep='\t', dtype=str)
    paths, checks = {}, []
    for row in source.itertuples(index=False):
        path = metadata_dir/Path(row.path).name
        digest = sha256(path)
        if path.stat().st_size != int(row.bytes) or digest != row.sha256:
            raise SystemExit(f'{path} does not match manifests/v5/source.tsv')
        paths[row.role] = path
        checks.append(dict(role=row.role, bytes=int(row.bytes), sha256=digest, revision=row.revision))
    return paths, checks


def sciplex_and_op3_parents():
    """Canonical parents of the compounds scored in v1-v4, re-standardized in this environment."""
    sciplex = pd.read_csv(ROOT/'manifests/chemistry_manifest.tsv', sep='\t', keep_default_na=False)
    op3 = pd.read_csv(ROOT/'manifests/v4/op3_compounds.tsv', sep='\t', keep_default_na=False)
    parents = {}
    for name, smiles in [('sciplex3', sciplex.loc[sciplex.status.eq('accepted'), 'original_smiles']),
                         ('op3', op3.original_smiles)]:
        parents[name] = {standardize_smiles(s)['canonical_smiles'] for s in smiles}
    return parents


def stereo_free(smiles):
    from rdkit import Chem
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)


def tables(paths, protocol):
    spec = protocol['split']
    selection = protocol['dataset']['contexts']['cell_line_selection']
    rule = dict(min_median_cells=selection['min_median_vehicle_cells'], n_lines=selection['n_lines'],
                seed=selection['seed'])
    samples = build_samples(pd.read_parquet(paths['sample_metadata'], columns=list(SAMPLE_COLUMNS)))
    drugs = pd.read_parquet(paths['drug_metadata'], columns=list(DRUG_COLUMNS))
    compounds, report = split_compounds(compound_table(drugs, samples), spec)
    parents = sciplex_and_op3_parents()
    accepted = compounds.status.eq('accepted')
    for name, known in parents.items():
        compounds[f'in_{name}'] = accepted & compounds.canonical_smiles.isin(known)
    replicates = replicate_table(samples, compounds)
    lines = pd.read_parquet(paths['cell_line_metadata'], columns=list(CELL_LINE_COLUMNS)).drop_duplicates()
    if lines.Cell_ID_Cellosaur.duplicated().any():
        raise ValueError('a Cellosaurus ID maps to more than one name or organ')
    vehicle = samples.loc[samples.role.eq('vehicle'), 'sample'].tolist()
    obs = pq.read_table(paths['obs_metadata'], columns=list(OBS_COLUMNS),
                        filters=[('sample', 'in', vehicle)]).to_pandas()
    if not obs['sample'].isin(vehicle).all():
        raise ValueError('the obs read returned rows of non-vehicle samples')
    for name in ('cell_line', 'pass_filter'):
        obs[name] = obs[name].astype(str)
    # cell_line_metadata also lists lines that are not in Tahoe-100M; the study's lines are those in its wells.
    present = sorted(obs.cell_line.unique())
    if not set(present) <= set(lines.Cell_ID_Cellosaur):
        raise ValueError('vehicle cells carry cell lines missing from cell_line_metadata')
    passing = obs[obs.pass_filter.eq(PASS_VALUE)].rename(columns={'cell_line': 'cell_line_id'})
    units = vehicle_units(passing, samples, present)
    cell_lines, chosen = select_cell_lines(units, **rule)
    cell_lines = cell_lines.merge(lines.rename(columns={'Cell_ID_Cellosaur': 'cell_line_id'}), on='cell_line_id',
                                  how='left')[['cell_line_id', 'cell_name', 'Organ', 'median_vehicle_cells',
                                               'eligible', 'draw_order', 'selected']]
    contexts = context_table(chosen)
    facts = design_facts(samples, compounds, replicates, obs, units, parents, report, rule)
    return dict(samples=samples, compounds=compounds, replicates=replicates, vehicle_units=units,
                cell_lines=cell_lines, contexts=contexts), report, facts


def design_facts(samples, compounds, replicates, obs, units, parents, report, rule):
    roles = samples.role.value_counts().to_dict()
    compound_samples = samples[samples.role.eq('compound')]
    per_plate = pd.crosstab(compound_samples.plate, compound_samples.dose_nM)
    per_plate = per_plate.loc[sorted(per_plate.index, key=lambda p: int(p[5:]))]
    accepted = compounds[compounds.status.eq('accepted')]
    excluded = compounds[compounds.status.eq('excluded')]
    duplicate = excluded.exclusion_reason.str.startswith('duplicate_parent')
    stereo_free_known = {name: {stereo_free(s) for s in known} for name, known in parents.items()}
    overlap = {}
    for name in parents:
        hit = accepted[accepted[f'in_{name}']]
        loose = accepted[accepted.canonical_smiles.map(stereo_free).isin(stereo_free_known[name])]
        overlap[name] = dict(exact_parent=len(hit), exact_parent_by_split=hit.split.value_counts().to_dict(),
                             stereo_free_parent=len(loose), compounds=sorted(hit.compound_id))
    plates_per_compound = (compound_samples[compound_samples.drug.isin(accepted.compound_id)]
                           .groupby('drug').plate.nunique())
    return dict(
        samples=len(samples), plates=samples.plate.nunique(), roles=roles,
        vehicle_samples_per_plate=samples[samples.role.eq('vehicle')].groupby('plate').size().unique().tolist(),
        reference_control_samples=samples[samples.role.eq('reference_control')]
        .groupby('dose_nM').size().to_dict(),
        compound_samples_by_plate_and_dose_nM=per_plate.to_dict(orient='index'),
        drugs_in_compound_samples=int(compound_samples.drug.nunique()),
        compounds=dict(accepted=len(accepted), excluded_structure=int((~duplicate).sum()),
                       excluded_duplicate_parent=int(duplicate.sum()),
                       scaffold_groups=int(accepted.scaffold_group.nunique()),
                       acyclic_groups=int(accepted.scaffold_group.str.startswith('acyclic:').sum()),
                       exclusions=excluded[['compound_id', 'exclusion_reason']].to_dict('records')),
        replicate_samples_per_compound_dose=replicates.n_samples.value_counts().sort_index().to_dict(),
        plates_per_accepted_compound=plates_per_compound.value_counts().sort_index().to_dict(),
        overlap_with_scored=overlap,
        vehicle_cells=dict(rows_read=len(obs), samples_read=int(obs['sample'].nunique()),
                           cell_lines=int(units.cell_line_id.nunique()),
                           pass_filter=obs.pass_filter.value_counts().to_dict(), counted_value=PASS_VALUE,
                           unit_cells=units.vehicle_cells.describe().round(1).to_dict(),
                           units_below_30=int(units.vehicle_cells.lt(30).sum()),
                           units_below_200=int(units.vehicle_cells.lt(200).sum())),
        cell_line_rule=rule)


def roundtrip(frame):
    """Serialize and re-read so dtype differences cannot hide or invent mismatches."""
    buffer = io.StringIO()
    frame.to_csv(buffer, sep='\t', index=False)
    buffer.seek(0)
    return pd.read_csv(buffer, sep='\t', keep_default_na=False)


def jsonable(value):
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-dir', type=Path, required=True)
    parser.add_argument('--check', action='store_true', help='compare with committed manifests instead of writing')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    protocol = yaml.safe_load((ROOT/'configs/protocol_v5.yaml').read_text())
    paths, checks = verified_paths(args.metadata_dir)
    frames, report, facts = tables(paths, protocol)
    if args.check:
        mismatches = {name: not pd.read_csv(OUT/f'{name}.tsv', sep='\t', keep_default_na=False)
                      .equals(roundtrip(frame)) for name, frame in frames.items()}
        if any(mismatches.values()):
            raise SystemExit(f'regenerated design differs from committed manifests: {mismatches}')
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        for name, frame in frames.items():
            frame.to_csv(OUT/f'{name}.tsv', sep='\t', index=False)
        (OUT/'split_report.json').write_text(json.dumps(report, indent=2)+'\n')
    code = [Path(__file__), ROOT/'src/drugperturb/v5/design.py', ROOT/'src/drugperturb/v4/design.py',
            ROOT/'src/drugperturb/chemistry.py']
    receipt = dict(
        analysis='v5_design_audit', mode='check' if args.check else 'write', inputs_verified=checks,
        reads=dict(sample_metadata=list(SAMPLE_COLUMNS), drug_metadata=list(DRUG_COLUMNS),
                   cell_line_metadata=list(CELL_LINE_COLUMNS),
                   obs_metadata=dict(columns=list(OBS_COLUMNS), row_filter='sample in the 28 vehicle samples')),
        expression_read=False, compound_sample_cell_counts_read=False, split=report, design=facts,
        outputs={f'manifests/v5/{name}.tsv': hashlib.sha256((OUT/f'{name}.tsv').read_bytes()).hexdigest()
                 for name in frames},
        code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in code},
        versions=dict(python=sys.version.split()[0], platform=platform.platform(), pandas=pd.__version__,
                      numpy=np.__version__, pyarrow=pyarrow.__version__, rdkit=RDKIT_VERSION),
        command=' '.join(['python', *sys.argv]))
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, default=jsonable)+'\n')
    print(json.dumps(dict(split=report, compounds=facts['compounds'] | {'exclusions': len(facts['compounds']['exclusions'])},
                          selected_lines=frames['contexts'].cell_line_id.unique().tolist()), indent=2, default=jsonable))


if __name__ == '__main__':
    main()
