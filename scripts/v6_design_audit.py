"""Draw the v6 cell lines from the committed v5 design audit; reads no Tahoe-100M file.

The v5 audit already counted vehicle cells for all 50 lines on all 14 plates
(manifests/v5/vehicle_units.tsv, manifests/v5/cell_lines.tsv), so the v6 line
draw needs no new read of the dataset. Both inputs are verified against the
SHA256 recorded in receipts/v5/design_audit.json before they are used.

Writes manifests/v6 - the 50-line table with the pool and the draw, the drawn
lines' vehicle units with the split gate flag, and the 30 contexts - plus a
receipt. ``--check`` regenerates the tables and requires them to equal the
committed manifests (protocol_v6 gate G1).

No expression value, no expression summary and no cell count of a compound
sample is read here, as in v5.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd

from drugperturb.v6.design import (MIN_HALF_CELLS, MIN_VEHICLE_CELLS, MIN_WELL_CELLS, context_table,
                                   gate_feasibility, select_new_cell_lines, vehicle_gate)

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT/'manifests/v5'
OUT = ROOT/'manifests/v6'
V5_RECEIPT = ROOT/'receipts/v5/design_audit.json'
INPUTS = ('cell_lines', 'vehicle_units')
SEED = 6
N_LINES = 10


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_inputs():
    """The v5 manifests this audit reads, each checked against the v5 audit receipt."""
    recorded = json.loads(V5_RECEIPT.read_text())['outputs']
    frames, checks = {}, {}
    for name in INPUTS:
        path = V5/f'{name}.tsv'
        key = f'manifests/v5/{name}.tsv'
        digest = sha256(path)
        expected = recorded.get(key)
        if expected is None:
            raise SystemExit(f'{key} has no SHA256 in {V5_RECEIPT.relative_to(ROOT)}')
        if digest != expected:
            raise SystemExit(f'{key} does not match the v5 receipt: {digest} != {expected}')
        frames[name] = pd.read_csv(path, sep='\t')
        checks[key] = digest
    return frames, checks


def build(frames):
    v5_lines = sorted(frames['cell_lines'].loc[frames['cell_lines'].selected, 'cell_line_id'])
    lines, chosen = select_new_cell_lines(frames['vehicle_units'], exclude=v5_lines, n_lines=N_LINES, seed=SEED)
    gate = vehicle_gate(frames['vehicle_units'], chosen)
    names = frames['cell_lines'].set_index('cell_line_id')
    lines = lines.assign(cell_name=lines.cell_line_id.map(names.cell_name), Organ=lines.cell_line_id.map(names.Organ))
    lines = lines[['cell_line_id', 'cell_name', 'Organ', 'median_vehicle_cells', 'eligible', 'scored_in_v5',
                   'in_pool', 'draw_order', 'selected']]
    gate = gate.assign(cell_name=gate.cell_line_id.map(names.cell_name))
    per_line, overall = gate_feasibility(gate)
    frames_out = dict(cell_lines=lines, vehicle_units=gate, contexts=context_table(chosen))
    facts = dict(
        v5_lines=v5_lines, pool=int(lines.in_pool.sum()), eligible=int(lines.eligible.sum()),
        drawn=[dict(cell_line_id=c, cell_name=names.cell_name[c], organ=names.Organ[c],
                    median_vehicle_cells=float(names.median_vehicle_cells[c])) for c in chosen],
        contexts=len(frames_out['contexts']),
        split_gate=dict(min_half_cells=MIN_HALF_CELLS, min_well_cells=MIN_WELL_CELLS,
                        min_vehicle_cells=MIN_VEHICLE_CELLS, **overall, per_line=per_line),
        organs={k: int(v) for k, v in lines.loc[lines.selected, 'Organ'].value_counts().sort_index().items()})
    return frames_out, chosen, facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='regenerate and compare with the committed manifests')
    parser.add_argument('--receipt', type=Path, default=ROOT/'receipts/v6/design_audit.json')
    args = parser.parse_args()
    frames, checks = verified_inputs()
    built, chosen, facts = build(frames)
    if args.check:
        mismatches = {}
        for name, frame in built.items():
            committed = pd.read_csv(OUT/f'{name}.tsv', sep='\t')
            if not frame.reset_index(drop=True).equals(committed.reset_index(drop=True)):
                mismatches[name] = 'differs'
        if mismatches:
            raise SystemExit(f'regenerated design differs from committed manifests: {mismatches}')
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        for name, frame in built.items():
            frame.to_csv(OUT/f'{name}.tsv', sep='\t', index=False)
    code = [Path(__file__), ROOT/'src/drugperturb/v6/design.py', ROOT/'src/drugperturb/v5/design.py']
    receipt = dict(
        analysis='v6_design_audit', mode='check' if args.check else 'write', inputs_verified=checks,
        reads='manifests/v5 cell_lines.tsv and vehicle_units.tsv only; no Tahoe-100M file',
        expression_read=False, compound_sample_cell_counts_read=False,
        draw=dict(seed=SEED, n_lines=N_LINES, rule='v5 mechanism over the v5-eligible lines v5 did not select'),
        design=facts, selected_lines=chosen,
        outputs={f'manifests/v6/{name}.tsv': sha256(OUT/f'{name}.tsv') for name in built},
        code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in code},
        versions=dict(python=sys.version.split()[0], platform=platform.platform(), pandas=pd.__version__,
                      numpy=np.__version__),
        command=' '.join(['python', *sys.argv]))
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(drawn=facts['drawn'], pool=facts['pool'], contexts=facts['contexts'],
                          organs=facts['organs'], split_gate={k: v for k, v in facts['split_gate'].items()
                                                              if k != 'per_line'}), indent=2))


if __name__ == '__main__':
    main()
