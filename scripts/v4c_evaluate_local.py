"""Score the post hoc corrected OP3 seals locally, after replaying the v4 evaluation (configs/v4_corrected_posthoc.yaml).

Every input must equal its committed hash or seal: the v4 O1 extraction and O2
preparation outputs, the sealed context and anchored baselines, the four sealed v4
arms, and the corrected seals and selection (committed before this runs). The v4
evaluation is first replayed from the v4 seals and compared with reports/v4/op3, to
show that this environment reproduces the Kaggle evaluation. Then the corrected seals
are scored (v4_evaluate_op3.py --descriptor-tolerance), the v4 post hoc exclusion
analysis is repeated on them, and the side-by-side tables are written. The receipt
records commands, versions and input/output hashes.
"""
from pathlib import Path
import argparse
import datetime
import json
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from drugperturb.v3.runner import sha256  # noqa: E402

ARMS = ['A0_v3_reference', 'I1_nb_likelihood', 'I2_counterfactual_pseudobulk', 'I3_nb_plus_pseudobulk']
TABLES = ['per_compound.tsv', 'summary.tsv', 'contrasts.tsv', 'test_diagnostics.tsv']


def committed_seal(local, committed):
    """A local seal must equal the committed one, and its archive must match the sealed hash."""
    seal = json.loads(local.read_text())
    if seal != json.loads(committed.read_text()):
        raise ValueError(f'{local} differs from the committed {committed}')
    if sha256(local.parent/seal['predictions']['path']) != seal['predictions']['sha256']:
        raise ValueError(f'{local}: archive hash differs from the seal')
    return {str(committed.relative_to(ROOT)): seal['predictions']['sha256']}


def verify_inputs(args):
    checked = {}
    prepare = json.loads((ROOT/'receipts/v4/op3/prepare_receipt.json').read_text())
    for name, digest in prepare['outputs'].items():
        if sha256(args.prepared/name) != digest:
            raise ValueError(f'prepared {name} differs from receipts/v4/op3/prepare_receipt.json')
        checked[f'prepared/{name}'] = digest
    if sha256(args.extraction_root/'unit_counts.npz') != prepare['inputs']['unit_counts.npz']:
        raise ValueError('extraction unit_counts.npz differs from the one the preparation read')
    checked['extraction/unit_counts.npz'] = prepare['inputs']['unit_counts.npz']
    checked.update(committed_seal(args.baselines_sealed/'context_baselines_op3_seal.json',
                                  ROOT/'receipts/v4/op3/context_baselines_op3_seal.json'))
    checked.update(committed_seal(args.anchored_sealed/'anchored_baselines_op3_seal.json',
                                  ROOT/'receipts/v4/anchored/anchored_baselines_op3_seal.json'))
    for root, committed in [(args.v4_sealed, 'receipts/v4/op3/sealed'), (args.corrected_sealed, 'receipts/v4_corrected/op3/sealed')]:
        for arm in ARMS:
            checked.update(committed_seal(root/f'{arm}_op3_seal.json', ROOT/committed/f'{arm}_op3_seal.json'))
        if json.loads((root/'selection_op3.json').read_text()) != json.loads((ROOT/committed/'selection_op3.json').read_text()):
            raise ValueError(f'{root}/selection_op3.json differs from {committed}')
        checked[f'{committed}/selection_op3.json'] = sha256(ROOT/committed/'selection_op3.json')
    return checked


def compare(replay, committed):
    """Byte identity per table, else the largest numeric difference; summary.json compared as data."""
    result = {}
    for name in TABLES:
        a, b = (replay/name).read_bytes(), (committed/name).read_bytes()
        if a == b:
            result[name] = dict(bytes_identical=True)
            continue
        x, y = (pd.read_csv(p/name, sep='\t', keep_default_na=False, na_values=['']) for p in (replay, committed))
        numeric = x.select_dtypes('number').columns
        same_text = x.drop(columns=numeric).astype(str).equals(y.drop(columns=numeric).astype(str))
        diff = float(np.nanmax(np.abs(x[numeric].to_numpy(float) - y[numeric].to_numpy(float)))) if len(numeric) else 0.0
        result[name] = dict(bytes_identical=False, same_shape=x.shape == y.shape, text_columns_equal=same_text,
                            max_abs_numeric_difference=diff)
    a, b = (json.loads((p/'summary.json').read_text()) for p in (replay, committed))
    result['summary.json'] = dict(equal_as_data=a == b)
    return result


def run(command, log):
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print('$', ' '.join(map(str, command)), flush=True)
    done = subprocess.run(list(map(str, command)), cwd=ROOT, capture_output=True, text=True,
                          env={'PYTHONPATH': str(ROOT/'src'), 'PYTHONDONTWRITEBYTECODE': '1', 'PATH': '/usr/bin:/bin'})
    log.append(dict(started_utc=started, command=[str(c) for c in command], returncode=done.returncode,
                    stdout_tail=done.stdout[-2000:], stderr_tail=done.stderr[-2000:]))
    if done.returncode:
        raise SystemExit(f'command failed: {command}\n{done.stdout[-3000:]}{done.stderr[-3000:]}')
    return done.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extraction-root', type=Path, required=True, help='dp-v4-op3-extract output extraction/')
    parser.add_argument('--prepared', type=Path, required=True, help='dp-v4-op3-prepare output prepared/')
    parser.add_argument('--baselines-sealed', type=Path, required=True, help='dp-v4-op3-prepare output sealed/')
    parser.add_argument('--anchored-sealed', type=Path, required=True, help='dp-v4-anchored output sealed/')
    parser.add_argument('--v4-sealed', type=Path, required=True, help='dp-v4-fit-op3 version 2 output sealed/')
    parser.add_argument('--corrected-sealed', type=Path, required=True, help='dp-v4c-fit-op3 output sealed/')
    parser.add_argument('--replay-root', type=Path, required=True, help='scratch directory for the v4 replay')
    parser.add_argument('--commit', required=True, help='repository commit the scripts ran from')
    args = parser.parse_args()
    report = ROOT/'reports/v4_corrected/op3'
    receipt = ROOT/'receipts/v4_corrected/evaluate_local.json'
    if (report/'summary.json').exists() or args.replay_root.exists():
        raise ValueError('refusing to overwrite an evaluation output or replay directory')
    inputs = verify_inputs(args)
    log = []
    shared = ['--extraction-root', args.extraction_root, '--prepared', args.prepared,
              '--sealed-root', args.baselines_sealed, '--sealed-root', args.anchored_sealed]
    run([sys.executable, 'scripts/v4_evaluate_op3.py', *shared, '--sealed-root', args.v4_sealed,
         '--output-root', args.replay_root], log)
    replay = compare(args.replay_root, ROOT/'reports/v4/op3')
    reproduced = replay['summary.json']['equal_as_data'] and all(
        v['bytes_identical'] or (v['same_shape'] and v['text_columns_equal'] and v['max_abs_numeric_difference'] <= 1e-9)
        for k, v in replay.items() if k != 'summary.json')
    print(json.dumps(replay, indent=1), flush=True)
    if not reproduced:
        raise SystemExit('the local replay does not reproduce reports/v4/op3; corrected scoring not run')
    run([sys.executable, 'scripts/v4_evaluate_op3.py', *shared, '--sealed-root', args.corrected_sealed,
         '--output-root', report, '--descriptor-tolerance'], log)
    run([sys.executable, 'scripts/v4_posthoc_op3.py', report/'per_compound.tsv', ROOT/'scripts',
         report/'posthoc_sensitivity.tsv'], log)
    run([sys.executable, 'scripts/v4c_side_by_side.py', '--v4-report', 'reports/v4/op3', '--corrected-report', report,
         '--v4-selection', 'receipts/v4/op3/sealed/selection_op3.json',
         '--corrected-selection', 'receipts/v4_corrected/op3/sealed/selection_op3.json',
         '--output-root', report/'side_by_side'], log)
    import scipy
    import torch
    outputs = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(report.rglob('*')) if p.is_file()}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(dict(
        analysis='v4_corrected_posthoc', plan='configs/v4_corrected_posthoc.yaml', commit=args.commit,
        machine=dict(platform=platform.platform(), python=sys.version, processor=platform.processor()),
        packages=dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, torch=torch.__version__),
        input_paths={k: str(v) for k, v in vars(args).items() if k != 'commit'}, inputs_verified=inputs,
        v4_replay=dict(compared_with='reports/v4/op3', reproduced=reproduced, files=replay),
        commands=log, outputs=outputs, test_outcomes_previously_opened_by='dp-v4-evaluate version 1',
        written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=2)+'\n')
    print(f'wrote {receipt.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
