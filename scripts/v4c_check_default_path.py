"""Check that the default v4 OP3 path is unchanged by the --descriptor-tolerance option (synthetic fixture).

Runs the synthetic OP3 fixture of tests/test_v4_op3_pipeline.py through fit (all 12
arm/seed fits, 2 epochs), seal and evaluation twice, without the option: once with the
scripts of a base commit and once with the scripts under test, both on the same
src/ and prepared inputs. Prediction arrays, histories and evaluation tables must be
identical; JSON records may differ only in wall-clock fields and file paths.
Software check only; needs descriptastorus and torch (cell-model environment).
"""
from pathlib import Path
import argparse
import json
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARMS = ['A0_v3_reference', 'I1_nb_likelihood', 'I2_counterfactual_pseudobulk', 'I3_nb_plus_pseudobulk']
WALL_CLOCK = {'seconds', 'minutes'}
EVALUATION = ['per_compound.tsv', 'summary.tsv', 'contrasts.tsv', 'test_diagnostics.tsv']


def run(repo, script, *args):
    env = {'PYTHONPATH': str(ROOT/'src'), 'PYTHONDONTWRITEBYTECODE': '1', 'PATH': '/usr/bin:/bin'}
    done = subprocess.run([sys.executable, str(repo/'scripts'/script), *map(str, args)], cwd=repo, env=env,
                          capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f'{repo.name} {script} failed\n{done.stdout[-3000:]}{done.stderr[-3000:]}')


def differences(a, b, path=''):
    """Paths at which two JSON values differ, ignoring wall-clock keys; paths are reported, not ignored."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key in WALL_CLOCK:
                continue
            if key not in a or key not in b:
                yield f'{path}/{key}'
            else:
                yield from differences(a[key], b[key], f'{path}/{key}')
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            yield from differences(x, y, f'{path}[{i}]')
    elif a != b:
        yield path


def arrays_equal(a, b):
    with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
        return x.files == y.files and all(np.array_equal(x[k], y[k]) for k in x.files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True, help='scripts/ and configs/ of the base commit')
    parser.add_argument('--new', type=Path, required=True, help='scripts/ and configs/ under test')
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT/'tests'))
    from test_v4_op3_pipeline import synthetic_op3
    if args.work.exists():
        raise ValueError(f'{args.work} exists')
    args.work.mkdir(parents=True)
    for repo in (args.base, args.new):  # the fixture's lowered minimums, as in the pipeline test
        protocol = yaml.safe_load((repo/'configs/protocol_v4.yaml').read_text())
        protocol['datasets']['op3']['split'].update(min_test_compounds=2, min_test_groups=2)
        protocol['datasets']['op3']['gene_space']['n_genes'] = 40
        (repo/'configs/protocol_v4.yaml').write_text(yaml.safe_dump(protocol, sort_keys=False))
    w = args.work
    synthetic_op3(w/'op3.h5ad')
    run(args.base, 'v4_op3_design.py', '--h5ad', w/'op3.h5ad')
    shutil.copytree(args.base/'manifests', args.new/'manifests', dirs_exist_ok=True)
    run(args.base, 'v4_op3_extract.py', '--h5ad', w/'op3.h5ad', '--output-root', w/'ext', '--rows-per-block', 1000)
    run(args.base, 'v4_op3_prepare.py', '--extraction-root', w/'ext', '--output-root', w/'prep')
    run(args.base, 'v4_context_baselines.py', '--prepared', w/'prep', '--output-root', w/'sealed_base')
    run(args.base, 'v4_anchored_baselines.py', '--setting', 'op3', '--prepared', w/'prep', '--baselines-root',
        w/'sealed_base', '--output-root', w/'sealed_anchored')
    seeds = protocol['training']['seeds']['op3']
    for version, repo in [('base', args.base), ('new', args.new)]:
        for arm in ARMS:
            for seed in seeds:
                run(repo, 'v4_fit.py', '--setting', 'op3', '--arm', arm, '--seed', seed, '--prepared', w/'prep',
                    '--output-root', w/version, '--device', 'cpu', '--max-epochs', 2)
        run(repo, 'v4_seal.py', '--setting', 'op3', '--fit-root', w/version/'fits', '--prepared', w/'prep',
            '--output-root', w/version/'sealed', '--allow-smoke')
        run(repo, 'v4_evaluate_op3.py', '--extraction-root', w/'ext', '--prepared', w/'prep', '--sealed-root',
            w/'sealed_base', '--sealed-root', w/version/'sealed', '--sealed-root', w/'sealed_anchored',
            '--output-root', w/version/'eval')
    checks, json_differences = {}, set()

    def compare_json(name, a, b):
        found = list(differences(json.loads(a.read_text()), json.loads(b.read_text())))
        json_differences.update(re.sub(r'\[\d+\]', '[]', p) for p in found)
        # Paths of inputs differ (two repositories, two output roots); so do archive hashes, because
        # np.savez_compressed stamps the zip entries with the write time. Contents are compared directly.
        checks[name] = all(p.endswith('/path') or p.endswith('/sha256') or p.startswith('/seals/') for p in found)

    for arm in ARMS:
        for seed in seeds:
            name = f'{arm}_op3_s{seed}'
            a, b = w/'base/fits', w/'new/fits'
            checks[f'fit {name}: arrays'] = arrays_equal(a/f'{name}.npz', b/f'{name}.npz')
            ha, hb = (pd.read_csv(p/f'{name}_history.tsv', sep='\t') for p in (a, b))
            checks[f'fit {name}: history except minutes'] = ha.drop(columns='minutes').equals(hb.drop(columns='minutes'))
            compare_json(f'fit {name}: record except wall clock', a/f'{name}.json', b/f'{name}.json')
            checks[f'fit {name}: no corrected labels'] = not {'analysis', 'descriptor_min_std'} & set(
                json.loads((b/f'{name}.json').read_text()))
        checks[f'seal {arm}: arrays'] = arrays_equal(w/'base/sealed'/f'{arm}_op3_predictions.npz',
                                                      w/'new/sealed'/f'{arm}_op3_predictions.npz')
        compare_json(f'seal {arm}: json except wall clock, paths, archive hashes',
                     w/'base/sealed'/f'{arm}_op3_seal.json', w/'new/sealed'/f'{arm}_op3_seal.json')
    compare_json('selection except archive hashes', w/'base/sealed/selection_op3.json', w/'new/sealed/selection_op3.json')
    for name in EVALUATION:
        checks[f'evaluation {name}: bytes'] = (w/'base/eval'/name).read_bytes() == (w/'new/eval'/name).read_bytes()
    compare_json('evaluation summary.json except seal hashes', w/'base/eval/summary.json', w/'new/eval/summary.json')
    result = dict(unchanged=all(checks.values()), checks=checks, json_paths_that_differ=sorted(json_differences),
                  ignored_keys=sorted(WALL_CLOCK), fits_per_version=len(ARMS)*len(seeds), max_epochs=2,
                  python=sys.version, fixture='tests/test_v4_op3_pipeline.py synthetic_op3')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(unchanged=result['unchanged'], json_paths_that_differ=result['json_paths_that_differ']), indent=1))
    if not result['unchanged']:
        raise SystemExit('the default path changed')


if __name__ == '__main__':
    main()
