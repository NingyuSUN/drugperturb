"""Stage the code dataset and kernels for Kaggle and push them with the kaggle CLI.

Usage (credentials come from the user's own kaggle CLI configuration):
  python kaggle/push.py code              # create or version the private code dataset
  python kaggle/push.py kernel k1         # push one kernel: k1, k2, k3-chemcpa, k3-biolord, k5
  python kaggle/push.py kernel k3-chemcpa --smoke 2
Everything is private. Raw data, models and outputs never enter the code archive.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get('KAGGLE_BUILD_ROOT', ROOT/'work/kaggle_build'))/ROOT.name
EXCLUDE = ('data/', 'models/', 'work/', 'logs/', '.git/')
KERNELS = {
    'k1': dict(slug='dp-v3-extract', file='k1_extract.py', language='python', gpu=False, sources=[]),
    'k2': dict(slug='dp-v3-prepare', file='k2_prepare.py', language='python', gpu=False, sources=['dp-v3-extract']),
    'k3-chemcpa': dict(slug='dp-v3-chemcpa', file='k3_fit.py', language='python', gpu=True, sources=['dp-v3-prepare']),
    'k3-biolord': dict(slug='dp-v3-biolord', file='k3_fit.py', language='python', gpu=True, sources=['dp-v3-prepare']),
    'k3d': dict(slug='dp-v3-diagnose', file='k3d_diagnose.py', language='python', gpu=False, sources=['dp-v3-prepare']),
    'k4': dict(slug='dp-v3-anchors', file='k4_anchors.py', language='python', gpu=False,
               sources=['dp-v3-extract', 'dp-v3-prepare']),
    'k5': dict(slug='dp-v3-evaluate', file='k5_evaluate.py', language='python', gpu=False,
               sources=['dp-v3-extract', 'dp-v3-prepare', 'dp-v3-chemcpa', 'dp-v3-biolord']),
    # v4: separate code dataset so the v3 kernels keep their code version.
    'v4-o1': dict(slug='dp-v4-op3-extract', file='v4_op3_extract.py', language='python', gpu=False, sources=[],
                  code='drugperturb-v4-code'),
    'v4-o2': dict(slug='dp-v4-op3-prepare', file='v4_op3_prepare.py', language='python', gpu=False,
                  sources=['dp-v4-op3-extract'], code='drugperturb-v4-code'),
    'v4-fit-op3': dict(slug='dp-v4-fit-op3', file='v4_fit.py', language='python', gpu=True,
                       sources=['dp-v4-op3-prepare'], code='drugperturb-v4-code'),
    'v4-fit-sciplex': dict(slug='dp-v4-fit-sciplex', file='v4_fit.py', language='python', gpu=True,
                           sources=['dp-v3-prepare'], code='drugperturb-v4-code'),
    'v4-anchored': dict(slug='dp-v4-anchored', file='v4_anchored.py', language='python', gpu=False,
                        sources=['dp-v4-op3-prepare', 'dp-v3-prepare'], code='drugperturb-v4-code'),
    'v4-k5': dict(slug='dp-v4-evaluate', file='v4_evaluate.py', language='python', gpu=False,
                  sources=['dp-v4-op3-extract', 'dp-v4-op3-prepare', 'dp-v4-fit-op3', 'dp-v4-fit-sciplex',
                           'dp-v4-anchored', 'dp-v3-extract', 'dp-v3-prepare', 'dp-v3-chemcpa'],
                  code='drugperturb-v4-code'),
}
CODE_DATASETS = ('drugperturb-v3-code', 'drugperturb-v4-code')


def kaggle(*args):
    print('$ kaggle', *args, flush=True)
    return subprocess.run(['kaggle', *args], check=True, text=True, capture_output=True).stdout


def username():
    for line in kaggle('config', 'view').splitlines():
        if line.strip().startswith('- username:'):
            return line.split(':', 1)[1].strip()
    raise SystemExit('kaggle CLI has no configured username')


def tracked_files():
    listed = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT,
                            check=True, text=True, capture_output=True).stdout.splitlines()
    return [p for p in listed if not p.startswith(EXCLUDE) and (ROOT/p).is_file()]


def push_code(user, dataset='drugperturb-v3-code'):
    target = BUILD/dataset
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target/'drugperturb-src.archive', 'w:gz') as tar:
        for path in tracked_files():
            tar.add(ROOT/path, arcname=path)
    meta = dict(title=dataset, id=f'{user}/{dataset}', licenses=[dict(name='other')])
    (target/'dataset-metadata.json').write_text(json.dumps(meta, indent=2))
    exists = dataset in kaggle('datasets', 'list', '--mine', '--search', dataset)
    if exists:
        print(kaggle('datasets', 'version', '-p', str(target), '-m', 'code update', '-q', '-t'))
    else:
        print(kaggle('datasets', 'create', '-p', str(target), '-q', '-t'))


def push_kernel(user, name, smoke):
    spec = KERNELS[name]
    target = BUILD/name
    target.mkdir(parents=True, exist_ok=True)
    text = (ROOT/'kaggle'/spec['file']).read_text()
    if spec['file'] == 'k3_fit.py':
        text = text.replace("'__METHOD__'", repr(name.split('-')[1])).replace('__SMOKE__', repr(smoke))
    if spec['file'] == 'v4_fit.py':
        text = text.replace("'__DATASET__'", repr(name.split('-')[-1])).replace('__SMOKE__', repr(smoke))
    (target/spec['file']).write_text(text)
    meta = dict(id=f'{user}/{spec["slug"]}', title=spec['slug'], code_file=spec['file'], language=spec['language'],
                kernel_type='script', is_private=True, enable_gpu=spec['gpu'], enable_internet=True,
                dataset_sources=[f'{user}/{spec.get("code", "drugperturb-v3-code")}'],
                kernel_sources=[f'{user}/{s}' for s in spec['sources']], competition_sources=[])
    if spec['gpu']:
        meta['machine_shape'] = 'NvidiaTeslaT4'
    (target/'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    print(kaggle('kernels', 'push', '-p', str(target)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('what', choices=['code', 'kernel'])
    parser.add_argument('name', nargs='?', choices=list(KERNELS))
    parser.add_argument('--smoke', type=int, help='engineering smoke epochs for k3 kernels')
    parser.add_argument('--dataset', choices=CODE_DATASETS, default='drugperturb-v3-code', help='code dataset to version')
    args = parser.parse_args()
    user = username()
    if args.what == 'code':
        push_code(user, args.dataset)
    else:
        push_kernel(user, args.name, args.smoke)


if __name__ == '__main__':
    main()
